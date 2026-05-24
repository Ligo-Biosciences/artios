"""
Simple modulate function for absorbing AdaLN into the DiT block.

Do we combine diffusion transformer and transfomer?

Deepmind's pattern is one custom class and one custom forward for every specific function. There are no if else conditional flows. This is the pattern
we follow.

Cross attention and self attention are separated.
"""

from dataclasses import dataclass
from functools import partial

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from artios.model.modules.components import Attention, SwiGLU
from artios.model.modules.layout import (
    rearrange_to_key_subset,
    rearrange_to_query_subset,
)
from artios.model.modules.primitives import BiasInitLinear, LayerNorm, LinearNoBias
from artios.utils.typing import Bool, Float, typecheck


@dataclass(frozen=True, slots=True, kw_only=True)
class DiffusionTransformerConfig:
    """Architecture settings for the global diffusion transformer."""

    num_blocks: int = 16
    num_heads: int = 16

    def __post_init__(self) -> None:
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be greater than 0")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be greater than 0")


class DiTBlock(nn.Module):
    """Diffusion Transformer Block, used for global attention within DiffusionTransformer."""

    def __init__(self, c_token: int, num_heads: int):
        """
        Args:
            c_token: (int) token embedding dimension
            num_heads: (int) number of attention heads
        """
        super().__init__()

        assert c_token % num_heads == 0, "Input dim should be divisible by num heads."

        self.c_token = c_token
        self.num_heads = num_heads

        # Attention
        self.attn_layernorm = LayerNorm(c_token, elementwise_affine=False)
        self.attention = Attention(
            q_dim=c_token,
            kv_dim=c_token,
            head_dim=c_token // num_heads,
            num_heads=num_heads,
            gating=True,
            final_linear_init="default",
        )
        # Transition
        self.mlp_layernorm = LayerNorm(c_token, elementwise_affine=False)
        self.mlp = nn.Sequential(
            SwiGLU(c_in=c_token, c_out=2 * c_token),
            LinearNoBias(2 * c_token, c_token, init="default"),
        )

    @torch.compile(dynamic=False)
    @typecheck
    def forward(
        self,
        a: Float[Tensor, "... n c_token"],
        *,
        scale_attn: Float[Tensor, "... n c_token"],
        shift_attn: Float[Tensor, "... n c_token"],
        gate_attn: Float[Tensor, "... n c_token"],
        scale_mlp: Float[Tensor, "... n c_token"],
        shift_mlp: Float[Tensor, "... n c_token"],
        gate_mlp: Float[Tensor, "... n c_token"],
        pair_bias: Float[Tensor, "... h n n"],
        pair_mask: Bool[Tensor, "... n n"],
    ) -> Float[Tensor, "... n c_token"]:
        """
        Forward pass of the transformer block.

        Args:
            a: Input token representations.
            scale_attn: attention scaling.
            shift_attn: attention shift.
            gate_attn: attention gating.
            scale_mlp: MLP scaling.
            shift_mlp: MLP shift.
            gate_mlp: MLP gating.
            pair_bias: Pairwise attention bias.
            pair_mask: Pairwise attention mask.
        """
        assert (
            pair_bias.shape[-3] == self.num_heads
        ), f"Pair bias dimension mismatch. Expected {self.num_heads}, got {pair_bias.shape[-3]}"

        # Adaln + Attention
        # shift_attn + scale_attn * LayerNorm(a).
        attn_input = torch.addcmul(shift_attn, scale_attn, self.attn_layernorm(a))
        attn_output = self.attention(
            q_x=attn_input, kv_x=attn_input, pair_bias=pair_bias, pair_mask=pair_mask
        )
        # Gated residual update: a + gate_attn * attn_output.
        a = torch.addcmul(a, gate_attn, attn_output)

        # Adaln + MLP
        # shift_mlp + scale_mlp * LayerNorm(a).
        mlp_input = torch.addcmul(shift_mlp, scale_mlp, self.mlp_layernorm(a))
        mlp_output = self.mlp(mlp_input)

        # a + gate_mlp * mlp_output.
        a = torch.addcmul(a, gate_mlp, mlp_output)
        return a


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        c_token: int,
        c_s: int,
        c_z: int,
        num_heads: int,
        checkpoint_grads: bool = True,
    ):
        """
        Args:
            num_blocks: (int) number of diffusion transformer blocks
            c_token: (int) token embedding dimension
            c_s: (int) condition embedding dimension
            c_z: (int) pair bias embedding dimension
            num_heads: (int) number of attention heads
            checkpoint_grads (bool): whether to use gradient checkpointing during training
        """
        super().__init__()

        assert num_blocks > 0, "Number of blocks must be greater than 0."

        self.num_blocks = num_blocks
        self.c_token = c_token
        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads
        self.checkpoint_grads = checkpoint_grads

        self.condition_layernorm = LayerNorm(c_s)

        # AdaLN parameters
        self.scale_proj = nn.Sequential(
            BiasInitLinear(c_s, 2 * num_blocks * c_token, biasinit=1.0),
            nn.Sigmoid(),
        )
        self.shift_proj = LinearNoBias(c_s, 2 * num_blocks * c_token, init="zeros")

        # Gate parameters
        self.gate_proj = nn.Sequential(
            BiasInitLinear(c_s, 2 * num_blocks * c_token, biasinit=-2.0),
            nn.Sigmoid(),
        )

        # Pair biases
        self.z_layernorm = LayerNorm(c_z)
        self.z_proj = LinearNoBias(c_z, num_heads * num_blocks)

        # DiT blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(c_token, num_heads) for _ in range(num_blocks)]
        )

    def _chunk_block_params(
        self, fused_params: Tensor
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Split fused parameters into MLP and attention components across blocks.

        Args:
            fused_params: Tensor with fused MLP and attention parameters [..., 2*num_blocks*c_token]

        Returns:
            Tuple containing:
                - attn_params: Attention parameters split across blocks
                - mlp_params: MLP parameters split across blocks
        """
        attn_fused, mlp_fused = fused_params.chunk(2, dim=-1)
        attn_params = attn_fused.chunk(self.num_blocks, dim=-1)
        mlp_params = mlp_fused.chunk(self.num_blocks, dim=-1)

        return attn_params, mlp_params

    def _prep_blocks(
        self,
        *,
        scale_attns: tuple[Tensor, ...],
        shift_attns: tuple[Tensor, ...],
        gate_attns: tuple[Tensor, ...],
        scale_mlps: tuple[Tensor, ...],
        shift_mlps: tuple[Tensor, ...],
        gate_mlps: tuple[Tensor, ...],
        pair_biases: tuple[Tensor, ...],
        pair_mask: Tensor,
    ):
        """Prepare the DiT blocks."""
        blocks = [
            partial(
                block,
                scale_attn=scale_attns[i],
                shift_attn=shift_attns[i],
                gate_attn=gate_attns[i],
                scale_mlp=scale_mlps[i],
                shift_mlp=shift_mlps[i],
                gate_mlp=gate_mlps[i],
                pair_bias=pair_biases[i],
                pair_mask=pair_mask,
            )
            for i, block in enumerate(self.blocks)
        ]

        return blocks

    # @profile_function("diffusion_transformer")
    @typecheck
    # @torch.compile
    def forward(
        self,
        a: Float[Tensor, "... n c_token"],
        s: Float[Tensor, "... n c_s"],
        z: Float[Tensor, "... n n c_z"],
        pair_mask: Bool[Tensor, "... n n"],
    ) -> Float[Tensor, "... n c_token"]:
        """
        Args:
            a: single representation
            s: single conditioning
            z: pair representation
            pair_mask: pair mask for the attention

        Notes:
            We currently hand the DiT with an N_sample dimension in its inputs.
            For computational efficiency, z has only a 1 N_sample dimension.
        """

        # Conditioning
        s = self.condition_layernorm(s)

        # Chunk block parameters
        scale_attns, scale_mlps = self._chunk_block_params(self.scale_proj(s))
        shift_attns, shift_mlps = self._chunk_block_params(self.shift_proj(s))
        gate_attns, gate_mlps = self._chunk_block_params(self.gate_proj(s))

        # Project and chunk pair bias
        z = self.z_layernorm(z)
        pair_bias_fused = self.z_proj(z)
        pair_bias_fused = rearrange(pair_bias_fused, "... i j nb -> ... nb i j")
        # TODO: repeat pair_bias_fused so its N_sample dimension is N_sample, not just 1.
        pair_biases = pair_bias_fused.chunk(self.num_blocks, dim=-3)

        # Prepare blocks
        blocks = self._prep_blocks(
            scale_attns=scale_attns,
            shift_attns=shift_attns,
            gate_attns=gate_attns,
            scale_mlps=scale_mlps,
            shift_mlps=shift_mlps,
            gate_mlps=gate_mlps,
            pair_biases=pair_biases,
            pair_mask=pair_mask.bool(),
        )

        # DiT Tower
        for block in blocks:
            if self.checkpoint_grads:
                a = checkpoint(block, a, use_reentrant=False)
            else:
                a = block(a)

        return a


class LocalAttnTransformerBlock(nn.Module):
    """Local Attention Transformer Block, used for local attention within LocalAttnTransformer."""

    def __init__(
        self,
        c_atom: int,
        num_heads: int,
        num_queries: int = 32,
        num_keys: int = 128,
    ):
        """
        Args:
            c_atom: (int) atom embedding dimension
            num_heads: (int) number of attention heads
            num_queries: (int) number of queries
            num_keys: (int) number of keys

        Note:
            if we write a local attention class, we might have better performance due to not replicating
            the key tensor with the rearrangement. We have 4x more keys and values than we need to.
        """
        super().__init__()

        assert c_atom % num_heads == 0, "Input dim should be divisible by num heads."

        self.c_atom = c_atom
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.num_keys = num_keys

        self.attn_layernorm = LayerNorm(c_atom, elementwise_affine=False)
        # Attention
        self.attention = Attention(
            q_dim=c_atom,
            kv_dim=c_atom,
            head_dim=c_atom // num_heads,
            num_heads=num_heads,
            gating=True,
            final_linear_init="default",
        )
        # AdaLN MLP
        self.mlp_layernorm = LayerNorm(c_atom, elementwise_affine=False)
        self.mlp = nn.Sequential(
            SwiGLU(c_in=c_atom, c_out=2 * c_atom),
            LinearNoBias(2 * c_atom, c_atom, init="default"),
        )

    @typecheck
    def forward(
        self,
        q_l: Float[Tensor, "... n_atom c_atom"],
        *,
        scale_attn: Float[Tensor, "... n_atom c_atom"],
        shift_attn: Float[Tensor, "... n_atom c_atom"],
        gate_attn: Float[Tensor, "... n_atom c_atom"],
        scale_mlp: Float[Tensor, "... n_atom c_atom"],
        shift_mlp: Float[Tensor, "... n_atom c_atom"],
        gate_mlp: Float[Tensor, "... n_atom c_atom"],
        trunked_pair_bias: Float[Tensor, "... n_subset h n_q n_k"],
        trunked_pair_mask: Bool[Tensor, "... n_subset n_q n_k"],
    ) -> Float[Tensor, "... n_atom c_atom"]:
        """
        Args:
            q_l: atom representation
            scale_attn: attention scaling
            shift_attn: attention shift
            gate_attn: attention gating
            scale_mlp: MLP scaling
            shift_mlp: MLP shift
            gate_mlp: MLP gating
            trunked_pair_bias: trunked pair bias
            trunked_pair_mask: trunked pair mask
        """
        assert (
            trunked_pair_bias.shape[-3] == self.num_heads
        ), f"Pair bias dimension mismatch. Expected {self.num_heads}, got {trunked_pair_bias.shape[-3]}"

        # AdaLN on q_l first
        # AdaLN modulation: shift_attn + scale_attn * LayerNorm(q_l).
        attn_input = torch.addcmul(shift_attn, scale_attn, self.attn_layernorm(q_l))

        # Rearrange q_l and do attention with trunked
        q_x = rearrange_to_query_subset(
            attn_input, rearrange_dim=-2, num_queries=self.num_queries
        )  # [..., N_subset, N_query, c_atom]
        kv_x = rearrange_to_key_subset(
            attn_input,
            rearrange_dim=-2,
            num_keys=self.num_keys,
            num_queries=self.num_queries,
        )  # [..., n_subset, n_key, c_atom]

        attn_out = self.attention(
            q_x=q_x,
            kv_x=kv_x,
            pair_bias=trunked_pair_bias,
            pair_mask=trunked_pair_mask,
        )  # [..., n_subset, n_query, c_atom]

        # Rearrange back to atom from query layout
        attn_out = rearrange(
            attn_out, "... n_subset n_query c_atom -> ... (n_subset n_query) c_atom"
        )

        # Gate the attention output and add as residual
        # Gated residual update: q_l + gate_attn * attn_out.
        q_l = torch.addcmul(q_l, gate_attn, attn_out)

        # MLP with AdaLN
        # AdaLN modulation: shift_mlp + scale_mlp * LayerNorm(q_l).
        mlp_input = torch.addcmul(shift_mlp, scale_mlp, self.mlp_layernorm(q_l))
        mlp_output = self.mlp(mlp_input)
        # Gated residual update: q_l + gate_mlp * mlp_output.
        q_l = torch.addcmul(q_l, gate_mlp, mlp_output)

        return q_l


class LocalAttnTransformer(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        c_atom: int,
        c_cond: int,
        c_atompair: int,
        num_heads: int,
        num_queries: int = 32,
        num_keys: int = 128,
        checkpoint_grads: bool = True,
    ):
        """
        Args:
            num_blocks: (int) number of diffusion transformer blocks
            c_token: (int) token embedding dimension
            c_s: (int) condition embedding dimension
            c_atompair: (int) pair bias embedding dimension
            num_heads: (int) number of attention heads
            num_queries: (int) number of queries
            num_keys: (int) number of keys
            checkpoint_grads (bool): whether to use gradient checkpointing during training
        """
        super().__init__()

        self.num_blocks = num_blocks
        self.c_atom = c_atom
        self.c_cond = c_cond
        self.c_atompair = c_atompair
        self.num_heads = num_heads
        self.checkpoint_grads = checkpoint_grads

        self.condition_layernorm = LayerNorm(c_cond)

        # AdaLN parameters
        self.scale_proj = nn.Sequential(
            BiasInitLinear(c_cond, 2 * num_blocks * c_atom, biasinit=1.0),
            nn.Sigmoid(),
        )
        self.shift_proj = LinearNoBias(c_cond, 2 * num_blocks * c_atom, init="zeros")

        # Gate parameters
        self.gate_proj = nn.Sequential(
            BiasInitLinear(c_cond, 2 * num_blocks * c_atom, biasinit=-2.0),
            nn.Sigmoid(),
        )

        # Pair biases
        self.p_layernorm = LayerNorm(c_atompair)
        self.p_proj = LinearNoBias(c_atompair, num_heads * num_blocks)

        self.blocks = nn.ModuleList(
            [
                LocalAttnTransformerBlock(
                    c_atom=c_atom,
                    num_heads=num_heads,
                    num_queries=num_queries,
                    num_keys=num_keys,
                )
                for _ in range(num_blocks)
            ]
        )

    def _chunk_block_params(
        self, fused_params: Tensor
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Split fused parameters into MLP and attention components across blocks.

        Args:
            fused_params: Tensor with fused MLP and attention parameters [..., 2*num_blocks*c_token]

        Returns:
            Tuple containing:
                - attn_params: Attention parameters split across blocks
                - mlp_params: MLP parameters split across blocks
        """
        attn_fused, mlp_fused = fused_params.chunk(2, dim=-1)
        attn_params = attn_fused.chunk(self.num_blocks, dim=-1)
        mlp_params = mlp_fused.chunk(self.num_blocks, dim=-1)

        return attn_params, mlp_params

    def _prep_blocks(
        self,
        *,
        scale_attns: tuple[Tensor, ...],
        shift_attns: tuple[Tensor, ...],
        gate_attns: tuple[Tensor, ...],
        scale_mlps: tuple[Tensor, ...],
        shift_mlps: tuple[Tensor, ...],
        gate_mlps: tuple[Tensor, ...],
        trunked_pair_biases: tuple[Tensor, ...],
        trunked_pair_mask: Tensor,
    ):
        """Prepare the DiT blocks."""
        blocks = [
            partial(
                block,
                scale_attn=scale_attns[i],
                shift_attn=shift_attns[i],
                gate_attn=gate_attns[i],
                scale_mlp=scale_mlps[i],
                shift_mlp=shift_mlps[i],
                gate_mlp=gate_mlps[i],
                trunked_pair_bias=trunked_pair_biases[i],
                trunked_pair_mask=trunked_pair_mask,
            )
            for i, block in enumerate(self.blocks)
        ]

        return blocks

    # @profile_function("atom_transformer")
    @typecheck
    def forward(
        self,
        *,
        q: Float[Tensor, "... n_atom c_atom"],
        c: Float[Tensor, "... n_atom c_cond"],
        p: Float[Tensor, "... n_subsets n_queries n_keys c_atompair"],
        trunked_pair_mask: Bool[Tensor, "... n_subsets n_queries n_keys"],
    ) -> Float[Tensor, "... n_atom c_atom"]:
        """
        Args:
            q: [..., n_atom, c_atom]
            c: [..., n_atom, c_cond]
            p: [..., n_subsets, n_queries, n_keys, c_atompair]
            trunked_pair_mask: [..., n_subsets, n_queries, n_keys]

        Returns:
            [..., n_atom, c_atom]
        """

        # Conditioning prep
        c = self.condition_layernorm(c)
        scale_attns, scale_mlps = self._chunk_block_params(self.scale_proj(c))
        shift_attns, shift_mlps = self._chunk_block_params(self.shift_proj(c))
        gate_attns, gate_mlps = self._chunk_block_params(self.gate_proj(c))

        # Embed and chunk pair bias
        p = self.p_layernorm(p)
        pair_bias_fused = self.p_proj(
            p
        )  # [..., n_subsets, n_queries, n_keys, n_heads * num_blocks]
        pair_bias_fused = rearrange(pair_bias_fused, "... i j nb -> ... nb i j")
        trunked_pair_biases = pair_bias_fused.chunk(
            self.num_blocks, dim=-3
        )  # list of [..., n_subsets, n_heads, n_queries, n_keys]

        # Prepare blocks
        blocks = self._prep_blocks(
            scale_attns=scale_attns,
            shift_attns=shift_attns,
            gate_attns=gate_attns,
            scale_mlps=scale_mlps,
            shift_mlps=shift_mlps,
            gate_mlps=gate_mlps,
            trunked_pair_biases=trunked_pair_biases,
            trunked_pair_mask=trunked_pair_mask.bool(),
        )

        # LocalAttn Tower
        for block in blocks:
            if self.checkpoint_grads:
                q = checkpoint(block, q, use_reentrant=False)
            else:
                q = block(q)

        return q
