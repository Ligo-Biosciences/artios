"""

Simple modulate function for absorbing AdaLN into the DiT block.

Do we combine diffusion transformer and transfomer?

Deepmind's pattern is one custom class and one custom forward for every specific function. There are no if else conditional flows. This is the pattern
we follow.

Cross attention and self attention are separated.

"""

from functools import partial
from typing import Tuple

from torch import Tensor, nn

from protenix.model.modules.components import Attention, Transition
from protenix.model.modules.primitives import BiasInitLinear, LayerNorm, LinearNoBias
from protenix.model.utils import permute_final_dims


class DiTBlock(nn.Module):
    """Diffusion Transformer Block."""

    def __init__(self, c_token: int, num_heads: int):
        assert c_token % num_heads == 0, "Input dim should be divisible by num heads."
        self.attn_layernorm = LayerNorm(c_token, elementwise_affine=False)
        # Attention
        self.attention = Attention(
            q_dim=c_token,
            kv_dim=c_token,
            head_dim=c_token // num_heads,
            num_heads=num_heads,
            gating=True,
            final_linear_init="default",
        )
        self.mlp_layernorm = LayerNorm(c_token, elementwise_affine=False)
        # Transition
        self.transition = Transition(c_in=c_token, n=2, output_linear_init="default")

    def forward(
        self,
        *,
        a: Tensor,
        scale_attn: Tensor,
        shift_attn: Tensor,
        gate_attn: Tensor,
        scale_mlp: Tensor,
        shift_mlp: Tensor,
        gate_mlp: Tensor,
        pair_bias: Tensor,
        pair_mask: Tensor,
    ):
        """
        Forward pass of the transformer block.

        Args:
            a (Tensor): Input token representations.
                [..., c_token]
            scale_attn (Tensor): attention scaling.
                [..., c_token]
            shift_attn (Tensor): attention shift.
                [..., c_token]
            gate_attn (Tensor): attention gating.
                [..., c_token]
            scale_mlp (Tensor): MLP scaling.
                [..., c_token]
            shift_mlp (Tensor): MLP shift.
                [..., c_token]
            gate_mlp (Tensor): MLP gating.
                [..., c_token]
            pair_bias (Tensor): Pairwise attention bias.
                [..., num_heads, N, N]
            pair_mask (Tensor): Pairwise attention mask.
                [..., N, N]
        """
        # Adaln + Attention
        attn_input = scale_attn * self.attn_layernorm(a) + shift_attn
        attn_output = self.attention(
            q_x=attn_input, kv_x=attn_input, pair_bias=pair_bias, pair_mask=pair_mask
        )
        a = a + (gate_attn * attn_output)

        # Adaln + MLP
        mlp_input = scale_mlp * self.mlp_layernorm(a) + shift_mlp
        mlp_output = self.transition(mlp_input)
        a = a + (gate_mlp * mlp_output)
        return a


class DiffusionTransformer(nn.Module):
    """"""

    def __init__(
        self,
        num_blocks: int,
        c_token: int,
        c_s: int,
        c_z: int,
        num_heads: int,
    ):
        self.num_blocks = num_blocks
        self.c_token = c_token
        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads

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
        self.z_linear = LinearNoBias(c_z, num_heads * num_blocks)

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
                - mlp_params: MLP parameters split across blocks
                - attn_params: Attention parameters split across blocks
        """
        mlp_fused, attn_fused = fused_params.chunk(2, dim=-1)
        mlp_params = mlp_fused.chunk(self.num_blocks, dim=-1)
        attn_params = attn_fused.chunk(self.num_blocks, dim=-1)

        return mlp_params, attn_params

    def _prep_blocks(
        self,
        *,
        scale_attns: Tuple[Tensor, ...],
        shift_attns: Tuple[Tensor, ...],
        gate_attns: Tuple[Tensor, ...],
        scale_mlps: Tuple[Tensor, ...],
        shift_mlps: Tuple[Tensor, ...],
        gate_mlps: Tuple[Tensor, ...],
        pair_biases: Tuple[Tensor, ...],
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

    def forward(self, a: Tensor, s: Tensor, z: Tensor, pair_mask: Tensor):
        """
        Args:
            a: [..., N, c_token]
            s: [..., c_s]
            z: [..., N, N, c_z]
            pair_mask: [..., N, N]
        TODO: add units tests
        TODO: add asserts
        """
        # Conditioning
        s = self.condition_layernorm(s)

        # Chunk block parameters
        scale_mlps, scale_attns = self._chunk_block_params(self.scale_linear(s))
        shift_mlps, shift_attns = self._chunk_block_params(self.shift_linear(s))
        gate_mlps, gate_attns = self._chunk_block_params(self.gate_linear(s))

        # Embed and chunk pair bias
        z = self.z_layernorm(z)
        pair_bias_fused = self.z_linear(z)
        pair_bias_fused = permute_final_dims(
            pair_bias_fused, [2, 0, 1]
        )  # [..., num_heads*num_blocks, N, N]
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
            pair_mask=pair_mask,
        )

        # DiT Tower
        for block in blocks:
            a = block(a=a)

        return a


class LocalAttnTransformerBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self):
        # Attention
        # MLP
        pass


class LocalAttnTransformer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self):
        # Put the rearranges in here within the for loop.
        pass
