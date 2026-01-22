"""Atom attention encoder and decoder modules."""

from dataclasses import dataclass
from typing import Optional

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn import functional as F

from protenix.data.features import FullContext, ReferenceConformerContext
from protenix.model.modules.layout import (
    broadcast_token_to_local_atom_pair,
    rearrange_to_key_subset,
    rearrange_to_query_subset,
)
from protenix.model.modules.primitives import LayerNorm, LinearNoBias
from protenix.model.modules.transformer import LocalAttnTransformer
from protenix.model.utils import average_atom_to_token, broadcast_token_to_atom
from protenix.utils.tensor_utils import expand_at_dim
from protenix.utils.typing import Bool, Float, Int, typecheck


@typecheck
@dataclass
class AtomAttentionSkipConnections:
    """Skip connections from AtomAttentionEncoder for AtomAttentionDecoder."""

    q: Float[Tensor, "... n_sample n_atom c_atom"]
    c: Float[Tensor, "... #n_sample n_atom c_atom"]
    p: Float[
        Tensor, "... #n_sample n_blocks n_queries n_keys c_atompair"
    ]  # TODO: p should never be n_sample?
    local_attn_mask: Bool[Tensor, "... #n_sample n_blocks n_queries n_keys"]


class AtomAttentionEncoder(nn.Module):
    """Encode atom-wise features and coordinates into token-wise represenation."""

    def __init__(
        self,
        c_token: int,  # 384 or 768
        has_coords: bool = True,
        has_s: bool = True,
        has_z: bool = True,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_s: int = 384,
        c_z: int = 128,
        num_blocks: int = 3,
        num_heads: int = 4,
        num_queries: int = 32,
        num_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
        checkpoint_grads: bool = True,
    ) -> None:
        """
        Args:
            c_token (int): token embedding dim.
            has_coords (bool): whether to initialize components for coordinate embedding (r_l).
            has_s (bool): whether to initialize components for single embedding (s). Defaults to True.
            has_z (bool): whether to initialize components for pair embedding (z). Defaults to True.
            c_atom (int, optional): atom embedding dim. Defaults to 128.
            c_atompair (int, optional): atompair embedding dim. Defaults to 16.
            c_s (int, optional):  single embedding dim. Defaults to 384.
            c_z (int, optional): pair embedding dim. Defaults to 128.
            num_blocks (int, optional): number of blocks in AtomTransformer. Defaults to 3.
            num_heads (int, optionall): number of heads in AtomTransformer. Defaults to 4.
            blocks_per_ckpt: number of AtomAttentionEncoder/AtomTransformer blocks in each activation checkpoint
            checkpoint_grads (bool): whether to use gradient checkpointing. Defaults to True.
        """
        super(AtomAttentionEncoder, self).__init__()

        self.has_coords = has_coords  # Keep for backward compatibility
        self.has_s = has_s
        self.has_z = has_z
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s = c_s
        self.c_z = c_z
        self.num_queries = num_queries
        self.num_keys = num_keys

        # Atom single features
        self.embed_ref_pos = LinearNoBias(
            in_features=3, out_features=c_atom, high_precision=True
        )
        self.embed_ref_mask = LinearNoBias(1, c_atom)
        self.embed_ref_element = LinearNoBias(128, c_atom)
        self.embed_ref_charge = LinearNoBias(1, c_atom)
        self.embed_ref_atom_name = LinearNoBias(4 * 64, c_atom)

        # Atom pair features
        self.embed_pair_offsets = LinearNoBias(
            in_features=3, out_features=self.c_atompair, high_precision=True
        )
        self.embed_inv_pair_distances = LinearNoBias(1, self.c_atompair)
        self.embed_pair_offsets_valid = LinearNoBias(1, self.c_atompair)

        # Initialize components based on granular flags
        if self.has_s:
            self.layernorm_s = LayerNorm(self.c_s)
            self.linear_no_bias_s = LinearNoBias(
                in_features=self.c_s,
                out_features=self.c_atom,
                init="zeros",
                high_precision=True,
            )

        if self.has_z:
            self.layernorm_z = LayerNorm(self.c_z)
            self.linear_no_bias_z = LinearNoBias(
                in_features=self.c_z,
                out_features=self.c_atompair,
                high_precision=True,
            )

        if self.has_coords:
            self.embed_noisy_coords = LinearNoBias(
                in_features=3, out_features=self.c_atom, high_precision=True
            )

        self.linear_no_bias_cl = LinearNoBias(
            in_features=self.c_atom,
            out_features=self.c_atompair,
            init="relu",  # TODO: orthogonal init
        )
        self.linear_no_bias_cm = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atompair, init="relu"
        )
        self.small_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(
                in_features=self.c_atompair, out_features=self.c_atompair, init="relu"
            ),
            nn.ReLU(),
            LinearNoBias(
                in_features=self.c_atompair, out_features=self.c_atompair, init="relu"
            ),
            nn.ReLU(),
            LinearNoBias(
                in_features=self.c_atompair, out_features=self.c_atompair, init="zeros"
            ),
        )
        self.atom_transformer = LocalAttnTransformer(
            num_blocks=num_blocks,
            num_heads=num_heads,
            c_atom=c_atom,
            c_cond=c_atom,
            c_atompair=c_atompair,
            num_queries=num_queries,
            num_keys=num_keys,
            checkpoint_grads=checkpoint_grads,
        )
        self.final_linear = LinearNoBias(
            in_features=self.c_atom,
            out_features=self.c_token,
        )

    @torch.compile
    @typecheck
    def _per_atom_conditioning(
        self,
        reference: ReferenceConformerContext,
    ) -> tuple[
        Float[Tensor, "... n_atom c_atom"],
        Float[Tensor, "... n_blocks n_queries n_keys c_atompair"],
    ]:
        """Generate per-atom single and (local) pair conditioning.

        Args:
            reference: Reference conformer context containing:
                - ref_pos: reference conformer positions
                - ref_atom_name_chars: atom name characters
                - ref_element: element one-hot
                - ref_mask: conformer mask
                - ref_charge: atomic charge
                - ref_space_uid: unique conformer ID

        Returns:
            c_l: Atom-wise single conditioning
            p_lm: Atom pair-wise conditioning
        """
        # Extract and prepare features
        ref_pos = reference.ref_pos
        ref_atom_name_chars = rearrange(
            reference.ref_atom_name_chars, "... n c -> ... (n c)"
        ).to(ref_pos.dtype)
        ref_element = reference.ref_element.to(ref_pos.dtype)
        ref_mask = reference.ref_mask.unsqueeze(-1).to(ref_pos.dtype)
        ref_charge = reference.ref_charge.unsqueeze(-1).to(ref_pos.dtype)
        ref_space_uid = reference.ref_space_uid

        # Create atom single conditioning: embed per-atom metadata
        # [..., n_atom, c_atom]
        c_l = self.embed_ref_pos(ref_pos)
        c_l = c_l + self.embed_ref_mask(ref_mask)
        c_l = c_l + self.embed_ref_element(ref_element)
        c_l = c_l + self.embed_ref_charge(torch.arcsinh(ref_charge))
        c_l = c_l + self.embed_ref_atom_name(ref_atom_name_chars)
        c_l = c_l * ref_mask

        # Pair conditioning: rearrange to local attention subsets
        queries_ref_pos = rearrange_to_query_subset(
            ref_pos, rearrange_dim=-2, num_queries=self.num_queries
        )  # [..., n_blocks, n_queries, 3]
        keys_ref_pos = rearrange_to_key_subset(
            ref_pos,
            rearrange_dim=-2,
            num_keys=self.num_keys,
            num_queries=self.num_queries,
        )  # [..., n_blocks, n_keys, 3]
        queries_ref_uid = rearrange_to_query_subset(
            ref_space_uid, rearrange_dim=-1, num_queries=self.num_queries
        )  # [..., n_blocks, n_queries]
        keys_ref_uid = rearrange_to_key_subset(
            ref_space_uid,
            rearrange_dim=-1,
            num_keys=self.num_keys,
            num_queries=self.num_queries,
        )  # [..., n_blocks, n_keys]

        # Compute pairwise displacements
        d_lm = (
            queries_ref_pos[..., :, None, :] - keys_ref_pos[..., None, :, :]
        )  # [..., n_blocks, n_queries, n_keys, 3]

        # Valid pair mask: atoms in same conformer
        v_lm = (
            queries_ref_uid[..., :, None].int() == keys_ref_uid[..., None, :].int()
        ).unsqueeze(dim=-1)  # [..., n_blocks, n_queries, n_keys, 1]

        # Create pair representation
        # [..., n_blocks, n_queries, n_keys, c_atompair]
        p_lm = self.embed_pair_offsets(d_lm)
        p_lm = p_lm + self.embed_inv_pair_distances(
            1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))
        )
        p_lm = p_lm + self.embed_pair_offsets_valid(v_lm.to(dtype=p_lm.dtype))
        p_lm = p_lm * v_lm  # Mask inter-conformer pairs

        return c_l, p_lm

    @torch.compile
    @typecheck
    def forward(
        self,
        context: FullContext,
        r_l: Optional[Float[Tensor, "... n_sample n_atom 3"]] = None,
        s: Optional[Float[Tensor, "... #n_sample n_token c_s"]] = None,
        z: Optional[Float[Tensor, "... #n_sample n_token n_token c_z"]] = None,
    ) -> tuple[
        Float[Tensor, "... n_sample n_token c_token"],
        AtomAttentionSkipConnections,
    ]:
        """Encode atom features into token-level representations.

        Args:
            context: Full feature context containing reference conformer, masks, and indices.
            r_l: Noisy atom positions for diffusion.
            s: Single (token) embedding.
            z: Pair (token-token) embedding.

        Returns:
            a: Token-level output.
            skip: Skip connections for the decoder.
        """
        # Extract features from context
        atom_mask = context.mask.atom_mask
        atom_to_token_idx = context.index.atom_to_token_idx
        n_token = context.token.restype.shape[-2]

        # Per-atom conditioning from reference conformer
        c_l, p_lm = self._per_atom_conditioning(context.reference)

        # Compute the truncated pair mask for local attention
        queries_mask = rearrange_to_query_subset(
            atom_mask, rearrange_dim=-1, num_queries=self.num_queries
        )  # [..., n_blocks, n_queries]
        keys_mask = rearrange_to_key_subset(
            atom_mask,
            rearrange_dim=-1,
            num_keys=self.num_keys,
            num_queries=self.num_queries,
        )  # [..., n_blocks, n_keys]
        trunked_pair_mask = queries_mask[..., :, None] & keys_mask[..., None, :]

        # Mask the representations
        p_lm = p_lm * trunked_pair_mask[..., None]
        c_l = c_l * atom_mask[..., None]

        # Embed single features into the pair conditioning
        queries_c_l = rearrange_to_query_subset(
            q_x=self.linear_no_bias_cl(F.relu(c_l)),
            rearrange_dim=-2,
            num_queries=self.num_queries,
        )
        keys_c_m = rearrange_to_key_subset(
            k_x=self.linear_no_bias_cm(F.relu(c_l)),
            rearrange_dim=-2,
            num_keys=self.num_keys,
            num_queries=self.num_queries,
        )
        p_lm = p_lm + queries_c_l[..., :, None, :] + keys_c_m[..., None, :, :]

        # Add n_sample dimension
        c_l = c_l.unsqueeze(-3)
        p_lm = p_lm.unsqueeze(-5)
        atom_mask = atom_mask.unsqueeze(-2)
        trunked_pair_mask = trunked_pair_mask.unsqueeze(-4)

        # Incorporate single embedding (s) if provided
        if self.has_s and s is not None:
            c_l = c_l + broadcast_token_to_atom(
                x_token=self.linear_no_bias_s(self.layernorm_s(s)),
                atom_to_token_idx=repeat(
                    atom_to_token_idx,
                    "... n_atom -> ... n_sample n_atom",
                    n_sample=s.shape[-3],  # n_sample | 1
                ),
            )
            c_l = c_l * atom_mask[..., None]

        # Incorporate pair embedding (z) if provided
        if self.has_z and z is not None:
            z_local_pairs = broadcast_token_to_local_atom_pair(
                z_token=self.linear_no_bias_z(self.layernorm_z(z)),
                atom_to_token_idx=repeat(
                    atom_to_token_idx,
                    "... n_atom -> ... n_sample n_atom",
                    n_sample=z.shape[-4],  # n_sample | 1
                ),
                num_queries=self.num_queries,
                num_keys=self.num_keys,
            )
            p_lm = p_lm + z_local_pairs
            p_lm = p_lm * trunked_pair_mask[..., None]

        # Incorporate noisy coordinates if provided
        if self.has_coords and r_l is not None:
            q_l = c_l + self.embed_noisy_coords(r_l)
            q_l = q_l * atom_mask[..., None]
        else:
            q_l = c_l.clone()

        # Small MLP on the pair activations
        p_lm = p_lm + self.small_mlp(p_lm)

        # Local attention transformer
        q_l = self.atom_transformer(
            q=q_l, c=c_l, p=p_lm, trunked_pair_mask=trunked_pair_mask
        )

        # Aggregate atom features to token level
        a = average_atom_to_token(
            x_atom=F.relu(self.final_linear(q_l)),
            atom_mask=atom_mask,
            atom_to_token_idx=atom_to_token_idx.unsqueeze(-2),
            n_token=n_token,
        )

        skip = AtomAttentionSkipConnections(
            q=q_l,
            c=c_l,
            p=p_lm,
            local_attn_mask=trunked_pair_mask,
        )
        return a, skip


class AtomAttentionDecoder(nn.Module):
    """Decode atom representations back to coordinate updates (Algorithm 6 in AF3)."""

    def __init__(
        self,
        c_token: int = 384,
        c_atom: int = 128,
        c_atompair: int = 16,
        num_blocks: int = 3,
        num_heads: int = 4,
        num_queries: int = 32,
        num_keys: int = 128,
        checkpoint_grads: bool = True,
    ) -> None:
        super(AtomAttentionDecoder, self).__init__()
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.num_queries = num_queries
        self.num_keys = num_keys

        self.broadcast_linear = LinearNoBias(c_token, c_atom)
        self.atom_transformer = LocalAttnTransformer(
            num_blocks=num_blocks,
            c_atom=c_atom,
            c_cond=c_atom,
            c_atompair=c_atompair,
            num_heads=num_heads,
            num_queries=num_queries,
            num_keys=num_keys,
            checkpoint_grads=checkpoint_grads,
        )
        self.final_layernorm = LayerNorm(c_atom)
        self.final_linear = LinearNoBias(c_atom, 3, init="zeros", high_precision=True)

    @typecheck
    def forward(
        self,
        a: Float[Tensor, "... n_sample n_token c_token"],
        skip: AtomAttentionSkipConnections,
        atom_to_token_idx: Int[Tensor, "... n_atom"],
        atom_mask: Bool[Tensor, "... n_atom"],
    ) -> Float[Tensor, "... n_sample n_atom 3"]:
        """Decode atom features to coordinate updates.

        Args:
            a: Token-level representation from encoder.
            skip: Skip connections from encoder.
            atom_to_token_idx: Mapping from atoms to their parent tokens.
            atom_mask: Mask indicating valid atoms.

        Returns:
            Coordinate updates for each atom.
        """
        # Add n_sample and channel dim for broadcasting
        atom_mask = rearrange(atom_mask, "... n_atom -> ... 1 n_atom 1")

        # Broadcast per-token activations to per-atom activations and add skip connection
        q = skip.q + broadcast_token_to_atom(
            x_token=self.broadcast_linear(a),
            atom_to_token_idx=expand_at_dim(atom_to_token_idx, dim=-2, n=a.shape[-3]),
        )
        q = q * atom_mask

        # Local attention transformer
        q = self.atom_transformer(
            q=q, c=skip.c, p=skip.p, trunked_pair_mask=skip.local_attn_mask
        )

        # Map to position updates
        r = self.final_linear(self.final_layernorm(q))
        r = r * atom_mask
        return r
