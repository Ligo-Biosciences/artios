"""
TODO: switch to jaxtyping for readability.
"""

from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from protenix.model.modules.layout import (
    broadcast_token_to_local_atom_pair,
    rearrange_to_key_subset,
    rearrange_to_query_subset,
)
from protenix.model.modules.primitives import LayerNorm, LinearNoBias
from protenix.model.modules.transformer import LocalAttnTransformer
from protenix.model.utils import (
    average_atom_to_token,
    broadcast_token_to_atom,
    expand_at_dim,
    flatten_final_dims,
)


class AtomAttentionEncoder(nn.Module):
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
        )
        self.final_linear = LinearNoBias(
            in_features=self.c_atom,
            out_features=self.c_token,
        )

    def _per_atom_conditioning(
        self,
        input_feature_dict: Dict[str, Tensor],
    ):
        """
        Generate per-atom single and (local) pair conditioning.

        Args:
            input_feature_dict (dict) that contains the following features:
                * ref_pos: reference conformers
                    [..., N_atom, 3]
                * ref_atom_name_chars: reference atom name character
                    [..., N_atom, 4, 64]
                * ref_element: reference element
                    [..., N_atom, 128]
                * ref_mask: reference conformer mask
                    [..., N_atom]
                * ref_charge: reference charge
                    [..., N_atom]
                * ref_space_uid: unique id for the ligand / residue
                    [..., N_atom]

        Returns:
            - Tensor of atom-wise single conditioning
                [..., n_atoms, c_atom]
            - Tensor of atom pair-wise conditioning
                [..., n_blocks, n_queries, n_keys, c_atompair]
        """

        # Extract features for special handling
        # TODO: make this prettier
        ref_pos = input_feature_dict["ref_pos"]
        ref_atom_name_chars = flatten_final_dims(
            input_feature_dict["ref_atom_name_chars"], num_dims=2
        ).to(ref_pos.dtype)
        ref_element = input_feature_dict["ref_element"].to(ref_pos.dtype)
        ref_mask = input_feature_dict["ref_mask"].unsqueeze(-1).to(ref_pos.dtype)
        ref_charge = input_feature_dict["ref_charge"].unsqueeze(-1)
        ref_space_uid = input_feature_dict["ref_space_uid"]

        # TODO: Verify inputs with asserts

        # Create the atom single conditioning: Embed per-atom meta data
        # [..., N_atom, C_atom]
        c_l = self.embed_ref_pos(ref_pos)
        c_l = c_l + self.embed_ref_mask(ref_mask)
        c_l = c_l + self.embed_ref_element(ref_element)
        c_l = c_l + self.embed_ref_charge(torch.arcsinh(ref_charge))
        c_l = c_l + self.embed_ref_atom_name(ref_atom_name_chars)
        c_l = c_l * ref_mask

        # Pair conditioning
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
        # Compute displacements
        d_lm = (
            queries_ref_pos[..., :, None, :] - keys_ref_pos[..., None, :, :]
        )  # [..., n_blocks, n_queries, n_keys, 3]
        v_lm = (
            queries_ref_uid[..., :, None].int() == keys_ref_uid[..., None, :].int()
        ).unsqueeze(dim=-1)  # [..., n_blocks, n_queries, n_keys, 1]

        # Create pair repr
        # [..., n_blocks, n_queries, n_keys, c_atompair]
        p_lm = self.embed_pair_offsets(d_lm)
        p_lm = p_lm + self.embed_inv_pair_distances(
            1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))
        )
        p_lm = p_lm + self.embed_pair_offsets_valid(v_lm.to(dtype=p_lm.dtype))
        p_lm = p_lm * v_lm  # mask inter-conformer nonsense
        return c_l, p_lm

    # @profile_function("atom_attention_encoder")
    def forward(
        self,
        input_feature_dict: Dict[str, Tensor],
        r_l: Tensor = None,
        s: Tensor = None,
        z: Tensor = None,
        return_trunked_pair_mask: bool = False,
    ):
        """
        Args:
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            r_l (Tensor, optional): noisy position. Used for diffusion module.
                [..., N_sample, N_atom, 3] if provided.
            s (Tensor, optional): single embedding
                [..., N_sample | 1, N_token, c_s]
            z (Tensor, optional): pair embedding
                [..., N_sample | 1, N_token, N_token, c_z]

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: the output of AtomAttentionEncoder
            a:
                [..., N_sample, N_token, c_token]
            q_l:
                [..., N_sample | 1, N_atom, c_atom]
            c_l:
                [..., N_sample | 1, N_atom, c_atom]
            p_lm:
                [..., N_sample | 1, N_atom, N_atom, c_atompair]
        """
        # Validate input dimensions
        if s is not None:
            if s.shape[-3] != 1:
                assert (
                    s.shape[-3] == r_l.shape[-3]
                ), f"s sample dimension {s.shape[-3]} must match r_l sample dimension {r_l.shape[-3]}"
            assert len(r_l.shape[:-3]) == len(s.shape[:-3])  # batch dims
        if z is not None:
            if z.shape[-4] != 1:
                assert (
                    z.shape[-4] == r_l.shape[-3]
                ), f"z sample dimension {z.shape[-4]} must match r_l sample dimension {r_l.shape[-3]}"
            assert len(r_l.shape[:-3]) == len(z.shape[:-4])

        atom_mask = input_feature_dict["atom_mask"]
        atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
        n_token = input_feature_dict["restype"].shape[-2]

        # Per-atom conditioning
        c_l, p_lm = self._per_atom_conditioning(input_feature_dict)
        # [..., N_atom, c_atom]

        # Compute the trunked pair mask
        # [..., N_atom]
        queries_mask = rearrange_to_query_subset(
            atom_mask, rearrange_dim=-1, num_queries=self.num_queries
        )  # [..., n_blocks, n_queries]
        keys_mask = rearrange_to_key_subset(
            atom_mask,
            rearrange_dim=-1,
            num_keys=self.num_keys,
            num_queries=self.num_queries,
        )  # [..., n_blocks, n_keys]
        trunked_pair_mask = queries_mask[..., :, None] * keys_mask[..., None, :]

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
        # [..., n_blocks, n_queries, n_keys, c_atompair]

        # N_sample dimension gets added
        c_l = c_l.unsqueeze(-3)
        p_lm = p_lm.unsqueeze(-5)
        atom_mask = atom_mask.unsqueeze(-2)
        trunked_pair_mask = trunked_pair_mask.unsqueeze(-4)

        # N_SAMPLE BEGINS
        if self.has_s:
            c_l = c_l + broadcast_token_to_atom(
                x_token=self.linear_no_bias_s(self.layernorm_s(s)),
                atom_to_token_idx=atom_to_token_idx.unsqueeze(-2),  # expand_at_dim(
                #     atom_to_token_idx,
                #     dim=-2,
                #     n=s.shape[-3],  # N_sample | 1
                # ),
            )  # [..., N_sample | 1, N_atom, c_atom], depending on s shape
            c_l = c_l * atom_mask[..., None]

        if self.has_z:
            # Broadcast token pair vectors to local atom pairs
            z_local_pairs = broadcast_token_to_local_atom_pair(
                z_token=self.linear_no_bias_z(self.layernorm_z(z)),
                atom_to_token_idx=expand_at_dim(
                    atom_to_token_idx,
                    dim=-2,
                    n=z.shape[-4],  # N_sample | 1
                ),
                num_queries=self.num_queries,
                num_keys=self.num_keys,
            )  # [..., N_sample | 1, n_blocks, n_queries, n_keys, c_atompair]

            p_lm = p_lm + z_local_pairs
            p_lm = p_lm * trunked_pair_mask[..., None]

        if self.has_coords:
            q_l = c_l + self.embed_noisy_coords(r_l)
            q_l = q_l * atom_mask[..., None]
        else:
            q_l = c_l.clone()

        # Small MLP on the pair activations
        p_lm = p_lm + self.small_mlp(p_lm)

        # Local Attn Transformer
        q_l = self.atom_transformer(
            q=q_l, c=c_l, p=p_lm, trunked_pair_mask=trunked_pair_mask
        )

        # Aggregate to token
        a = average_atom_to_token(
            x_atom=F.relu(self.final_linear(q_l)),
            atom_mask=atom_mask,
            atom_to_token_idx=atom_to_token_idx.unsqueeze(-2),
            n_token=n_token,
        )
        if return_trunked_pair_mask:
            return a, q_l, c_l, p_lm, trunked_pair_mask
        else:
            return a, q_l, c_l, p_lm


class AtomAttentionDecoder(nn.Module):
    """
    Implements Algorithm 6 in AF3
    """

    def __init__(
        self,
        c_token: int = 384,
        c_atom: int = 128,
        c_atompair: int = 16,
        num_blocks: int = 3,
        num_heads: int = 4,
        num_queries: int = 32,
        num_keys: int = 128,
    ) -> None:
        """
        Args:
            c_token (int): feature channel of token (single a). Defaults to 384.
            c_atom (int): embedding dim for atom embedding. Defaults to 128.
            c_atompair (int): embedding dim for atom pair embedding.
            c_s (int):
            num_blocks (int): number of blocks for AtomTransformer. Defaults to 3.
            num_heads (int): number of heads for AtomTransformer. Defaults to 4.
            num_queries (int): Number of queries. Defaults to 32.
            num_keys (int): Number of keys. Defaults to 128.
        """
        super(AtomAttentionDecoder, self).__init__()
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.num_queries = num_queries
        self.num_keys = num_keys

        self.broadcast_linear = LinearNoBias(c_token, c_atom)
        # Local atom transformer
        self.atom_transformer = LocalAttnTransformer(
            num_blocks=num_blocks,
            c_atom=c_atom,
            c_cond=c_atom,
            c_atompair=c_atompair,
            num_heads=num_heads,
            num_queries=num_queries,
            num_keys=num_keys,
        )
        self.final_layernorm = LayerNorm(c_atom)
        self.final_linear = LinearNoBias(c_atom, 3, init="zeros", high_precision=True)

    # @profile_function("atom_attention_decoder")
    def forward(
        self,
        a: Tensor,
        q_skip: Tensor,
        c_skip: Tensor,
        p_skip: Tensor,
        atom_to_token_idx: Tensor,
        atom_mask: Tensor,
        trunked_pair_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            a (Tensor): the single feature aggregate per-atom representation
                [..., N_sample, N_token, c_token]
            q_skip (Tensor): atom single embedding
                [..., N_sample, N_atom, c_atom]
            c_skip (Tensor): atom single embedding
                [..., 1, N_atom, c_atom]
            p_skip (Tensor): atompair single embedding
                [..., 1, n_blocks, n_queries, n_keys, c_atompair]
            atom_to_token_idx (Tensor): atom to token index for broadcasting.
                [..., N_atom]
            atom_mask (Tensor): per-atom mask indicating valid atoms
                [..., N_atom]
            trunked_pair_mask (Tensor): local atom-atom pair mask for local attention transformer.
                [..., n_blocks, n_queries, n_keys]
        Returns:
            Tensor: the updated noisy coordinates
                [..., N_atom, 3]
        # TODO: atom attention decoder has a different input signature than the decoder. Is this optimal?
        """
        # Add N_sample dims
        atom_mask = atom_mask.unsqueeze(-2)
        trunked_pair_mask = trunked_pair_mask.unsqueeze(-4)

        # Broadcast per-token activations to per-atom activations and add the skip connection
        q = q_skip + broadcast_token_to_atom(
            x_token=self.broadcast_linear(a),
            atom_to_token_idx=atom_to_token_idx.unsqueeze(
                -2
            ),  # expand_at_dim(atom_to_token_idx, dim=-2, n=a.shape[-3]),
        )
        q = q * atom_mask[..., None]

        # Cross attention transformer
        q = self.atom_transformer(
            q=q, c=c_skip, p=p_skip, trunked_pair_mask=trunked_pair_mask
        )

        # Map to positions update
        r = self.final_linear(self.final_layernorm(q))
        r = r * atom_mask[..., None]
        return r
