from typing import Dict, Optional

import torch
from torch import Tensor, nn

from protenix.model.modules.layout import (
    broadcast_token_to_local_atom_pair,
    rearrange_to_key_subset,
    rearrange_to_query_subset,
)
from protenix.model.modules.primitives import LayerNorm, LinearNoBias
from protenix.model.modules.transformer_v2 import LocalAttnTransformer
from protenix.model.utils import (
    average_atom_to_token,
    broadcast_token_to_atom,
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
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
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
            n_blocks (int, optional): number of blocks in AtomTransformer. Defaults to 3.
            n_heads (int, optionall): number of heads in AtomTransformer. Defaults to 4.
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
        self.n_queries = n_queries
        self.n_keys = n_keys

        # TODO: complete the precision and initialization to mimic DeepMind's.
        self.embed_ref_pos = LinearNoBias(
            in_features=3, out_features=c_atom, high_precision=True
        )
        self.embed_ref_charge = LinearNoBias(1, c_atom)
        self.embed_ref_mask = LinearNoBias(1, c_atom)
        self.embed_ref_element = LinearNoBias(128, c_atom)
        self.embed_ref_atom_name = LinearNoBias(4 * 64, c_atom)
        self.embed_pair_offsets = LinearNoBias(
            in_features=3, out_features=self.c_atompair, high_precision=True
        )
        self.embed_inv_pair_distances = LinearNoBias(1, self.c_atompair)
        self.embed_pair_offsets_valid = LinearNoBias(1, self.c_atompair)

        # Initialize components based on granular flags
        if self.has_s:
            self.layernorm_s = LayerNorm(self.c_s)
            self.linear_no_bias_s = LinearNoBias(
                in_features=self.c_s, out_features=self.c_atom, init="zeros"
            )

        if self.has_coords:
            self.embed_noisy_coords = LinearNoBias(
                in_features=3, out_features=self.c_atom, init="default"
            )

        if self.has_z:
            self.layernorm_z = LayerNorm(self.c_z)
            self.linear_no_bias_z = LinearNoBias(
                in_features=self.c_z,
                out_features=self.c_atompair,
                high_precision=True,
                init="zeros",
            )

        self.linear_no_bias_cl = LinearNoBias(
            in_features=self.c_atom,
            out_features=self.c_atompair,
        )
        self.linear_no_bias_cm = LinearNoBias(
            in_features=self.c_atom,
            out_features=self.c_atompair,
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
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_queries=n_queries,
            n_keys=n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.linear_no_bias_q = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_token, init="relu"
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
        """

        # Extract features for special handling
        ref_pos = input_feature_dict["ref_pos"]
        ref_atom_name_chars = flatten_final_dims(
            input_feature_dict["ref_atom_name_chars"], num_dims=2
        )
        ref_element = input_feature_dict["ref_element"]
        ref_mask = input_feature_dict["ref_mask"].unsqueeze(-1)
        ref_charge = input_feature_dict["ref_charge"].unsqueeze(-1)
        ref_space_uid = input_feature_dict["ref_space_uid"]

        # Verify inputs with asserts

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
            ref_pos, rearrange_dim=-2, num_queries=self.n_queries
        )  # [..., n_blocks, n_queries, 3]
        keys_ref_pos = rearrange_to_key_subset(
            ref_pos, rearrange_dim=-2, num_keys=self.n_keys, num_queries=self.n_queries
        )  # [..., n_blocks, n_keys, 3]
        queries_ref_uid = rearrange_to_query_subset(
            ref_space_uid, rearrange_dim=-1, num_queries=self.n_queries
        )  # [..., n_blocks, n_queries]
        keys_ref_uid = rearrange_to_key_subset(
            ref_space_uid,
            rearrange_dim=-1,
            num_keys=self.n_keys,
            num_queries=self.n_queries,
        )  # # [..., n_blocks, n_keys]
        # Compute displacements
        d_lm = (
            queries_ref_pos[..., :, None, :] - keys_ref_pos[..., None, :, :]
        )  # [..., n_blocks, n_queries, n_keys, 3]
        v_lm = (
            queries_ref_uid[..., None, :, :].int()
            == keys_ref_uid[..., None, :, :].int()
        ).unsqueeze(dim=-1)  # [..., n_blocks, n_queries, n_keys, 1]

        # Create pair repr
        # [..., n_blocks, n_queries, n_keys, c_atompair]
        p_lm = self.embed_pair_offsets(d_lm) * v_lm
        p_lm = (
            p_lm
            + self.embed_inv_pair_distances(
                1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))
            )
            * v_lm
        )
        p_lm = p_lm + self.embed_pair_offsets_valid(v_lm.to(dtype=p_lm.dtype)) * v_lm
        return c_l, p_lm

    # Featurize function for doing the featurization - allows iteration in the future.
    def forward(
        self,
        input_feature_dict: Dict[str, Tensor],
        r_l: Tensor = None,
        s: Tensor = None,
        z: Tensor = None,
    ):
        """
        Args:
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            r_l (Tensor, optional): noisy position. Used for diffusion module.
                [..., N_sample | 1, N_atom, 3] if provided.
            s (Tensor, optional): single embedding
                [..., N_sample | 1, N_token, c_s]
            z (Tensor, optional): pair embedding
                [..., N_sample | 1, N_token, N_token, c_z]

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: the output of AtomAttentionEncoder
            a:
                [..., N_sample | 1, N_token, c_token]
            q_l:
                [..., N_sample | 1, N_atom, c_atom]
            c_l:
                [..., N_sample | 1, N_atom, c_atom]
            p_lm:
                [..., N_sample | 1, N_atom, N_atom, c_atompair]
        """
        atom_mask = input_feature_dict["atom_mask"]
        atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
        c_l, p_lm = self._per_atom_conditioning(input_feature_dict)

        # Compute and apply mask
        # [..., N_atom]
        queries_mask = rearrange_to_query_subset(
            atom_mask, rearrange_dim=-1, num_queries=self.n_queries
        )  # [..., n_blocks, n_queries]
        keys_mask = rearrange_to_key_subset(
            atom_mask,
            rearrange_dim=-1,
            num_keys=self.n_keys,
            num_queries=self.n_queries,
        )  # [..., n_blocks, n_keys]
        c_l = c_l * atom_mask[..., None]

        # Mask the pair repr
        pair_mask = queries_mask[..., :, None] * keys_mask[..., None, :]
        p_lm = p_lm * pair_mask[..., None]

        # If has s
        if self.has_s:
            c_l = c_l + broadcast_token_to_atom(
                x_token=self.linear_no_bias_s(
                    (self.layernorm_s(s)), atom_to_token_idx=atom_to_token_idx
                )
            )  # [..., N_sample | 1, N_atom, c_atom], depending on s shape

        # If z
        if self.has_z:
            # Broadcast token pair vectors to local atom pairs
            z_local_pairs = broadcast_token_to_local_atom_pair(
                z_token=self.linear_no_bias_z(self.layernorm_z(z)),
                atom_to_token_idx=atom_to_token_idx,
                n_queries=self.n_queries,
                n_keys=self.n_keys,
            )  # [..., N_sample | 1, n_blocks, n_queries, n_keys, c_atompair]
            p_lm = p_lm + z_local_pairs

        # If r_l
        # if self.has_coords:
        #     q_l = c_l + self.embed_noisy_coords(r_l)
        # else:
        #     q_l = c_l.clone()

        # Local Attn Transformer
        self.atom_transformer()

        # Aggregate to token
        average_atom_to_token()
        # return a, q_skip, c_skip, p_lm
        pass


class AtomAttentionDecoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward():
        pass
