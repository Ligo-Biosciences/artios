from typing import Dict, Optional

import torch
from torch import Tensor, nn

from protenix.model.modules.primitives import LayerNorm, LinearNoBias
from protenix.model.modules.transformer_v2 import LocalAttnTransformer


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

        self.input_feature = {
            # "ref_pos": 3,
            # "ref_charge": 1,
            "ref_mask": 1,
            "ref_element": 128,
            "ref_atom_name_chars": 4 * 64,
        }
        # TODO: complete the precision and initialization to mimic DeepMind's.
        self.linear_no_bias_ref_pos = LinearNoBias(
            in_features=3, out_features=c_atom, high_precision=True
        )

        self.linear_no_bias_ref_charge = LinearNoBias(
            in_features=1, out_features=c_atom
        )

        self.linear_no_bias_f = LinearNoBias(
            in_features=sum(self.input_feature.values()), out_features=self.c_atom
        )
        self.linear_no_bias_d = LinearNoBias(
            in_features=3, out_features=self.c_atompair, high_precision=True
        )
        self.linear_no_bias_invd = LinearNoBias(
            in_features=1,
            out_features=self.c_atompair,
        )
        self.linear_no_bias_v = LinearNoBias(
            in_features=1,
            out_features=self.c_atompair,
        )

        # Initialize components based on granular flags
        if self.has_s:
            self.layernorm_s = LayerNorm(self.c_s)
            self.linear_no_bias_s = LinearNoBias(
                in_features=self.c_s, out_features=self.c_atom, init="zeros"
            )

        if self.has_z:
            self.layernorm_z = LayerNorm(self.c_z)
            self.linear_no_bias_z = LinearNoBias(
                in_features=self.c_z,
                out_features=self.c_atompair,
                high_precision=True,
                init="zeros",
            )

        if self.has_coords:
            self.linear_no_bias_r = LinearNoBias(
                in_features=3, out_features=self.c_atom, init="default"
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
        # atom_mask = input_feature_dict["atom_mask"]
        # atom_to_token_idx = input_feature_dict["atom_to_token_idx"]

        # Extract ref_pos for special handling
        ref_pos = input_feature_dict["ref_pos"]
        batch_shape = ref_pos.shape[:-2]
        N_atom = ref_pos.shape[-2]

        # Verify inputs with asserts

        # Featurizations
        # Create the atom single conditioning: Embed per-atom meta data
        # TODO: separate the ref_pos to be done in FP32 precision.
        # [..., N_atom, C_atom]
        c_l = self.linear_no_bias_ref_pos(
            input_feature_dict["ref_pos"]
        ) + self.linear_no_bias_ref_charge(
            # use arcsinh for ref_charge
            torch.arcsinh(input_feature_dict["ref_charge"]).reshape(
                *batch_shape, N_atom, 1
            )
        )
        c_l = self.linear_no_bias_f(
            torch.cat(
                [
                    input_feature_dict[name].reshape(
                        *batch_shape, N_atom, self.input_feature[name]
                    )
                    for name in self.input_feature
                ],
                dim=-1,
            )
        )
        c_l = c_l * input_feature_dict["ref_mask"].reshape(*batch_shape, N_atom, 1)

        # Bunch of rearranges for atom pair representation
        # Mask the pair repr
        # Local Attn Transformer
        # Aggregate to token
        # return a, q_skip, c_skip, p_lm
        pass


class AtomAttentionDecoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward():
        pass
