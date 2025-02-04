import torch
import torch.nn as nn
from typing import Any
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.model.modules.primitives import LinearNoBias
from .modules.diffusion import DiffusionModule
from .modules.embedders import InputFeatureEmbedder, RelativePositionEncoding
from typing import Optional


class ArtiosLigand(nn.Module):
    def __init__(self, configs):
        super(ArtiosLigand, self).__init__()
        self.configs = configs
        self.c_s, self.c_z, self.c_s_inputs = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
        )

        self.input_feature_embedder = InputFeatureEmbedder(
            **configs.model.input_feature_embedder
        )

        self.linear_no_bias_s_init = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_z_init1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_z_init2 = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_z
        )
        self.linear_no_bias_token_bonds = LinearNoBias(
            in_features=1, out_features=self.c_z
        )

        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )

        self.diffusion_module = DiffusionModule(
            **configs.model.diffusion_module
        )

    def forward(
            self,
            noised_coords: torch.Tensor,
            input_feature_dict: dict[str, Any],
            label_full_dict: dict[str, Any],
            label_dict: dict[str, Any],
            symmetric_permutation: SymmetricPermutation = None,
            inplace_safe: bool = False,
            chunk_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        
        N_token = input_feature_dict["residue_index"].shape[-1]
        if N_token <= 16:
            # Deepspeed_evo_attention do not support token <= 16
            deepspeed_evo_attention_condition_satisfy = False
        else:
            deepspeed_evo_attention_condition_satisfy = True

        # Initial single and pair embeddings
        s_inputs = self.input_feature_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )  # [..., N_token, 449] TODO: is 449 the c_token? Why not set in configs?
        s_init = self.linear_no_bias_s_init(s_inputs)  #  [..., N_token, c_s]
        z_init = (
            self.linear_no_bias_z_init1(s_init)[..., None, :]
            + self.linear_no_bias_z_init2(s_init)[..., None, :, :]
        )  #  [..., N_token, N_token, c_z]

        # Add relative position encoding and token bonds
        if inplace_safe:
            z_init += self.relative_position_encoding(input_feature_dict)
            z_init += self.linear_no_bias_token_bonds(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        else:
            z_init = z_init + self.relative_position_encoding(input_feature_dict)
            z_init = z_init + self.linear_no_bias_token_bonds(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )

        # TODO: Diffusion logic 

        return # pred_dict, label_dict, log_dict

    



        
    
