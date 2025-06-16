# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from artios.model.modules.primitives import LinearNoBias
from artios.model.modules.transformer import AtomAttentionEncoder


class InputFeatureEmbedder(nn.Module):
    """
    Implements Algorithm 2 in AF3
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 384,
        n_blocks: int = 3,
    ) -> None:
        """
        Args:
            c_atom (int, optional): atom embedding dim. Defaults to 128.
            c_atompair (int, optional): atom pair embedding dim. Defaults to 16.
            c_token (int, optional): token embedding dim. Defaults to 384.
        """
        super(InputFeatureEmbedder, self).__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.atom_attention_encoder = AtomAttentionEncoder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            n_blocks=n_blocks,
            has_coords=False,
        )
        # Line2
        # self.input_feature = {"restype": 32} # {"restype": 32, "profile": 32, "deletion_mean": 1}

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_feature_dict (Dict[str, Any]): dict of input features
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: token embedding
                [..., N_token, 384 (c_token) + 32 (c_atompair) := 416]
        """
        # Embed per-atom features.
        a, _, _, _ = self.atom_attention_encoder(
            input_feature_dict=input_feature_dict,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )  # [..., N_token, c_token]
        # Concatenate the per-token features.
        batch_shape = input_feature_dict["restype"].shape[:-1]
        #s_inputs = torch.cat(
        #    [a]
        #    + [
        #        input_feature_dict[name].reshape(*batch_shape, d)
        #        for name, d in self.input_feature.items()
        #    ],
        #    dim=-1,
        #)
        
        return a


class RelativePositionEncoding(nn.Module):
    """
    Implements Algorithm 3 in AF3
    """

    def __init__(self, r_max: int = 32, s_max: int = 2, c_z: int = 128) -> None:
        """
        Args:
            r_max (int, optional): Relative position indices clip value. Defaults to 32.
            s_max (int, optional): Relative chain indices clip value. Defaults to 2.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        """
        super(RelativePositionEncoding, self).__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        self.linear_no_bias = LinearNoBias(
            in_features=(4 * self.r_max + 2 * self.s_max + 7), out_features=self.c_z
        )
        self.input_feature = {
            "asym_id": 1,
            "residue_index": 1,
            "entity_id": 1,
            "sym_id": 1,
            "token_index": 1,
        }

    def forward(self, input_feature_dict: dict[str, Any]) -> torch.Tensor:
        """
        Args:
            input_feature_dict (Dict[str, Any]): input meta feature dict.
            asym_id / residue_index / entity_id / sym_id / token_index
                [..., N_tokens]
        Returns:
            torch.Tensor: relative position encoding
                [..., N_token, N_token, c_z]
        """
        b_same_chain = (
            input_feature_dict["asym_id"][..., :, None]
            == input_feature_dict["asym_id"][..., None, :]
        ).long()  # [..., N_token, N_token]
        b_same_residue = (
            input_feature_dict["residue_index"][..., :, None]
            == input_feature_dict["residue_index"][..., None, :]
        ).long()  # [..., N_token, N_token]
        b_same_entity = (
            input_feature_dict["entity_id"][..., :, None]
            == input_feature_dict["entity_id"][..., None, :]
        ).long()  # [..., N_token, N_token]
        d_residue = torch.clip(
            input=input_feature_dict["residue_index"][..., :, None]
            - input_feature_dict["residue_index"][..., None, :]
            + self.r_max,
            min=0,
            max=2 * self.r_max,
        ) * b_same_chain + (1 - b_same_chain) * (
            2 * self.r_max + 1
        )  # [..., N_token, N_token]
        a_rel_pos = F.one_hot(d_residue, 2 * (self.r_max + 1))
        d_token = torch.clip(
            input=input_feature_dict["token_index"][..., :, None]
            - input_feature_dict["token_index"][..., None, :]
            + self.r_max,
            min=0,
            max=2 * self.r_max,
        ) * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (
            2 * self.r_max + 1
        )  # [..., N_token, N_token]
        a_rel_token = F.one_hot(d_token, 2 * (self.r_max + 1))
        d_chain = torch.clip(
            input=input_feature_dict["sym_id"][..., :, None]
            - input_feature_dict["sym_id"][..., None, :]
            + self.s_max,
            min=0,
            max=2 * self.s_max,
        ) * b_same_entity + (1 - b_same_entity) * (
            2 * self.s_max + 1
        )  # [..., N_token, N_token]
        a_rel_chain = F.one_hot(d_chain, 2 * (self.s_max + 1))

        p = self.linear_no_bias(
            torch.cat(
                [a_rel_pos, a_rel_token, b_same_entity[..., None], a_rel_chain],
                dim=-1,
            ).float()
        )  # [..., N_token, N_token, 2 * (self.r_max + 1)+ 2 * (self.r_max + 1)+ 1 + 2 * (self.s_max + 1)] -> [..., N_token, N_token, c_z]
        return p


class FourierEmbedding(nn.Module):
    """
    Implements Algorithm 22 in AF3
    """

    def __init__(self, c: int, seed: int = 42) -> None:
        """
        Args:
            c (int): embedding dim.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        super(FourierEmbedding, self).__init__()
        self.c = c
        
        # Register buffers instead of parameters since they don't need gradients
        # This is more memory efficient and semantically clearer
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer('w', torch.randn(c, generator=generator))
        self.register_buffer('b', torch.randn(c, generator=generator))

    def forward(self, t_hat_timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_hat_timestep (torch.Tensor): the noise level
                [..., N_sample]

        Returns:
            torch.Tensor: the output fourier embedding
                [..., N_sample, c]
        """
        return torch.cos(
            input=2 * torch.pi * (t_hat_timestep.unsqueeze(dim=-1) * self.w + self.b)
        )
