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

from typing import Optional, Union

import torch
import torch.nn as nn

from protenix.model.modules.embedders import FourierEmbedding, RelativePositionEncoding
from protenix.model.modules.primitives import LinearNoBias, Transition
from protenix.model.modules.transformer import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    DiffusionTransformer,
)
from protenix.model.utils import expand_at_dim, add
from protenix.model.modules.primitives import LayerNorm
from protenix.openfold_local.utils.checkpointing import get_checkpoint_fn


class DiffusionConditioning(nn.Module):
    """
    Implements Algorithm 21 in AF3
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_z: int = 128,
        c_s: int = 384,
        c_s_inputs: int = 386,
        c_timestep_embedding: int = 256,
    ) -> None:
        """
        Args:
            sigma_data (torch.float, optional): the standard deviation of the data. Defaults to 16.0.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_s_inputs (int, optional): input embedding dim from InputEmbedder. Defaults to 449.
            c_noise_embedding (int, optional): noise embedding dim. Defaults to 256.
        """
        super(DiffusionConditioning, self).__init__()
        self.sigma_data = sigma_data
        self.c_z = c_z
        self.c_s = c_s
        self.c_s_inputs = c_s_inputs

        ### From Main Inference Loop
        # Line 2-3
        self.layernorm_s_init = LayerNorm(self.c_s_inputs)
        self.linear_no_bias_s_init = LinearNoBias(in_features=self.c_s_inputs, out_features=self.c_s) 
        self.linear_no_bias_z_init_i = LinearNoBias(in_features=self.c_s, out_features=self.c_z)
        self.linear_no_bias_z_init_j = LinearNoBias(in_features=self.c_s, out_features=self.c_z)

        # Line 4-5
        self.linear_no_bias_token_bonds = LinearNoBias(in_features=1, out_features=self.c_z)

        ### From Diffusion Conditioning
        #  Line 1-2
        self.relpe = RelativePositionEncoding(c_z=c_z)
        self.layernorm_z = LayerNorm(2 * self.c_z)
        self.linear_no_bias_z = LinearNoBias(
            in_features=2 * self.c_z, out_features=self.c_z
        )
        # Line 3-5:
        self.transition_z = Transition(c_in=self.c_z, n=2)

        # Line 8-9
        self.fourier_embedding = FourierEmbedding(c=c_timestep_embedding)
        self.layernorm_t = LayerNorm(c_timestep_embedding)
        self.linear_no_bias_t = LinearNoBias(
            in_features=c_timestep_embedding, out_features=self.c_s
        )
        # Line 10-12
        self.transition_s1 = Transition(c_in=self.c_s, n=2)
        self.transition_s2 = Transition(c_in=self.c_s, n=2)
        print(f"Diffusion Module has {self.sigma_data}")

    def forward(
        self,
        timestep: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        inplace_safe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            timestep (torch.Tensor): the noising timestep (the t such that x_t = (1-t)*x_0 + t*x_1)
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            inplace_safe (bool): Whether it is safe to use inplace operations.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: embeddings s and z
                - s (torch.Tensor): [..., N_sample, N_tokens, c_s]
                - z (torch.Tensor): [..., N_N_tokens, N_tokens, c_z]
        """
        ### Logic from Main Inference Loop (Algorithm 1)
        # Create single embedding from s_inputs (Line 2)
        s = self.linear_no_bias_s_init(self.layernorm_s_init(s_inputs)) 

        # Create pair embedding from s_init (Line 3)
        z = add(
            self.linear_no_bias_z_init_i(s)[..., None, :],
            self.linear_no_bias_z_init_j(s)[..., None, :, :],
        )  #  [..., N_token, N_token, c_z]

        # Add token bonds to pair embedding (Line 5)
        z = add(
            z,
            self.linear_no_bias_token_bonds(input_feature_dict["token_bonds"].unsqueeze(-1)),
            inplace_safe=inplace_safe
        )

        ### Logic from Diffusion Conditioning (Algorithm 21)
        # Concatenate z_init with relative position encoding and embed (Line 1)
        z = torch.cat(
            tensors=[z, self.relpe(input_feature_dict)], dim=-1 
        ) # [..., N_tokens, N_tokens, 2*c_z]

        # Normalise and embed pair embedding (Line 2)
        z = self.linear_no_bias_z(self.layernorm_z(z)) 

        # Send pair embedding through one transition (Line 3-5)
        z = add(z, self.transition_z(z), inplace_safe=inplace_safe)

        # Fourier embed the noise level (Line 8) 
        t_hat_timestep = 2 * timestep - 1 # Rescale to [-1, 1] for smoother embedding
        timestep_t = self.fourier_embedding(t_hat_timestep=t_hat_timestep).to(s.dtype)  # [..., N_sample, c_noise_embedding]
        
        # Normalise, embed noise and add to single embed (Line 9)
        s = add(
            s.unsqueeze(dim=-3),
            self.linear_no_bias_t(
                self.layernorm_t(timestep_t)
            ).unsqueeze(dim=-2)
        )  # [..., N_sample, N_tokens, c_s]
        
        # Send single embedding through two transitions (Line 10-12)
        s = add(s, self.transition_s1(s), inplace_safe=inplace_safe)
        s = add(s, self.transition_s2(s), inplace_safe=inplace_safe)
 
        return s, z


class DiffusionModule(nn.Module):
    """
    Implements Algorithm 20 in AF3
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        atom_encoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        transformer: dict[str, int] = {"n_blocks": 24, "n_heads": 16},
        atom_decoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        blocks_per_ckpt: Optional[int] = None,
        use_fine_grained_checkpoint: bool = False,
        initialization: Optional[dict[str, Union[str, float, bool]]] = None,
    ) -> None:
        """
        Args:
            sigma_data (torch.float, optional): the standard deviation of the data. Defaults to 16.0.
            c_atom (int, optional): embedding dim for atom feature. Defaults to 128.
            c_atompair (int, optional): embedding dim for atompair feature. Defaults to 16.
            c_token (int, optional): feature channel of token (single a). Defaults to 768.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            c_s_inputs (int, optional): hidden dim [for single input embedding]. Defaults to 449.
            atom_encoder (dict[str, int], optional): configs in AtomAttentionEncoder. Defaults to {"n_blocks": 3, "n_heads": 4}.
            transformer (dict[str, int], optional): configs in DiffusionTransformer. Defaults to {"n_blocks": 24, "n_heads": 16}.
            atom_decoder (dict[str, int], optional): configs in AtomAttentionDecoder. Defaults to {"n_blocks": 3, "n_heads": 4}.
            blocks_per_ckpt: number of atom_encoder/transformer/atom_decoder blocks in each activation checkpoint
                Size of each chunk. A higher value corresponds to fewer
                checkpoints, and trades memory for speed. If None, no checkpointing is performed.
            use_fine_grained_checkpoint: whether use fine-gained checkpoint for finetuning stage 2
                only effective if blocks_per_ckpt is not None.
            initialization: initialize the diffusion module according to initialization config.
        """

        super(DiffusionModule, self).__init__()
        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s_inputs = c_s_inputs
        self.c_s = c_s
        self.c_z = c_z

        # Grad checkpoint setting
        self.blocks_per_ckpt = blocks_per_ckpt
        self.use_fine_grained_checkpoint = use_fine_grained_checkpoint

        self.diffusion_conditioning = DiffusionConditioning(
            sigma_data=self.sigma_data, c_z=c_z, c_s=c_s, c_s_inputs=c_s_inputs
        )
        self.atom_attention_encoder = AtomAttentionEncoder(
            **atom_encoder,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            has_coords=True,
            c_s=c_s,
            c_z=c_z,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        # Alg20: line4
        self.layernorm_s = LayerNorm(c_s)
        self.linear_no_bias_s = LinearNoBias(in_features=c_s, out_features=c_token)
        self.diffusion_transformer = DiffusionTransformer(
            **transformer,
            c_a=c_token,
            c_s=c_s,
            c_z=c_z,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.layernorm_a = LayerNorm(c_token)
        self.atom_attention_decoder = AtomAttentionDecoder(
            **atom_decoder,
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            blocks_per_ckpt=blocks_per_ckpt,
        )

    def f_forward( #TODO: change to s and z's instead of single_s and pair_z
        self,
        r_noisy: torch.Tensor,
        timestep: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """The raw network to be trained.
        This is v_theta(x_t, t)
        Here, c_noise(sigma) is computed in Conditioning module.

        Args:
            r_noisy (torch.Tensor): scaled x_noisy (i.e., c_in * x)
                [..., N_sample, N_atom, 3]
            timestep (torch.Tensor): the noising timestep (the t such that x_t = (1-t)*x_0 + t*x_1)
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input feature
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: denoised coords
                [..., N_sample, N_atom, 3]
        """
        N_sample = r_noisy.size(-3)
        assert timestep.size(-1) == N_sample

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        # Conditioning, shared across difference samples
        # Diffusion_conditioning consumes 7-8G when token num is 768,
        # use checkpoint here if blocks_per_ckpt is not None.
        if blocks_per_ckpt:
            checkpoint_fn = get_checkpoint_fn()
            s_single, z_pair = checkpoint_fn(
                self.diffusion_conditioning,
                timestep,
                input_feature_dict,
                s_inputs,
                inplace_safe,
            )
        else:
            s_single, z_pair = self.diffusion_conditioning(
                timestep=timestep,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                inplace_safe=inplace_safe,
            )  # [..., N_sample, N_token, c_s], [..., N_token, N_token, c_z]

        # Expand embeddings to match N_sample
        z_pair = expand_at_dim(
            z_pair, dim=-4, n=N_sample
        )  # [..., N_sample, N_token, N_token, c_z]

        # Fine-grained checkpoint for finetuning stage 2 (token num: 768) for avoiding OOM
        if blocks_per_ckpt and self.use_fine_grained_checkpoint:
            checkpoint_fn = get_checkpoint_fn()
            a_token, q_skip, c_skip, p_skip = checkpoint_fn(
                self.atom_attention_encoder,
                input_feature_dict,
                r_noisy,
                s_single, # Replaced s_trunk
                z_pair,
                inplace_safe,
                chunk_size,
            )
        else:
            # Sequence-local Atom Attention and aggregation to coarse-grained tokens
            a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(
                input_feature_dict=input_feature_dict,
                r_l=r_noisy,
                s=s_single, # Replaced s_trunk
                z=z_pair,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        # Full self-attention on token level.
        a_token = add(
            a_token, 
            self.linear_no_bias_s(self.layernorm_s(s_single)), 
            inplace_safe=inplace_safe
        ) # [..., N_sample, N_token, c_token]
        
        a_token = self.diffusion_transformer(
            a=a_token,
            s=s_single,
            z=z_pair,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        a_token = self.layernorm_a(a_token)

        # Fine-grained checkpoint for finetuning stage 2 (token num: 768) for avoiding OOM
        if blocks_per_ckpt and self.use_fine_grained_checkpoint:
            checkpoint_fn = get_checkpoint_fn()
            r_update = checkpoint_fn(
                self.atom_attention_decoder,
                input_feature_dict,
                a_token,
                q_skip,
                c_skip,
                p_skip,
                inplace_safe,
                chunk_size,
            )
        else:
            # Broadcast token activations to atoms and run Sequence-local Atom Attention
            r_update = self.atom_attention_decoder(
                input_feature_dict=input_feature_dict,
                a=a_token,
                q_skip=q_skip,
                c_skip=c_skip,
                p_skip=p_skip,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )

        return r_update 

    def c_in(self, timestep: torch.Tensor) -> torch.Tensor:
        return 1 / (self.sigma_data * (timestep ** 2 + (1 - timestep) ** 2).sqrt())

    def c_out(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.sigma_data * (1 - timestep) / (timestep ** 2 + (1 - timestep) ** 2).sqrt()

    def c_skip(self, timestep: torch.Tensor) -> torch.Tensor:
        return timestep / (timestep ** 2 + (1 - timestep) ** 2)

    def forward(
        self,
        x_noisy: torch.Tensor,
        timestep: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        # Scale input noisy coords to unit variance
        r_noisy = x_noisy * self.c_in(timestep[..., None, None])

        # Predict velocity from noisy coords
        r_update = self.f_forward(
            r_noisy=r_noisy,
            timestep=timestep,  # Return to original shape for f_forward
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        # Scale output with direct function calls
        coords_pred = x_noisy * self.c_skip(timestep[..., None, None]) + \
                      r_update * self.c_out(timestep[..., None, None])

        return coords_pred
