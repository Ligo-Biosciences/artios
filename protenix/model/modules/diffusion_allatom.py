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
from protenix.model.modules.primitives import LayerNorm, LinearNoBias, Transition
from protenix.model.modules.transformer import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    DiffusionTransformer,
)
from protenix.model.utils import (
    add,
    aggregate_atom_to_token,
    centre_random_augmentation,
    expand_at_dim,
    one_hot,
)
from protenix.openfold_local.utils.checkpointing import get_checkpoint_fn


class DiffusionConditioning(nn.Module):
    """
    Implements Algorithm 21 in AF3
    """

    def __init__(
        self,
        sigma_data: float = 12.0,
        c_z: int = 128,
        c_s: int = 384,
        c_timestep_embedding: int = 256,
        target_dist_min: float = 1.0,
        target_dist_max: float = 30.0,
        target_dist_bin_size: float = 0.5,
    ) -> None:
        """
        Args:
            sigma_data (torch.float, optional): the standard deviation of the data. Defaults to 16.0.
            c_z (int, optional): hidden dim [for pair embedding].
            c_s (int, optional):  hidden dim [for single embedding].
            c_timestep_embedding (int, optional): timestep embedding dim.
            target_dist_min (float, optional): minimum distance for target pair distance binning. Defaults to 1.0.
            target_dist_max (float, optional): maximum distance for target pair distance binning. Defaults to 30.0.
            target_dist_bin_size (float, optional): bin size for target pair distance binning. Defaults to 0.5.
        """
        super(DiffusionConditioning, self).__init__()
        self.sigma_data = sigma_data
        self.c_z = c_z
        self.c_s = c_s

        # Target distance binning parameters
        self.target_dist_min = target_dist_min
        self.target_dist_max = target_dist_max
        self.target_dist_bin_size = target_dist_bin_size
        self.n_target_dist_bins = int(
            (target_dist_max - target_dist_min) / target_dist_bin_size
        )

        ### Ligand reference position embeddings
        self.linear_no_bias_d = LinearNoBias(in_features=3, out_features=self.c_z)
        self.linear_no_bias_invd = LinearNoBias(
            in_features=1, out_features=self.c_z, init="zeros"
        )

        # Token bonds
        self.linear_no_bias_token_bonds = LinearNoBias(
            in_features=1, out_features=self.c_z, init="zeros"
        )

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

        # Latent and coordinate timestep embeddings
        self.layernorm_latent_t = LayerNorm(c_timestep_embedding)
        self.linear_no_bias_latent_t = LinearNoBias(
            in_features=c_timestep_embedding, out_features=self.c_s
        )
        self.layernorm_coord_t = LayerNorm(c_timestep_embedding)
        self.linear_no_bias_coord_t = LinearNoBias(
            in_features=c_timestep_embedding, out_features=self.c_s
        )

        # Line 10-12
        self.transition_s1 = Transition(c_in=self.c_s, n=2)
        self.transition_s2 = Transition(c_in=self.c_s, n=2)

        # Target feature embeddings
        self.linear_target_restype = LinearNoBias(
            in_features=32, out_features=self.c_s
        )  # Target restype one-hot -> single embedding
        self.linear_target_coord = LinearNoBias(
            in_features=3, out_features=self.c_s
        )  # Target center coords -> single embedding
        self.linear_target_pair_dist = LinearNoBias(
            in_features=self.n_target_dist_bins, out_features=self.c_z
        )  # Target pair distances -> pair embedding

        # Noisy coordinate feature embeddings
        self.linear_noisy_pair_dist = LinearNoBias(
            in_features=self.n_target_dist_bins, out_features=self.c_z
        )  # Noisy pair distances -> pair embedding

    def get_ligand_condition(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """Generate ligand conditioning for pair embedding initialization.

        Args:
            input_feature_dict: Input feature dictionary containing ligand data
            inplace_safe: Whether it is safe to use inplace operations

        Returns:
            torch.Tensor: Initial pair embedding z with ligand conditioning
                [..., n_token, n_token, c_z]
        """
        atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
        n_token = input_feature_dict["residue_index"].shape[-1]

        # 1. Get token-wise ligand reference positions
        # [..., N_atom, 3]
        is_ligand = input_feature_dict["is_ligand"].unsqueeze(-1)  # [..., n_atom, 1]
        ligand_ref_pos = (
            input_feature_dict["ref_pos"] * is_ligand
        )  # (zero out non-ligand atoms)

        # 2. Aggregate per-atom reference positions to per-token reference positions
        # [..., N_atom, 3] -> [..., N_token, 3]
        ligand_ref_pos = aggregate_atom_to_token(
            x_atom=ligand_ref_pos,
            atom_to_token_idx=atom_to_token_idx,
            n_token=n_token,
            reduce="mean",
        )

        # Compute v_lm mask
        token_ref_space_uid = aggregate_atom_to_token(
            x_atom=input_feature_dict["ref_space_uid"].unsqueeze(-1),
            atom_to_token_idx=atom_to_token_idx,
            n_token=n_token,
            reduce="mean",
        )  # [..., N_token, 1]
        v_lm = (
            token_ref_space_uid[..., None, :, :] == token_ref_space_uid[..., None, :]
        )  # [..., n_token, n_token, 1]

        # 3. Compute ligand ref pos displacement
        d_lm = (
            ligand_ref_pos[..., None, :] - ligand_ref_pos[..., None, :, :]
        )  # [..., n_token, n_token, 3]
        z = self.linear_no_bias_d(d_lm) * v_lm  # [..., n_token, n_token, c_z]

        # 4. Embed pairwise inverse squared distances
        invd_emb = (
            self.linear_no_bias_invd(1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True)))
            * v_lm
        )
        z = add(z, invd_emb, inplace_safe=inplace_safe)

        return z

    def get_target_condition(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        N_sample: int,
        inplace_safe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate target embeddings from target features.

        Args:
            input_feature_dict: Input feature dictionary containing target data
            N_sample (int): number of samples for target coordinate augmentation
            inplace_safe (bool): Whether it is safe to use inplace operations

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Target single and pair embeddings
                - s_target (torch.Tensor): [..., N_token, c_s]
                - z_target (torch.Tensor): [..., N_token, N_token, c_z]
        """
        target_coord = input_feature_dict[
            "target_centre_coordinate"
        ]  # [..., N_token, 3]
        target_restype = input_feature_dict["target_restype"]  # [..., N_token, 32]
        is_target_token = input_feature_dict["is_target_token"]  # [..., N_token]

        # Augment the coordinates
        target_coord_augment = centre_random_augmentation(
            target_coord, mask=is_target_token, N_sample=N_sample, s_trans=1.0
        )

        # 1. Target restype embedding (single embedding)
        s_target_restype = self.linear_target_restype(
            target_restype
        )  # [..., N_token, c_s]

        # 2. Target center coordinate embedding (single embedding)
        s_target_coord = self.linear_target_coord(
            centre_random_augmentation(
                target_coord, mask=is_target_token, N_sample=N_sample, s_trans=1.0
            )
            / self.sigma_data
        )  # [..., N_sample, N_token, c_s]

        # Combine single embeddings
        s_target = (
            s_target_restype.unsqueeze(-3) + s_target_coord
        )  # [..., N_sample, N_token, c_s]

        # 3. Target pair distance embedding (pair embedding)
        # Compute pairwise distances between target center coordinates
        target_coord_i = target_coord_augment.unsqueeze(
            -2
        )  # [..., N_sample, N_token, 1, 3]
        target_coord_j = target_coord_augment.unsqueeze(
            -3
        )  # [..., N_sample, 1, N_token, 3]
        pair_dists = torch.norm(
            target_coord_i - target_coord_j, dim=-1
        )  # [..., N_token, N_token]

        # Create distance bins using the one_hot utility
        lower_bins = torch.arange(
            self.target_dist_min,
            self.target_dist_max,
            self.target_dist_bin_size,
            device=pair_dists.device,
            dtype=pair_dists.dtype,
        )  # [n_bins]
        upper_bins = lower_bins + self.target_dist_bin_size  # [n_bins]

        # One-hot encode distances
        pair_dist_onehot = one_hot(
            pair_dists, lower_bins, upper_bins
        )  # [..., N_token, N_token, n_bins]

        # Embed one-hot distances
        z_target = self.linear_target_pair_dist(
            pair_dist_onehot
        )  # [..., N_token, N_token, c_z]

        # Mask pair embeddings for target tokens only
        # TODO: safety measure necessary here?
        # target_mask = is_target_token.unsqueeze(-1)  # [..., N_token, 1]
        # target_mask_pair = (
        #    target_mask.unsqueeze(-2) * target_mask.unsqueeze(-3)
        # )  # [..., N_token, N_token, 1]
        # z_target = z_target_dist * target_mask_pair  # [..., N_token, N_token, c_z]

        return s_target, z_target

    def forward(
        self,
        r_noisy: torch.Tensor,
        coord_timestep: torch.Tensor,
        latent_timestep: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        inplace_safe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            r_noisy (torch.Tensor): noisy coordinates
                [..., N_sample, 3]
            coord_timestep (torch.Tensor): the noising timestep for coords
                [..., N_sample]
            latent_timestep (torch.Tensor): the noising timestep for latents
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            inplace_safe (bool): Whether it is safe to use inplace operations.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: embeddings s and z
                - s (torch.Tensor): [..., N_sample, N_tokens, c_s]
                - z (torch.Tensor): [..., N_sample, N_tokens, N_tokens, c_z]

        TODO: Change all reference to lm indexing to ij indexing since now we are using tokenwise representations
        TODO: this will be quite memory intensive conditioning.
        """
        N_sample = coord_timestep.shape[-1]

        # Grab data on inputs
        n_token = input_feature_dict["residue_index"].shape[-1]

        ### Initialize pair embedding with ligand conformer features
        z = self.get_ligand_condition(input_feature_dict, inplace_safe)

        # Add token bonds to pair embedding
        z = add(
            z,
            self.linear_no_bias_token_bonds(
                input_feature_dict["token_bonds"].unsqueeze(-1)
            ),
            inplace_safe=inplace_safe,
        )

        ### Relative position encodings
        # Concatenate z_init with relative position encoding and embed (Line 1)
        z = torch.cat(
            tensors=[z, self.relpe(input_feature_dict)], dim=-1
        )  # [..., N_sample, N_tokens, N_tokens, 2*c_z]

        # Normalise and embed pair embedding (Line 2)
        z = self.linear_no_bias_z(self.layernorm_z(z))

        ### Distance featurization for noisy coordinates
        # Compute pairwise distances between noisy coordinates
        r_noisy_i = r_noisy.unsqueeze(-2)  # [..., N_sample, N_token, 1, 3]
        r_noisy_j = r_noisy.unsqueeze(-3)  # [..., N_sample, 1, N_token, 3]
        noisy_pair_dists = torch.norm(
            r_noisy_i - r_noisy_j, dim=-1
        )  # [..., N_sample, N_token, N_token]

        # Create distance bins using the one_hot utility
        lower_bins = torch.arange(
            self.target_dist_min,
            self.target_dist_max,
            self.target_dist_bin_size,
            device=noisy_pair_dists.device,
            dtype=noisy_pair_dists.dtype,
        )  # [n_bins]
        upper_bins = lower_bins + self.target_dist_bin_size  # [n_bins]

        # One-hot encode distances
        noisy_pair_dist_onehot = one_hot(
            noisy_pair_dists, lower_bins, upper_bins
        )  # [..., N_sample, N_token, N_token, n_bins]

        # Embed one-hot distances
        z_noisy = self.linear_noisy_pair_dist(
            noisy_pair_dist_onehot
        )  # [..., N_sample, N_token, N_token, c_z]

        # Add noisy distance features to pair embedding
        z = add(z.unsqueeze(-4), z_noisy, inplace_safe=inplace_safe)

        ### Initialize single embedding
        # 1. Fourier embed the timestep
        timestep_coord = self.fourier_embedding(
            t_hat_timestep=(
                2 * coord_timestep - 1
            )  # Rescale to [-1, 1] for smoother embedding
        ).to(z.dtype)  # [..., N_sample, c_noise_embedding]
        timestep_latent = self.fourier_embedding(
            t_hat_timestep=(2 * latent_timestep - 1)
        ).to(z.dtype)

        # 2. Normalise and embed both timesteps
        s = expand_at_dim(
            self.linear_no_bias_latent_t(self.layernorm_latent_t(timestep_latent)),
            dim=-2,
            n=n_token,
        )  # [..., N_sample, N_tokens, c_s]
        s = add(
            s,
            expand_at_dim(
                self.linear_no_bias_coord_t(self.layernorm_coord_t(timestep_coord)),
                dim=-2,
                n=n_token,
            ),
            inplace_safe=inplace_safe,
        )  # [..., N_sample, N_token, c_s]

        # Target features
        s_target, z_target = self.get_target_condition(
            input_feature_dict, N_sample=N_sample, inplace_safe=inplace_safe
        )

        # Add target embeedings
        s = add(s, s_target, inplace_safe=inplace_safe)
        z = add(z, z_target, inplace_safe=inplace_safe)

        ### Transition for both single and pair embeddings
        # Send single embedding through two transitions
        s = add(s, self.transition_s1(s), inplace_safe=inplace_safe)
        s = add(s, self.transition_s2(s), inplace_safe=inplace_safe)

        # Send pair embedding through one transition
        z = add(z, self.transition_z(z), inplace_safe=inplace_safe)

        return s, z


class DiffusionModule(nn.Module):
    """
    Implements Algorithm 20 in AF3
    """

    def __init__(
        self,
        sigma_data: float = 12.0,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        atom_encoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        transformer: dict[str, int] = {"n_blocks": 24, "n_heads": 16},
        atom_decoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        dropout: float = 0.0,
        blocks_per_ckpt: Optional[int] = None,
        use_fine_grained_checkpoint: bool = False,
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
            dropout (float, optional): dropout probability for attention weights and transition block. Defaults to 0.0.
            blocks_per_ckpt: number of atom_encoder/transformer/atom_decoder blocks in each activation checkpoint
                Size of each chunk. A higher value corresponds to fewer
                checkpoints, and trades memory for speed. If None, no checkpointing is performed.
            use_fine_grained_checkpoint: whether use fine-gained checkpoint for finetuning stage 2
                only effective if blocks_per_ckpt is not None.
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
            sigma_data=self.sigma_data,
            c_z=c_z,
            c_s=c_s,  # c_s_inputs=c_s_inputs
        )
        self.atom_attention_encoder = (
            AtomAttentionEncoder(  # TODO: no need for this anymore
                **atom_encoder,
                c_atom=c_atom,
                c_atompair=c_atompair,
                c_token=c_token,
                has_noisy_coords=True,
                c_s=c_s,
                c_z=c_z,
                blocks_per_ckpt=blocks_per_ckpt,
            )
        )
        # Alg20: line4
        # self.layernorm_s = LayerNorm(c_s)
        # self.linear_no_bias_s = LinearNoBias(in_features=c_s, out_features=c_token, init="zeros")
        self.diffusion_transformer = DiffusionTransformer(
            **transformer,
            c_a=c_token,
            c_s=c_s,
            c_z=c_z,
            dropout=dropout,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.layernorm_a = LayerNorm(c_token)
        self.atom_attention_decoder = (
            AtomAttentionDecoder(  # TODO: no need for this either
                **atom_decoder,
                c_token=c_token,
                c_atom=c_atom,
                c_atompair=c_atompair,
                blocks_per_ckpt=blocks_per_ckpt,
            )
        )

        # TODO: two lightweight heads for coord and latent prediction here

    def f_forward(
        self,
        r_noisy: torch.Tensor,
        timestep: torch.Tensor,  # TODO: two different timesteps for coord and latent
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
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
            s, z = checkpoint_fn(
                self.diffusion_conditioning,
                timestep,
                input_feature_dict,
                inplace_safe,
            )
        else:
            s, z = self.diffusion_conditioning(
                timestep=timestep,
                input_feature_dict=input_feature_dict,
                inplace_safe=inplace_safe,
            )  # [..., N_sample, N_token, c_s], [..., N_token, N_token, c_z]

        # Expand embeddings to match N_sample
        z = expand_at_dim(
            z,
            dim=-4,  # TODO: since we'll embed the distances, there will be no more broadcasting here.
            n=1,  # rely on broadcasting for N_sample
        )  # [..., 1, N_token, N_token, c_z]

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(  # TODO: delete
            input_feature_dict=input_feature_dict,
            r_l=r_noisy,
            s=s,  # Replaced s_trunk
            z=z,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        # Add token-level conditioning from atom encoder
        # TODO: put this back in since it'll have target coords etc.
        # s = add(s, c_token, inplace_safe=inplace_safe) # [..., N_sample, N_token, c_s]

        # Mix single conditioning into token representation
        # a_token = add(
        #    a_token,
        #    self.linear_no_bias_s(self.layernorm_s(s)),
        #    inplace_safe=inplace_safe
        # ) # [..., N_sample, N_token, c_token]

        # Full self-attention on token level.
        a_token = self.diffusion_transformer(  # NOTE: this looks like it'll remain unchanged for now.
            a=a_token,
            s=s,
            z=z,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        a_token = self.layernorm_a(a_token)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        velocity_pred = self.atom_attention_decoder(  # TODO: two different velocity preds for coords and latents
            input_feature_dict=input_feature_dict,
            a=a_token,
            q_skip=q_skip,
            c_skip=c_skip,
            p_skip=p_skip,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        return velocity_pred

    def c_in(self, timestep: torch.Tensor) -> torch.Tensor:
        return 1 / (self.sigma_data * (timestep**2 + (1 - timestep) ** 2).sqrt())

    def c_out(self, timestep: torch.Tensor) -> torch.Tensor:
        # return 1 - timestep
        return self.sigma_data * (1 - timestep)

    def c_skip(self, timestep: torch.Tensor) -> torch.Tensor:
        return 1.0

    def forward(
        self,
        x_noisy: torch.Tensor,
        timestep: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
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
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        # Scale output with direct function calls
        coords_pred = x_noisy * self.c_skip(
            timestep[..., None, None]
        ) + r_update * self.c_out(
            timestep[..., None, None]
        )  # return 2 separate entities (in a dict potentially)

        return coords_pred
