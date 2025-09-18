from typing import Optional, Union

import torch
import torch.nn as nn

from protenix.model.modules.components import Transition
from protenix.model.modules.embedders import FourierEmbedding, RelativePositionEncoding
from protenix.model.modules.primitives import LayerNorm, LinearNoBias
from protenix.model.modules.transformer import DiffusionTransformer
from protenix.model.utils import (
    average_atom_to_token,
    centre_random_augmentation,
    expand_at_dim,
    one_hot,
)


class DiffusionConditioning(nn.Module):
    """
    Implements Algorithm 21 in AF3
    """

    def __init__(
        self,
        sigma_coord: float = 12.0,
        c_z: int = 128,
        c_s: int = 384,
        c_timestep_embedding: int = 256,
        target_dist_min: float = 1.0,
        target_dist_max: float = 30.0,
        target_dist_bin_size: float = 0.5,
    ) -> None:
        """
        Args:
            sigma_coord (torch.float, optional): the standard deviation of the coordinates. Defaults to 16.0.
            c_z (int, optional): hidden dim [for pair embedding].
            c_s (int, optional):  hidden dim [for single embedding].
            c_timestep_embedding (int, optional): timestep embedding dim.
            target_dist_min (float, optional): minimum distance for target pair distance binning. Defaults to 1.0.
            target_dist_max (float, optional): maximum distance for target pair distance binning. Defaults to 30.0.
            target_dist_bin_size (float, optional): bin size for target pair distance binning. Defaults to 0.5.
        """
        super(DiffusionConditioning, self).__init__()
        self.sigma_coord = sigma_coord
        self.c_z = c_z
        self.c_s = c_s

        ### Ligand features
        # Ligand reference position embeddings
        self.embed_ligand_ref_pos = LinearNoBias(3, self.c_s, init="lecun")
        self.embed_ligand_pair_offsets = LinearNoBias(
            3, self.c_z, init="lecun", high_precision=False
        )
        self.embed_ligand_pair_invd = LinearNoBias(1, self.c_z, init="lecun")
        self.embed_ref_mask = LinearNoBias(1, self.c_s, init="normal")
        self.embed_pair_offsets_valid = LinearNoBias(1, self.c_z, init="normal")

        # Ligand condition embeddings
        self.ligand_input_feature = {
            "ref_charge": 1,
            "ref_mask": 1,
            "ref_element": 128,
            "ref_atom_name_chars": 4 * 64,
        }
        self.ligand_condition_embed = LinearNoBias(
            sum(self.ligand_input_feature.values()), self.c_s, init="lecun"
        )

        # Token bonds
        self.token_bonds_embed = LinearNoBias(
            in_features=1, out_features=self.c_z, init="normal"
        )

        ### Relpos
        self.relpos = RelativePositionEncoding(c_z=c_z)
        self.layernorm_z = LayerNorm(2 * self.c_z)
        self.linear_no_bias_z = LinearNoBias(2 * self.c_z, self.c_z)

        ### Timestep embeddings
        self.fourier_embedding = FourierEmbedding(c=c_timestep_embedding)
        self.layernorm_latent_t = LayerNorm(c_timestep_embedding)
        self.latent_t_embed = LinearNoBias(c_timestep_embedding, self.c_s, init="lecun")
        self.layernorm_coord_t = LayerNorm(c_timestep_embedding)
        self.coord_t_embed = LinearNoBias(c_timestep_embedding, self.c_s, init="lecun")

        # Entity type embedding
        self.entity_features = {
            "is_protein_token": 1,
            "is_ligand_token": 1,
            "is_dna_token": 1,
            "is_rna_token": 1,
            "is_target_token": 1,
            "is_hotspot": 1,
        }
        self.entity_type_embed = LinearNoBias(
            in_features=sum(self.entity_features.values()),
            out_features=self.c_s,
            init="lecun",
        )

        ### Target features
        # Target distance binning parameters
        self.target_dist_min = target_dist_min
        self.target_dist_max = target_dist_max
        self.target_dist_bin_size = target_dist_bin_size
        self.n_target_dist_bins = int(
            (target_dist_max - target_dist_min) / target_dist_bin_size
        )

        # Initialize distance bins as parameters during initialization
        lower_bins = torch.arange(
            target_dist_min, target_dist_max, target_dist_bin_size
        )
        upper_bins = torch.cat([lower_bins[1:], lower_bins.new_tensor([1e6])], dim=-1)
        self.lower_bins = nn.Parameter(lower_bins, requires_grad=False)
        self.upper_bins = nn.Parameter(upper_bins, requires_grad=False)
        self.num_bins = len(lower_bins)

        # Target feature embeddings
        self.embed_target_restype = LinearNoBias(32, self.c_s, init="normal")
        self.embed_target_coord = LinearNoBias(3, self.c_s, init="lecun")
        self.embed_target_pair_dist = LinearNoBias(
            self.n_target_dist_bins, self.c_z, init="normal"
        )

        ### Transitions
        self.transition_z = Transition(
            c_in=self.c_z, n=2
        )  # TODO: are zero init's appropriate here with Muon?
        self.transition_s1 = Transition(c_in=self.c_s, n=2)
        self.transition_s2 = Transition(c_in=self.c_s, n=2)

    # @torch.compiler.disable
    def get_ligand_condition(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
    ) -> torch.Tensor:
        """Generate ligand conditioning for pair embedding initialization.

        Args:
            input_feature_dict: Input feature dictionary containing ligand data
            inplace_safe: Whether it is safe to use inplace operations

        Returns:
            torch.Tensor: Initial single and pair embeddingwith ligand conditioning
                 * [..., n_token, c_s]
                 * [..., n_token, n_token, c_z]
        """
        atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
        atom_mask = input_feature_dict["atom_mask"]
        n_token = input_feature_dict["residue_index"].shape[-1]
        n_atom = input_feature_dict["ref_pos"].shape[-2]
        batch_shape = input_feature_dict["ref_pos"].shape[:-2]

        ### Create ligand pair conditioning
        # 1. Get token-wise ligand reference positions
        # [..., N_atom, 3]
        is_ligand = input_feature_dict["is_ligand"].unsqueeze(-1)  # [..., n_atom, 1]
        ligand_ref_pos = (
            input_feature_dict["ref_pos"] * is_ligand
        )  # (zero out non-ligand atoms)

        # 2. Aggregate per-atom reference positions to per-token reference positions
        # [..., N_atom, 3] -> [..., N_token, 3]
        ligand_ref_pos = average_atom_to_token(
            x_atom=ligand_ref_pos,
            atom_to_token_idx=atom_to_token_idx,
            atom_mask=atom_mask,
            n_token=n_token,
        )

        # Compute v_lm mask
        token_ref_space_uid = average_atom_to_token(
            x_atom=input_feature_dict["ref_space_uid"].unsqueeze(-1),
            atom_to_token_idx=atom_to_token_idx,
            atom_mask=atom_mask,
            n_token=n_token,
        )  # [..., N_token, 1]
        v_lm = (
            token_ref_space_uid[..., None, :, :].int()
            == token_ref_space_uid[..., :, None, :].int()
        ).to(ligand_ref_pos.dtype)  # [..., n_token, n_token, 1]

        # 3. Compute ligand ref pos displacement
        d_lm = (
            ligand_ref_pos[..., :, None, :] - ligand_ref_pos[..., None, :, :]
        )  # [..., n_token, n_token, 3]
        z = self.embed_ligand_pair_offsets(d_lm)
        z = z + self.embed_pair_offsets_valid(v_lm)

        # 4. Embed pairwise inverse squared distances
        z = z + self.embed_ligand_pair_invd(
            1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))
        )  # [..., n_token, n_token, c_z]
        z = z * v_lm  # mask out inter-conformers

        ### Create ligand single conditioning
        # Embed ligand atom-wise features
        c_s = (
            self.ligand_condition_embed(
                torch.cat(
                    [
                        input_feature_dict[name]
                        .reshape(*batch_shape, n_atom, self.ligand_input_feature[name])
                        .to(d_lm.dtype)
                        for name in self.ligand_input_feature
                    ],
                    dim=-1,
                )
            )
            * is_ligand
        )  # [..., N_atom, c_s]

        c_s = average_atom_to_token(
            x_atom=c_s,
            atom_to_token_idx=atom_to_token_idx,
            atom_mask=atom_mask,
            n_token=n_token,
        )  # [..., N_token, c_s]

        c_s = c_s + self.embed_ligand_ref_pos(ligand_ref_pos)
        return c_s, z

    # @torch.compiler.disable
    def get_target_condition(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
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
            target_coord, mask=is_target_token, N_sample=1, s_trans=1.0
        ).squeeze(-3)  # [..., N_token, c_s]

        # 1. Target restype embedding (single embedding)
        s_target_restype = self.embed_target_restype(
            target_restype
        )  # [..., N_token, c_s]

        # 2. Target center coordinate embedding (single embedding)
        s_target_coord = self.embed_target_coord(
            target_coord_augment / self.sigma_coord
        )  # [..., N_token, c_s]
        s_target = s_target_restype + s_target_coord

        # 3. Target pair distance embedding (pair embedding)
        # Compute pairwise distances between target center coordinates
        pair_dists = torch.norm(
            target_coord_augment[..., None, :, :]
            - target_coord_augment[..., :, None, :],
            dim=-1,
        )  # [..., N_token, N_token]

        # One-hot encode distances
        pair_dist_onehot = one_hot(
            pair_dists,
            self.lower_bins,
            self.upper_bins,
        )  # [..., N_token, N_token, n_bins]

        # Embed one-hot distances
        z_target = self.embed_target_pair_dist(
            pair_dist_onehot
        )  # [..., N_token, N_token, c_z]

        # Mask pair embeddings for target tokens only
        target_mask_pair = (
            is_target_token[..., None, :] * is_target_token[..., :, None]
        )  # [..., N_token, N_token, 1]
        z_target = (
            z_target * target_mask_pair[..., None]
        )  # [..., N_token, N_token, c_z]

        return s_target, z_target

    # @torch.compile
    def forward(
        self,
        r_noisy_coord: torch.Tensor,
        coord_timestep: torch.Tensor,
        latent_timestep: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        inplace_safe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            r_noisy_coord (torch.Tensor): noisy coordinates
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
                - z (torch.Tensor): [..., N_tokens, N_tokens, c_z]
        """
        dtype = r_noisy_coord.dtype

        # Grab data on inputs
        n_token = input_feature_dict["residue_index"].shape[-1]

        # Initialize pair embedding with ligand conformer features
        s_ligand, z_ligand = self.get_ligand_condition(input_feature_dict)

        # Target features
        s_target, z_target = self.get_target_condition(input_feature_dict)

        # Add target embeddings
        s = s_target + s_ligand
        z = z_target + z_ligand

        # Add token bonds to pair embedding
        z = z + self.token_bonds_embed(
            input_feature_dict["token_bonds"].unsqueeze(-1)
        )  # [..., N_token, N_token, c_z]

        ### Relative position encodings
        # Concatenate z_init with relative position encoding and embed (Line 1)
        relpos_embed = self.relpos(
            asym_id=input_feature_dict["asym_id"],
            residue_index=input_feature_dict["residue_index"],
            entity_id=input_feature_dict["entity_id"],
            sym_id=input_feature_dict["sym_id"],
            token_index=input_feature_dict["token_index"],
        )
        z = torch.cat(
            tensors=[z, relpos_embed],
            dim=-1,
        )  # [..., N_tokens, N_tokens, 2*c_z]

        # Normalise and embed pair embedding (Line 2)
        z = self.linear_no_bias_z(self.layernorm_z(z))

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
        s = s.unsqueeze(-3)
        s = s + expand_at_dim(
            self.latent_t_embed(self.layernorm_latent_t(timestep_latent)),
            dim=-2,
            n=n_token,
        )  # [..., N_sample, N_tokens, c_s]
        s = s + expand_at_dim(
            self.coord_t_embed(self.layernorm_coord_t(timestep_coord)),
            dim=-2,
            n=n_token,
        )  # [..., N_sample, N_token, c_s]

        # Embed entity type and target using the is_X_token features
        s_entity_type = self.entity_type_embed(
            torch.stack(
                [input_feature_dict[name].to(dtype) for name in self.entity_features],
                dim=-1,
            )
        )  # [..., N_token, c_s]
        s = s + s_entity_type.unsqueeze(-3)

        ### Transition for both single and pair embeddings
        # Send single embedding through two transitions
        s = s + self.transition_s1(s)
        s = s + self.transition_s2(s)

        # Send pair embedding through one transition
        z = z + self.transition_z(z)

        return s, z


class DiffusionModule(nn.Module):
    """
    Implements Algorithm 20 in AF3
    """

    def __init__(
        self,
        sigma_coord: float = 12.0,
        sigma_latent: float = 3.0,
        c_latent: int = 8,
        c_z: int = 128,
        c_token: int = 768,
        c_s: int = 384,
        c_s_inputs: int = 449,
        # atom_encoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        transformer: dict[str, int] = {"num_blocks": 24, "num_heads": 16},
        # atom_decoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        dropout: float = 0.0,
        blocks_per_ckpt: Optional[int] = None,
        use_fine_grained_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            sigma_coord (torch.float, optional): the standard deviation of the coordinates. Defaults to 12.0.
            sigma_latent (torch.float, optional): the standard deviation of the latents. Defaults to 3.0.
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
        self.sigma_coord = sigma_coord
        self.sigma_latent = sigma_latent
        self.c_token = c_token
        self.c_s_inputs = c_s_inputs
        self.c_s = c_s
        self.c_z = c_z

        # Grad checkpoint setting
        self.blocks_per_ckpt = blocks_per_ckpt
        self.use_fine_grained_checkpoint = use_fine_grained_checkpoint

        self.diffusion_conditioning = DiffusionConditioning(
            sigma_coord=sigma_coord,
            c_z=c_z,
            c_s=c_s,
        )
        # Coordinate embedding layer
        self.linear_no_bias_r_embed = LinearNoBias(
            3,
            c_token,  # init="lecun"
        )

        # Latent embedding layer
        # self.layernorm_latent = LayerNorm(c_latent)
        self.linear_no_bias_latent_embed = LinearNoBias(
            c_latent,
            c_token,  # init="lecun"
        )

        # Alg20: line4
        self.layernorm_s = LayerNorm(c_s)
        self.linear_no_bias_s = LinearNoBias(
            in_features=c_s, out_features=c_token, init="zeros"
        )
        self.diffusion_transformer = DiffusionTransformer(
            **transformer, c_token=c_token, c_s=c_s, c_z=c_z
        )
        self.layernorm_a = LayerNorm(c_token)

        # Coordinate velocity head
        self.coord_head_transition = Transition(c_in=c_token, n=2)
        self.coord_final_layernorm = LayerNorm(c_token)
        self.coord_final_linear = LinearNoBias(c_token, 3, init="zeros")

        # Latent velocity head
        self.latent_head_transition = Transition(c_in=c_token, n=2)
        self.latent_final_layernorm = LayerNorm(c_token)
        self.latent_final_linear = LinearNoBias(c_token, c_latent, init="zeros")

    def coord_head(self, a_token: torch.Tensor):
        """Coordinate head for velocity prediction."""
        a_token = a_token + self.coord_head_transition(a_token)
        return self.coord_final_linear(self.coord_final_layernorm(a_token))

    def latent_head(self, a_token: torch.Tensor):
        """Latent head for velocity prediction."""
        a_token = a_token + self.latent_head_transition(a_token)
        return self.latent_final_linear(self.latent_final_layernorm(a_token))

    def f_forward(
        self,
        r_noisy_coord: torch.Tensor,
        r_noisy_latent: torch.Tensor,
        coord_timestep: torch.Tensor,
        latent_timestep: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        inplace_safe: bool = False,
        use_fine_grained_checkpoint: bool = False,  # TODO: temporary
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """The raw network to be trained.
        This is v_theta(x_t, t)
        Here, c_noise(sigma) is computed in Conditioning module.

        Args:
            r_noisy_coord (torch.Tensor): scaled noisy coords (i.e., c_in * x)
                [..., N_sample, N_token, 3]
            r_noisy_latent (torch.Tensor): scaled noisy latents
                [..., N_sample, N_token, c_latent]
            coord_timestep (torch.Tensor): the noising timestep for coordinates
                [..., N_sample]
            latent_timestep (torch.Tensor): the noising timestep for latents
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input feature dictionary
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: coordinate and latent velocities
                - coord_velocity: [..., N_sample, N_token, 3]
                - latent_velocity: [..., N_sample, N_token, c_latent]
        """
        N_sample = r_noisy_coord.size(-3)
        assert coord_timestep.size(-1) == N_sample
        assert latent_timestep.size(-1) == N_sample

        # Compute pair mask
        token_mask = input_feature_dict["token_mask"]  # [..., N_token]
        token_mask = token_mask.unsqueeze(-2)  # add N_sample dim
        pair_mask = token_mask[..., :, None] * token_mask[..., None, :]
        # [..., 1, N_token, N_token]

        s, z = self.diffusion_conditioning(
            r_noisy_coord=r_noisy_coord,
            coord_timestep=coord_timestep,
            latent_timestep=latent_timestep,
            input_feature_dict=input_feature_dict,
            inplace_safe=inplace_safe,
        )  # [..., N_sample, N_token, c_s], [..., N_token, N_token, c_z]

        # Embed noisy coordinates and latents into token space
        a_coord = self.linear_no_bias_r_embed(r_noisy_coord)
        a_latent = self.linear_no_bias_latent_embed(r_noisy_latent)

        a_token = a_coord + a_latent  # [..., N_sample, N_token, c_token]

        # Mix single conditioning into token representation
        a_token = a_token + self.linear_no_bias_s(self.layernorm_s(s))
        # [..., N_sample, N_token, c_token]

        # Apply diffusion transformer on token representations
        a_token = a_token * token_mask[..., None]
        z = z.unsqueeze(-4)
        z = z * pair_mask[..., None]

        a_token = self.diffusion_transformer(a=a_token, s=s, z=z, pair_mask=pair_mask)
        a_token = a_token * token_mask[..., None]
        a_token = self.layernorm_a(a_token)

        # Apply separate velocity prediction heads for coordinates and latents
        latent_velocity = self.latent_head(a_token)
        coord_velocity = self.coord_head(a_token)

        return coord_velocity, latent_velocity

    def c_in(self, timestep: torch.Tensor) -> torch.Tensor:
        return 1 / (timestep**2 + (1 - timestep) ** 2).sqrt()

    def c_out(self, timestep: torch.Tensor) -> torch.Tensor:
        return 1 - timestep  # velocity parametrization

    def c_skip(self, timestep: torch.Tensor) -> torch.Tensor:
        return 1.0

    def velocity_c_out(self, timestep):
        # Carlos, forgive me.
        cutoff = 0.8
        t_n = timestep
        std = (
            self.sigma_coord
            / (1 - cutoff)
            * ((1 - cutoff) ** 2 + (cutoff - t_n) ** 2 + (1 - t_n) ** 2).sqrt()
        )
        # TODO: remove the jank, unify with proper c_out
        return torch.where(timestep < cutoff, std, 1.0)

    def forward(
        self,
        noisy_coord: torch.Tensor,
        noisy_latent: torch.Tensor,
        coord_timestep: torch.Tensor,
        latent_timestep: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        inplace_safe: bool = False,
        return_velocities: bool = False,
        chunk_size: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        # Expand timesteps
        coord_timestep_ = coord_timestep[..., None, None]
        latent_timestep_ = latent_timestep[..., None, None]

        # Scale input noisy coords to unit variance
        r_noisy_coord = (noisy_coord / self.sigma_coord) * self.c_in(coord_timestep_)
        r_noisy_latent = (noisy_latent / self.sigma_latent) * self.c_in(
            latent_timestep_
        )

        # Predict velocities using separate heads for coordinates and latents
        r_coord, r_latent = self.f_forward(
            r_noisy_coord=r_noisy_coord,
            r_noisy_latent=r_noisy_latent,
            coord_timestep=coord_timestep,  # Return to original shape for f_forward
            latent_timestep=latent_timestep,
            input_feature_dict=input_feature_dict,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        # Bring to correct scale
        coord_v = self.sigma_coord * r_coord * self.velocity_c_out(coord_timestep_)
        latent_v = self.sigma_latent * r_latent

        # Apply skip connections and output scaling for both coordinate and latent predictions
        coord_pred = noisy_coord * self.c_skip(coord_timestep_) + coord_v * self.c_out(
            coord_timestep_
        )
        latent_pred = noisy_latent * self.c_skip(
            latent_timestep_
        ) + latent_v * self.c_out(latent_timestep_)

        if return_velocities:
            return coord_v, latent_v

        return coord_pred, latent_pred
