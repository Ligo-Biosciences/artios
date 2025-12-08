"""
TODO: this filename should eventually just be called diffusion.py I think.
"""

from typing import Optional, Union

import torch
import torch.nn as nn

from protenix.model.modules.components import Transition
from protenix.model.modules.embedders import FourierEmbedding, RelativePositionEncoding
from protenix.model.modules.primitives import LayerNorm, LinearNoBias
from protenix.model.modules.transformer import DiffusionTransformer
from protenix.model.utils import (
    average_atom_to_token,
    broadcast_token_to_atom,
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
        motif_dist_min: float = 1.0,
        motif_dist_max: float = 15.0,
        motif_dist_bin_size: float = 0.25,
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
            # TODO: halved the target distance configs for now. Give a think if sensible.
            motif_dist_min (float, optional): minimum distance for motif pair distance binning. Defaults to 1.0.
            motif_dist_max (float, optional): maximum distance for motif pair distance binning. Defaults to 15.0.
            motif_dist_bin_size (float, optional): bin size for motif pair distance binning. Defaults to 0.25.
        """
        super(DiffusionConditioning, self).__init__()
        self.sigma_coord = sigma_coord
        self.c_z = c_z
        self.c_s = c_s

        ### Atom level features
        # Reference position embeddings
        self.embed_atom_ref_pos = LinearNoBias(3, self.c_s, init="lecun")
        self.embed_atom_pair_offsets = LinearNoBias(
            3, self.c_z, init="lecun", high_precision=False
        )
        self.embed_atom_pair_invd = LinearNoBias(1, self.c_z, init="lecun")
        self.embed_atom_pair_offsets_valid = LinearNoBias(1, self.c_z, init="normal")

        # Atom level condition embeddings
        self.atom_input_feature = {
            "ref_charge": 1,
            "ref_mask": 1,
            "ref_element": 128,
            "ref_atom_name_chars": 4 * 64,
        }
        self.atom_condition_embed = LinearNoBias(
            sum(self.atom_input_feature.values()), self.c_s, init="lecun"
        )

        ### Motif features
        # Motif distance binning parameters
        self.motif_dist_min = motif_dist_min
        self.motif_dist_max = motif_dist_max
        self.motif_dist_bin_size = motif_dist_bin_size
        self.n_motif_dist_bins = int(
            (motif_dist_max - motif_dist_min) / motif_dist_bin_size
        )

        motif_lower_bins = torch.arange(
            motif_dist_min, motif_dist_max, motif_dist_bin_size
        )
        motif_upper_bins = torch.cat(
            [motif_lower_bins[1:], motif_lower_bins.new_tensor([1e6])], dim=-1
        )
        self.motif_lower_bins = nn.Parameter(motif_lower_bins, requires_grad=False)
        self.motif_upper_bins = nn.Parameter(motif_upper_bins, requires_grad=False)

        # Motif feature embeddings
        self.embed_motif_restype = LinearNoBias(32, self.c_s, init="normal")
        self.embed_motif_coord = LinearNoBias(3, self.c_s, init="lecun")
        self.embed_motif_pair_dist = LinearNoBias(
            self.n_motif_dist_bins, self.c_z, init="normal"
        )

        ### Token bonds
        self.token_bonds_embed = LinearNoBias(
            in_features=1, out_features=self.c_z, init="normal"
        )

        ### Relpos
        self.relpos = RelativePositionEncoding(c_z=c_z)
        self.layernorm_z = LayerNorm(2 * self.c_z)
        self.linear_no_bias_z = LinearNoBias(2 * self.c_z, self.c_z)

        ### Timestep embeddings
        self.fourier_embedding = FourierEmbedding(c=c_timestep_embedding)
        self.layernorm_t = LayerNorm(c_timestep_embedding)
        self.linear_t = LinearNoBias(c_timestep_embedding, self.c_s, init="lecun")

        # Entity type embedding
        self.entity_features = {
            "is_protein_token": 1,
            "is_ligand_token": 1,
            "is_dna_token": 1,
            "is_rna_token": 1,
            "is_target_token": 1,
            "is_hotspot": 1,
            "is_motif_token": 1,
            "is_motif_related_token": 1,
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
        # TODO: soft-binning with different parameters can work better here
        # TODO: we likely need displacements and finer distances such as 1 / (1 + d**2)
        target_lower_bins = torch.arange(
            target_dist_min, target_dist_max, target_dist_bin_size
        )
        target_upper_bins = torch.cat(
            [target_lower_bins[1:], target_lower_bins.new_tensor([1e6])], dim=-1
        )
        self.target_lower_bins = nn.Parameter(target_lower_bins, requires_grad=False)
        self.target_upper_bins = nn.Parameter(target_upper_bins, requires_grad=False)
        self.num_bins = len(target_lower_bins)

        # Target feature embeddings
        self.embed_target_restype = LinearNoBias(32, self.c_s, init="normal")
        self.embed_target_coord = LinearNoBias(3, self.c_s, init="lecun")
        self.embed_target_pair_dist = LinearNoBias(
            self.n_target_dist_bins, self.c_z, init="normal"
        )

        ### Transitions
        self.transition_z = Transition(c_in=self.c_z, n=2)
        self.transition_s1 = Transition(c_in=self.c_s, n=2)
        self.transition_s2 = Transition(c_in=self.c_s, n=2)

    # @torch.compiler.disable
    def get_atom_condition(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
    ) -> torch.Tensor:
        """Generate token level single and pair conditioning on the tokens that
        are atoms. This is the ligand and motif atoms only.

        Args:
            input_feature_dict: Input feature dictionary.
            inplace_safe: Whether it is safe to use inplace operations

        Returns:
            torch.Tensor: Initial single and pair embedding with atom level conditioning
                 * [..., n_token, c_s]
                 * [..., n_token, n_token, c_z]
        """
        atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
        atom_mask = input_feature_dict["atom_mask"]
        n_token = input_feature_dict["residue_index"].shape[-1]
        n_atom = input_feature_dict["ref_pos"].shape[-2]
        batch_shape = input_feature_dict["ref_pos"].shape[:-2]

        is_ligand = input_feature_dict["is_ligand"]

        is_motif_related = input_feature_dict["is_motif_related"]  # [..., N_atom]

        ### Create atom pair conditioning
        # 1. Get token-wise atom reference positions
        # [..., N_atom, 3]
        is_atom = (
            (is_ligand | is_motif_related).bool().unsqueeze(-1)
        )  # [..., n_atom, 1]
        ref_pos = (
            input_feature_dict["ref_pos"] * is_atom
        )  # (zero out that are non-atomized)

        # 2. Aggregate per-atom reference positions to per-token reference positions
        # [..., N_atom, 3] -> [..., N_token, 3]
        ref_pos = average_atom_to_token(
            x_atom=ref_pos,
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

        # Check if ref_space_uid values are the same (or very similar for floating point)
        v_lm = (
            torch.abs(
                token_ref_space_uid[..., None, :, :]
                - token_ref_space_uid[..., :, None, :]
            )
            < 1e-2
        ).to(ref_pos.dtype)  # [..., n_token, n_token, 1]

        # 3. Compute ref pos displacement
        d_lm = (
            ref_pos[..., :, None, :] - ref_pos[..., None, :, :]
        )  # [..., n_token, n_token, 3]
        z = self.embed_atom_pair_offsets(d_lm)
        z = z + self.embed_atom_pair_offsets_valid(v_lm)

        # 4. Embed pairwise inverse squared distances
        z = z + self.embed_atom_pair_invd(
            1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))
        )  # [..., n_token, n_token, c_z]
        z = z * v_lm  # mask out inter-conformers

        ### Create atom single conditioning
        # Embed atom atom-wise features
        c_s = (
            self.atom_condition_embed(
                torch.cat(
                    [
                        input_feature_dict[name]
                        .reshape(*batch_shape, n_atom, self.atom_input_feature[name])
                        .to(d_lm.dtype)
                        for name in self.atom_input_feature
                    ],
                    dim=-1,
                )
            )
            * is_atom
        )  # [..., N_atom, c_s]

        c_s = average_atom_to_token(
            x_atom=c_s,
            atom_to_token_idx=atom_to_token_idx,
            atom_mask=atom_mask,
            n_token=n_token,
        )  # [..., N_token, c_s]

        c_s = c_s + self.embed_atom_ref_pos(ref_pos)

        return c_s, z

    def get_motif_condition(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate motif conditioning embeddings.

        Args:
            input_feature_dict: Input feature dictionary containing motif data

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Motif single and pair embeddings
                - s_motif (torch.Tensor): [..., N_token, c_s]
                - z_motif (torch.Tensor): [..., N_token, N_token, c_z]
        """
        is_motif_token = input_feature_dict["is_motif_token"]  # [..., N_token]
        motif_coord = input_feature_dict["motif_coordinate"]  # [..., N_token, 3]
        motif_restype = input_feature_dict["motif_restype"]  # [..., N_token, 32]

        # Augment motif coordinates
        motif_coord_augment = centre_random_augmentation(
            motif_coord,
            mask=is_motif_token,
            N_sample=1,
            s_trans=1.0,  # TODO: increase s_trans?
        ).squeeze(-3)  # [..., N_token, 3]

        # Single embeddings
        s_motif = self.embed_motif_restype(motif_restype)
        s_motif = s_motif + self.embed_motif_coord(motif_coord_augment)
        s_motif = s_motif * is_motif_token[..., None]

        # Pair embeddings - compute motif distances
        motif_pair_dists = torch.norm(
            motif_coord_augment[..., None, :, :] - motif_coord_augment[..., :, None, :],
            dim=-1,
        )  # [..., N_token, N_token]

        # One-hot encode distances
        motif_pair_dist_onehot = one_hot(
            motif_pair_dists,
            self.motif_lower_bins,
            self.motif_upper_bins,
        )  # [..., N_token, N_token, n_bins]

        # Mask pair embeddings for motif tokens only
        motif_mask_pair = is_motif_token[..., None, :] * is_motif_token[..., :, None]
        motif_pair_dist_onehot = motif_pair_dist_onehot * motif_mask_pair[..., None]

        # Embed one-hot distances
        z_motif = self.embed_motif_pair_dist(motif_pair_dist_onehot)

        return s_motif, z_motif

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
            self.target_lower_bins,
            self.target_upper_bins,
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
        s_target = s_target * is_target_token[..., None]  # [..., N_token, c_s]

        return s_target, z_target

    # @torch.compile
    def forward(
        self,
        r_noisy: torch.Tensor,
        timestep: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        inplace_safe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            r_noisy (torch.Tensor): noisy coordinates
                [..., N_sample, N_atom, 3]
            timestep (torch.Tensor): the noising timestep
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            inplace_safe (bool): Whether it is safe to use inplace operations.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: embeddings s and z
                - s (torch.Tensor): [..., N_sample, N_tokens, c_s]
                - z (torch.Tensor): [..., N_tokens, N_tokens, c_z]
        """
        assert (
            timestep.shape[-1] == r_noisy.shape[-3]
        ), f"Got timestep shape {timestep.shape} and r_noisy {r_noisy.shape}"
        dtype = r_noisy.dtype

        # Grab data on inputs
        n_token = input_feature_dict["residue_index"].shape[-1]

        # Initialize pair embedding with atom level (motif + ligand) conformer features
        s_atom, z_atom = self.get_atom_condition(input_feature_dict)

        # Motif features
        s_motif, z_motif = self.get_motif_condition(input_feature_dict)

        # Target features
        s_target, z_target = self.get_target_condition(input_feature_dict)

        # Add single and pair embeddings
        s = s_target + s_atom + s_motif
        z = z_target + z_atom + z_motif

        # Add token bonds to pair embedding
        # z = z + self.token_bonds_embed(
        #     input_feature_dict["token_bonds"].unsqueeze(-1)
        # )  # [..., N_token, N_token, c_z]

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
        timestep_embed = self.fourier_embedding(
            t_hat_timestep=(
                2 * timestep - 1
            )  # Rescale to [-1, 1] for smoother embedding
        ).to(z.dtype)  # [..., N_sample, c_noise_embedding]

        # 2. Normalise and embed both timesteps
        s = s.unsqueeze(-3)
        s = s + expand_at_dim(
            self.linear_t(self.layernorm_t(timestep_embed)),
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
        c_token: int = 768,
        c_atom: int = 128,
        c_s: int = 384,
        c_z: int = 128,
        transformer: dict[str, int] = {"num_blocks": 24, "num_heads": 16},
        dropout: float = 0.0,
        blocks_per_ckpt: Optional[int] = None,
        use_fine_grained_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            sigma_coord (torch.float, optional): the standard deviation of the coordinates. Defaults to 12.0.
            c_token (int, optional): feature channel of token (single a). Defaults to 768.
            c_atom (int, optional): feature channel of atom. Defaults to 128.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            transformer (dict[str, int], optional): configs in DiffusionTransformer. Defaults to {"n_blocks": 24, "n_heads": 16}.
            dropout (float, optional): dropout probability for attention weights and transition block. Defaults to 0.0.
            blocks_per_ckpt: number of atom_encoder/transformer/atom_decoder blocks in each activation checkpoint
                Size of each chunk. A higher value corresponds to fewer
                checkpoints, and trades memory for speed. If None, no checkpointing is performed.
            use_fine_grained_checkpoint: whether use fine-gained checkpoint for finetuning stage 2
                only effective if blocks_per_ckpt is not None.
        """

        super(DiffusionModule, self).__init__()
        self.sigma_coord = sigma_coord
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_s = c_s
        self.c_z = c_z

        self.atom_input_feature = {
            "ref_charge": 1,
            "ref_element": 128,
            "ref_atom_name_chars": 4 * 64,
        }

        # Grad checkpoint setting
        self.blocks_per_ckpt = blocks_per_ckpt
        self.use_fine_grained_checkpoint = use_fine_grained_checkpoint

        # Coordinate embedding layer
        self.linear_no_bias_r_embed = LinearNoBias(
            3,
            c_token,  # init="lecun"
        )

        self.diffusion_conditioning = DiffusionConditioning(
            sigma_coord=sigma_coord,
            c_z=c_z,
            c_s=c_s,
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

        self.linear_no_bias_a = LinearNoBias(
            c_token,
            c_atom,
        )

        self.linear_no_bias_f = LinearNoBias(
            sum(self.atom_input_feature.values()),
            c_atom,
        )

        self.readout_transition = Transition(c_in=c_atom, n=2)
        self.readout_layernorm = LayerNorm(c_atom)
        self.readout_linear = LinearNoBias(c_atom, 3, init="zeros")

    def readout(self, q_atom: torch.Tensor):
        """Readout layer."""
        q_atom = q_atom + self.readout_transition(q_atom)
        return self.readout_linear(self.readout_layernorm(q_atom))

    def f_forward(
        self,
        r_noisy: torch.Tensor,
        timestep: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        The raw network to be trained.
        This is v_theta(x_t, t)
        Here, c_noise(sigma) is computed in Conditioning module.

        Args:
            r_noisy (torch.Tensor): scaled noisy coordinates (i.e., c_in * x)
                [..., N_sample, N_token, 3]
            timestep (torch.Tensor): the noising timestep
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input feature dictionary
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.

        Returns:
            torch.Tensor: Network output. This will either be velocity prediction or denoising coord prediction,
                depending on network parameterisation.
                [..., N_sample, N_token, 3]
        """
        N_sample = r_noisy.shape[-3]
        assert (
            timestep.shape[-1] == N_sample
        ), f"Got timestep shape {timestep.shape} but expected {N_sample}"

        atom_mask = expand_at_dim(
            input_feature_dict["atom_mask"],
            dim=-2,
            n=N_sample,
        )
        atom_to_token_idx = expand_at_dim(
            input_feature_dict["atom_to_token_idx"], dim=-2, n=N_sample
        )

        n_token = input_feature_dict["residue_index"].shape[-1]
        n_atom = atom_mask.shape[-1]
        batch_shape = input_feature_dict["residue_index"].shape[:-1]

        # Compute pair mask
        token_mask = input_feature_dict["token_mask"]  # [..., N_token]
        token_mask = token_mask.unsqueeze(-2)  # add N_sample dim
        pair_mask = token_mask[..., :, None] * token_mask[..., None, :]
        # [..., 1, N_token, N_token]

        # Average coordinates per token
        q_skip = self.linear_no_bias_r_embed(r_noisy)

        a_token = average_atom_to_token(
            x_atom=q_skip,  # [..., N_sample, N_atom, 3]
            atom_mask=atom_mask,  # [..., N_atom]
            atom_to_token_idx=atom_to_token_idx,  # [..., N_sample, N]
            n_token=n_token,
        )  # [..., N_sample, N_token, c_token]

        s, z = self.diffusion_conditioning(
            r_noisy=r_noisy,
            timestep=timestep,
            input_feature_dict=input_feature_dict,
            inplace_safe=inplace_safe,
        )  # [..., N_sample, N_token, c_s], [..., N_token, N_token, c_z]

        # Mix single conditioning into token representation
        a_token = a_token + self.linear_no_bias_s(self.layernorm_s(s))
        a_token = a_token * token_mask[..., None]  # [..., N_sample, N_token, c_token]

        z = z.unsqueeze(-4)
        z = z * pair_mask[..., None]

        a_token = self.diffusion_transformer(a=a_token, s=s, z=z, pair_mask=pair_mask)
        a_token = a_token * token_mask[..., None]
        a_token = self.layernorm_a(a_token)

        # Convert back to atom space
        q_atom = self.linear_no_bias_a(
            broadcast_token_to_atom(
                x_token=a_token, atom_to_token_idx=atom_to_token_idx
            )
            + q_skip
        )  # [..., N_sample, N_atom, c_atom]

        # Break symmetry in the backbone atoms with atom level information
        q_atom = q_atom + expand_at_dim(
            self.linear_no_bias_f(
                torch.cat(
                    [
                        input_feature_dict[name]
                        .reshape(*batch_shape, n_atom, self.atom_input_feature[name])
                        .to(q_atom.dtype)
                        for name in self.atom_input_feature
                    ],
                    dim=-1,
                )
            ),
            dim=-3,
            n=N_sample,
        )

        return self.readout(q_atom)

    def c_in(self, timestep: torch.Tensor) -> torch.Tensor:
        return 1 / (timestep**2 + (1 - timestep) ** 2).sqrt()

    def c_out(self, timestep: torch.Tensor) -> torch.Tensor:
        return 1 - timestep  # velocity parametrization

    def c_skip(self, timestep: torch.Tensor) -> torch.Tensor:
        return 1.0

    def forward(
        self,
        r_noisy: torch.Tensor,
        timestep: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        inplace_safe: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Performs a forward pass of the diffusion module.

        Args:
            r_noisy (torch.Tensor): Noisy coordinates.
            timestep (torch.Tensor): The noising timestep.
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): Input feature dictionary.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.

        Returns:
            torch.Tensor: Predicted coordinates.
                [..., N_sample, N_token, 3]
        """
        # Expand timesteps
        timestep_ = timestep[..., None, None]

        # Scale input noisy coords to unit variance
        r_noisy_unit = (r_noisy / self.sigma_coord) * self.c_in(timestep_)

        # Predict velocity
        out = self.f_forward(
            r_noisy=r_noisy_unit,
            timestep=timestep,
            input_feature_dict=input_feature_dict,
            inplace_safe=inplace_safe,
        )

        # Apply skip connections and output scaling
        r_pred = r_noisy * self.c_skip(timestep_) + out * self.sigma_coord * self.c_out(
            timestep_
        )

        return r_pred
