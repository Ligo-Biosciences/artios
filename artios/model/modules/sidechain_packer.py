import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from artios.data.features import FullContext
from artios.model.generator import (
    Deterministic,
    FlowMatchingSampler,
    InferenceTimestepScheduler,
    SamplingResult,
    StochasticitySchedule,
    VelocityScorePredictor,
)
from artios.model.modules.atom_attention import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
)
from artios.model.modules.embedders import FourierEmbedding, RelativePositionEncoding
from artios.model.modules.primitives import LayerNorm, LinearNoBias
from artios.model.modules.transformer import DiffusionTransformer
from artios.model.utils import average_atom_to_token, one_hot
from artios.utils.geometry import centre_random_augmentation
from artios.utils.typing import Float, typecheck


class SidechainPacker(nn.Module):
    """A lightweight non-equivariant sidechain decoder for all-atom sequence design."""

    def __init__(
        self,
        c_s: int = 384,
        c_token: int = 384,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_z: int = 32,
        atom_encoder: dict[str, int] = {"num_blocks": 3, "num_heads": 4},
        transformer: dict[str, int] = {"num_blocks": 10, "num_heads": 12},
        atom_decoder: dict[str, int] = {"num_blocks": 3, "num_heads": 4},
        num_queries: int = 32,
        num_keys: int = 128,
        sigma_coord: float = 10.0,
        local_sigma_coord: float = 1.0,
        centre_dist_min: float = 2.0,
        centre_dist_max: float = 22.0,
        centre_dist_bin_size: float = 0.5,
    ) -> None:
        super().__init__()
        self.sigma_coord = sigma_coord
        self.local_sigma_coord = local_sigma_coord
        self.relpos = RelativePositionEncoding(r_max=32, s_max=2, c_z=c_z)

        # Centre atom distance binning for pair representation
        n_centre_dist_bins = int(
            (centre_dist_max - centre_dist_min) / centre_dist_bin_size
        )
        centre_lower_bins = torch.arange(
            centre_dist_min, centre_dist_max, centre_dist_bin_size
        )
        centre_upper_bins = torch.cat(
            [centre_lower_bins[1:], torch.tensor([centre_dist_max])], dim=-1
        )
        self.register_buffer("centre_lower_bins", centre_lower_bins)
        self.register_buffer("centre_upper_bins", centre_upper_bins)

        # Centre atom conditioning embeddings (non-ligand: protein/DNA/RNA)
        self.embed_centre_coords = LinearNoBias(3, c_s)
        self.embed_centre_pair_dist = LinearNoBias(n_centre_dist_bins, c_z)

        # Ligand reference conformer conditioning embeddings
        self.embed_ligand_ref_pos = LinearNoBias(3, c_s, init="lecun")
        self.embed_ligand_pair_offsets = LinearNoBias(
            3, c_z, init="lecun", high_precision=False
        )
        self.embed_ligand_pair_invd = LinearNoBias(1, c_z, init="lecun")
        self.embed_ligand_pair_valid = LinearNoBias(1, c_z, init="normal")

        # Encoder
        self.atom_attention_encoder = AtomAttentionEncoder(
            **atom_encoder,
            c_token=c_token,
            has_coords=True,
            has_s=True,
            has_z=True,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_s=c_s,
            c_z=c_z,
            num_queries=num_queries,
            num_keys=num_keys,
        )

        # Single conditioning embedding
        self.embed_restype = LinearNoBias(32, c_s)
        self.fourier_embedding = FourierEmbedding(c=c_s)
        self.embed_timestep = LinearNoBias(c_s, c_s)

        # DiT
        self.diffusion_transformer = DiffusionTransformer(
            **transformer,  # type: ignore[invalid-argument-type]
            c_token=c_token,
            c_s=c_s,
            c_z=c_z,
        )
        self.layernorm_a = LayerNorm(c_token)

        # Decoder
        self.atom_attention_decoder = AtomAttentionDecoder(
            **atom_decoder,
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            num_queries=num_queries,
            num_keys=num_keys,
        )

    @typecheck
    def get_ligand_conditioning(
        self,
        context: FullContext,
    ) -> tuple[
        Float[Tensor, "... n_token c_s"], Float[Tensor, "... n_token n_token c_z"]
    ]:
        """Generate ligand-specific single and pair conditioning from reference conformer.

        Embeds ligand reference conformer positions aggregated to token level, and
        pairwise offsets/distances between ligand tokens. Uses ref_space_uid to
        prevent different ligands from interacting with each other.

        Args:
            context: Full context containing all features and labels.

        Returns:
            Tuple of single and pair embeddings with ligand conditioning:
                - s_ligand: [..., n_token, c_s]
                - z_ligand: [..., n_token, n_token, c_z]
        """
        atom_to_token_idx = context.index.atom_to_token_idx
        atom_mask = context.mask.atom_mask
        n_token = context.token.residue_index.shape[-1]

        is_ligand = context.entity.is_ligand  # [..., n_atom]
        is_ligand_token = context.entity.is_ligand_token  # [..., n_token]

        # Get ligand reference positions (atom level), zero out non-ligand atoms
        is_ligand_expanded = rearrange(is_ligand, "... n_atom -> ... n_atom 1")
        ref_pos = context.reference.ref_pos * is_ligand_expanded  # [..., n_atom, 3]

        # Aggregate per-atom reference positions to per-token reference positions
        ligand_ref_pos = average_atom_to_token(
            x_atom=ref_pos,
            atom_to_token_idx=atom_to_token_idx,
            atom_mask=atom_mask,
            n_token=n_token,
        )  # [..., n_token, 3]

        # Compute ref_space_uid validity mask
        token_ref_space_uid = average_atom_to_token(
            x_atom=context.reference.ref_space_uid.float().unsqueeze(-1),
            atom_to_token_idx=atom_to_token_idx,
            atom_mask=atom_mask,
            n_token=n_token,
        )  # [..., n_token, 1]

        v_lm = (
            torch.abs(
                token_ref_space_uid[..., None, :, :]
                - token_ref_space_uid[..., :, None, :]
            )
            < 1e-2
        ).to(ligand_ref_pos.dtype)  # [..., n_token, n_token, 1]

        # Single embedding: ligand reference positions
        s_ligand = self.embed_ligand_ref_pos(ligand_ref_pos)
        s_ligand = s_ligand * is_ligand_token[..., None]  # [..., n_token, c_s]

        # Pair embedding: displacement vectors between ligand tokens
        d_lm = (
            ligand_ref_pos[..., :, None, :] - ligand_ref_pos[..., None, :, :]
        )  # [..., n_token, n_token, 3]

        z_ligand = self.embed_ligand_pair_offsets(d_lm)
        z_ligand = z_ligand + self.embed_ligand_pair_valid(v_lm)

        # Add inverse squared distance embedding
        z_ligand = z_ligand + self.embed_ligand_pair_invd(
            1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))
        )  # [..., n_token, n_token, c_z]

        # Mask out inter-ligand (different ref_space_uid) pairs
        z_ligand = z_ligand * v_lm

        # Mask to only ligand-ligand token pairs
        ligand_mask_i = rearrange(is_ligand_token, "... n -> ... n 1")
        ligand_mask_j = rearrange(is_ligand_token, "... n -> ... 1 n")
        ligand_pair_mask = ligand_mask_i & ligand_mask_j
        z_ligand = z_ligand * ligand_pair_mask[..., None]

        return s_ligand, z_ligand

    # @typecheck
    @torch.compile
    def f_forward(
        self,
        r_noisy: Float[Tensor, "... n_sample n_atom 3"],
        timestep: Float[Tensor, "... n_sample"],
        context: FullContext,
    ) -> Float[Tensor, "... n_sample n_atom 3"]:
        """All-atom denoiser forward pass.

        Args:
            r_noisy: Noisy atom coordinates.
            timestep: Diffusion timestep.
            context: Full feature context.

        Returns:
            Denoised atom coordinates.
        """
        # Extract features from context
        dtype = context.reference.ref_pos.dtype
        atom_mask = context.mask.atom_mask
        atom_to_token_idx = context.index.atom_to_token_idx
        token_mask = context.mask.token_mask
        restype = context.token.restype
        n_token = restype.shape[-2]
        n_sample = timestep.shape[-1]

        # Compute pair mask
        token_mask = token_mask.unsqueeze(-2)  # add n_sample dim
        mask_i = rearrange(token_mask, "... n -> ... n 1")
        mask_j = rearrange(token_mask, "... n -> ... 1 n")
        pair_mask = mask_i & mask_j

        # Compute relative position encoding
        z = self.relpos(
            asym_id=context.token.asym_id,
            residue_index=context.token.residue_index,
            entity_id=context.token.entity_id,
            sym_id=context.token.sym_id,
            token_index=context.token.token_index,
        )

        # Add centre atom pairwise distance encoding (non-ligand only)
        centre_coords = context.coordinate.centre_coordinate  # [..., n_token, 3]
        is_ligand_token = context.entity.is_ligand_token  # [..., n_token]
        non_ligand_mask = ~is_ligand_token

        # Zero out ligand centre coordinates
        centre_coords_masked = centre_coords * non_ligand_mask[..., None]

        # Compute pairwise distances
        centre_i = rearrange(centre_coords_masked, "... n c -> ... n 1 c")
        centre_j = rearrange(centre_coords_masked, "... n c -> ... 1 n c")
        centre_pair_dists = torch.norm(
            centre_i - centre_j, dim=-1
        )  # [..., n_token, n_token]

        # One-hot encode distances
        centre_pair_dist_onehot = one_hot(
            centre_pair_dists, self.centre_lower_bins, self.centre_upper_bins
        )

        # Zero out pairs involving ligands
        mask_i = rearrange(non_ligand_mask, "... n -> ... n 1")
        mask_j = rearrange(non_ligand_mask, "... n -> ... 1 n")
        non_ligand_pair_mask = mask_i & mask_j
        centre_pair_dist_onehot = (
            centre_pair_dist_onehot * non_ligand_pair_mask[..., None]
        )

        # Embed and add to pair representation
        z_centre = self.embed_centre_pair_dist(centre_pair_dist_onehot)
        z = z + z_centre

        # Add ligand reference conformer conditioning
        s_ligand, z_ligand = self.get_ligand_conditioning(context)
        z = z + z_ligand

        z = rearrange(z, "... i j c -> ... 1 i j c")

        # Compute sequence embedding
        s = s_ligand + self.embed_restype(restype.to(dtype))
        s = rearrange(s, "... n c -> ... 1 n c")  # [..., 1, n_token, c]

        # Add centre coordinate embedding (non-ligand only)
        centre_coords_augmented = centre_random_augmentation(
            centre_coords_masked,
            mask=context.mask.centre_atom_is_resolved,
            n_sample=n_sample,
            s_trans=1.0,
        )  # [..., n_sample, n_token, 3]
        s = s + self.embed_centre_coords(centre_coords_augmented)

        # Add timestep embedding
        timestep_emb = self.embed_timestep(
            self.fourier_embedding(t_hat_timestep=2 * timestep - 1)
        )
        s = s + repeat(timestep_emb, "... n_sample c -> ... n_sample n c", n=n_token)

        # Encode atoms
        a, skip = self.atom_attention_encoder(
            context=context,
            r_l=r_noisy,  # / self.sigma_coord,
            s=s,
            z=z,
        )

        a = self.diffusion_transformer(a=a, s=s, z=z, pair_mask=pair_mask)
        a = self.layernorm_a(a)

        # Decode atoms
        r_update = self.atom_attention_decoder(
            a=a,
            skip=skip,
            atom_to_token_idx=atom_to_token_idx,
            atom_mask=atom_mask,
        )
        return r_update

    def as_predictor(
        self,
        context: FullContext,
        eps: float = 1e-5,
    ) -> VelocityScorePredictor:
        """Create a VelocityScorePredictor bound to this packer and context.

        Args:
            context: Full feature context to use for predictions.
            eps: Small constant for numerical stability.

        Returns:
            A callable satisfying VelocityScorePredictor protocol.
        """

        def predictor(
            x_noisy: Float[Tensor, "... n_atom 3"],
            timestep: Float[Tensor, "..."],
        ) -> tuple[Float[Tensor, "... n_atom 3"], Float[Tensor, "... n_atom 3"]]:
            coords_pred = self.forward(
                context=context,
                r_noisy=x_noisy,
                timestep=timestep,
            )

            t = repeat(timestep, "... -> ... 1 1")
            velocity = (coords_pred - x_noisy) / torch.clamp(1 - t, min=eps)
            score = t * velocity - x_noisy
            return velocity, score

        return predictor

    def sample(
        self,
        context: FullContext,
        timestep_scheduler: InferenceTimestepScheduler,
        n_steps: int,
        residual_centre_atoms: Float[Tensor, "... n_sample n_atom 3"],
        stochasticity: StochasticitySchedule = Deterministic(),
        gamma: float = 1.0,
        n_sample: int = 1,
    ) -> SamplingResult:
        """All-atom denoising sampling using flow matching.

        Args:
            context: Full feature context.
            timestep_scheduler: Scheduler for generating timestep sequence.
            n_steps: Number of integration steps.
            residual_centre_atoms: Residue center atom coordinates for residual connection.
            stochasticity: Schedule defining g(t) in the SDE.
            gamma: Temperature parameter for the stochastic term.
            n_sample: Number of samples to generate.

        Returns:
            SamplingResult containing final denoised coordinates.
        """
        n_atom = context.index.atom_to_token_idx.shape[-1]
        batch_shape = context.token.residue_index.shape[:-1]
        device = context.token.residue_index.device
        dtype = torch.float32

        # Generate initial noise with residual offset
        x_init = self.local_sigma_coord * torch.randn(
            size=(*batch_shape, n_sample, n_atom, 3), device=device, dtype=dtype
        )
        x_init = x_init + residual_centre_atoms

        # Create sampler and predictor
        sampler = FlowMatchingSampler(
            timestep_scheduler=timestep_scheduler,
            stochasticity=stochasticity,
            gamma=gamma,
            second_order=False,
            centre_coords=True,
        )
        predictor = self.as_predictor(context)

        mask = repeat(
            context.mask.is_resolved,
            "... n_atom -> ... n_sample n_atom",
            n_sample=n_sample,
        )

        return sampler.sample(
            predictor=predictor,
            x_init=x_init,
            n_steps=n_steps,
            mask=mask,
        )

    @typecheck
    def forward(
        self,
        r_noisy: Float[Tensor, "... n_sample n_atom 3"],
        timestep: Float[Tensor, "... n_sample"],
        context: FullContext,
    ) -> Float[Tensor, "... n_sample n_atom 3"]:
        """Performs a forward pass of the sidechain packer.

        Args:
            r_noisy: Noisy atom coordinates.
            timestep: Diffusion timestep.
            context: Full context containing all features and labels.
            inplace_safe: Whether it is safe to use inplace operations.

        Returns:
            Denoised atom coordinates.
        """
        # Expand timesteps
        timestep_ = timestep[..., None, None]

        # Scale input coords to unit variance (no timestep dependence)
        r_noisy_unit = r_noisy / self.sigma_coord

        # Predict velocity
        out = self.f_forward(
            r_noisy=r_noisy_unit,
            timestep=timestep,
            context=context,
        )

        # Apply skip connection and output scaling
        r_pred = r_noisy + out * (1 - timestep_)

        return r_pred
