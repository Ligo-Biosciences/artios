"""
TODO: this filename should eventually just be called diffusion.py I think.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from artios.data.features import FullContext
from artios.metrics.rmsd import align_pred_to_true
from artios.model.generator import (
    SIGMA_DATA,
    CFGSchedule,
    EDMCFGPredictor,
    EDMDenoisePredictor,
    EDMInferenceNoiseScheduler,
    EDMSampler,
    NoCFG,
    SamplingResult,
)
from artios.model.modules.components import Transition
from artios.model.modules.embedders import FourierEmbedding, RelativePositionEncoding
from artios.model.modules.primitives import LayerNorm, LinearNoBias
from artios.model.modules.transformer import (
    DiffusionTransformer,
    DiffusionTransformerConfig,
)
from artios.model.utils import average_atom_to_token, broadcast_token_to_atom, one_hot
from artios.utils.geometry import centre_random_augmentation
from artios.utils.typing import typecheck


@dataclass
class CachedConditioning:
    """Timestep-independent conditioning, reusable across sampling steps."""

    s_base: Float[Tensor, "... n_token c_s"]
    z: Float[Tensor, "... n_token n_token c_z"]


def _stable_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class DiffusionConditioningConfig:
    """Architecture settings for diffusion conditioning."""

    c_timestep_embedding: int = 256
    motif_dist_min: float = 1.0
    motif_dist_max: float = 15.0
    motif_dist_bin_size: float = 0.25

    def __post_init__(self) -> None:
        if self.c_timestep_embedding <= 0:
            raise ValueError("c_timestep_embedding must be greater than 0")
        if self.motif_dist_max <= self.motif_dist_min:
            raise ValueError("motif_dist_max must be greater than motif_dist_min")
        if self.motif_dist_bin_size <= 0:
            raise ValueError("motif_dist_bin_size must be greater than 0")


@dataclass(frozen=True, slots=True, kw_only=True)
class DiffusionModuleConfig:
    """Architecture settings for the EDM diffusion backbone module."""

    c_token: int = 768
    c_atom: int = 256
    c_s: int = 384
    c_z: int = 128
    conditioning: DiffusionConditioningConfig = field(
        default_factory=DiffusionConditioningConfig
    )
    transformer: DiffusionTransformerConfig = field(
        default_factory=DiffusionTransformerConfig
    )

    def __post_init__(self) -> None:
        for name in ("c_token", "c_atom", "c_s", "c_z"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than 0")
        if self.c_token % self.transformer.num_heads != 0:
            raise ValueError("c_token must be divisible by transformer.num_heads")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def architecture_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True, slots=True, kw_only=True)
class DiffusionRuntimeConfig:
    """Execution settings that do not define diffusion architecture."""

    checkpoint_grads: bool = True


class DiffusionConditioning(nn.Module):
    """
    Implements Algorithm 21 in AF3
    """

    def __init__(
        self,
        config: DiffusionConditioningConfig | None = None,
        c_z: int = 128,
        c_s: int = 384,
    ) -> None:
        """
        Args:
            config: Architecture settings for diffusion conditioning.
            c_z (int, optional): hidden dim [for pair embedding].
            c_s (int, optional):  hidden dim [for single embedding].
        """
        super(DiffusionConditioning, self).__init__()
        if config is None:
            config = DiffusionConditioningConfig()

        self.config = config
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
        self.motif_dist_min = config.motif_dist_min
        self.motif_dist_max = config.motif_dist_max
        self.motif_dist_bin_size = config.motif_dist_bin_size
        self.n_motif_dist_bins = int(
            (config.motif_dist_max - config.motif_dist_min) / config.motif_dist_bin_size
        )

        motif_lower_bins = torch.arange(
            config.motif_dist_min, config.motif_dist_max, config.motif_dist_bin_size
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
        # self.token_bonds_embed = LinearNoBias(
        #     in_features=1, out_features=self.c_z, init="normal"
        # )

        ### Relpos
        self.relpos = RelativePositionEncoding(c_z=c_z)
        self.layernorm_z = LayerNorm(2 * self.c_z)
        self.linear_no_bias_z = LinearNoBias(2 * self.c_z, self.c_z)

        ### Timestep embeddings
        self.fourier_embedding = FourierEmbedding(c=config.c_timestep_embedding)
        self.layernorm_t = LayerNorm(config.c_timestep_embedding)
        self.linear_t = LinearNoBias(
            config.c_timestep_embedding, self.c_s, init="lecun"
        )

        # Entity type embedding
        self.entity_features = {
            "is_protein_token": 1,
            "is_ligand_token": 1,
            "is_dna_token": 1,
            "is_rna_token": 1,
        }
        self.motif_entity_features = {
            "is_motif_token": 1,
            "is_motif_related_token": 1,
        }
        self.entity_type_embed = LinearNoBias(
            in_features=sum(self.entity_features.values()),
            out_features=self.c_s,
            init="lecun",
        )
        self.motif_entity_type_embed = LinearNoBias(
            in_features=sum(self.motif_entity_features.values()),
            out_features=self.c_s,
            init="lecun",
        )

        ### Transitions
        self.transition_z = Transition(c_in=self.c_z, n=2)
        self.transition_s1 = Transition(c_in=self.c_s, n=2)
        self.transition_s2 = Transition(c_in=self.c_s, n=2)

    # @torch.compiler.disable
    @typecheck
    def get_atom_condition(
        self,
        context: FullContext,
    ) -> tuple[
        Float[Tensor, "... n_token c_s"], Float[Tensor, "... n_token n_token c_z"]
    ]:
        """Generate token level single and pair conditioning on the tokens that
        are atoms. This is the ligand and motif atoms only.

        Args:
            context: Full context containing all features and labels.

        Returns:
            Tuple of single and pair embeddings with atom level conditioning:
                - c_s: [..., n_token, c_s]
                - z: [..., n_token, n_token, c_z]
        """
        atom_to_token_idx = context.index.atom_to_token_idx
        atom_mask = context.mask.atom_mask
        n_token = context.token.residue_index.shape[-1]
        n_atom = context.reference.ref_pos.shape[-2]
        batch_shape = context.reference.ref_pos.shape[:-2]

        is_ligand = context.entity.is_ligand  # [..., n_atom]

        is_motif_related = context.motif.is_motif_related  # [..., n_atom]

        ### Create atom pair conditioning
        # 1. Get token-wise atom reference positions
        # [..., n_atom, 3]
        is_atom = rearrange(
            (is_ligand | is_motif_related), "... n_atom -> ... n_atom 1"
        )
        ref_pos = context.reference.ref_pos * is_atom

        # 2. Aggregate per-atom reference positions to per-token reference positions
        # [..., n_atom, 3] -> [..., n_token, 3]
        ref_pos = average_atom_to_token(
            x_atom=ref_pos,
            atom_to_token_idx=atom_to_token_idx,
            atom_mask=atom_mask,
            n_token=n_token,
        )

        # Compute v_lm mask
        token_ref_space_uid = average_atom_to_token(
            x_atom=context.reference.ref_space_uid.float().unsqueeze(-1),
            atom_to_token_idx=atom_to_token_idx,
            atom_mask=atom_mask,
            n_token=n_token,
        )  # [..., n_token, 1]

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
                        getattr(context.reference, name)
                        .reshape(*batch_shape, n_atom, self.atom_input_feature[name])
                        .to(d_lm.dtype)
                        for name in self.atom_input_feature
                    ],
                    dim=-1,
                )
            )
            * is_atom
        )  # [..., n_atom, c_s]

        c_s = average_atom_to_token(
            x_atom=c_s,
            atom_to_token_idx=atom_to_token_idx,
            atom_mask=atom_mask,
            n_token=n_token,
        )  # [..., n_token, c_s]

        c_s = c_s + self.embed_atom_ref_pos(ref_pos)

        return c_s, z

    @typecheck
    def get_motif_condition(
        self,
        context: FullContext,
    ) -> tuple[
        Float[Tensor, "... n_token c_s"], Float[Tensor, "... n_token n_token c_z"]
    ]:
        """Generate motif conditioning embeddings.

        Args:
            context: Full context containing all features and labels.

        Returns:
            Tuple of motif single (s_motif) and pair (z_motif) embeddings

        # TODO: is_motif_related needs to be embedded
        is_motif needs to be embedded
        both of these in pair and single embeddings

        the motif distances should be embedded with the inverse distance
        embedding as well.
        """
        is_motif_token = context.motif.is_motif_token  # [..., n_token]
        motif_coord = context.motif.motif_coordinate  # [..., n_token, 3]
        motif_restype = context.motif.motif_restype  # [..., n_token, 32]

        # Augment motif coordinates
        motif_coord_augment = centre_random_augmentation(
            motif_coord,
            mask=is_motif_token,
            n_sample=1,
            s_trans=1.0,  # TODO: increase s_trans?
        ).squeeze(-3)  # [..., n_token, 3]

        # Single embeddings
        s_motif = self.embed_motif_restype(motif_restype)
        s_motif = s_motif + self.embed_motif_coord(motif_coord_augment)
        s_motif = s_motif * is_motif_token[..., None]

        # Pair embeddings - compute motif distances
        motif_pair_dists = torch.norm(
            motif_coord_augment[..., None, :, :] - motif_coord_augment[..., :, None, :],
            dim=-1,
        )  # [..., n_token, n_token]

        # One-hot encode distances
        motif_pair_dist_onehot = one_hot(
            motif_pair_dists,
            self.motif_lower_bins,
            self.motif_upper_bins,
        )  # [..., n_token, n_token, n_bins]

        # Mask pair embeddings for motif tokens only
        is_motif_i = rearrange(is_motif_token, "... n -> ... n 1")
        is_motif_j = rearrange(is_motif_token, "... n -> ... 1 n")
        motif_mask_pair = is_motif_i & is_motif_j
        motif_pair_dist_onehot = motif_pair_dist_onehot * motif_mask_pair[..., None]

        # Embed one-hot distances
        z_motif = self.embed_motif_pair_dist(motif_pair_dist_onehot)

        return s_motif, z_motif

    @typecheck
    def compute_cache(self, context: FullContext) -> CachedConditioning:
        """Compute timestep-independent conditioning that can be reused across sampling steps.

        Args:
            context: Full context containing all features and labels.

        Returns:
            CachedConditioning with precomputed s_base and z.
        """
        dtype = context.reference.ref_pos.dtype

        # Initialize pair embedding with atom level (motif + ligand) conformer features
        s_atom, z_atom = self.get_atom_condition(context)

        # Motif features
        s_motif, z_motif = self.get_motif_condition(context)

        # Pair: combine atom + motif, add relpos, layernorm + linear, transition
        z = z_atom + z_motif

        relpos_embed = self.relpos(
            asym_id=context.token.asym_id,
            residue_index=context.token.residue_index,
            entity_id=context.token.entity_id,
            sym_id=context.token.sym_id,
            token_index=context.token.token_index,
        )
        z = torch.cat(tensors=[z, relpos_embed], dim=-1)
        z = self.linear_no_bias_z(self.layernorm_z(z))
        z = z + checkpoint(self.transition_z, z, use_reentrant=False)

        # Single base: atom + motif + entity types
        s_base = s_atom + s_motif

        s_entity_type = self.entity_type_embed(
            torch.stack(
                [
                    getattr(context.entity, name).to(dtype)
                    for name in self.entity_features
                ],
                dim=-1,
            )
        )
        s_entity_type = s_entity_type + self.motif_entity_type_embed(
            torch.stack(
                [
                    getattr(context.motif, name).to(dtype)
                    for name in self.motif_entity_features
                ],
                dim=-1,
            )
        )
        s_base = s_base + s_entity_type

        return CachedConditioning(s_base=s_base, z=z)

    # @torch.compile
    @typecheck
    def forward(
        self,
        r_noisy: Float[Tensor, "... n_sample n_atom 3"],
        noise_level: Float[Tensor, "... n_sample"],
        context: FullContext,
        cache: Optional[CachedConditioning] = None,
        inplace_safe: bool = False,
    ) -> tuple[
        Float[Tensor, "... n_sample n_token c_s"],
        Float[Tensor, "... n_token n_token c_z"],
    ]:
        """
        Args:
            r_noisy: Noisy coordinates
            noise_level: EDM noising level.
            context: Full context containing all features and labels.
            cache: Precomputed timestep-independent conditioning. If None, computed internally.
            inplace_safe: Whether it is safe to use inplace operations.

        Returns:
            Tuple of single (s) and pair (z) embeddings
        """
        if cache is None:
            cache = self.compute_cache(context)

        n_token = context.token.residue_index.shape[-1]

        # Timestep-dependent single conditioning
        noise_embed = self.fourier_embedding(
            t_hat_timestep=torch.log(noise_level / SIGMA_DATA) / 4
        ).to(cache.z.dtype)

        s = rearrange(cache.s_base, "... n c -> ... 1 n c")
        s = s + repeat(
            self.linear_t(self.layernorm_t(noise_embed)),
            "... s c -> ... s n c",
            n=n_token,
        )

        s = s + self.transition_s1(s)
        s = s + self.transition_s2(s)

        return s, cache.z


class DiffusionModule(nn.Module):
    """
    Implements Algorithm 20 in AF3
    """

    def __init__(
        self,
        config: DiffusionModuleConfig | None = None,
        runtime_config: DiffusionRuntimeConfig | None = None,
    ) -> None:
        """
        Args:
            config: Architecture settings for the diffusion module.
            runtime_config: Execution settings that do not affect model identity.
        """

        super(DiffusionModule, self).__init__()
        if config is None:
            config = DiffusionModuleConfig()
        if runtime_config is None:
            runtime_config = DiffusionRuntimeConfig()

        self.config = config
        self.runtime_config = runtime_config
        self.c_token = config.c_token
        self.c_atom = config.c_atom
        self.c_s = config.c_s
        self.c_z = config.c_z

        self.atom_input_feature = {  # TODO: how could we better handle - dataclass?
            "ref_charge": 1,
            "ref_element": 128,
            "ref_atom_name_chars": 4 * 64,
        }

        self.linear_no_bias_r_embed = LinearNoBias(3, self.c_token)
        self.linear_no_bias_prev_pred_embed = LinearNoBias(
            3, self.c_token, init="zeros"
        )
        self.self_cond_merge = LinearNoBias(2 * self.c_token, self.c_token)

        self.diffusion_conditioning = DiffusionConditioning(
            config=config.conditioning,
            c_z=self.c_z,
            c_s=self.c_s,
        )

        self.layernorm_s = LayerNorm(self.c_s)
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_token, init="zeros"
        )
        self.diffusion_transformer = DiffusionTransformer(
            num_blocks=config.transformer.num_blocks,
            num_heads=config.transformer.num_heads,
            c_token=self.c_token,
            c_s=self.c_s,
            c_z=self.c_z,
            checkpoint_grads=runtime_config.checkpoint_grads,
        )
        self.layernorm_a = LayerNorm(self.c_token)

        self.linear_no_bias_a = LinearNoBias(
            self.c_token,
            self.c_atom,
        )

        self.linear_no_bias_f = LinearNoBias(
            sum(self.atom_input_feature.values()),
            self.c_atom,
        )

        self.readout_transition = Transition(c_in=self.c_atom, n=2)
        self.readout_layernorm = LayerNorm(self.c_atom)
        self.readout_linear = LinearNoBias(self.c_atom, 3, init="zeros")

    @typecheck
    def readout(
        self, q_atom: Float[Tensor, "... n_sample n_atom c_atom"]
    ) -> Float[Tensor, "... n_sample n_atom 3"]:
        """Readout layer.

        Args:
            q_atom: Atom features

        Returns:
            Predicted atom coordinates
        """
        q_atom = q_atom + self.readout_transition(q_atom)
        return self.readout_linear(self.readout_layernorm(q_atom))

    def as_predictor(
        self,
        context: FullContext,
        inplace_safe: bool = False,
        cfg_schedule: CFGSchedule = NoCFG(),
        self_conditioning: bool = True,
    ) -> EDMDenoisePredictor:
        """Create an EDM denoising predictor bound to this module and context.

        Optionally wraps with classifier-free guidance if cfg_schedule is not NoCFG.

        Args:
            context: Full feature context to use for predictions.
            inplace_safe: Whether to use inplace operations safely.
            cfg_schedule: CFG schedule. Use NoCFG() to skip classifier-free guidance
                entirely (no unconditional forward pass). Defaults to NoCFG().
            self_conditioning: Whether to chain predictions across sampling steps.
                When True, each step receives the aligned previous prediction.
                When False, prev_pred is always None (no self-conditioning).

        Returns:
            A callable satisfying EDMDenoisePredictor protocol.
            If cfg_schedule is not NoCFG, returns a CFGPredictor.
        """

        def make_predictor(ctx: FullContext) -> EDMDenoisePredictor:
            """Create a predictor bound to a specific context."""
            conditioning_cache = self.diffusion_conditioning.compute_cache(ctx)
            prev_pred: Optional[Float[Tensor, "... n_atom 3"]] = None

            def predictor(
                x_noisy: Float[Tensor, "... n_atom 3"],
                noise_level: Float[Tensor, "..."],
            ) -> Float[Tensor, "... n_atom 3"]:
                nonlocal prev_pred  # hidden statefulness for self conditioning

                # Align previous prediction to current noisy coords (handles frame changes)
                aligned_prev = None
                if self_conditioning and prev_pred is not None:
                    n_sample = x_noisy.shape[-3]
                    mask = repeat(ctx.mask.is_resolved, "... n -> ... s n", s=n_sample)
                    aligned_prev, _, _ = align_pred_to_true(
                        pred_pose=prev_pred,
                        true_pose=x_noisy,
                        atom_mask=mask,
                    )

                coords_pred = self.forward(
                    r_noisy=x_noisy,
                    noise_level=noise_level,
                    context=ctx,
                    cache=conditioning_cache,
                    inplace_safe=inplace_safe,
                    prev_pred=aligned_prev,
                )

                if self_conditioning:
                    prev_pred = coords_pred.detach()

                return coords_pred

            return predictor

        cond_predictor = make_predictor(context)

        if isinstance(cfg_schedule, NoCFG):
            return cond_predictor

        uncond_context = context.to_unconditional()
        uncond_predictor = make_predictor(uncond_context)
        return EDMCFGPredictor(
            cond_predictor=cond_predictor,
            uncond_predictor=uncond_predictor,
            guidance_schedule=cfg_schedule,
        )

    @typecheck
    def sample(
        self,
        context: FullContext,
        noise_scheduler: EDMInferenceNoiseScheduler,
        n_steps: int,
        cfg_schedule: CFGSchedule = NoCFG(),
        gamma0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale_lambda: float = 1.003,
        step_scale_eta: float = 1.5,
        n_sample: int = 1,
        inplace_safe: bool = False,
        centre_coords: bool = False,
        track_trajectory: bool = False,
        self_conditioning: bool = True,
    ) -> SamplingResult:
        """Backbone denoising sampling using AF3/Protenix-style EDM.

        Args:
            context: Full feature context.
            noise_scheduler: Scheduler for generating EDM noise levels.
            n_steps: Number of integration steps.
            cfg_schedule: CFG schedule.
            gamma0: AF3 churn factor.
            gamma_min: Noise threshold below which churn is disabled.
            noise_scale_lambda: Scale factor for added churn noise.
            step_scale_eta: Euler step scale factor.
            n_sample: Number of samples to generate.
            inplace_safe: Whether to use inplace operations safely.
            track_trajectory: If True, store coordinates at each step.

        Returns:
            SamplingResult containing final coordinates and optional trajectory.
        """
        n_atom = context.reference.ref_pos.shape[-2]
        batch_shape = context.token.residue_index.shape[:-1]
        device = context.token.residue_index.device
        dtype = torch.float32

        # Generate initial noise
        noise_schedule = noise_scheduler.get_schedule(n_steps, device, dtype)
        x_init = noise_schedule[0] * torch.randn(
            size=(*batch_shape, n_sample, n_atom, 3), device=device, dtype=dtype
        )

        # Create sampler and predictor
        sampler = EDMSampler(
            noise_scheduler=noise_scheduler,
            gamma0=gamma0,
            gamma_min=gamma_min,
            noise_scale_lambda=noise_scale_lambda,
            step_scale_eta=step_scale_eta,
            centre_coords=centre_coords,
        )
        predictor = self.as_predictor(
            context,
            inplace_safe=inplace_safe,
            cfg_schedule=cfg_schedule,
            self_conditioning=self_conditioning,
        )

        return sampler.sample(
            predictor=predictor,
            x_init=x_init,
            n_steps=n_steps,
            mask=context.mask.is_resolved,
            track_trajectory=track_trajectory,
        )

    @typecheck
    def f_forward(
        self,
        r_noisy: Float[Tensor, "... n_sample n_atom 3"],
        noise_level: Float[Tensor, "... n_sample"],
        context: FullContext,
        cache: Optional[CachedConditioning] = None,
        inplace_safe: bool = False,
        prev_pred: Optional[Float[Tensor, "... n_sample n_atom 3"]] = None,
    ) -> Float[Tensor, "... n_sample n_atom 3"]:
        """The raw network to be trained. This is F_theta(c_in * x, c_noise).
        c_noise(sigma) is computed in Conditioning module.

        Args:
            r_noisy: Scaled noisy coordinates (i.e., c_in * x)
            noise_level: EDM noising level.
            context: Full context containing all features and labels.
            cache: Precomputed timestep-independent conditioning from DiffusionConditioning.
            inplace_safe: Whether it is safe to use inplace operations.

        Returns:
            Network output (velocity prediction or denoising coord prediction,
            depending on network parameterisation)
        """
        n_sample = r_noisy.shape[-3]

        atom_mask = repeat(
            context.mask.atom_mask,
            "... a -> ... s a",
            s=n_sample,
        )
        atom_to_token_idx = repeat(
            context.index.atom_to_token_idx,
            "... a -> ... s a",
            s=n_sample,
        )

        n_token = context.token.residue_index.shape[-1]
        n_atom = atom_mask.shape[-1]
        batch_shape = context.token.residue_index.shape[:-1]

        # Compute pair mask
        token_mask = context.mask.token_mask  # [..., N_token]
        token_mask = token_mask.unsqueeze(-2)  # add N_sample dim
        pair_mask = token_mask[..., :, None] * token_mask[..., None, :]
        # [..., 1, N_token, N_token]

        # Average coordinates per token
        q_r = self.linear_no_bias_r_embed(r_noisy)
        if prev_pred is not None:
            q_prev = self.linear_no_bias_prev_pred_embed(prev_pred)
            q_skip = self.self_cond_merge(torch.cat([q_r, q_prev], dim=-1))
        else:
            q_skip = q_r

        a_token = average_atom_to_token(
            x_atom=q_skip,  # [..., N_sample, N_atom, 3]
            atom_mask=atom_mask,  # [..., N_atom]
            atom_to_token_idx=atom_to_token_idx,  # [..., N_sample, N]
            n_token=n_token,
        )  # [..., N_sample, N_token, c_token]

        s, z = self.diffusion_conditioning(
            r_noisy=r_noisy,
            noise_level=noise_level,
            context=context,
            cache=cache,
            inplace_safe=inplace_safe,
        )  # [..., N_sample, N_token, c_s], [..., N_token, N_token, c_z]

        # Mix single conditioning into token representation
        a_token = a_token + self.linear_no_bias_s(self.layernorm_s(s))
        a_token = a_token * token_mask[..., None]  # [..., N_sample, N_token, c_token]

        z = rearrange(z, "... i j c_z -> ... 1 i j c_z")
        z = z * rearrange(pair_mask, "... 1 i j -> ... 1 i j 1")

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
        atom_features = self.linear_no_bias_f(
            torch.cat(
                [
                    getattr(context.reference, name)
                    .reshape(*batch_shape, n_atom, self.atom_input_feature[name])
                    .to(q_atom.dtype)
                    for name in self.atom_input_feature
                ],
                dim=-1,
            )
        )
        q_atom = q_atom + repeat(atom_features, "... a c -> ... s a c", s=n_sample)

        return self.readout(q_atom)

    def c_in(self, noise_level: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        """EDM input scaling factor for noisy coordinates."""
        return 1 / torch.sqrt(SIGMA_DATA**2 + noise_level**2)

    def c_out(self, noise_level: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        """EDM output scaling factor."""
        return SIGMA_DATA * noise_level / torch.sqrt(SIGMA_DATA**2 + noise_level**2)

    def c_skip(self, noise_level: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        """EDM skip connection scaling factor."""
        return SIGMA_DATA**2 / (SIGMA_DATA**2 + noise_level**2)

    @typecheck
    def forward(
        self,
        r_noisy: Float[Tensor, "... n_sample n_atom 3"],
        noise_level: Float[Tensor, "... n_sample"],
        context: FullContext,
        cache: Optional[CachedConditioning] = None,
        inplace_safe: bool = False,
        prev_pred: Optional[Float[Tensor, "... n_sample n_atom 3"]] = None,
    ) -> Float[Tensor, "... n_sample n_atom 3"]:
        """Performs a forward pass of the diffusion module.

        Args:
            r_noisy: Noisy coordinates
            noise_level: EDM noising level.
            context: Full context containing all features and labels.
            cache: Precomputed timestep-independent conditioning from DiffusionConditioning.
            inplace_safe: Whether it is safe to use inplace operations.
            prev_pred: Previous denoised prediction for self-conditioning.

        Returns:
            Predicted coordinates
        """
        # Expand noise levels
        noise_level_ = noise_level[..., None, None]

        # Scale input noisy coords to unit variance
        r_noisy_unit = r_noisy * self.c_in(noise_level_)

        # Previous predictions are already denoised coordinates, so scale them
        # by the fixed data scale rather than the current noise-dependent c_in.
        prev_pred_unit = None
        if prev_pred is not None:
            prev_pred_unit = prev_pred / SIGMA_DATA

        # Predict EDM residual update
        out = self.f_forward(
            r_noisy=r_noisy_unit,
            noise_level=noise_level,
            context=context,
            cache=cache,
            inplace_safe=inplace_safe,
            prev_pred=prev_pred_unit,
        )

        # Apply skip connections and output scaling
        r_pred = r_noisy * self.c_skip(noise_level_) + out * self.c_out(noise_level_)

        return r_pred
