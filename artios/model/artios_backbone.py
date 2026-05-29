from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
from einops import repeat
from torch import Tensor
from typing_extensions import assert_never

from artios.data.features import FullContext
from artios.model.generator import (
    SIGMA_DATA,
    EDMInferenceNoiseScheduler,
    EDMTrainingNoiseSampler,
)
from artios.model.loss import (
    LossWeightConfig,
    align_to_prediction,
    build_atom_weights,
    compute_entity_metrics,
    edm_noise_level_scale,
    mse_loss,
    mse_per_atom,
)
from artios.model.modules.diffusion_backbone import (
    DiffusionModule,
    DiffusionModuleConfig,
    DiffusionRuntimeConfig,
)
from artios.model.output import BackboneOutput, DiffusionOutput, ModelOutput
from artios.model.utils import Mode
from artios.utils.casting import disable_autocast
from artios.utils.geometry import centre_random_augmentation


def _has_config_key(config: Any, key: str) -> bool:
    if config is None:
        return False
    try:
        return key in config
    except TypeError:
        return hasattr(config, key)


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtiosBackboneConfig:
    """Immutable model identity for ArtiosBackbone."""

    n_model_seed: int = 1
    diffusion_module: DiffusionModuleConfig = field(
        default_factory=DiffusionModuleConfig
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtiosBackboneRuntimeConfig:
    """Runtime settings that do not define ArtiosBackbone architecture."""

    diffusion: DiffusionRuntimeConfig = field(default_factory=DiffusionRuntimeConfig)


@dataclass(frozen=True, slots=True, kw_only=True)
class BackboneTrainingConfig:
    """Training recipe for ArtiosBackbone.

    Defaults are production SOTA from train_backbone_edm_motif_ema_0.9999.sh.
    """

    diffusion_batch_size: int = 1
    self_conditioning_prob: float = 0.5
    loss_weights: LossWeightConfig = field(
        default_factory=lambda: LossWeightConfig(
            protein=1.0,
            dna=0.0,
            rna=0.0,
            ligand=10.0,
            motif=0.0,
            motif_related=5.0,
        )
    )

    def __post_init__(self) -> None:
        if self.diffusion_batch_size < 1:
            raise ValueError("diffusion_batch_size must be >= 1")
        if not 0.0 <= self.self_conditioning_prob <= 1.0:
            raise ValueError("self_conditioning_prob must be in [0, 1]")


class ArtiosBackbone(nn.Module):
    def __init__(
        self,
        configs=None,
        *,
        model_config: ArtiosBackboneConfig | None = None,
        runtime_config: ArtiosBackboneRuntimeConfig | None = None,
        training_config: BackboneTrainingConfig | None = None,
    ):
        super(ArtiosBackbone, self).__init__()
        if _has_config_key(configs, "sigma_data"):
            raise ValueError(
                "SIGMA_DATA is a code constant; remove the sigma_data config override."
            )
        if _has_config_key(configs, "model"):
            legacy_model_config = configs.model
            legacy_architecture_keys = [
                key
                for key in ("N_model_seed", "diffusion_module")
                if _has_config_key(legacy_model_config, key)
            ]
            if legacy_architecture_keys:
                joined_keys = ", ".join(
                    f"model.{key}" for key in legacy_architecture_keys
                )
                raise ValueError(
                    f"{joined_keys} are no longer config fields. "
                    "Edit ArtiosBackboneConfig or DiffusionModuleConfig in code instead."
                )
        self.configs = configs
        if model_config is None:
            model_config = ArtiosBackboneConfig()
        if runtime_config is None:
            runtime_config = ArtiosBackboneRuntimeConfig(
                diffusion=DiffusionRuntimeConfig(
                    checkpoint_grads=bool(configs.checkpoint_grads)
                )
            )
        if training_config is None:
            mse_cfg = configs.loss.backbone.mse
            training_config = BackboneTrainingConfig(
                loss_weights=LossWeightConfig(
                    protein=mse_cfg.weight_protein,
                    dna=mse_cfg.weight_dna,
                    rna=mse_cfg.weight_rna,
                    ligand=mse_cfg.weight_ligand,
                    motif=mse_cfg.weight_motif,
                    motif_related=mse_cfg.weight_motif_related,
                ),
            )
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.training_config = training_config

        self.loss_weight_config = self.training_config.loss_weights
        self.N_model_seed = self.model_config.n_model_seed
        self.noise_sampler = EDMTrainingNoiseSampler()
        self.inference_noise_scheduler = EDMInferenceNoiseScheduler()
        self.diffusion_batch_size = self.training_config.diffusion_batch_size

        self.diffusion_module = DiffusionModule(
            config=self.model_config.diffusion_module,
            runtime_config=self.runtime_config.diffusion,
        )

    def main_inference_loop(
        self,
        context: FullContext,
        inplace_safe: bool = False,
        *,
        n_sample: int | None = None,
        n_step: int | None = None,
        track_trajectory: bool | None = None,
    ) -> BackboneOutput:
        if n_sample is None:
            n_sample = self.configs.sample_diffusion.N_sample
        if n_step is None:
            n_step = self.configs.sample_diffusion.N_step
        if track_trajectory is None:
            track_trajectory = self.configs.sample_diffusion.track_trajectory

        result = self.diffusion_module.sample(
            context=context,
            noise_scheduler=self.inference_noise_scheduler,
            n_steps=n_step,
            n_sample=n_sample,
            inplace_safe=inplace_safe,
            track_trajectory=track_trajectory,
        )

        return BackboneOutput(
            coordinate=result.final,
            trajectory=result.trajectory,
        )

    def main_train_loop(
        self,
        context: FullContext,
        inplace_safe: bool = False,
        seed: Optional[int] = None,
    ) -> DiffusionOutput:
        """
        Main training (and evaluation) loop for the Artios All-Atom model.

        Args:
            context (FullContext): Full context containing all features and labels.
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
            seed (Optional[int]): Seed for random number generation. Defaults to None.

        Returns:
            ModelOutput: Model output containing the prediction.
        """
        device = context.coordinate.coordinate.device
        dtype = context.coordinate.coordinate.dtype
        batch_shape = context.coordinate.coordinate.shape[:-2]

        n_sample = self.diffusion_batch_size
        generator = (
            torch.Generator(device).manual_seed(seed)
            if not self.training and seed is not None
            else None
        )

        # Augment target coordinates
        x_target_augment = centre_random_augmentation(
            x_input_coords=context.coordinate.coordinate,
            n_sample=n_sample,
            mask=context.mask.is_resolved,
        ).to(dtype)

        # Generate EDM noise levels
        noise_level = self.noise_sampler.sample(
            size=(*batch_shape, n_sample), device=device, generator=generator
        ).to(dtype)

        # Generate noise
        noise = (
            torch.randn(
                x_target_augment.shape,
                dtype=dtype,
                device=device,
                generator=generator,
            )
            * noise_level[..., None, None]
        )

        # EDM additive noising
        r_noisy = x_target_augment + noise

        # Self-conditioning: with probability p, warm-start with a no-grad forward pass
        prev_pred = None
        if torch.rand(1).item() < self.training_config.self_conditioning_prob:
            with torch.no_grad():
                prev_pred = self.diffusion_module(
                    r_noisy=r_noisy,
                    noise_level=noise_level,
                    context=context,
                    inplace_safe=inplace_safe,
                ).detach()

        # Denoising
        pred = self.diffusion_module(
            r_noisy=r_noisy,
            noise_level=noise_level,
            context=context,
            inplace_safe=inplace_safe,
            prev_pred=prev_pred,
        )

        # Correct symmetric atoms
        # result = atom_permutation.run_batched(
        #     pred_coord=pred,
        #     true_coord=context.coordinate.coordinate,
        #     true_coord_mask=context.mask.atom_loss_mask,
        #     ref_space_uid=context.reference.ref_space_uid,
        #     atom_perm_list=context.atom_perm.perms,
        #     pdb_id=context.basic.pdb_id,
        #     permute_label=False,
        # )

        # symmetry_corrected = result.coordinate if result else pred

        return DiffusionOutput(coordinate=pred, noise_level=noise_level)

    def forward(
        self,
        context: FullContext,
        mode: Mode,
        *,
        seed: Optional[int] = None,
        n_sample: int | None = None,
        n_step: int | None = None,
        track_trajectory: bool | None = None,
    ) -> ModelOutput:
        inplace_safe = not (self.training or torch.is_grad_enabled())

        match mode:
            case Mode.TRAIN | Mode.EVAL:
                output = self.main_train_loop(
                    context=context,
                    inplace_safe=inplace_safe,
                    seed=seed,
                )
            case Mode.INFERENCE:
                output = self.main_inference_loop(
                    context=context,
                    inplace_safe=False,
                    n_sample=n_sample,
                    n_step=n_step,
                    track_trajectory=track_trajectory,
                )
            case _:
                assert_never(mode)

        return output

    @disable_autocast
    def compute_loss(
        self,
        output: DiffusionOutput,
        context: FullContext,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute loss from model output and context.

        Args:
            output: Model output containing predicted coordinates and noise level.
            context: Full context containing ground truth and masks.

        Returns:
            Tuple of (loss tensor, metrics dict for logging).
        """
        pred = output.coordinate  # [batch, n_sample, n_atom, 3]
        n_sample = pred.shape[-3]

        # Build atom weights
        atom_weights = build_atom_weights(
            config=self.loss_weight_config,
            is_protein=context.entity.is_protein,
            is_dna=context.entity.is_dna,
            is_rna=context.entity.is_rna,
            is_ligand=context.entity.is_ligand,
            is_motif=context.motif.is_motif,
            is_motif_related=context.motif.is_motif_related,
        )
        atom_weights = repeat(atom_weights, "b n -> b s n", s=n_sample)

        # Expand ground truth and masks to match samples
        true = repeat(context.coordinate.coordinate, "b n c -> b s n c", s=n_sample)
        mask = repeat(context.mask.atom_loss_mask, "b n -> b s n", s=n_sample)

        # Align ground truth to prediction
        true_aligned = align_to_prediction(pred, true, mask, weight=atom_weights)

        # Compute per-sample MSE loss
        per_sample_mse = mse_loss(pred, true_aligned, mask, weight=atom_weights)

        assert output.noise_level is not None, "noise_level required for EDM loss"

        # Scale by EDM noise level
        scale = edm_noise_level_scale(output.noise_level, sigma_data=SIGMA_DATA)
        scaled_mse = scale * per_sample_mse

        # Reduce to scalar
        loss = scaled_mse.mean()

        # Compute entity metrics for logging
        per_atom_se = mse_per_atom(pred, true_aligned)  # [batch, n_sample, n_atom]
        per_atom_se = per_atom_se * scale[..., None]  # scale by noise level
        per_atom_se_avg = per_atom_se.mean(dim=-2)  # [batch, n_atom]

        entity_masks = {
            "protein": context.entity.is_protein,
            "dna": context.entity.is_dna,
            "rna": context.entity.is_rna,
            "ligand": context.entity.is_ligand,
            "motif": context.motif.is_motif,
            "motif_related": context.motif.is_motif_related,
        }
        metrics = compute_entity_metrics(
            per_atom_se_avg, context.mask.atom_loss_mask, entity_masks
        )
        # Return tensor directly to avoid CUDA sync - will be converted in metrics aggregator
        metrics["mse_loss"] = scaled_mse.mean(dim=-1)  # [batch]

        return loss, metrics
