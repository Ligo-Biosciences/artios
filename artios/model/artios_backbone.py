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
    cfg_schedule_from_config,
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


class ArtiosBackbone(nn.Module):
    def __init__(
        self,
        configs=None,
        *,
        model_config: ArtiosBackboneConfig | None = None,
        runtime_config: ArtiosBackboneRuntimeConfig | None = None,
    ):
        super(ArtiosBackbone, self).__init__()
        if configs is None:
            raise ValueError(
                "configs are still required for loss, noise, and sampling settings"
            )
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
        self.model_config = model_config
        self.runtime_config = runtime_config

        mse_cfg = configs.loss.backbone.mse
        self.loss_weight_config = LossWeightConfig(
            protein=mse_cfg.weight_protein,
            dna=mse_cfg.weight_dna,
            rna=mse_cfg.weight_rna,
            ligand=mse_cfg.weight_ligand,
            motif=mse_cfg.weight_motif,
            motif_related=mse_cfg.weight_motif_related,
        )

        self.N_model_seed = self.model_config.n_model_seed

        self.noise_sampler = EDMTrainingNoiseSampler(**configs.train_noise_sampler)

        self.inference_noise_scheduler = EDMInferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )

        self.sampling_configs = configs.sample_diffusion

        self.diffusion_batch_size = self.configs.diffusion_batch_size

        self.diffusion_module = DiffusionModule(
            config=self.model_config.diffusion_module,
            runtime_config=self.runtime_config.diffusion,
        )

    def main_inference_loop(
        self,
        context: FullContext,
        inplace_safe: bool = False,
    ) -> BackboneOutput:
        """Main inference loop for ArtiosBackbone.

        Args:
            context: Full feature context.
            inplace_safe: Whether to use inplace operations safely.

        Returns:
            BackboneOutput: Model output containing the prediction.
        """
        N_sample = self.sampling_configs.N_sample
        N_step = self.sampling_configs.N_step

        cfg_schedule = cfg_schedule_from_config(
            cfg_type=self.sampling_configs.cfg_type,
            weight=self.sampling_configs.cfg_weight,
            lower_guidance_limit_weight=self.sampling_configs.lower_guidance_limit_weight,
            upper_guidance_limit_weight=self.sampling_configs.upper_guidance_limit_weight,
            lower_guidance_limit=self.sampling_configs.lower_guidance_limit,
            upper_guidance_limit=self.sampling_configs.upper_guidance_limit,
        )

        result = self.diffusion_module.sample(
            context=context,
            noise_scheduler=self.inference_noise_scheduler,
            n_steps=N_step,
            gamma0=self.sampling_configs.gamma0,
            gamma_min=self.sampling_configs.gamma_min,
            noise_scale_lambda=self.sampling_configs.noise_scale_lambda,
            step_scale_eta=self.sampling_configs.step_scale_eta,
            n_sample=N_sample,
            inplace_safe=inplace_safe,
            cfg_schedule=cfg_schedule,
            centre_coords=self.sampling_configs.centre_and_augment_coords_during_sampling,
            track_trajectory=self.sampling_configs.track_trajectory,
            self_conditioning=self.sampling_configs.self_conditioning,
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
        if torch.rand(1).item() < self.configs.self_conditioning_prob:
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
    ) -> ModelOutput:
        """
        Forward pass of the Artios backbone model.

        Args:
            context (FullContext): Full context containing all features and labels.
            mode (Mode): Mode of operation.
            seed (Optional[int]): Seed for random number generation. Defaults to None.

        Returns:
            ModelOutput: Model output containing the prediction and noise level.
        """
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
