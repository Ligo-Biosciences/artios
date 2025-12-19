from typing import Optional

import torch
import torch.nn as nn
from einops import repeat
from torch import Tensor
from typing_extensions import assert_never

from protenix.data.features import FullContext
from protenix.model.generator import (
    LaProteinaCoordScheduler,
    ProteinaCoordSampler,
    stochasticity_from_config,
)
from protenix.model.loss import (
    LossWeightConfig,
    align_to_prediction,
    build_atom_weights,
    compute_entity_metrics,
    mse_loss,
    mse_per_atom,
    timestep_scale,
)
from protenix.model.modules.diffusion_backbone import DiffusionModule
from protenix.model.output import DiffusionOutput, ModelOutput
from protenix.model.utils import Mode, noise_w_flow_matching
from protenix.utils.casting import disable_autocast
from protenix.utils.geometry import centre_random_augmentation


class ArtiosBackbone(nn.Module):
    def __init__(self, configs):
        super(ArtiosBackbone, self).__init__()
        self.configs = configs

        mse_cfg = configs.loss.backbone.mse
        self.loss_weight_config = LossWeightConfig(
            protein=mse_cfg.weight_protein,
            dna=mse_cfg.weight_dna,
            rna=mse_cfg.weight_rna,
            ligand=mse_cfg.weight_ligand,
            motif=mse_cfg.weight_motif,
            motif_related=mse_cfg.weight_motif_related,
        )

        self.N_model_seed = self.configs.model.N_model_seed

        self.timestep_sampler = ProteinaCoordSampler(**configs.train_timestep_sampler)

        self.inference_timestep_scheduler = LaProteinaCoordScheduler(
            **configs.inference_timestep_scheduler.la_proteina.coord
        )

        self.sampling_configs = configs.sample_diffusion

        self.diffusion_batch_size = self.configs.diffusion_batch_size

        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        self.sigma_coord = configs.sigma_coord

    def main_inference_loop(
        self,
        context: FullContext,
        inplace_safe: bool = False,
    ) -> ModelOutput:
        """Main inference loop for ArtiosBackbone.

        Args:
            context: Full feature context.
            inplace_safe: Whether to use inplace operations safely.

        Returns:
            ModelOutput: Model output containing the prediction.
        """
        N_sample = self.sampling_configs.N_sample
        N_step = self.sampling_configs.N_step

        stochasticity = stochasticity_from_config(
            stoch_type=self.sampling_configs.stoch_type,
            alpha=self.sampling_configs.alpha,
        )

        pred = self.diffusion_module.sample(
            context=context,
            timestep_scheduler=self.inference_timestep_scheduler,
            n_steps=N_step,
            stochasticity=stochasticity,
            gamma=self.sampling_configs.gamma,
            n_sample=N_sample,
            second_order=self.sampling_configs.second_order,
            inplace_safe=inplace_safe,
        )

        return ModelOutput(coordinate=pred)

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

        # Generate timesteps
        timestep = self.timestep_sampler.sample(
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
            * self.sigma_coord
        )

        # Apply flow matching interpolation
        r_noisy = noise_w_flow_matching(
            x_target_augment, noise, timestep[..., :, None, None]
        )

        # Denoising
        pred = self.diffusion_module(
            r_noisy=r_noisy,
            timestep=timestep,
            context=context,
            inplace_safe=inplace_safe,
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

        return DiffusionOutput(coordinate=pred, timestep=timestep)

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
            ModelOutput: Model output containing the prediction and timestep.
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
            output: Model output containing predicted coordinates and timestep.
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

        # Scale by timestep
        scale = timestep_scale(output.timestep, sigma=self.sigma_coord)
        scaled_mse = scale * per_sample_mse

        # Reduce to scalar
        loss = scaled_mse.mean()

        # Compute entity metrics for logging
        per_atom_se = mse_per_atom(pred, true_aligned)  # [batch, n_sample, n_atom]
        per_atom_se = per_atom_se * scale[..., None]  # scale by timestep
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
