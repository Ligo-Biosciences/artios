from typing import Optional

import torch
import torch.nn as nn
from typing_extensions import assert_never

from protenix.data.features import FullContext
from protenix.model.generator import (
    LaProteinaCoordScheduler,
    ProteinaCoordSampler,
    stochasticity_from_config,
)
from protenix.model.modules.diffusion_backbone import DiffusionModule
from protenix.model.output import ModelOutput
from protenix.model.utils import Mode, centre_random_augmentation, noise_w_flow_matching
from protenix.utils.permutation import atom_permutation


class ArtiosBackbone(nn.Module):
    def __init__(self, configs):
        super(ArtiosBackbone, self).__init__()
        self.configs = configs

        self.N_model_seed = self.configs.model.N_model_seed

        self.timestep_sampler = ProteinaCoordSampler(
            **configs.train_timestep_sampler.coord
        )

        self.inference_timestep_scheduler = LaProteinaCoordScheduler(
            **configs.inference_timestep_scheduler.la_proteina.coord
        )

        self.sampling_configs = configs.sample_diffusion

        self.diffusion_batch_size = self.configs.diffusion_batch_size
        self.cfg_drop_probability = self.configs.cfg_drop_probability

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
    ) -> ModelOutput:
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
            torch.Generator(device).manual_seed(seed) if not self.training else None
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
        result = atom_permutation.run_batched(
            pred_coord=pred,
            true_coord=context.coordinate.coordinate,
            true_coord_mask=context.mask.atom_loss_mask,
            ref_space_uid=context.reference.ref_space_uid,
            atom_perm_list=context.atom_perm.perms,
            pdb_id=context.basic.pdb_id,
            permute_label=False,
        )

        symmetry_corrected = result.coordinate

        return ModelOutput(coordinate=symmetry_corrected, timestep=timestep)

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
