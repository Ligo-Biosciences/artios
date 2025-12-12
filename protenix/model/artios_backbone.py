from typing import Optional

import torch
import torch.nn as nn
from typing_extensions import assert_never

from protenix.data.features import FullContext
from protenix.model.generator import (
    FlowMatchingSampler,
    LaProteinaCoordScheduler,
    ProteinaCoordSampler,
    StochasticitySchedule,
    VelocityScorePredictor,
    stochasticity_from_config,
)
from protenix.model.modules.diffusion_backbone import DiffusionModule
from protenix.model.output import ModelOutput
from protenix.model.utils import Mode, centre_random_augmentation, noise_w_flow_matching


class ArtiosBackbone(nn.Module):
    def __init__(self, configs):
        super(ArtiosBackbone, self).__init__()
        self.configs = configs

        self.N_model_seed = self.configs.model.N_model_seed

        # Training timestep sampler
        self.timestep_sampler = ProteinaCoordSampler(
            **configs.train_timestep_sampler.coord
        )

        # Inference timestep scheduler
        self.inference_timestep_scheduler = LaProteinaCoordScheduler(
            **configs.inference_timestep_scheduler.la_proteina.coord
        )

        # Stochasticity schedule for inference
        sample_cfg = configs.sample_diffusion
        self.stochasticity: StochasticitySchedule = stochasticity_from_config(
            stoch_type=sample_cfg.get("stoch_type", "deterministic"),
            alpha=sample_cfg.get("alpha", 1.0),
            beta_mean=sample_cfg.get("beta_mean", 0.5),
            beta_std=sample_cfg.get("beta_std", 0.224),
            cutoff=sample_cfg.get("cutoff", 0.95),
        )

        self.diffusion_batch_size = self.configs.diffusion_batch_size
        self.cfg_drop_probability = self.configs.cfg_drop_probability

        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        self.sigma_coord = configs.sigma_coord

    def _make_predictor(
        self,
        context: FullContext,
        inplace_safe: bool = False,
        eps: float = 1e-5,
    ) -> VelocityScorePredictor:
        """Create a VelocityScorePredictor bound to this model and context.

        Args:
            context: Full feature context for predictions.
            inplace_safe: Whether to use inplace operations safely.
            eps: Small constant for numerical stability.

        Returns:
            A callable satisfying VelocityScorePredictor protocol.
        """

        def predictor(
            x_noisy: torch.Tensor,
            timestep: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Run diffusion module
            pred = self.diffusion_module(
                r_noisy=x_noisy,
                timestep=timestep,
                context=context,
                inplace_safe=inplace_safe,
            )

            # Compute velocity and score
            t = timestep[..., None, None]
            velocity = (pred - x_noisy) / torch.clamp(1 - t, min=eps)
            score = t * velocity - x_noisy
            return velocity, score

        return predictor

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
        device = context.token.residue_index.device
        dtype = torch.float32

        n_sample = self.configs.sample_diffusion["N_sample"]
        n_steps = self.configs.sample_diffusion["N_step"]
        gamma = self.configs.sample_diffusion.get("gamma", 1.0)
        second_order = self.configs.sample_diffusion.get("second_order", False)

        # Get dimensions for initial noise
        batch_shape = context.token.residue_index.shape[:-1]
        n_atom = context.reference.ref_pos.shape[-2]

        # Generate initial noise
        x_init = self.sigma_coord * torch.randn(
            size=(*batch_shape, n_sample, n_atom, 3), device=device, dtype=dtype
        )

        # Create sampler and predictor
        sampler = FlowMatchingSampler(
            timestep_scheduler=self.inference_timestep_scheduler,
            stochasticity=self.stochasticity,
            gamma=gamma,
            second_order=second_order,
            centre_coords=True,
        )
        predictor = self._make_predictor(context, inplace_safe=inplace_safe)

        # Sample
        pred = sampler.sample(
            predictor=predictor,
            x_init=x_init,
            n_steps=n_steps,
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

        return ModelOutput(coordinate=pred, timestep=timestep)

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
