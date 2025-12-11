from typing import Any, Optional

import torch
import torch.nn as nn

from protenix.data.features import FullContext
from protenix.model.generator import (
    FlowMatchingSampler,
    LaProteinaCoordScheduler,
    ProteinaCoordSampler,
    StochasticitySchedule,
    VelocityScorePredictor,
    sample_noising,
    stochasticity_from_config,
)
from protenix.model.modules.diffusion_backbone import DiffusionModule
from protenix.model.output import ModelOutput


class ArtiosBackbone(nn.Module):
    def __init__(self, configs):
        super(ArtiosBackbone, self).__init__()
        self.configs = configs

        self.N_model_seed = self.configs.model.N_model_seed

        # Training timestep sampler
        self.timestep_sampler = ProteinaCoordSampler(**configs.train_timestep_sampler)

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
            ModelOutput containing predicted coordinates.
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
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main training (and evaluation) loop for the Artios All-Atom model.

        Args:
            context (FullContext): Full context containing all features and labels.
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
            seed (Optional[int]): Seed for random number generation. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """
        device = context.coordinate.coordinate.device

        N_sample = self.diffusion_batch_size
        generator = (
            torch.Generator(device).manual_seed(seed) if not self.training else None
        )

        # Noising
        noisy_coords, timestep, _ = sample_noising(
            input=context.coordinate.coordinate,
            mask=context.mask.is_resolved,
            timestep_sampler=self.timestep_sampler,
            num_sample=N_sample,
            scale=self.sigma_coord,
            generator=generator,
        )

        # # With CFG drop probability, train the unconditional model
        # if torch.rand(1).item() < self.cfg_drop_probability:
        #     context.entity.is_ligand = torch.zeros_like(context.entity.is_ligand)
        #     context.entity.is_ligand_token = torch.zeros_like(context.entity.is_ligand_token)
        #     context.entity.is_target_token = torch.zeros_like(context.entity.is_target_token)
        #     context.entity.is_hotspot = torch.zeros_like(context.entity.is_hotspot)
        #     context.entity.is_motif_related = torch.zeros_like(context.entity.is_motif_related)
        #     context.entity.is_motif_related_token = torch.zeros_like(context.entity.is_motif_related_token)

        # Denoising
        pred = self.diffusion_module(
            r_noisy=noisy_coords,  # TODO: can we maintain one of r_noisy or noisy_coords plz?
            timestep=timestep,
            context=context,
            inplace_safe=inplace_safe,
        )

        return ModelOutput(coordinate=pred, timestep=timestep)

    def forward(
        self,
        context: FullContext,
        mode: str,
        *,
        current_step: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> ModelOutput:
        """
        Forward pass of the Artios backbone model.

        Args:
            context (FullContext): Full context containing all features and labels.
            mode (str): Mode of operation ('train', 'inference', 'eval').
            current_step (Optional[int]): Current training step. Defaults to None.
            seed (Optional[int]): Seed for random number generation. Defaults to None.

        Returns:
            ModelOutput: Model output containing the prediction and timestep.
        """

        assert mode in ["train", "inference", "eval"]
        inplace_safe = not (self.training or torch.is_grad_enabled())

        if mode == "train" or mode == "eval":
            output = self.main_train_loop(
                context=context,
                inplace_safe=inplace_safe,
                seed=seed,
            )
        elif mode == "inference":
            output = self.main_inference_loop(
                context=context,
                inplace_safe=False,
            )

        return output
