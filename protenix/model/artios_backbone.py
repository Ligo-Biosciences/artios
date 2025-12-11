from typing import Any, Optional

import torch
import torch.nn as nn

from protenix.data.features import FullContext
from protenix.model.generator_backbone import (
    LaProteinaCoordScheduler,
    TrainingTimestepSampler,
    sample_diffusion,
    sample_noising,
)
from protenix.model.modules.diffusion_backbone import DiffusionModule
from protenix.model.output import ModelOutput
from protenix.utils.torch_utils import autocasting_disable_decorator


class ArtiosBackbone(nn.Module):
    def __init__(self, configs):
        super(ArtiosBackbone, self).__init__()
        self.configs = configs

        self.N_model_seed = self.configs.model.N_model_seed

        # Fixed eval generator for reproducibility
        # self.eval_generator = torch.Generator(device=self.device).manual_seed(42)

        self.timestep_sampler = TrainingTimestepSampler(
            **configs.train_timestep_sampler
        )

        self.inference_timestep_scheduler = LaProteinaCoordScheduler(
            **configs.inference_timestep_scheduler.la_proteina.coord
        )

        self.diffusion_batch_size = self.configs.diffusion_batch_size
        self.cfg_drop_probability = self.configs.cfg_drop_probability

        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        self.sigma_coord = configs.sigma_coord

    def sample_diffusion(self, **kwargs) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
        torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "sigma_coord",
                "stoch_type",
                "alpha",
                "gamma",
                "second_order",
                "guidance_weight",
                "beta_mean",
                "beta_std",
                "lower_guidance_limit",
                "upper_guidance_limit",
            ]
        }
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)

    def main_inference_loop(
        self,
        context: FullContext,
        inplace_safe: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop for ArtiosBackbone.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        device = context.token.residue_index.device
        dtype = context.token.residue_index.dtype

        N_sample = self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]

        timesteps = self.inference_timestep_scheduler(
            N_step=N_step, device=device, dtype=dtype
        )

        pred = self.sample_diffusion(
            denoise_net=self.diffusion_module,
            context=context,
            timesteps=timesteps,
            N_sample=N_sample,
            inplace_safe=inplace_safe,
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
