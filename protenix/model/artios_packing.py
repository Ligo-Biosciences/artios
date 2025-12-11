from typing import Optional

import torch
import torch.nn as nn
from einops import repeat

from protenix.data.features import FullContext
from protenix.model.generator_backbone import (
    LaProteinaCoordScheduler,
    TrainingTimestepSampler,
    sample_local_noising,
)
from protenix.model.modules.sidechain_packer import SidechainPacker
from protenix.model.output import ModelOutput
from protenix.model.utils import broadcast_token_to_atom


class ArtiosPacking(nn.Module):
    def __init__(self, configs):
        super(ArtiosPacking, self).__init__()
        self.configs = configs

        # Fixed eval generator for reproducibility
        # self.eval_generator = torch.Generator(device=self.device).manual_seed(42)

        self.timestep_sampler = TrainingTimestepSampler(
            **configs.artios_packing.train_timestep_sampler
        )

        self.inference_timestep_scheduler = LaProteinaCoordScheduler(
            **configs.artios_packing.inference_timestep_scheduler.la_proteina.coord
        )
        self.sampling_configs = configs.artios_packing.sample_diffusion

        self.diffusion_batch_size = self.configs.diffusion_batch_size
        self.cfg_drop_probability = self.configs.cfg_drop_probability

        self.sigma_coord = configs.sigma_coord
        self.local_sigma_coord = configs.artios_packing.model.local_sigma_coord

        self.sidechain_packer = SidechainPacker(**configs.artios_packing.model)

    def main_inference_loop(
        self,
        context: FullContext,
    ) -> ModelOutput:
        """
        Main inference loop for ArtiosBackbone.

        Returns:
            ModelOutput: Model output containing the prediction.
        """
        device = context.token.residue_index.device
        dtype = context.token.residue_index.dtype

        N_sample = self.sampling_configs.N_sample
        N_step = self.sampling_configs.N_step

        timesteps = self.inference_timestep_scheduler(
            N_step=N_step, device=device, dtype=dtype
        )

        residual_centre_atoms = broadcast_token_to_atom(
            x_token=context.coordinate.centre_coordinate,
            atom_to_token_idx=context.index.atom_to_token_idx,
        )
        residual_centre_atoms = repeat(
            residual_centre_atoms,
            "... n_atom c -> ... n_sample n_atom c",
            n_sample=N_sample,
        )

        pred = self.sidechain_packer.sample(
            context=context,
            timesteps=timesteps,
            residual_centre_atoms=residual_centre_atoms,
            n_sample=N_sample,
            stoch_type=self.sampling_configs.stoch_type,
            alpha=self.sampling_configs.alpha,
            gamma=self.sampling_configs.gamma,
            eps=self.sampling_configs.eps,
        )

        return ModelOutput(coordinate=pred)

    def main_train_loop(
        self,
        context: FullContext,
        seed: Optional[int] = None,
    ) -> ModelOutput:
        """
        Main training (and evaluation) loop for the Artios sidechain packing model.

        Args:
            context (FullContext): Full context containing all features and labels.
            seed (Optional[int]): Seed for random number generation. Defaults to None.

        Returns:
            ModelOutput: Model output containing the prediction.
        """
        device = context.coordinate.coordinate.device

        n_sample = self.diffusion_batch_size
        n_token = context.token.token_index.shape[-1]

        generator = (
            torch.Generator(device).manual_seed(seed)
            if seed is not None and not self.training
            else None
        )

        # Noising
        r_noisy, timestep, _ = sample_local_noising(
            input=context.coordinate.coordinate,
            mask=context.mask.is_resolved,
            centre_atom_mask=context.mask.centre_atom_mask,
            atom_to_token_idx=context.index.atom_to_token_idx,
            timestep_sampler=self.timestep_sampler,
            n_sample=n_sample,
            n_token=n_token,
            scale=self.local_sigma_coord,
            generator=generator,
        )

        # Denoising
        pred = self.sidechain_packer(
            r_noisy=r_noisy,
            timestep=timestep,
            context=context,
        )

        return ModelOutput(coordinate=pred, timestep=timestep)

    def forward(
        self,
        context: FullContext,
        mode: str,
        *,
        seed: Optional[int] = None,
    ) -> ModelOutput:
        """
        Forward pass of the Artios sidechain packing model.

        Args:
            context (FullContext): Full context containing all features and labels.
            mode (str): Mode of operation ('train', 'inference', 'eval').
            seed (Optional[int]): Seed for random number generation. Defaults to None.

        Returns:
            ModelOutput: Model output containing the prediction and timestep.
        """

        assert mode in ["train", "inference", "eval"]

        if mode == "train" or mode == "eval":
            output = self.main_train_loop(
                context=context,
                seed=seed,
            )
        elif mode == "inference":
            output = self.main_inference_loop(
                context=context,
            )

        return output
