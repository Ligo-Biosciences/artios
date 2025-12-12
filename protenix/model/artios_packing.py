from typing import Optional

import torch
import torch.nn as nn
from einops import repeat
from typing_extensions import assert_never

from protenix.data.features import FullContext
from protenix.model.generator import (
    BetaWithUniform,
    LaProteinaCoordScheduler,
    stochasticity_from_config,
)
from protenix.model.modules.sidechain_packer import SidechainPacker
from protenix.model.output import ModelOutput
from protenix.model.utils import (
    Mode,
    average_atom_to_token,
    broadcast_token_to_atom,
    centre_random_augmentation,
    expand_at_dim,
    noise_w_flow_matching,
)
from protenix.utils.permutation import atom_permutation


class ArtiosPacking(nn.Module):
    def __init__(self, configs):
        super(ArtiosPacking, self).__init__()
        self.configs = configs

        # Fixed eval generator for reproducibility
        # self.eval_generator = torch.Generator(device=self.device).manual_seed(42)

        self.timestep_sampler = BetaWithUniform(
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
        N_sample = self.sampling_configs.N_sample
        N_step = self.sampling_configs.N_step

        residual_centre_atoms = broadcast_token_to_atom(
            x_token=context.coordinate.centre_coordinate,
            atom_to_token_idx=context.index.atom_to_token_idx,
        )
        residual_centre_atoms = repeat(
            residual_centre_atoms,
            "... n_atom c -> ... n_sample n_atom c",
            n_sample=N_sample,
        )

        stochasticity = stochasticity_from_config(
            stoch_type=self.sampling_configs.stoch_type,
            alpha=self.sampling_configs.alpha,
        )

        pred = self.sidechain_packer.sample(
            context=context,
            timestep_scheduler=self.inference_timestep_scheduler,
            n_steps=N_step,
            residual_centre_atoms=residual_centre_atoms,
            stochasticity=stochasticity,
            gamma=self.sampling_configs.gamma,
            n_sample=N_sample,
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
        dtype = context.coordinate.coordinate.dtype
        batch_shape = context.coordinate.coordinate.shape[:-2]

        n_sample = self.diffusion_batch_size
        n_token = context.token.token_index.shape[-1]

        generator = (
            torch.Generator(device).manual_seed(seed)
            if seed is not None and not self.training
            else None
        )

        # Augment target coordinates
        x_target_augment = centre_random_augmentation(
            x_input_coords=context.coordinate.coordinate,
            n_sample=n_sample,
            mask=context.mask.is_resolved,
        ).to(dtype)

        # Get centre atoms (expand masks for n_sample dimension)
        centre_atom_mask = expand_at_dim(
            context.mask.centre_atom_mask, dim=-2, n=n_sample
        )
        atom_to_token_idx = expand_at_dim(
            context.index.atom_to_token_idx, dim=-2, n=n_sample
        )

        centre_atoms = average_atom_to_token(
            x_atom=x_target_augment,
            atom_mask=centre_atom_mask,
            atom_to_token_idx=atom_to_token_idx,
            n_token=n_token,
        )

        residual_centre_atoms = broadcast_token_to_atom(
            x_token=centre_atoms, atom_to_token_idx=atom_to_token_idx
        )

        # Generate timesteps
        timestep = self.timestep_sampler.sample(
            size=(*batch_shape, n_sample), device=device, generator=generator
        ).to(dtype)

        # Generate noise centered at residue centres
        noise = (
            torch.randn(
                x_target_augment.shape,
                dtype=dtype,
                device=device,
                generator=generator,
            )
            * self.local_sigma_coord
        )
        noise = residual_centre_atoms + noise

        # Apply flow matching interpolation
        r_noisy = noise_w_flow_matching(
            x_target_augment, noise, timestep[..., :, None, None]
        )

        pred = self.sidechain_packer(
            r_noisy=r_noisy,
            timestep=timestep,
            context=context,
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
        Forward pass of the Artios sidechain packing model.

        Args:
            context (FullContext): Full context containing all features and labels.
            mode (Mode): Mode of operation.
            seed (Optional[int]): Seed for random number generation. Defaults to None.

        Returns:
            ModelOutput: Model output containing the prediction and timestep.
        """
        match mode:
            case Mode.TRAIN | Mode.EVAL:
                output = self.main_train_loop(
                    context=context,
                    seed=seed,
                )
            case Mode.INFERENCE:
                output = self.main_inference_loop(
                    context=context,
                )
            case _:
                assert_never(mode)

        return output
