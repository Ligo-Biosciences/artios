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
from protenix.model.modules.sidechain_packer import SidechainPacker
from protenix.model.output import DiffusionOutput, ModelOutput
from protenix.model.utils import Mode, average_atom_to_token, broadcast_token_to_atom
from protenix.utils.casting import disable_autocast
from protenix.utils.geometry import centre_random_augmentation
from protenix.utils.tensor_utils import expand_at_dim


class ArtiosPacking(nn.Module):
    def __init__(self, configs):
        super(ArtiosPacking, self).__init__()
        self.configs = configs

        mse_cfg = configs.loss.packing.mse
        self.loss_weight_config = LossWeightConfig(
            protein=mse_cfg.weight_protein,
            dna=mse_cfg.weight_dna,
            rna=mse_cfg.weight_rna,
            ligand=mse_cfg.weight_ligand,
        )

        # Fixed eval generator for reproducibility
        # self.eval_generator = torch.Generator(device=self.device).manual_seed(42)

        self.timestep_sampler = ProteinaCoordSampler(
            **configs.artios_packing.train_timestep_sampler
        )

        self.inference_timestep_scheduler = LaProteinaCoordScheduler(
            **configs.artios_packing.inference_timestep_scheduler.la_proteina.coord
        )
        self.sampling_configs = configs.artios_packing.sample_diffusion

        self.diffusion_batch_size = self.configs.diffusion_batch_size

        self.sigma_coord = configs.sigma_coord
        self.local_sigma_coord = configs.artios_packing.model.local_sigma_coord

        self.sidechain_packer = SidechainPacker(**configs.artios_packing.model)

    def main_inference_loop(
        self,
        context: FullContext,
    ) -> ModelOutput:
        """
        Main inference loop for ArtiosPacking.

        Returns:
            ModelOutput: Model output containing the prediction.
        """
        n_sample = self.sampling_configs.N_sample
        n_step = self.sampling_configs.N_step

        residual_centre_atoms = broadcast_token_to_atom(
            x_token=context.coordinate.centre_coordinate,
            atom_to_token_idx=context.index.atom_to_token_idx,
        )
        residual_centre_atoms = repeat(
            residual_centre_atoms,
            "... n_atom c -> ... n_sample n_atom c",
            n_sample=n_sample,
        )

        stochasticity = stochasticity_from_config(
            stoch_type=self.sampling_configs.stoch_type,
            alpha=self.sampling_configs.alpha,
        )

        pred = self.sidechain_packer.sample(
            context=context,
            timestep_scheduler=self.inference_timestep_scheduler,
            n_steps=n_step,
            residual_centre_atoms=residual_centre_atoms,
            stochasticity=stochasticity,
            gamma=self.sampling_configs.gamma,
            n_sample=n_sample,
        )

        return ModelOutput(coordinate=pred)

    def main_train_loop(
        self,
        context: FullContext,
        seed: Optional[int] = None,
    ) -> DiffusionOutput:
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

        # Flow matching interpolation
        t = timestep[..., :, None, None]
        r_noisy = t * x_target_augment + (1 - t) * noise

        pred = self.sidechain_packer(
            r_noisy=r_noisy,
            timestep=timestep,
            context=context,
        )

        # # Correct symmetric atoms
        # result = atom_permutation.run_batched(
        #     pred_coord=pred,
        #     true_coord=context.coordinate.coordinate,
        #     true_coord_mask=context.mask.atom_loss_mask,
        #     ref_space_uid=context.reference.ref_space_uid,
        #     atom_perm_list=context.atom_perm.perms,
        #     pdb_id=context.basic.pdb_id,
        #     permute_label=False,
        # )

        # symmetry_corrected = result.coordinate

        return DiffusionOutput(coordinate=pred, timestep=timestep)

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
        )
        atom_weights = repeat(atom_weights, "b n -> b s n", s=n_sample)

        # Expand ground truth and masks to match samples
        true = repeat(context.coordinate.coordinate, "b n c -> b s n c", s=n_sample)
        mask = repeat(context.mask.atom_loss_mask, "b n -> b s n", s=n_sample)

        # Align ground truth to prediction
        true_aligned = align_to_prediction(pred, true, mask, weight=atom_weights)

        # Compute per-sample MSE loss
        per_sample_mse = mse_loss(pred, true_aligned, mask, weight=atom_weights)

        # Scale by timestep (using local sigma for packing)
        scale = timestep_scale(output.timestep, sigma=self.local_sigma_coord)
        scaled_mse = scale * per_sample_mse

        # Reduce to scalar
        loss = scaled_mse.mean()

        # Compute entity metrics for logging
        per_atom_se = mse_per_atom(pred, true_aligned)  # [batch, n_sample, n_atom]
        per_atom_se = per_atom_se * scale[..., None]  # scale by timestep
        per_atom_se_avg = per_atom_se.mean(dim=-2)

        entity_masks = {
            "protein": context.entity.is_protein,
            "dna": context.entity.is_dna,
            "rna": context.entity.is_rna,
            "ligand": context.entity.is_ligand,
        }
        metrics = compute_entity_metrics(
            per_atom_se_avg, context.mask.atom_loss_mask, entity_masks
        )
        # Return tensor directly to avoid CUDA sync - will be converted in metrics aggregator
        metrics["mse_loss"] = scaled_mse.mean(dim=-1)  # [batch]

        return loss, metrics
