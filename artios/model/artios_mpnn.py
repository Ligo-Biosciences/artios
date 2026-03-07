from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from typing_extensions import assert_never

from artios.data.features import FullContext
from artios.model.generator import CosineScheduler, LinearMaskingTimestepSampler
from artios.model.loss import nll_loss
from artios.model.modules.mpnn import MPNN
from artios.model.output import ModelOutput, SequenceOutput
from artios.model.utils import Mode
from artios.utils.casting import disable_autocast


@dataclass(frozen=True)
class MPNNLossConfig:
    """Configuration for MPNN sequence loss."""

    label_smoothing: float = 0.1
    weight_protein: float = 1.0
    weight_dna: float = 1.0
    weight_rna: float = 1.0
    weight_ligand: float = 1.0


class ArtiosMPNN(nn.Module):
    """All-atom message passing neural network for sequence design."""

    def __init__(self, configs):
        super(ArtiosMPNN, self).__init__()

        self.train_zero_shot = configs.artios_mpnn.train_zero_shot
        self.sample_zero_shot = configs.artios_mpnn.sampling.zero_shot
        self.temp = configs.artios_mpnn.sampling.temperature
        self.smoothing_factor = configs.artios_mpnn.sampling.smoothing_factor
        self.n_steps = configs.artios_mpnn.sampling.n_steps

        mpnn_loss_cfg = configs.loss.mpnn
        self.loss_config = MPNNLossConfig(
            label_smoothing=mpnn_loss_cfg.label_smoothing,
            weight_protein=mpnn_loss_cfg.weight_protein,
            weight_dna=mpnn_loss_cfg.weight_dna,
            weight_rna=mpnn_loss_cfg.weight_rna,
            weight_ligand=mpnn_loss_cfg.weight_ligand,
        )

        self.masking_sampler = LinearMaskingTimestepSampler()
        self.inference_scheduler = CosineScheduler(
            configs.artios_mpnn.sampling.cosine_scale
        )

        self.mpnn = MPNN(**configs.artios_mpnn.model)

    def main_inference_loop(
        self,
        context: FullContext,
        seed: Optional[int] = None,
    ) -> SequenceOutput:
        seq_logits = self.mpnn.sample(
            context=context,
            timestep_scheduler=self.inference_scheduler,
            masking_timestep_sampler=self.masking_sampler,
            n_steps=self.n_steps,
            temperature=self.temp,
            seed=seed,
        )

        return SequenceOutput(seq_logits=seq_logits)

    def main_train_loop(
        self,
        context: FullContext,
        seed: Optional[int] = None,
    ) -> SequenceOutput:
        """
        Main training loop for the all-atom MPNN model.

        Args:
            context (FullContext): Full context containing all features and labels.
            seed (Optional[int]): Seed for reproducibility (used in eval mode).

        Returns:
            SequenceOutput: Model output containing logits, masking info, and timestep.
        """
        device = context.token.restype.device
        batch_size = context.token.restype.shape[0]

        generator = (
            torch.Generator(device).manual_seed(seed)
            if not self.training and seed is not None
            else None
        )

        # Sample timesteps and convert to mask rates
        timestep = self.masking_sampler.sample(
            size=(batch_size,), device=device, generator=generator
        )
        mask_rate = self.masking_sampler.get_mask_rates(timestep)

        logits, is_masked, _, _ = self.mpnn(
            context=context,
            mask_rate=mask_rate,
        )

        return SequenceOutput(seq_logits=logits, is_masked=is_masked, timestep=timestep)

    def forward(
        self,
        context: FullContext,
        mode: Mode,
        *,
        seed: Optional[int] = None,
    ) -> ModelOutput:
        """
        Forward pass for all-atom MPNN module.

        Args:
            context (FullContext): Full context containing all features and labels.
            mode (Mode): Mode of operation.
            seed (Optional[int]): Seed for random number generation.

        Returns:
            ModelOutput: Model output containing the prediction.
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
                    seed=seed,
                )
            case _:
                assert_never(mode)

        return output

    @disable_autocast
    def compute_loss(
        self,
        output: SequenceOutput,
        context: FullContext,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute sequence loss from model output and context.

        Args:
            output: Model output containing seq_logits.
            context: Full context containing ground truth sequence and masks.

        Returns:
            Tuple of (loss tensor, metrics dict for logging).
        """
        assert output.seq_logits is not None, "seq_logits required for loss computation"
        assert output.timestep is not None, "timestep required for loss computation"

        logits = output.seq_logits  # [batch, n_token, num_residue_types]
        labels = context.token.restype.float()  # [batch, n_token, num_residue_types]
        is_resolved = context.mask.centre_atom_is_resolved  # [batch, n_token]
        diffusion_mask = output.is_masked  # [batch, n_token]
        mask = is_resolved & diffusion_mask

        # Get ELBO loss weight from masking sampler
        loss_weight = self.masking_sampler.get_loss_weights(output.timestep)  # [batch]

        # Compute NLL loss
        nll = nll_loss(
            logits=logits,
            labels=labels,
            mask=mask,
            label_smoothing=self.loss_config.label_smoothing,
        )  # [batch]

        # Weight by masking schedule
        weighted_nll = loss_weight * nll  # [batch]
        loss = weighted_nll.mean()

        # Compute metrics for logging
        metrics: dict[str, Tensor] = {}
        metrics["weighted_nll_loss"] = loss.detach()

        # Per-entity NLL metrics
        entity_masks = {
            "protein": context.entity.is_protein_token,
            "dna": context.entity.is_dna_token,
            "rna": context.entity.is_rna_token,
            "ligand": context.entity.is_ligand_token,
        }

        # Compute per-position NLL for entity metrics
        per_position_nll = -torch.sum(
            labels * F.log_softmax(logits, dim=-1), dim=-1
        )  # [batch, n_token]

        # Accuracy computation
        pred_tokens = logits.argmax(dim=-1)  # [batch, n_token]
        true_tokens = labels.argmax(dim=-1)  # [batch, n_token]
        correct = (pred_tokens == true_tokens).float()  # [batch, n_token]

        eps = 1e-6
        for entity_name, entity_mask in entity_masks.items():
            combined_mask = mask & entity_mask  # [batch, n_token]

            # Compute per-batch NLL (following compute_entity_metrics pattern)
            masked_nll = per_position_nll * combined_mask
            sum_nll = masked_nll.sum(dim=-1)  # [batch]
            count = combined_mask.sum(dim=-1).clamp(min=eps)  # [batch]
            per_batch_nll = sum_nll / count  # [batch]

            # Compute per-batch accuracy
            masked_correct = correct * combined_mask
            sum_correct = masked_correct.sum(dim=-1)  # [batch]
            per_batch_acc = sum_correct / count  # [batch]

            # Set NaN where no tokens of this type exist
            no_tokens = combined_mask.sum(dim=-1) == 0
            per_batch_nll = torch.where(
                no_tokens,
                torch.full_like(per_batch_nll, float("nan")),
                per_batch_nll,
            )
            per_batch_acc = torch.where(
                no_tokens,
                torch.full_like(per_batch_acc, float("nan")),
                per_batch_acc,
            )

            metrics[f"nll_{entity_name}_loss"] = per_batch_nll
            metrics[f"accuracy_{entity_name}"] = per_batch_acc

        # Overall accuracy (per-batch)
        masked_correct_all = correct * mask
        sum_correct_all = masked_correct_all.sum(dim=-1)  # [batch]
        count_all = mask.sum(dim=-1).clamp(min=eps)  # [batch]
        metrics["accuracy"] = sum_correct_all / count_all  # [batch]

        return loss, metrics
