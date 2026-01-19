from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from typing_extensions import assert_never

from protenix.data.features import FullContext
from protenix.model.loss import nll_loss
from protenix.model.modules.mpnn import MPNN
from protenix.model.output import ModelOutput
from protenix.model.utils import Mode, masked_mean
from protenix.utils.casting import disable_autocast


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

        mpnn_loss_cfg = configs.loss.mpnn
        self.loss_config = MPNNLossConfig(
            label_smoothing=mpnn_loss_cfg.label_smoothing,
            weight_protein=mpnn_loss_cfg.weight_protein,
            weight_dna=mpnn_loss_cfg.weight_dna,
            weight_rna=mpnn_loss_cfg.weight_rna,
            weight_ligand=mpnn_loss_cfg.weight_ligand,
        )

        self.mpnn = MPNN(**configs.artios_mpnn.model)

    def main_inference_loop(
        self,
        context: FullContext,
        seed: Optional[int] = None,
    ) -> ModelOutput:
        seq_logits, _ = self.mpnn.sample(
            context=context,
            temperature=self.temp,
            zero_shot=self.sample_zero_shot,
            smoothing_factor=self.smoothing_factor,
            seed=seed,
        )

        return ModelOutput(seq_logits=seq_logits)

    def main_train_loop(
        self,
        context: FullContext,
        seed: Optional[int] = None,
    ) -> ModelOutput:
        """
        Main training loop for the all-atom MPNN model.

        Args:
            context (FullContext): Full context containing all features and labels.
            seed (Optional[int]): Seed for random number generation. Defaults to None.

        Returns:
            ModelOutput: Model output containing the prediction.
        """

        logits, _, _ = self.mpnn(
            context=context,
            zero_shot=self.train_zero_shot,
            seed=seed,
        )

        return ModelOutput(seq_logits=logits)

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
            seed (Optional[int]): Seed for random number generation. Defaults to None.

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
        output: ModelOutput,
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

        logits = output.seq_logits  # [batch, n_token, num_residue_types]
        labels = context.token.restype.float()  # [batch, n_token, num_residue_types]
        mask = context.mask.centre_atom_is_resolved  # [batch, n_token]

        # Compute NLL loss
        loss = nll_loss(
            logits=logits,
            labels=labels,
            mask=mask,
            label_smoothing=self.loss_config.label_smoothing,
        )

        # Compute metrics for logging
        metrics: dict[str, Tensor] = {}
        metrics["nll_loss"] = loss.detach()

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

        for entity_name, entity_mask in entity_masks.items():
            combined_mask = mask & entity_mask
            if combined_mask.any():
                entity_nll = masked_mean(per_position_nll, combined_mask, axes=-1)
                metrics[f"nll_{entity_name}_loss"] = entity_nll.mean()

        # Accuracy metrics
        pred_tokens = logits.argmax(dim=-1)  # [batch, n_token]
        true_tokens = labels.argmax(dim=-1)  # [batch, n_token]
        correct = (pred_tokens == true_tokens) & mask

        # Overall accuracy
        accuracy = correct.sum().float() / mask.sum().float().clamp(min=1e-6)
        metrics["accuracy"] = accuracy

        # Per-entity accuracy
        for entity_name, entity_mask in entity_masks.items():
            combined_mask = mask & entity_mask
            if combined_mask.any():
                entity_correct = correct & entity_mask
                entity_acc = (
                    entity_correct.sum().float()
                    / combined_mask.sum().float().clamp(min=1e-6)
                )
                metrics[f"accuracy_{entity_name}"] = entity_acc

        return loss, metrics
