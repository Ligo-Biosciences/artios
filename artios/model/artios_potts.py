from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from typing_extensions import assert_never

from artios.data.features import FullContext
from artios.model.generator import (
    LinearMaskingTimestepSampler,
    PottsProposal,
    PottsSampler,
)
from artios.model.loss import composite_nll_loss, nll_loss
from artios.model.modules.complexity import complexity_lcp
from artios.model.modules.mpnn import MPNN
from artios.model.modules.potts import (
    GraphPotts,
    GraphPottsParameterisation,
    compute_potts_joint_logits,
    compute_potts_single_logits,
)
from artios.model.output import PottsOutput
from artios.model.utils import Mode
from artios.utils.casting import disable_autocast


class PottsLossType(str, Enum):
    PSEUDO = "pseudo"
    COMPOSITE = "composite"


@dataclass(frozen=True)
class PottsLossConfig:
    loss_type: PottsLossType = PottsLossType.COMPOSITE
    label_smoothing: float = 0.0


class ArtiosPotts(nn.Module):
    """Potts model for sequence design using MPNN encoder/decoder features."""

    def __init__(self, configs):
        super(ArtiosPotts, self).__init__()

        self.masking_sampler = LinearMaskingTimestepSampler()
        self.mpnn = MPNN(**configs.artios_mpnn.model)

        node_dim = configs.artios_mpnn.model.node_dim
        edge_dim = configs.artios_mpnn.model.edge_dim
        num_residue_types = configs.artios_mpnn.model.num_residue_types
        dec_edge_dim = 2 * node_dim + edge_dim

        self.potts = GraphPotts(
            node_dim,
            dec_edge_dim,
            num_residue_types,
            parameterisation=GraphPottsParameterisation(
                configs.artios_mpnn.potts.parameterisation
            ),
            symmetric_J=configs.artios_mpnn.potts.symmetric_J,
        )

        potts_loss_cfg = configs.loss.potts
        self.loss_config = PottsLossConfig(
            loss_type=potts_loss_cfg.loss_type,
            label_smoothing=potts_loss_cfg.label_smoothing,
        )

        sampling_cfg = configs.artios_mpnn.potts.sampling
        self.potts_sampler = PottsSampler(
            proposal=PottsProposal(sampling_cfg.proposal),
            n_sweeps=sampling_cfg.n_sweeps,
            temperature=sampling_cfg.temperature,
            temperature_init=sampling_cfg.temperature_init,
            annealing_fraction=sampling_cfg.annealing_fraction,
            dt=sampling_cfg.dt,
            rejection_step=sampling_cfg.rejection_step,
        )
        self.complexity_penalty_weight = sampling_cfg.complexity_penalty_weight
        self.complexity_window_size = sampling_cfg.complexity_window_size
        self.complexity_entropy_min = sampling_cfg.complexity_entropy_min

    def main_inference_loop(
        self,
        context: FullContext,
        seed: Optional[int] = None,
    ) -> PottsOutput:
        batch_size = context.token.restype.shape[0]
        device = context.token.restype.device

        generator = (
            torch.Generator(device).manual_seed(seed) if seed is not None else None
        )

        mask_rate = torch.ones(batch_size, device=device)

        mpnn_output = self.mpnn(context=context, mask_rate=mask_rate)

        h, J = self.potts(
            mpnn_output.node_feats,
            mpnn_output.edge_feats,
            mpnn_output.edge_index,
            mpnn_output.edge_mask,
            context.mask.token_mask,
        )

        penalty_func = None
        if self.complexity_penalty_weight > 0:
            chain_id = context.token.asym_id
            token_mask = context.mask.token_mask
            weight = self.complexity_penalty_weight
            window_size = self.complexity_window_size
            entropy_min = self.complexity_entropy_min

            def penalty_func(S: torch.Tensor) -> torch.Tensor:
                return weight * complexity_lcp(
                    S,
                    chain_id,
                    token_mask,
                    window_size=window_size,
                    entropy_min=entropy_min,
                )

        result = self.potts_sampler.sample(
            h=h,
            J=J,
            edge_index=mpnn_output.edge_index,
            edge_mask=mpnn_output.edge_mask,
            token_mask=context.mask.token_mask,
            generator=generator,
            penalty_func=penalty_func,
        )

        return PottsOutput(
            h=h,
            J=J,
            edge_index=mpnn_output.edge_index,
            edge_mask=mpnn_output.edge_mask,
            is_masked=mpnn_output.is_masked,
            sampled_sequence=result.sequence,
            energy=result.energy,
        )

    def main_train_loop(
        self,
        context: FullContext,
        seed: Optional[int] = None,
    ) -> PottsOutput:
        device = context.token.restype.device
        batch_size = context.token.restype.shape[0]

        generator = (
            torch.Generator(device).manual_seed(seed)
            if not self.training and seed is not None
            else None
        )

        timestep = self.masking_sampler.sample(
            size=(batch_size,), device=device, generator=generator
        )
        mask_rate = self.masking_sampler.get_mask_rates(timestep)

        mpnn_output = self.mpnn(context=context, mask_rate=mask_rate)

        h, J = self.potts(
            mpnn_output.node_feats,
            mpnn_output.edge_feats,
            mpnn_output.edge_index,
            mpnn_output.edge_mask,
            context.mask.token_mask,
        )

        return PottsOutput(
            h=h,
            J=J,
            edge_index=mpnn_output.edge_index,
            edge_mask=mpnn_output.edge_mask,
            is_masked=mpnn_output.is_masked,
            timestep=timestep,
        )

    def forward(
        self,
        context: FullContext,
        mode: Mode,
        *,
        seed: Optional[int] = None,
    ) -> PottsOutput:
        match mode:
            case Mode.TRAIN | Mode.EVAL:
                output = self.main_train_loop(context=context, seed=seed)
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
        output: PottsOutput,
        context: FullContext,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute loss from Potts parameters (pseudo or composite likelihood).

        Args:
            output: PottsOutput containing h, J, masking info, and timestep.
            context: Full context containing ground truth sequence and masks.

        Returns:
            Tuple of (loss tensor, metrics dict for logging).
        """
        gt_seq = context.token.restype.argmax(dim=-1)
        labels = context.token.restype.float()
        is_resolved = context.mask.centre_atom_is_resolved
        mask = output.is_masked & is_resolved

        loss_weight = self.masking_sampler.get_loss_weights(output.timestep)
        eps = 1e-6

        if self.loss_config.loss_type == PottsLossType.PSEUDO:
            logits = compute_potts_single_logits(
                gt_seq,
                output.h,
                output.J,
                output.edge_index,
                output.edge_mask,
            )  # [B, N, R]

            nll = nll_loss(
                logits=logits,
                labels=labels,
                mask=mask,
                label_smoothing=self.loss_config.label_smoothing,
            )  # [B]
            per_position_nll = -torch.sum(
                labels * F.log_softmax(logits, dim=-1), dim=-1
            )  # [B, N]

        elif self.loss_config.loss_type == PottsLossType.COMPOSITE:
            joint_logits, edge_mask = compute_potts_joint_logits(
                gt_seq, output.h, output.J, output.edge_index, output.edge_mask
            )
            nll, per_position_nll = composite_nll_loss(
                joint_logits,
                gt_seq,
                output.edge_index,
                edge_mask,
                mask,
            )
            # get logits for accuracy
            logits = compute_potts_single_logits(
                gt_seq,
                output.h,
                output.J,
                output.edge_index,
                output.edge_mask,
            )  # [B, N, R]

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_config.loss_type}")

        weighted_nll = loss_weight * nll  # [B]
        loss = weighted_nll.mean()

        # Metrics
        metrics: dict[str, Tensor] = {}
        metrics["weighted_nll_loss"] = loss.detach()

        entity_masks = {
            "protein": context.entity.is_protein_token,
            "dna": context.entity.is_dna_token,
            "rna": context.entity.is_rna_token,
            "ligand": context.entity.is_ligand_token,
        }

        # Accuracy computation
        pred_tokens = logits.argmax(dim=-1)  # [B, N]
        correct = (pred_tokens == gt_seq).float()  # [B, N]

        for entity_name, entity_mask in entity_masks.items():
            combined_mask = mask & entity_mask  # [B, N]

            masked_nll_entity = per_position_nll * combined_mask
            sum_nll = masked_nll_entity.sum(dim=-1)  # [B]
            count = combined_mask.sum(dim=-1).clamp(min=eps)  # [B]
            per_batch_nll = sum_nll / count  # [B]

            masked_correct = correct * combined_mask
            sum_correct = masked_correct.sum(dim=-1)  # [B]
            per_batch_acc = sum_correct / count  # [B]

            no_tokens = combined_mask.sum(dim=-1) == 0
            per_batch_nll = torch.where(
                no_tokens, torch.full_like(per_batch_nll, float("nan")), per_batch_nll
            )
            per_batch_acc = torch.where(
                no_tokens, torch.full_like(per_batch_acc, float("nan")), per_batch_acc
            )

            metrics[f"nll_{entity_name}_loss"] = per_batch_nll
            metrics[f"accuracy_{entity_name}"] = per_batch_acc

        # Overall accuracy
        masked_correct_all = correct * mask
        sum_correct_all = masked_correct_all.sum(dim=-1)  # [B]
        count_all = mask.sum(dim=-1).clamp(min=eps)  # [B]
        metrics["accuracy"] = sum_correct_all / count_all  # [B]

        return loss, metrics
