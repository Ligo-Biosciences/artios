from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from einops import repeat
from torch import Tensor
from torch.nn import functional as F
from typing_extensions import assert_never

from artios.data.features import FullContext
from artios.model.generator import LinearMaskingTimestepSampler
from artios.model.loss import composite_nll_loss, nll_loss
from artios.model.modules.message_passing import gather_nodes
from artios.model.modules.mpnn import MPNN
from artios.model.modules.potts import GraphPotts, PottsOutput
from artios.model.output import ModelOutput
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
            symmetric_J=configs.artios_mpnn.potts.symmetric_J,
        )

        potts_loss_cfg = configs.loss.potts
        self.loss_config = PottsLossConfig(
            loss_type=potts_loss_cfg.loss_type,
            label_smoothing=potts_loss_cfg.label_smoothing,
        )

    def main_inference_loop(
        self,
        context: FullContext,
        seed: Optional[int] = None,
    ) -> PottsOutput:
        raise NotImplementedError("Inference loop not implemented for Potts model")

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
    ) -> ModelOutput | PottsOutput:
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

    def get_logits(
        self,
        seq: Tensor,
        output: PottsOutput,
    ) -> Tensor:
        """Compute Potts pseudo-loglikelihood logits.
        The logits are the negative energy of the states.

        For each node i and candidate state a, computes:
            logit_i(a) = -(h_i(a) + sum_j J_ij(a, s_j))
        where s_j is the ground truth state of neighbor j.

        Args:
            seq: Ground truth sequence indices [B, N].
            output: PottsOutput containing h, J, and edge_index.

        Returns:
            Logits for each position and state [B, N, R].
        """
        R = output.h.shape[-1]

        # Mask out self-edges (i->i) so node i's own state doesn't leak into its conditional
        ii = torch.arange(output.edge_index.shape[1], device=output.edge_index.device)
        not_self = output.edge_index != ii[None, :, None]  # [B, N, K]

        # Get neighbour GT states [B, N, K, 1]
        S_neighbours = gather_nodes(seq.unsqueeze(-1).float(), output.edge_index).long()

        # Index into J with GT neighbour states J[..., s_j]
        S_neighbours = repeat(S_neighbours, "... n k 1 -> ... n k r 1", r=R)
        J_at_S = torch.gather(output.J, -1, S_neighbours).squeeze(-1)  # [B, N, K, R]

        # Zero out self-edge contributions before summing
        J_at_S = J_at_S * not_self.unsqueeze(-1)

        # For each node, sum coupling contributions over neighbours
        J_node = J_at_S.sum(dim=-2)  # [B, N, R]

        return -(output.h + J_node)

    def get_joint_logits(
        self,
        seq: Tensor,
        output: PottsOutput,
    ) -> tuple[Tensor, Tensor]:
        """Compute joint Potts logits for composite likelihood.
        The logits are the negative energy of the joint state pairs.

        For each edge (i, j), computes the joint logit over all state pairs
        (a, b), accounting for background contributions from all other nodes.

        Args:
            seq: Ground truth sequence indices [batch, n_token].
            output: PottsOutput containing h, J, edge_index, edge_mask.

        Returns:
            Tuple of (joint_logits, edge_mask):
                joint_logits: [batch, n_token, neighbours, R, R]
                edge_mask: [batch, n_token, neighbours] (self-edges excluded)
        """
        h = output.h
        J = output.J  # [batch, n_token, neighbours, R, R]
        edge_index = output.edge_index

        _, N, K = edge_index.shape
        R = h.shape[-1]

        # Remove self edges (prevents leakage)
        ii = torch.arange(N, device=edge_index.device)
        not_self = edge_index != ii[None, :, None]
        edge_mask = not_self & output.edge_mask

        ### Get interactions between node states and GT neighbour states
        S_j = gather_nodes(seq.unsqueeze(-1).float(), edge_index).long()
        # Index into J with GT neighbour states J[..., s_j] -> [B, N, K, R]
        S_j = repeat(S_j, "... n k 1 -> ... n k r 1", r=R)
        J_j = torch.gather(J, -1, S_j).squeeze(-1)  # [B, N, K, R]

        ### Get interactions between GT node state and neighbour states
        # Index into J with GT node states J[..., s_i, :] -> [B, N, K, 1, R]
        S_i = repeat(seq.long(), "b n -> b n k 1 r", k=K, r=R)
        J_i = torch.gather(J, -2, S_i)  # [B, N, K, 1, R]

        # Get node energies from GT neighbour states
        J_j_self_edge_masked = J_j * not_self.unsqueeze(-1)  # zero out self-edges
        E_i = h + J_j_self_edge_masked.sum(dim=2)  # [B, N, R]

        # Get neighbour energies by gathering node energies
        E_j = gather_nodes(E_i, edge_index)  # [B, N, K, R]

        # Remove J_ij contribution from E_i (prevents leakage)
        E_i_minus_ij = repeat(E_i, "b n r -> b n k r 1", k=K) - J_j.unsqueeze(
            -1
        )  # [B, N, K, R, 1]

        # Remove J_ji contribution from E_j (prevents leakage)
        E_j_minus_ji = repeat(E_j, "b n k r -> b n k 1 r") - J_i  # [B, N, K, 1, R]

        # Joint logits (negative energy)
        joint_logits = -(E_i_minus_ij + E_j_minus_ji + J)  # [B, N, K, R, R]

        return joint_logits, edge_mask

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

        if self.loss_config.loss_type == "pseudo":
            logits = self.get_logits(gt_seq, output)  # [B, N, R]
            nll = nll_loss(
                logits=logits,
                labels=labels,
                mask=mask,
                label_smoothing=self.loss_config.label_smoothing,
            )  # [B]
            per_position_nll = -torch.sum(
                labels * F.log_softmax(logits, dim=-1), dim=-1
            )  # [B, N]

        elif self.loss_config.loss_type == "composite":
            joint_logits, edge_mask = self.get_joint_logits(gt_seq, output)
            nll, per_position_nll = composite_nll_loss(
                joint_logits,
                gt_seq,
                output.edge_index,
                edge_mask,
                mask,
            )
            # Get logits for accuracy (h, J already computed, this is cheap)
            logits = self.get_logits(gt_seq, output)  # [B, N, R]

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
