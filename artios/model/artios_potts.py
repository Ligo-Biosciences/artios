import logging
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from typing_extensions import assert_never

from artios.data.features import FullContext
from artios.model.generator import (
    ExponentialMaskingTimestepSampler,
    LinearMaskingTimestepSampler,
    PottsProposal,
    PottsSampler,
)
from artios.model.loss import composite_nll_loss, nll_loss
from artios.model.modules.mpnn import MPNN
from artios.model.modules.penalty import aromatic_enrichment_penalty, complexity_lcp
from artios.model.modules.potts import (
    GraphPotts,
    GraphPottsParameterisation,
    build_alphabet_mask,
    build_mutable_mask,
    build_s_init,
    compute_potts_joint_logits,
    compute_potts_single_logits,
)
from artios.model.output import PottsOutput
from artios.model.utils import Mode
from artios.utils.casting import disable_autocast

logger = logging.getLogger(__name__)

ALPHABET_BIAS = 1e4


class MaskingSamplerType(str, Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


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
            zero_sum_gauge=configs.artios_mpnn.potts.zero_sum_gauge,
        )

        potts_loss_cfg = configs.loss.potts
        self.loss_config = PottsLossConfig(
            loss_type=potts_loss_cfg.loss_type,
            label_smoothing=potts_loss_cfg.label_smoothing,
        )

        masking_sampler_type = MaskingSamplerType(
            configs.artios_mpnn.potts.masking_sampler
        )
        match masking_sampler_type:
            case MaskingSamplerType.LINEAR:
                self.masking_sampler = LinearMaskingTimestepSampler()
            case MaskingSamplerType.EXPONENTIAL:
                self.masking_sampler = ExponentialMaskingTimestepSampler(b=0.01)
            case _:
                assert_never(masking_sampler_type)

        sampling_cfg = configs.artios_mpnn.potts.sampling
        self.potts_sampler = PottsSampler(
            proposal=PottsProposal(sampling_cfg.proposal),
            n_sweeps=sampling_cfg.n_sweeps,
            temperature=sampling_cfg.temperature,
            temperature_init=sampling_cfg.temperature_init,
            annealing_fraction=sampling_cfg.annealing_fraction,
            dt=sampling_cfg.dt,
            rejection_step=sampling_cfg.rejection_step,
            track_trajectory=sampling_cfg.track_trajectory,
            trajectory_interval=sampling_cfg.trajectory_interval,
        )
        self.complexity_penalty_weight = sampling_cfg.complexity_penalty_weight
        self.complexity_window_size = sampling_cfg.complexity_window_size
        self.complexity_entropy_min = sampling_cfg.complexity_entropy_min
        self.exclude_cys = sampling_cfg.exclude_cys
        self.exclude_met = sampling_cfg.exclude_met
        self.ensemble_size = sampling_cfg.ensemble_size
        self.ensemble_noise_scale = sampling_cfg.ensemble_noise_scale
        self.aromatic_penalty_weight = sampling_cfg.aromatic_penalty_weight
        self.aromatic_target_fraction = sampling_cfg.aromatic_target_fraction
        self.aromatic_distance_cutoff = sampling_cfg.aromatic_distance_cutoff

    def _noise_backbone(
        self, context: FullContext, generator: torch.Generator | None
    ) -> FullContext:
        """Return a copy of context with Gaussian noise added to backbone coordinates."""
        noised_backbone = context.coordinate.backbone_coordinate + (
            torch.randn(
                context.coordinate.backbone_coordinate.shape,
                device=context.coordinate.backbone_coordinate.device,
                dtype=context.coordinate.backbone_coordinate.dtype,
                generator=generator,
            )
            * self.ensemble_noise_scale
        )
        noised_coord = replace(context.coordinate, backbone_coordinate=noised_backbone)
        return replace(context, coordinate=noised_coord)

    # TODO: this guy needs a lot of tidying before merging to main.
    def main_inference_loop(
        self,
        context: FullContext,
        seed: Optional[int] = None,
    ) -> PottsOutput:
        device = context.token.restype.device
        restypes = context.token.restype
        B, N, R = restypes.shape

        generator = (
            torch.Generator(device).manual_seed(seed) if seed is not None else None
        )

        # # TODO: this should be changed to a get_masked_restypes function we write in potts.py
        if context.stitched is not None:
            is_masked = ~context.stitched.stitched_mask & context.mask.token_mask
            masked_restypes = self.mpnn.pad_and_mask_restypes(
                context.stitched.stitched_restype, is_masked
            )
        else:
            is_masked = context.mask.token_mask
            masked_restypes = self.mpnn.pad_and_mask_restypes(restypes, is_masked)

        ### Ensemble logic
        h_sum = None
        J_sum = None
        for i in range(self.ensemble_size):
            ctx = (
                self._noise_backbone(context, generator)
                if self.ensemble_size > 1
                else context
            )
            mpnn_out = self.mpnn(
                context=ctx, masked_restypes=masked_restypes, is_masked=is_masked
            )

            h_i, J_i = self.potts(
                mpnn_out.node_feats,
                mpnn_out.edge_feats,
                mpnn_out.edge_index,
                mpnn_out.edge_mask,
                ctx.mask.token_mask,
            )

            if i == 0:
                h_sum, J_sum = h_i, J_i
                edge_index = mpnn_out.edge_index
                edge_mask = mpnn_out.edge_mask
            else:
                h_sum = h_sum + h_i
                J_sum = J_sum + J_i

        h = h_sum / self.ensemble_size
        J = J_sum / self.ensemble_size

        # TODO: I don't like how this looks
        ### Initialiation logic
        stitched_mask = (
            context.stitched.stitched_mask if context.stitched is not None else None
        )
        stitched_restype = (
            context.stitched.stitched_restype[..., :R].float()
            if context.stitched is not None
            else None
        )

        mutable_mask = build_mutable_mask(context.mask.token_mask, stitched_mask)
        alphabet_mask = build_alphabet_mask(
            mutable_mask, R, self.exclude_cys, self.exclude_met
        )

        h = h + ~alphabet_mask * ALPHABET_BIAS

        S_init = build_s_init(
            h,
            self.potts_sampler.temperature_init,
            stitched_mask,
            stitched_restype,
            generator,
        )

        ### Penalty logic
        active_site_mask = None
        if self.aromatic_penalty_weight > 0:
            lig_mask = context.entity.is_ligand_token  # [B, N]
            centre_coords = context.coordinate.centre_coordinate  # [B, N, 3]

            # Ligand center of mass per batch
            lig_coords = centre_coords * lig_mask.unsqueeze(-1).float()
            lig_com = lig_coords.sum(dim=1) / lig_mask.float().sum(
                dim=1, keepdim=True
            ).clamp(min=1)  # [B, 3]

            # Distance from each token to ligand CoM
            prot_mask = context.entity.is_protein_token  # [B, N]
            dist = (centre_coords - lig_com.unsqueeze(1)).norm(dim=-1)  # [B, N]

            active_site_mask = (
                (dist < self.aromatic_distance_cutoff)
                & prot_mask
                & context.mask.token_mask
            )

            n_lig = lig_mask.sum().item()
            n_active = active_site_mask.sum(dim=-1).tolist()
            logger.info(
                f"Aromatic penalty: {n_lig} ligand tokens, "
                f"active site sizes per batch: {n_active}, "
                f"ligand CoM: {lig_com[0].tolist()}"
            )

        penalties = []

        if self.complexity_penalty_weight > 0:
            chain_id = context.token.asym_id
            token_mask = context.mask.token_mask
            lcp_weight = self.complexity_penalty_weight
            lcp_window_size = self.complexity_window_size
            lcp_entropy_min = self.complexity_entropy_min

            def lcp_penalty(S: torch.Tensor) -> torch.Tensor:
                return lcp_weight * complexity_lcp(
                    S,
                    chain_id,
                    token_mask,
                    window_size=lcp_window_size,
                    entropy_min=lcp_entropy_min,
                )

            penalties.append(lcp_penalty)

        if self.aromatic_penalty_weight > 0 and active_site_mask is not None:
            arom_weight = self.aromatic_penalty_weight
            arom_target = self.aromatic_target_fraction
            arom_mask = active_site_mask

            def aromatic_penalty(S: torch.Tensor) -> torch.Tensor:
                return arom_weight * aromatic_enrichment_penalty(
                    S, arom_mask, target_fraction=arom_target
                )

            penalties.append(aromatic_penalty)

        penalty_func = None
        if penalties:

            def penalty_func(S: torch.Tensor) -> torch.Tensor:
                return sum(p(S) for p in penalties)

        ### Sampling logic
        result = self.potts_sampler.sample(
            h=h,
            J=J,
            edge_index=edge_index,
            edge_mask=edge_mask,
            token_mask=mutable_mask,
            S_init=S_init,
            generator=generator,
            penalty_func=penalty_func,
        )

        return PottsOutput(
            h=h,
            J=J,
            edge_index=edge_index,
            edge_mask=edge_mask,
            is_masked=is_masked,
            sampled_sequence=result.sequence,
            energy=result.energy,
            sequence_trajectory=result.sequence_trajectory,
            energy_trajectory=result.energy_trajectory,
            trajectory_sweep_indices=result.trajectory_sweep_indices,
        )

    def main_train_loop(
        self,
        context: FullContext,
        seed: Optional[int] = None,
    ) -> PottsOutput:
        device = context.token.restype.device
        B, N, _ = context.token.restype.shape

        generator = (
            torch.Generator(device).manual_seed(seed)
            if not self.training and seed is not None
            else None
        )

        timestep = self.masking_sampler.sample(
            size=(B,), device=device, generator=generator
        )
        mask_rate = self.masking_sampler.get_mask_rates(timestep)

        rand = torch.rand(B, N, device=device)
        is_masked = (rand < mask_rate[:, None]) & context.mask.token_mask
        masked_restypes = self.mpnn.pad_and_mask_restypes(
            context.token.restype, is_masked
        )

        mpnn_output = self.mpnn(
            context=context,
            masked_restypes=masked_restypes,
            is_masked=is_masked,
        )

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

        match self.loss_config.loss_type:
            case PottsLossType.PSEUDO:
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

            case PottsLossType.COMPOSITE:
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

            case _:
                assert_never(self.loss_config.loss_type)

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
