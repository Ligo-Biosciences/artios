import logging
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from einops import repeat
from torch import Tensor
from torch.nn import functional as F
from typing_extensions import assert_never

from artios.common.residue_stats import AA_FREQUENCIES, Organism
from artios.data.features import FullContext
from artios.model.generator import (
    ExponentialMaskingTimestepSampler,
    LinearMaskingTimestepSampler,
    PottsProposal,
    PottsSampler,
)
from artios.model.loss import composite_nll_loss, nll_loss
from artios.model.modules.mpnn import MPNN
from artios.model.modules.penalty import (
    PenaltyCollection,
    aromatic_enrichment_penalty,
    charged_enrichment_penalty,
    complexity_lcp,
    composition_chi_squared_penalty,
)
from artios.model.modules.potts import (
    GraphPotts,
    GraphPottsParameterisation,
    build_alphabet_mask,
    build_energy_bias_vector,
    build_mutable_mask,
    build_s_init,
    compute_potts_joint_logits,
    compute_potts_single_logits,
)
from artios.model.output import PottsOutput
from artios.model.utils import Mode, build_backbone_ligand_proximity_mask, masked_mean
from artios.utils.casting import disable_autocast

ALPHABET_BIAS = 1e4
logger = logging.getLogger(__name__)


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
    active_site_upweight: float = 1.0
    active_site_distance: float = 8.0


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
            num_factors=configs.artios_mpnn.potts.num_factors,
        )

        potts_loss_cfg = configs.loss.potts
        self.loss_config = PottsLossConfig(
            loss_type=potts_loss_cfg.loss_type,
            label_smoothing=potts_loss_cfg.label_smoothing,
            active_site_upweight=potts_loss_cfg.active_site_upweight,
            active_site_distance=potts_loss_cfg.active_site_distance,
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
        self.cfg_scale = sampling_cfg.cfg_scale
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
            track_u_i=sampling_cfg.track_u_i,
            u_i_interval=sampling_cfg.u_i_interval,
            cfg_scale=sampling_cfg.cfg_scale,
            active_site_only_cfg=sampling_cfg.active_site_only_cfg,
            granular_active_site_coloring=sampling_cfg.granular_active_site_coloring,
        )
        self.complexity_penalty_weight = sampling_cfg.complexity_penalty_weight
        self.complexity_window_size = sampling_cfg.complexity_window_size
        self.complexity_entropy_min = sampling_cfg.complexity_entropy_min
        self.exclude_cys = sampling_cfg.exclude_cys
        self.exclude_met = sampling_cfg.exclude_met
        self.energy_bias = dict(sampling_cfg.energy_bias)
        self.ensemble_size = sampling_cfg.ensemble_size
        self.ensemble_noise_scale = sampling_cfg.ensemble_noise_scale
        self.aromatic_penalty_weight = sampling_cfg.aromatic_penalty_weight
        self.aromatic_target_fraction = sampling_cfg.aromatic_target_fraction
        self.active_site_distance_cutoff = sampling_cfg.active_site_distance_cutoff
        self.charged_penalty_weight = sampling_cfg.charged_penalty_weight
        self.charged_target_fraction = sampling_cfg.charged_target_fraction
        self.composition_penalty_weight = sampling_cfg.composition_penalty_weight
        self.composition_organism = Organism(sampling_cfg.composition_organism)
        self.composition_tolerance_k = sampling_cfg.composition_tolerance_k

    def _build_masked_inputs(self, context: FullContext) -> tuple[Tensor, Tensor]:
        """Build masked restypes and is_masked from the context's fixed residues."""
        if context.fixed_residues is not None:
            is_masked = ~context.fixed_residues.fixed_mask & context.mask.token_mask
            masked_restypes = self.mpnn.pad_and_mask_restypes(
                context.fixed_residues.fixed_restype, is_masked
            )
        else:
            is_masked = context.mask.token_mask
            masked_restypes = self.mpnn.pad_and_mask_restypes(
                context.token.restype, is_masked
            )
        return masked_restypes, is_masked

    def _run_ensemble(
        self,
        context: FullContext,
        masked_restypes: Tensor,
        is_masked: Tensor,
        generator: Optional[torch.Generator],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run MPNN+Potts across ensemble members

        Returns:
            Averaged Potts fields (h, J) and the shared MPNN graph topoly (edge_index, edge_mask)
        """
        h_sum = J_sum = None
        edge_index = edge_mask = None
        for i in range(self.ensemble_size):
            if self.ensemble_noise_scale > 0 and self.ensemble_size > 1:
                bb = context.coordinate.backbone_coordinate
                noised = bb + self.ensemble_noise_scale * torch.randn(
                    bb.shape,
                    device=bb.device,
                    dtype=bb.dtype,
                    generator=generator,
                )
                ctx = replace(
                    context,
                    coordinate=replace(context.coordinate, backbone_coordinate=noised),
                )
            else:
                ctx = context

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

        return (
            h_sum / self.ensemble_size,
            J_sum / self.ensemble_size,
            edge_index,
            edge_mask,
        )

    def main_inference_loop(
        self,
        context: FullContext,
        seed: Optional[int] = None,
    ) -> PottsOutput:
        device = context.token.restype.device
        _, _, R = context.token.restype.shape

        generator = (
            torch.Generator(device).manual_seed(seed) if seed is not None else None
        )

        ### Conditional ensemble
        masked_restypes, is_masked = self._build_masked_inputs(context)
        h, J, edge_index, edge_mask = self._run_ensemble(
            context, masked_restypes, is_masked, generator
        )

        ### Unconditional ensemble
        h_uncond = J_uncond = edge_index_uncond = edge_mask_uncond = None
        if self.cfg_scale != 1.0:
            logger.info(
                f"Running unconditional ensemble for CFG (scale={self.cfg_scale})"
            )
            uncond_context = context.without_ligand()
            uncond_masked, uncond_is_masked = self._build_masked_inputs(uncond_context)
            h_uncond, J_uncond, edge_index_uncond, edge_mask_uncond = (
                self._run_ensemble(
                    uncond_context, uncond_masked, uncond_is_masked, generator
                )
            )

        ### Initialisation logic
        active_site_mask = build_backbone_ligand_proximity_mask(
            backbone_coordinate=context.coordinate.backbone_coordinate,
            centre_coordinate=context.coordinate.centre_coordinate,
            is_ligand_token=context.entity.is_ligand_token,
            token_mask=context.mask.token_mask,
            backbone_mask=context.mask.backbone_mask,
            distance_cutoff=self.active_site_distance_cutoff,
        )
        fixed_mask = (
            context.fixed_residues.fixed_mask
            if context.fixed_residues is not None
            else None
        )
        fixed_restype = (
            context.fixed_residues.fixed_restype
            if context.fixed_residues is not None
            else None
        )

        mutable_mask = build_mutable_mask(context.mask.token_mask, fixed_mask)
        alphabet_mask = build_alphabet_mask(
            mutable_mask, R, self.exclude_cys, self.exclude_met
        )
        energy_bias = build_energy_bias_vector(self.energy_bias, R, device)
        S_init = build_s_init(
            h,
            self.potts_sampler.temperature_init,
            fixed_mask,
            fixed_restype,
            generator,
        )

        ### Apply masks and biases
        h = h + ~alphabet_mask * ALPHABET_BIAS
        if h_uncond is not None:
            h_uncond = h_uncond + ~alphabet_mask * ALPHABET_BIAS

        if energy_bias is not None:
            h = h + energy_bias
            if h_uncond is not None:
                h_uncond = h_uncond + energy_bias

        ### Penalty logic
        penalty_dict: dict[str, callable] = {}

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

            penalty_dict["lcp"] = lcp_penalty

        if self.aromatic_penalty_weight > 0:

            def aromatic_penalty(S: torch.Tensor) -> torch.Tensor:
                return self.aromatic_penalty_weight * aromatic_enrichment_penalty(
                    S, active_site_mask, target_fraction=self.aromatic_target_fraction
                )

            penalty_dict["aromatic"] = aromatic_penalty

        if self.charged_penalty_weight > 0:

            def charged_penalty(S: torch.Tensor) -> torch.Tensor:
                return self.charged_penalty_weight * charged_enrichment_penalty(
                    S, active_site_mask, target_fraction=self.charged_target_fraction
                )

            penalty_dict["charged"] = charged_penalty

        if self.composition_penalty_weight > 0:
            comp_mask = mutable_mask & context.entity.is_protein_token
            org_freqs = AA_FREQUENCIES[self.composition_organism]
            k = self.composition_tolerance_k

            def composition_penalty(S: torch.Tensor) -> torch.Tensor:
                return (
                    self.composition_penalty_weight
                    * composition_chi_squared_penalty(S, comp_mask, org_freqs, k=k)
                )

            penalty_dict["composition"] = composition_penalty

        penalty = PenaltyCollection(penalty_dict) if penalty_dict else None

        ### Sampling logic
        result = self.potts_sampler.sample(
            h=h,
            J=J,
            edge_index=edge_index,
            edge_mask=edge_mask,
            token_mask=mutable_mask,
            S_init=S_init,
            generator=generator,
            penalty=penalty,
            h_uncond=h_uncond,
            J_uncond=J_uncond,
            edge_index_uncond=edge_index_uncond,
            edge_mask_uncond=edge_mask_uncond,
            active_site_mask=active_site_mask,
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
            u_i_trajectory=result.u_i_trajectory,
            u_i_sweep_indices=result.u_i_sweep_indices,
            penalty_grad_trajectories=result.penalty_grad_trajectories,
            penalty_names=result.penalty_names,
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

        _, N = gt_seq.shape

        # Get loss weights
        sampler_weight = self.masking_sampler.get_loss_weights(output.timestep)  # [B]
        eps = 1e-6

        if self.loss_config.active_site_upweight > 1.0:
            active_site = build_backbone_ligand_proximity_mask(
                backbone_coordinate=context.coordinate.backbone_coordinate,
                centre_coordinate=context.coordinate.centre_coordinate,
                is_ligand_token=context.entity.is_ligand_token,
                token_mask=context.mask.token_mask,
                backbone_mask=context.mask.backbone_mask,
                distance_cutoff=self.loss_config.active_site_distance,
            )  # [B, N]
            position_weight = torch.where(
                active_site, self.loss_config.active_site_upweight, 1.0
            )  # [B, N]
        else:
            position_weight = None

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

        # Get loss weight
        loss_weight = repeat(sampler_weight, "B -> B N", N=N)
        if position_weight is not None:
            loss_weight = loss_weight * position_weight

        loss = masked_mean(loss_weight * per_position_nll, mask, axes=-1).mean()

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
