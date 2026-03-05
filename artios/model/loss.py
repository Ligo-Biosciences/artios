"""Core loss functions for model training.

This module provides typechecked pure functions for computing losses.
Models should call these functions and compose their own `compute_loss` methods.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from artios.metrics.rmsd import weighted_rigid_align
from artios.model.utils import masked_mean
from artios.utils.typing import Bool, Float, typecheck


@dataclass(frozen=True)
class LossWeightConfig:
    """Configuration for per-atom loss weighting.

    All weights are additive adjustments to a base weight of 1.0.
    E.g., ligand=0.5 means ligand atoms get weight 1.5.
    """

    protein: float = 0.0
    dna: float = 0.0
    rna: float = 0.0
    ligand: float = 0.0
    motif: float = 0.0
    motif_related: float = 0.0


# =============================================================================
# Core Loss Functions
# =============================================================================


@typecheck
def mse_loss(
    pred: Float[Tensor, "*batch n_atom 3"],
    true: Float[Tensor, "*batch n_atom 3"],
    mask: Bool[Tensor, "*batch n_atom"],
    weight: Float[Tensor, "*batch n_atom"] | None = None,
    eps: float = 1e-6,
) -> Float[Tensor, "*batch"]:
    """Compute masked MSE loss over atoms.

    Masked coordinates are zeroed before computing squared error.

    Args:
        pred: Predicted coordinates.
        true: Ground truth coordinates.
        mask: Valid atom mask.
        weight: Optional per-atom weights applied before reduction.
        eps: Epsilon for numerical stability in masked mean.

    Returns:
        Per-batch MSE loss (weighted mean over atoms).
    """
    # Zero out masked coordinates to avoid large values from invalid atoms
    pred_masked = pred * mask[..., None]
    true_masked = true * mask[..., None]
    per_atom_se = ((pred_masked - true_masked) ** 2).sum(dim=-1)  # [..., n_atom]

    if weight is not None:
        per_atom_se = per_atom_se * weight

    return masked_mean(per_atom_se, mask, axes=-1, eps=eps)


@typecheck
def mse_per_atom(
    pred: Float[Tensor, "*batch n_atom 3"],
    true: Float[Tensor, "*batch n_atom 3"],
) -> Float[Tensor, "*batch n_atom"]:
    """Compute per-atom squared error (sum over xyz).

    Useful for computing entity-specific metrics.

    Args:
        pred: Predicted coordinates.
        true: Ground truth coordinates.

    Returns:
        Per-atom squared error.
    """
    return ((pred - true) ** 2).sum(dim=-1)


@typecheck
def nll_loss(
    logits: Float[Tensor, "*batch n n_class"],
    labels: Float[Tensor, "*batch n n_class"],
    mask: Bool[Tensor, "*batch n"],
    label_smoothing: float = 0.0,
    eps: float = 1e-6,
) -> Float[Tensor, "*batch"]:
    """Compute masked negative log-likelihood loss with optional label smoothing.

    Args:
        logits: Classification logits.
        labels: One-hot or soft labels (probabilities).
        mask: Valid position mask.
        label_smoothing: Label smoothing factor.
        eps: Epsilon for numerical stability.

    Returns:
        Scalar NLL loss.
    """
    n_class = labels.shape[-1]

    if label_smoothing > 0:
        labels_smoothed = labels + (label_smoothing / n_class)
        labels_smoothed = labels_smoothed / (
            labels_smoothed.sum(dim=-1, keepdim=True) + eps
        )
    else:
        labels_smoothed = labels

    per_position_loss = -torch.sum(
        labels_smoothed * F.log_softmax(logits, dim=-1), dim=-1
    )
    return masked_mean(per_position_loss, mask, axes=-1, eps=eps)


# =============================================================================
# Loss Utilities
# =============================================================================


@typecheck
def timestep_scale(
    timestep: Float[Tensor, "*batch"],
    sigma: float,
    clamp_max: float = 0.98,
) -> Float[Tensor, "*batch"]:
    """Compute per-sample scale factor from timestep.

    Implements the weighting: 1 / (sqrt(2) * (sigma * (1 - t))^2)

    Args:
        timestep: Timestep values in [0, 1].
        sigma: Sigma parameter for scaling.
        clamp_max: Maximum value to clamp timestep to avoid division by zero.

    Returns:
        Per-sample scale factors.
    """
    sqrt2 = 1.414
    t_clamped = torch.clamp(timestep, max=clamp_max)
    return 1.0 / (sqrt2 * (sigma * (1 - t_clamped)) ** 2)


@typecheck
def align_to_prediction(
    pred: Float[Tensor, "*batch n_atom 3"],
    true: Float[Tensor, "*batch n_atom 3"],
    mask: Bool[Tensor, "*batch n_atom"],
    weight: Float[Tensor, "*batch n_atom"] | None = None,
) -> Float[Tensor, "*batch n_atom 3"]:
    """Align ground truth to predicted coordinates using weighted rigid alignment.

    Masked coordinates are zeroed before alignment.

    Args:
        pred: Predicted coordinates.
        true: Ground truth coordinates.
        mask: Valid atom mask.
        weight: Optional per-atom weights for alignment. If None, uses mask as weights.

    Returns:
        Aligned ground truth coordinates (masked positions zeroed).
    """
    if weight is None:
        weight = mask.float()

    # Zero out masked coordinates to avoid large values from invalid atoms
    pred_masked = pred * mask[..., None]
    true_masked = true * mask[..., None]

    dtype = pred.dtype
    with torch.amp.autocast("cuda", enabled=False):
        true_aligned = weighted_rigid_align(
            x=true_masked.to(torch.float32),
            x_target=pred_masked.to(torch.float32),
            atom_weight=weight.to(torch.float32),
            mask=mask.bool(),
            stop_gradient=True,
        )
    return true_aligned.to(dtype)


@typecheck
def build_atom_weights(
    config: LossWeightConfig,
    is_protein: Bool[Tensor, "*batch n_atom"],
    is_dna: Bool[Tensor, "*batch n_atom"],
    is_rna: Bool[Tensor, "*batch n_atom"],
    is_ligand: Bool[Tensor, "*batch n_atom"],
    is_motif: Bool[Tensor, "*batch n_atom"] | None = None,
    is_motif_related: Bool[Tensor, "*batch n_atom"] | None = None,
) -> Float[Tensor, "*batch n_atom"]:
    """Build per-atom weights for loss computation.

    Uses additive semantics: starts with base weight of 1.0, then adds
    adjustments based on entity type and motif membership.

    Args:
        config: Weight configuration specifying adjustments per category.
        is_protein: Boolean mask for protein atoms.
        is_dna: Boolean mask for DNA atoms.
        is_rna: Boolean mask for RNA atoms.
        is_ligand: Boolean mask for ligand atoms.
        is_motif: Boolean mask for motif atoms.
        is_motif_related: Boolean mask for motif-related atoms.

    Returns:
        Per-atom weights (base 1.0 + adjustments).
    """
    weights = torch.ones(
        is_protein.shape, dtype=torch.float32, device=is_protein.device
    )

    weights = weights + is_protein.float() * config.protein
    weights = weights + is_dna.float() * config.dna
    weights = weights + is_rna.float() * config.rna
    weights = weights + is_ligand.float() * config.ligand

    if is_motif is not None:
        weights = weights + is_motif.float() * config.motif
    if is_motif_related is not None:
        weights = weights + is_motif_related.float() * config.motif_related

    return weights


@typecheck
def compute_entity_metrics(
    per_atom_loss: Float[Tensor, "*batch n_atom"],
    mask: Bool[Tensor, "*batch n_atom"],
    entity_masks: dict[str, Bool[Tensor, "*batch n_atom"]],
    eps: float = 1e-6,
) -> dict[str, Tensor]:
    """Compute per-entity-type loss metrics for logging.

    Args:
        per_atom_loss: Per-atom loss values (e.g., squared error).
        mask: Valid atom mask.
        entity_masks: Dictionary mapping entity names to their masks.

    Returns:
        Dictionary mapping metric names to tensors of per-batch-item values.
        Values are NaN if no atoms of that type exist in that batch item.

    Note:
        Returns tensors instead of Python floats to avoid CUDA synchronization.
        The tensors will be converted to floats when metrics are logged.
    """
    # batch_size = per_atom_loss.shape[0]
    # device = per_atom_loss.device
    metrics: dict[str, Tensor] = {}

    for entity_name, entity_mask in entity_masks.items():
        # Compute combined mask for all batch items at once
        combined_mask = mask & entity_mask  # [batch, n_atom]

        # Compute masked mean per batch item without .item() calls
        # Use masked operations to avoid sync
        masked_loss = per_atom_loss * combined_mask
        sum_loss = masked_loss.sum(dim=-1)  # [batch]
        count = combined_mask.sum(dim=-1).clamp(min=eps)  # [batch]
        per_batch_metrics = sum_loss / count  # [batch]

        # Set NaN where no atoms exist (count was clamped from 0)
        no_atoms = combined_mask.sum(dim=-1) == 0
        per_batch_metrics = torch.where(
            no_atoms,
            torch.full_like(per_batch_metrics, float("nan")),
            per_batch_metrics,
        )

        metrics[f"mse_{entity_name}_loss"] = per_batch_metrics

    return metrics
