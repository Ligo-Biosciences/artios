"""Core loss functions for model training.

This module provides typechecked pure functions for computing losses.
Models should call these functions and compose their own `compute_loss` methods.
"""

from dataclasses import dataclass

import torch
from einops import rearrange, repeat
from torch import Tensor
from torch.nn import functional as F

from artios.metrics.rmsd import weighted_rigid_align
from artios.model.modules.message_passing import gather_nodes
from artios.model.utils import masked_mean
from artios.utils.typing import Bool, Float, Int, typecheck


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


def composite_nll_loss_msa(
    h: Float[Tensor, "batch n_token num_residue_types"],
    J: Float[Tensor, "batch n_token neighbours num_residue_types num_residue_types"],
    edge_index: Int[Tensor, "batch n_token neighbours"],
    edge_mask: Bool[Tensor, "batch n_token neighbours"],
    msa_tokens: Int[Tensor, "batch n_token n_msa"],
    msa_gap_mask: Bool[Tensor, "batch n_token n_msa"],
    n_chunk: int = 8,
    eps: float = 1e-6,
) -> Float[Tensor, "batch n_token"]:
    """MSA-averaged composite NLL, returning the same per-position tensor
    shape as the non-MSA ``composite_nll_loss``.

    For each sample (b), MSA row (m), token (i), and edge to neighbour
    (k): computes ``log P(s_i, s_j | s_{-i,-j})`` under the row's own
    sequence context — exactly as the non-MSA composite NLL does, but
    now per row. Edges are averaged per (sample, row, token) with the
    ``/ 2`` undirected double-count correction, then rows are averaged
    per-position using ``masked_mean`` with ``~msa_gap_mask``.

    **The caller uses the returned ``per_position_nll`` exactly like the
    non-MSA ``composite_nll_loss`` output** — multiply by the per-token
    ``loss_weight`` and pass through ``masked_mean(..., node_mask)`` to
    reduce to a scalar. This keeps the compute_loss pipeline shared
    between the MSA and non-MSA paths.

    This relies on ``_build_msa_context`` filling every non-reference-
    chain token with its ground-truth residue id across all rows (so the
    per-row NLL is meaningful everywhere, not just on the reference
    chain) and on ``msa_gap_mask`` marking only real homolog gaps. With
    that arrangement, a null-MSA sample (every row identical to the
    query) reduces exactly to the non-MSA composite loss — no special
    casing needed.

    **Memory**: the per-chunk ``[R, R]`` joint-logits materialisation is
    the only significant allocation. At realistic training shapes
    (B=4, n_msa=64, N=600, K=30, R=32, fp32) the full-n_msa version
    would be ~19 GB; with ``n_chunk=8`` it is ~2.4 GB per chunk plus
    another ~2.4 GB for the log-softmax output, so ~5 GB peak.

    Args:
        h: Potts fields from the model [B, N, R].
        J: Potts couplings [B, N, K, R, R].
        edge_index: Neighbour indices [B, N, K].
        edge_mask: Valid edge mask [B, N, K].
        msa_tokens: MSA residue ids in STD_RESIDUES_AND_GAP [B, N, M] —
            matches ``MSAContext.tokens``. Column 0 is the query; columns
            1..M-1 are homologs. Non-reference-chain rows carry the gt
            residue id across every column.
        msa_gap_mask: True where ``msa_tokens == 31`` (real homolog gap
            on a reference-chain position) [B, N, M] — matches
            ``MSAContext.gap_mask``. False on non-reference-chain rows
            (they hold gt residues, not gaps).
        n_chunk: Chunk size along ``n_msa`` for the joint-logits step.
            Default 8. Reduce if OOM.
        eps: Numerical-stability epsilon.

    Returns:
        ``per_position_nll`` [B, N]: per-sample, per-token NLL ready to
        be consumed by the same pipeline as the non-MSA
        ``composite_nll_loss``.
    """
    from artios.model.modules.potts import compute_potts_joint_logits

    # [B, N, M] -> [B, M, N]
    msa_tokens = msa_tokens.transpose(-2, -1).contiguous().long()
    msa_gap_mask = msa_gap_mask.transpose(-2, -1).contiguous()

    M = msa_tokens.shape[1]
    K = edge_index.shape[-1]
    R = h.shape[-1]

    lp_chunks: list[Tensor] = []
    mask_chunks: list[Tensor] = []

    for m_start in range(0, M, n_chunk):
        m_end = min(m_start + n_chunk, M)
        c = m_end - m_start
        seq_chunk = msa_tokens[:, m_start:m_end]  # [B, c, N]

        h_c = repeat(h, "b n r -> b m n r", m=c)
        J_c = repeat(J, "b n k r1 r2 -> b m n k r1 r2", m=c)
        ei_c = repeat(edge_index, "b n k -> b m n k", m=c)
        em_c = repeat(edge_mask, "b n k -> b m n k", m=c)

        # BEWARE: this materialises joint_c shape [B, c, N, K, R, R]
        joint_c, mask_c = compute_potts_joint_logits(seq_chunk, h_c, J_c, ei_c, em_c)

        # Log-softmax over R x R joint states
        logp_flat = F.log_softmax(
            rearrange(joint_c, "... r1 r2 -> ... (r1 r2)"),
            dim=-1,
        )
        logp = rearrange(logp_flat, "... (r1 r2) -> ... r1 r2", r1=R)

        del joint_c

        # Gather neighbour GT states
        S_j = gather_nodes(seq_chunk.unsqueeze(-1).float(), ei_c).long()
        S_j = repeat(S_j, "... n k 1 -> ... n k r 1", r=R)

        ### Extract probs corresponding to GT states of the nodes and GT states of
        ### each of their neighbours.
        # Index using GT neighbour indices
        logp_at_sj = torch.gather(logp, -1, S_j).squeeze(-1)  # [B, c, N, K, R]
        del logp, logp_flat
        # Index using GT node indices
        S_i = repeat(seq_chunk.long(), "... n -> ... n k 1", k=K)
        lp_c = torch.gather(logp_at_sj, -1, S_i).squeeze(-1)
        del logp_at_sj

        lp_chunks.append(lp_c)
        mask_chunks.append(mask_c)

    logp_ij = torch.cat(lp_chunks, dim=1)  # [B, M, N, K]
    edge_no_self_mask = torch.cat(mask_chunks, dim=1)  # [B, M, N, K]

    # Mean over valid edges (with /2 for undirected double-counting)
    edge_count = edge_no_self_mask.float().sum(dim=-1).clamp(min=eps)  # [B, M, N]
    per_position_msa_nll = -(edge_no_self_mask.float() * logp_ij).sum(dim=-1) / (
        2 * edge_count
    )  # [B, M, N]

    # Mean over seqs in the MSA. Only non-gap tokens contribute.
    non_gap = ~msa_gap_mask  # [B, M, N]
    per_position_nll = masked_mean(
        per_position_msa_nll, non_gap, axes=1, eps=eps
    )  # [B, N]

    return per_position_nll


def composite_nll_loss(
    h: Float[Tensor, "batch n_token num_residue_types"],
    J: Float[Tensor, "batch n_token neighbours num_residue_types num_residue_types"],
    edge_index: Int[Tensor, "batch n_token neighbours"],
    edge_mask: Bool[Tensor, "batch n_token neighbours"],
    seq: Int[Tensor, "batch n_token"],
    eps: float = 1e-6,
) -> Float[Tensor, "batch n_token"]:
    """Compute per-position composite NLL from Potts fields and couplings.

    Computes joint Potts logits via ``compute_potts_joint_logits``, then
    log P(s_i, s_j | s_{-i,-j}) for each edge, and averages over valid
    edges per node (with /2 for undirected double-counting). Node-level
    masking and reduction to a scalar are the caller's responsibility.

    Args:
        h: Potts fields [B, N, R].
        J: Potts couplings [B, N, K, R, R].
        edge_index: Neighbour indices [B, N, K].
        edge_mask: Valid edge mask [B, N, K].
        seq: Ground truth sequence indices [B, N].
        eps: Epsilon for numerical stability.

    Returns:
        Per-position composite NLL [B, N].
    """
    from artios.model.modules.potts import compute_potts_joint_logits

    joint_logits, edge_no_self_mask = compute_potts_joint_logits(
        seq, h, J, edge_index, edge_mask
    )
    R = joint_logits.shape[-1]
    K = joint_logits.shape[2]

    # Log-softmax over R x R joint states
    logp_flat = F.log_softmax(
        rearrange(joint_logits, "b n k r1 r2 -> b n k (r1 r2)"),
        dim=-1,
    )
    logp = rearrange(logp_flat, "b n k (r1 r2) -> b n k r1 r2", r1=R)

    # Gather neighbour GT states
    S_j = gather_nodes(seq.unsqueeze(-1).float(), edge_index).long()
    S_j = repeat(S_j, "... n k 1 -> ... n k r 1", r=R)

    ### Extract probs corresponding to GT states of the nodes and GT states of
    ### each of their neighbours.
    # Index using GT neighbour indices
    logp_at_sj = torch.gather(logp, -1, S_j).squeeze(-1)  # [B, N, K, R]
    # Index using GT node indices
    S_i = repeat(seq.long(), "b n -> b n k 1", k=K)
    logp_ij = torch.gather(logp_at_sj, -1, S_i).squeeze(-1)  # [B, N, K]

    # Per-position NLL: average over valid edges, /2 since we double counted
    edge_count = edge_no_self_mask.float().sum(dim=-1).clamp(min=eps)  # [B, N]
    per_position_nll = -(edge_no_self_mask.float() * logp_ij).sum(dim=-1) / (
        2 * edge_count
    )  # [B, N]

    return per_position_nll


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
