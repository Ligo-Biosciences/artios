import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F

from artios.common import residue_names
from artios.common.constants import AROMATIC_INDICES, CHARGED_INDICES
from artios.common.hbond_propensities import (
    as_matrix as _hbond_pair_matrix,
)
from artios.common.hbond_propensities import (
    reach_array as _sidechain_reach_array,
)
from artios.common.residue_stats import OrganismAAFrequencies
from artios.common.sse_burial_stats import (
    SSE_BURIAL_BUCKETS,
)
from artios.common.sse_burial_stats import (
    as_mu_matrix as _sse_burial_mu_matrix,
)
from artios.common.sse_burial_stats import (
    as_sigma_matrix as _sse_burial_sigma_matrix,
)
from artios.utils.typing import Bool, Float, Int


@dataclass
class PenaltyCollection:
    """Named collection of penalty functions for Potts sampling.

    Callable (sums all penalties), but also exposes individual penalties
    by name for per-penalty gradient tracking.
    """

    penalties: dict[str, Callable[[Tensor], Tensor]]

    def __call__(self, S: Tensor) -> Tensor:
        """Sum all penalties."""
        return sum(fn(S) for fn in self.penalties.values())

    @property
    def names(self) -> list[str]:
        return list(self.penalties.keys())


def complexity_lcp(
    S: Float[Tensor, "batch n_token num_residue_types"],
    chain_id: Int[Tensor, "batch n_token"],
    token_mask: Bool[Tensor, "batch n_token"],
    window_size: int = 30,
    entropy_min: float = 2.32,
    num_natural_aas: int = 20,
    differentiable: bool = True,
    min_coverage: float = 0.9,
) -> Float[Tensor, "batch"]:
    """Compute local composition perplexity penalty.

    For each position, computes the Shannon entropy of amino acid composition
    in a local window (same chain only). Positions where entropy falls below
    `entropy_min` are penalized quadratically.

    When ``differentiable=True``, a straight-through estimator computes
    exact discrete counterfactual gradients: for each (position, amino acid)
    pair, what would the penalty be if that mutation were made. This gives
    more accurate gradient signal for MCMC proposals than naive autograd.

    Args:
        S: One-hot (or soft) sequence tensor [B, N, R]. Only the first
            `num_natural_aas` channels are used for composition.
        chain_id: Per-token chain identifier [B, N]. Positions with the
            same chain_id are considered part of the same chain.
        token_mask: Valid token mask [B, N].
        window_size: Size of the local sliding window.
        entropy_min: Minimum acceptable entropy. Positions with entropy
            below exp(entropy_min) perplexity are penalized.
        num_natural_aas: Number of natural amino acid types (first R channels).
        differentiable: If True, use straight-through estimator for
            discrete counterfactual gradients (following Chroma).
        min_coverage: Minimum fraction of window that must be valid
            for a position to be penalized.

    Returns:
        Penalty energy per sample [B]. Higher = more penalty.
    """
    B, N, R = S.shape
    device = S.device
    Q = num_natural_aas

    # Adjust window size for short sequences
    if N < window_size:
        window_size = N
    half_w = window_size // 2

    mask_i_float = token_mask.float()
    S_aa = S[..., :Q]  # [B, N, Q]
    S_aa_masked = (
        mask_i_float[..., None] * S_aa
    )  # zero out invalid positions for counting

    # Build sliding-window neighbor indices: [1, N, W]
    offsets = torch.arange(window_size, device=device) - half_w
    edge_idx = torch.arange(N, device=device)[None, :, None] + offsets[None, None, :]

    # Boundary mask computed BEFORE clamping to avoid double-counting at edges
    valid_boundary = (edge_idx >= 0) & (edge_idx < N)  # [1, N, W]
    edge_idx = edge_idx.clamp(min=0, max=N - 1)
    edge_idx = edge_idx.expand(B, -1, -1)  # [B, N, W]
    W = edge_idx.shape[-1]
    edge_flat = edge_idx.reshape(B, N * W)  # [B, N*W]

    # Gather chain_id and token_mask at neighbor positions
    chain_j = torch.gather(chain_id, 1, edge_flat).reshape(B, N, W)
    chain_i = chain_id[:, :, None]  # [B, N, 1]
    mask_j = torch.gather(token_mask.long(), 1, edge_flat).reshape(B, N, W).bool()

    # Combined validity mask: in-bounds, same chain, both valid
    mask_ij = (
        valid_boundary & (chain_i == chain_j) & mask_j & token_mask[:, :, None]
    ).float()  # [B, N, W]

    # Gather masked S at neighbor positions -> [B, N, W, Q]
    pos_for_gather = edge_flat.unsqueeze(-1).expand(-1, -1, Q)  # [B, N*W, Q]
    S_neighbors = torch.gather(S_aa_masked, 1, pos_for_gather).reshape(B, N, W, Q)

    # Local composition counts: [B, N, Q]
    counts = (mask_ij.unsqueeze(-1) * S_neighbors).sum(dim=2)

    # Min coverage: only penalize windows with sufficient valid positions
    total_counts = counts.sum(-1)  # [B, N]
    mask_coverage = total_counts > int(min_coverage * window_size)  # [B, N]

    # Valid position mask (has counts and is a valid token)
    mask_i_valid = ((total_counts > 0) & token_mask).float()  # [B, N]

    # Shannon entropy (naive estimator)
    P = counts / (total_counts.unsqueeze(-1) + 1e-8)
    H = -(P * torch.log(P + 1e-8)).sum(-1)  # [B, N]

    # Penalty per position
    U = (
        mask_coverage.float()
        * (torch.exp(H) - np.exp(entropy_min)).clamp(max=0).square()
    )

    # Straight-through estimator for discrete counterfactual gradients
    if differentiable:
        # Gather counts at neighbor positions: [B, N, W, Q]
        # counts_at_j[b, i, w] = counts for the window centered at neighbor w of position i
        counts_at_neighbors = torch.gather(
            counts, 1, edge_flat.unsqueeze(-1).expand(-1, -1, Q)
        ).reshape(B, N, W, Q)

        # Gather coverage mask at neighbor positions: [B, N, W, 1]
        mask_cov_j = (
            torch.gather(mask_coverage.long(), 1, edge_flat)
            .reshape(B, N, W)
            .float()
            .unsqueeze(-1)
        )

        # Counterfactual counts: if position i mutates to amino acid a,
        # how do the counts change for each neighboring window?
        # N_ij[b, i, w, a, q] = counts_at_neighbors[b, i, w, q] - S_aa[b, i, q] + delta(a, q)
        N_ij = (counts_at_neighbors - S_aa[:, :, None, :]).unsqueeze(-2) + torch.eye(
            Q, device=device
        )[None, None, None, ...]  # [B, N, W, Q, Q]
        N_ij = N_ij.clamp(min=0)

        # Entropy of counterfactual counts: [B, N, W, Q]
        N_ij_total = N_ij.sum(-1, keepdim=True)
        P_ij = N_ij / (N_ij_total + 1e-8)
        H_ij = -(P_ij * torch.log(P_ij + 1e-8)).sum(-1)

        # Counterfactual penalty: [B, N, W, Q]
        U_ij = (torch.exp(H_ij) - np.exp(entropy_min)).clamp(max=0).square()
        U_ij = mask_ij.unsqueeze(-1) * mask_cov_j * U_ij

        # Straight-through: dot with S to select actual aa, sum over windows
        U_differentiable = (U_ij.detach() * S_aa[:, :, None, :]).sum([-1, -2])  # [B, N]
        U = U.detach() + U_differentiable - U_differentiable.detach()

    # Mask and sum
    U = (mask_i_valid * U).sum(dim=1)  # [B]
    return U


def complexity_lcp_natural_prior(
    S: Float[Tensor, "batch n_token num_residue_types"],
    log_pi: Float[Tensor, "num_natural_aas"],
    chain_id: Int[Tensor, "batch n_token"],
    token_mask: Bool[Tensor, "batch n_token"],
    window_size: int = 30,
    nat_perp_min: float = 14.0,
    num_natural_aas: int = 20,
    min_coverage: float = 0.9,
) -> Float[Tensor, "batch"]:
    """Natural-prior LCP — Shannon-perplexity floor with a natural reference.

    Identical sliding-window structure to :func:`complexity_lcp`, but the
    reference distribution is the natural AA marginal ``π`` rather than the
    uniform implicit reference. For each window centred at position ``i``:

        CE[i] = -Σ_k p_k[i] · log(π_k)              (cross-entropy)
        Pp[i] = exp(CE[i])                          ∈ [exp(min surprisal), exp(max surprisal)]

    The penalty fires when ``Pp[i]`` falls below ``nat_perp_min`` —
    equivalent to the window using common (low-surprisal) AAs more than the
    natural background would predict. Because CE is *linear* in ``S``, the
    penalty is quadratic in ``S`` and autograd-exact; no straight-through
    estimator is needed.

    Compared to :func:`complexity_lcp`:
      * Reference distribution: π_natural instead of uniform.
      * Differentiability: autograd exact, no STE block.
      * Compute cost: O(N·W·Q) (one factor of Q cheaper than LCP's STE).
      * Target failure mode: over-use of common AAs, not low Shannon entropy.

    Args:
        S: ``[B, N, R]`` one-hot or soft sequence. Only the first
            ``num_natural_aas`` channels matter.
        log_pi: ``[Q]`` log of natural AA frequencies in PROTEIN_TYPES
            order. Precompute once from a reference set.
        chain_id: ``[B, N]``; cross-chain windows are excluded.
        token_mask: ``[B, N]`` valid-token mask.
        window_size: Sliding-window width (default 30).
        nat_perp_min: Natural-prior perplexity floor — windows whose
            ``exp(CE)`` falls below this are penalised. Calibrate against
            the natural distribution to match a desired offence rate.
        num_natural_aas: Number of natural AA types (default 20).
        min_coverage: Minimum window valid-fraction to count.

    Returns:
        Per-sample penalty ``[B]``. Higher = more low-surprisal-AA-heavy
        windows.
    """
    B, N, _ = S.shape
    device = S.device
    dtype = S.dtype
    Q = num_natural_aas

    if N < window_size:
        window_size = N
    half_w = window_size // 2

    # Per-residue surprisal: -log(π_{aa_j}). Linear in S.
    S_aa = S[..., :Q]
    surprisal = -(S_aa * log_pi).sum(dim=-1)  # [B, N]

    mask_i_float = token_mask.to(dtype)
    surprisal_masked = surprisal * mask_i_float  # invalid positions contribute 0

    # Same windowing machinery as complexity_lcp.
    offsets = torch.arange(window_size, device=device) - half_w
    edge_idx = torch.arange(N, device=device)[None, :, None] + offsets[None, None, :]
    valid_boundary = (edge_idx >= 0) & (edge_idx < N)
    edge_idx = edge_idx.clamp(min=0, max=N - 1).expand(B, -1, -1)
    W = edge_idx.shape[-1]
    edge_flat = edge_idx.reshape(B, N * W)

    chain_j = torch.gather(chain_id, 1, edge_flat).reshape(B, N, W)
    chain_i = chain_id[:, :, None]
    mask_j = torch.gather(token_mask.long(), 1, edge_flat).reshape(B, N, W).bool()
    mask_ij = (
        valid_boundary & (chain_i == chain_j) & mask_j & token_mask[:, :, None]
    ).to(dtype)

    surprisal_neighbors = torch.gather(surprisal_masked, 1, edge_flat).reshape(B, N, W)
    window_sum = (mask_ij * surprisal_neighbors).sum(dim=-1)  # [B, N]
    window_count = mask_ij.sum(dim=-1)  # [B, N]

    mask_coverage = window_count > int(min_coverage * window_size)
    mask_i_valid = ((window_count > 0) & token_mask & mask_coverage).to(dtype)

    ce_per_window = window_sum / (window_count + 1e-8)  # [B, N]
    perp_natural = torch.exp(ce_per_window)  # [B, N]

    U = (nat_perp_min - perp_natural).clamp(min=0).square()
    U = mask_i_valid * U  # [B, N]
    return U.sum(dim=1)  # [B]


def aromatic_enrichment_penalty(
    S: Float[Tensor, "batch n_token num_residue_types"],
    active_site_mask: Bool[Tensor, "batch n_token"],
    target_fraction: float = 0.25,
) -> Float[Tensor, "batch"]:
    """Penalize low aromatic residue fraction in the active site.

    Encourages aromatic residues (PHE, TRP, TYR, HIS) near the ligand
    binding site during MCMC sampling. Only penalizes when the aromatic
    fraction falls below the target — sequences that already meet or
    exceed the target receive zero penalty.

    Unlike LCP, this is a linear function of S, so autograd gives correct
    gradients even with one-hot inputs (no straight-through estimator needed).

    Args:
        S: One-hot (or soft) sequence tensor [B, N, R].
        active_site_mask: Boolean mask for active site protein residues [B, N].
        target_fraction: Target aromatic fraction in active site (default 0.25).

    Returns:
        Penalty energy per sample [B]. Higher = more penalty.
    """
    # Sum aromatic probabilities per position: [B, N]
    S_aromatic = S[:, :, AROMATIC_INDICES].sum(dim=-1)

    # Mask to active site and compute counts
    mask_float = active_site_mask.float()
    aromatic_count = (S_aromatic * mask_float).sum(dim=-1)  # [B]
    n_active = mask_float.sum(dim=-1)  # [B]

    # Fraction of aromatic residues in active site
    frac = aromatic_count / (n_active + 1e-8)  # [B]

    # Only penalize when below target (one-sided quadratic)
    penalty = (frac - target_fraction).clamp(max=0).square()  # [B]

    # Zero out for batches with no active site residues
    has_active_site = n_active > 0
    return penalty * has_active_site.float()


def charged_enrichment_penalty(
    S: Float[Tensor, "batch n_token num_residue_types"],
    active_site_mask: Bool[Tensor, "batch n_token"],
    target_fraction: float = 0.25,
) -> Float[Tensor, "batch"]:
    """Penalize low charged residue fraction in the active site.

    Encourages charged residues (ASP, GLU, ARG, LYS, HIS) near the ligand
    binding site during MCMC sampling. Only penalizes when the charged
    fraction falls below the target.

    Args:
        S: One-hot (or soft) sequence tensor [B, N, R].
        active_site_mask: Boolean mask for active site protein residues [B, N].
        target_fraction: Target charged fraction in active site (default 0.25).

    Returns:
        Penalty energy per sample [B]. Higher = more penalty.
    """
    S_charged = S[:, :, CHARGED_INDICES].sum(dim=-1)

    mask_float = active_site_mask.float()
    charged_count = (S_charged * mask_float).sum(dim=-1)
    n_active = mask_float.sum(dim=-1)

    frac = charged_count / (n_active + 1e-8)

    penalty = (frac - target_fraction).clamp(max=0).square()

    has_active_site = n_active > 0
    return penalty * has_active_site.float()


def build_active_site_mask(
    centre_coordinate: Float[Tensor, "batch n_token 3"],
    is_ligand_token: Bool[Tensor, "batch n_token"],
    is_protein_token: Bool[Tensor, "batch n_token"],
    token_mask: Bool[Tensor, "batch n_token"],
    distance_cutoff: float,
) -> Bool[Tensor, "batch n_token"]:
    """Identify protein residues near the ligand center of mass.
    Probably going to be used in penalties, like the aromatics penalty.

    Args:
        centre_coordinate: Per-token centre atom coordinates [B, N, 3].
        is_ligand_token: Ligand token mask [B, N].
        is_protein_token: Protein token mask [B, N].
        token_mask: Valid token mask [B, N].
        distance_cutoff: Distance threshold in Angstroms.

    Returns:
        Boolean mask of protein tokens within distance_cutoff of ligand CoM [B, N].
    """
    lig_coords = centre_coordinate * is_ligand_token.unsqueeze(-1).float()
    lig_com = lig_coords.sum(dim=1) / is_ligand_token.float().sum(
        dim=1, keepdim=True
    ).clamp(min=1)

    dist = (centre_coordinate - lig_com.unsqueeze(1)).norm(dim=-1)

    return (dist < distance_cutoff) & is_protein_token & token_mask


def _frequencies_to_tensor(
    freq_map: Mapping[str, float],
    device: torch.device,
    residue_order: tuple[str, ...] = residue_names.PROTEIN_TYPES,
) -> Tensor:
    """Convert residue-name-keyed frequencies to a tensor ordered by residue_order."""
    return torch.tensor(
        [freq_map[res] for res in residue_order], device=device, dtype=torch.float32
    )


@functools.lru_cache(maxsize=None)
def _aa_pair_propensity_tensor(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Empirical sc-sc H-bond propensity, 20×20, in PROTEIN_TYPES order. Cached per (device, dtype)."""
    return torch.tensor(
        _hbond_pair_matrix(order=residue_names.PROTEIN_TYPES),
        device=device,
        dtype=dtype,
    )


@functools.lru_cache(maxsize=None)
def _aa_reach_tensor(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Per-AA sidechain reach (Å), 20-element, in PROTEIN_TYPES order. Cached per (device, dtype)."""
    return torch.tensor(
        _sidechain_reach_array(order=residue_names.PROTEIN_TYPES),
        device=device,
        dtype=dtype,
    )


def hbond_density_penalty(
    S: Float[Tensor, "batch n_token num_residue_types"],
    backbone_coords: Float[Tensor, "batch n_token 5 3"],
    chain_id: Int[Tensor, "batch n_token"],
    token_mask: Bool[Tensor, "batch n_token"],
    distance_cutoff: float = 5.0,
    sharpness: float = 2.0,
    sequence_separation: int = 1,
    num_natural_aas: int = 20,
) -> Float[Tensor, "batch"]:
    """Open-ended sc-sc H-bond *encourager* — minus the expected H-bond score.

    Penalty per sample is::

        U = − Σ_{i < j, same chain, |i−j| ≥ sequence_separation}
                M[aa_i, aa_j] · sigmoid((cutoff − d_ij) / sharpness)

    ``M`` is the empirical sc–sc H-bond pair propensity (symmetric, 20×20) from
    ``artios.common.hbond_propensities``. ``d_ij`` is the distance between
    virtual sidechain functional tips at ``CA + reach[aa] · (CB − CA)/||CB − CA||``;
    the featurizer fills CB even for GLY (ProteinMPNN virtual Cβ).

    There is no target, no tolerance, no dead-zone, and no squared form:
    every additional in-range, chemically compatible neighbor lowers the
    energy by a fixed amount. The per-position gradient is

        ∂U/∂S[i, r] = − Σ_{j ≠ i, valid} M[r, aa_j] · gate_ij

    which scales with the number of in-range neighbors, not with 1/N. Large
    proteins get a proportionally larger total penalty but the same
    per-position pull, which is what the DLMC sampler consumes.

    The default ``distance_cutoff = 5 Å`` was chosen so the gate fires only
    when virtual tips are within plausible sc-sc H-bond geometry — wider
    cutoffs (8–15 Å) preserve Spearman correlation with biotite-detected
    H-bonds but spread the gradient thin across many distant pairs (see
    ``docs/potential_improvements/hbond_density_penalty_scale.md``).

    Args:
        S: Sequence tensor [B, N, R]; only the first ``num_natural_aas`` channels matter.
        backbone_coords: [B, N, 5, 3] in (N, CA, C, O, CB); only CA and CB are used.
        chain_id: [B, N]; cross-chain pairs are excluded.
        token_mask: [B, N] valid-token mask.
        distance_cutoff, sharpness: Soft sigmoid distance gate parameters, Å.
        sequence_separation: Minimum |i − j| for a pair to count.
        num_natural_aas: Number of natural amino acid types (default 20).

    Returns:
        Per-sample penalty [B]; more negative = more sc-sc H-bonds expected.
    """
    _, N, _ = S.shape
    device = S.device
    dtype = S.dtype
    Q = num_natural_aas

    reach = _aa_reach_tensor(device, dtype)  # [Q]
    chem_matrix = _aa_pair_propensity_tensor(device, dtype)  # [Q, Q]

    ca = backbone_coords[..., 1, :]
    cb = backbone_coords[..., 4, :]
    cb_dir = cb - ca
    cb_dir = cb_dir / (cb_dir.norm(dim=-1, keepdim=True) + 1e-8)

    S_aa = S[..., :Q]
    per_res_reach = (S_aa * reach).sum(dim=-1)
    centre = ca + per_res_reach.unsqueeze(-1) * cb_dir  # [B, N, 3]

    # Bilinear-in-S pair chemistry: chem_pair[b, i, j] = S_i^T M S_j.
    chem_pair = (S_aa @ chem_matrix) @ S_aa.transpose(-1, -2)  # [B, N, N]

    diff = rearrange(centre, "b n c -> b n 1 c") - rearrange(centre, "b n c -> b 1 n c")
    distance_gate = torch.sigmoid((distance_cutoff - diff.norm(dim=-1)) / sharpness)

    # Non-AA tokens (UNK, gap, nucleotides) have S_aa all zero and contribute
    # 0 to chem_pair anyway; gating pair_mask on aa_token_mask keeps the
    # pair set semantically clean (and matches LCP's treatment).
    aa_token_mask = token_mask & (S_aa.sum(dim=-1) > 0)

    intra_pair = torch.triu(
        torch.ones(N, N, dtype=torch.bool, device=device),
        diagonal=sequence_separation,
    )
    same_chain = rearrange(chain_id, "b n -> b n 1") == rearrange(
        chain_id, "b n -> b 1 n"
    )
    both_aa = rearrange(aa_token_mask, "b n -> b n 1") & rearrange(
        aa_token_mask, "b n -> b 1 n"
    )
    pair_mask = (same_chain & both_aa & intra_pair).to(dtype)

    hbond_score = (chem_pair * distance_gate * pair_mask).sum(dim=(-2, -1))
    return -hbond_score


def composition_chi_squared_penalty(
    S: Float[Tensor, "batch n_token num_residue_types"],
    protein_mutable_mask: Bool[Tensor, "batch n_token"],
    organism_freqs: OrganismAAFrequencies,
    k: float = 1.0,
    num_natural_aas: int = 20,
) -> Float[Tensor, "batch"]:
    """Dead-zone composition penalty against organism-specific AA frequencies.

    Computes empirical amino acid frequencies over mutable protein positions
    and penalizes deviation beyond k standard deviations from the reference mean.
    Compositions within the dead zone (mu +/- k*sigma) receive zero penalty.

    The penalty is normalized by sigma^2 so that each amino acid contributes
    in units of excess z-scores squared.

    Since composition p is linear in S, standard autograd gives exact
    gradients (no straight-through estimator needed).

    Args:
        S: One-hot (or soft) sequence tensor [B, N, R].
        protein_mutable_mask: Boolean mask for mutable protein positions [B, N].
        organism_freqs: Reference organism AA frequency statistics.
        k: Dead zone width in standard deviations (0 = no dead zone).
        num_natural_aas: Number of natural amino acid types (default 20).

    Returns:
        Penalty per sample [B]. Higher = more divergent composition.
    """
    Q = num_natural_aas
    mu_ref = _frequencies_to_tensor(organism_freqs.mean, S.device)
    sigma_ref = _frequencies_to_tensor(organism_freqs.std, S.device)

    S_aa = S[..., :Q]  # [B, N, Q]
    mask = protein_mutable_mask.float().unsqueeze(-1)  # [B, N, 1]

    counts = (S_aa * mask).sum(dim=1)  # [B, Q]
    n_eff = mask.sum(dim=1)  # [B, 1]
    p = counts / (n_eff + 1e-8)  # [B, Q]

    deviation = (p - mu_ref).abs() - k * sigma_ref  # [B, Q]
    penalty = (deviation.clamp(min=0) ** 2 / (sigma_ref**2 + 1e-8)).sum(dim=-1)  # [B]

    has_positions = n_eff.squeeze(-1) > 0
    return penalty * has_positions.float()


_POS_CHARGE_INDICES = tuple(
    residue_names.PROTEIN_TYPES.index(r) for r in ("LYS", "ARG")
)
_NEG_CHARGE_INDICES = tuple(
    residue_names.PROTEIN_TYPES.index(r) for r in ("ASP", "GLU")
)


def net_charge_penalty(
    S: Float[Tensor, "batch n_token num_residue_types"],
    token_mask: Bool[Tensor, "batch n_token"],
    target_per_residue: float = -0.013,
    num_natural_aas: int = 20,
) -> Float[Tensor, "batch"]:
    """Symmetric quadratic on per-residue protein net charge.

    At pH 7 with HIS partial protonation ignored::

        net_charge_per_residue = (#K + #R − #D − #E) / n_residues
        penalty                = (net_charge_per_residue − target)²

    Per-residue makes the function chain-count-invariant — a 170-residue
    monomer and a 1500-residue dimer at the same charge density score
    identically. Linear in S, so autograd gives correct gradients on soft
    (or one-hot) inputs without a straight-through estimator.

    The default ``target_per_residue ≈ -0.013`` matches the natural
    beta-pdb reference mean (≈ -1.3 per 100 residues).

    Args:
        S: One-hot (or soft) sequence tensor [B, N, R].
        token_mask: Valid token mask [B, N].
        target_per_residue: Target net charge per residue.
        num_natural_aas: Number of natural amino acid types (default 20).

    Returns:
        Per-sample penalty [B]. Higher = farther from target.
    """
    Q = num_natural_aas
    S_aa = S[..., :Q]
    mask = token_mask.to(S.dtype)

    pos_share = S_aa[..., list(_POS_CHARGE_INDICES)].sum(dim=-1)  # [B, N]
    neg_share = S_aa[..., list(_NEG_CHARGE_INDICES)].sum(dim=-1)  # [B, N]

    pos_count = (pos_share * mask).sum(dim=-1)  # [B]
    neg_count = (neg_share * mask).sum(dim=-1)  # [B]

    n_residues = mask.sum(dim=-1)  # [B]
    net_per_res = (pos_count - neg_count) / (n_residues + 1e-8)

    penalty = (net_per_res - target_per_residue).square()
    return penalty * (n_residues > 0).to(S.dtype)


# Matches artios.utils.geometry.NUM_SSE_BURIAL_BUCKETS and the row count of
# the (μ, σ) tables in artios.common.sse_burial_stats.
_NUM_SSE_BURIAL_BUCKETS = len(SSE_BURIAL_BUCKETS)


@functools.lru_cache(maxsize=None)
def _sse_burial_mu_tensor(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Natural per-bucket per-AA μ, shape ``[6, 20]`` in PROTEIN_TYPES order.

    Cached per (device, dtype) — mirrors ``_aa_pair_propensity_tensor``.
    """
    return torch.tensor(
        _sse_burial_mu_matrix(order=residue_names.PROTEIN_TYPES),
        device=device,
        dtype=dtype,
    )


@functools.lru_cache(maxsize=None)
def _sse_burial_sigma_tensor(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Natural per-bucket per-AA σ, shape ``[6, 20]`` in PROTEIN_TYPES order.

    Cached per (device, dtype).
    """
    return torch.tensor(
        _sse_burial_sigma_matrix(order=residue_names.PROTEIN_TYPES),
        device=device,
        dtype=dtype,
    )


def sse_burial_composition_penalty(
    S: Float[Tensor, "batch n_token num_residue_types"],
    bucket_idx: Int[Tensor, "batch n_token"],
    token_mask: Bool[Tensor, "batch n_token"],
    num_natural_aas: int = 20,
) -> Float[Tensor, "batch"]:
    """σ²-weighted regression-to-mean composition penalty per (SSE × burial) bucket.

    For each of the 6 buckets (α/β/coil × buried/exposed), computes the
    per-bucket AA frequency in the sample and penalises deviation from a
    natural reference (μ, σ) under a no-dead-zone χ² statistic::

        p_{b,q}  =  Σ_{i: bucket_i = b}  S_{i,q}  /  N_b
        L        =  Σ_{b, q}  (p_{b,q} − μ_{b,q})² / σ_{b,q}²

    Reference (μ, σ) are loaded from :mod:`artios.common.sse_burial_stats` —
    per-structure mean and standard deviation of AA frequency across natural
    monomers. Rare AAs (small σ) get amplified gradient force per unit
    deviation, so an aromatic deficit pushes the sampler as hard as an
    alanine surplus despite the much smaller absolute frequency gap.

    Positions with ``bucket_idx == -1`` (chain padding, missing backbone,
    DSSP-too-short, or otherwise masked) and buckets that end up empty in a
    sample contribute 0. p is linear in S so autograd is exact.

    Args:
        S: ``[B, N, R]`` one-hot or soft sequence; only the first
            ``num_natural_aas`` channels are read.
        bucket_idx: ``[B, N]`` int from
            :func:`artios.utils.geometry.compute_sse_burial_buckets` — values
            in ``{0..5}`` or ``-1`` for ignore. Backbone is fixed during
            Potts sampling, so this should be precomputed once per inference.
        token_mask: ``[B, N]`` valid-token mask.
        num_natural_aas: Number of natural AA types (default 20).

    Returns:
        Per-sample penalty ``[B]``; larger = farther from the natural
        per-bucket composition.
    """
    Q = num_natural_aas
    K = _NUM_SSE_BURIAL_BUCKETS

    mu = _sse_burial_mu_tensor(S.device, S.dtype)  # [K, Q]
    sigma = _sse_burial_sigma_tensor(S.device, S.dtype)  # [K, Q]

    S_aa = S[..., :Q]
    valid = (token_mask & (bucket_idx >= 0)).to(S.dtype)  # [B, N]
    bucket_one_hot = F.one_hot(bucket_idx.clamp(min=0), num_classes=K).to(S.dtype)
    bucket_one_hot = bucket_one_hot * valid.unsqueeze(-1)  # [B, N, K]

    counts = bucket_one_hot.transpose(-1, -2) @ S_aa  # [B, K, Q]
    n_per_bucket = bucket_one_hot.sum(dim=1)  # [B, K]

    p = counts / (n_per_bucket.unsqueeze(-1) + 1e-8)  # [B, K, Q]
    z_squared = ((p - mu) / (sigma + 1e-8)) ** 2  # [B, K, Q]
    per_bucket = z_squared.sum(dim=-1)  # [B, K]

    return (per_bucket * (n_per_bucket > 0).to(S.dtype)).sum(dim=-1)
