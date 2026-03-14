"""Local composition perplexity penalty for sequence sampling.

Penalizes sequences with low local amino acid diversity, encouraging
compositionally complex (natural-looking) sequences during MCMC sampling.

Adapted from Chroma (Ingraham et al., 2023).
"""

import numpy as np
import torch
from torch import Tensor

from artios.utils.typing import Bool, Float, Int


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
