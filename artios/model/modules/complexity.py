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
) -> Float[Tensor, "batch"]:
    """Compute local composition perplexity penalty.

    For each position, computes the Shannon entropy of amino acid composition
    in a local window (same chain only). Positions where entropy falls below
    `entropy_min` are penalized quadratically.

    This function is differentiable w.r.t. S, enabling gradient-based
    adjustment of MCMC proposal distributions.

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

    Returns:
        Penalty energy per sample [B]. Higher = more penalty.
    """
    B, N, R = S.shape
    device = S.device
    half_w = window_size // 2

    # Restrict to natural amino acids
    S_aa = S[..., :num_natural_aas]  # [B, N, 20]
    Q = num_natural_aas

    # Build 1D sliding-window neighbour indices: [N, W]
    offsets = torch.arange(-half_w, half_w + 1, device=device)  # [W]
    positions = torch.arange(N, device=device)[:, None] + offsets[None, :]  # [N, W]
    positions = positions.clamp(min=0, max=N - 1)  # [N, W]
    W = positions.shape[-1]

    # Expand for batch: [B, N*W]
    positions = positions.unsqueeze(0).expand(B, -1, -1)  # [B, N, W]
    pos_flat = positions.reshape(B, N * W)

    # Gather chain_id and token_mask at neighbour positions
    chain_j = torch.gather(chain_id, 1, pos_flat).reshape(B, N, W)  # [B, N, W]
    mask_j = torch.gather(token_mask.long(), 1, pos_flat).reshape(B, N, W).bool()

    # Same-chain mask
    chain_i = chain_id[:, :, None]  # [B, N, 1]
    same_chain = chain_i == chain_j  # [B, N, W]
    valid_mask = same_chain & mask_j & token_mask[:, :, None]  # [B, N, W]

    # Gather one-hot amino acids at neighbour positions -> [B, N, W, Q]
    pos_for_gather = pos_flat.unsqueeze(-1).expand(-1, -1, Q)  # [B, N*W, Q]
    S_aa_neighbours = torch.gather(S_aa, 1, pos_for_gather).reshape(B, N, W, Q)

    # Mask and sum to get local composition counts [B, N, Q]
    counts = (valid_mask.unsqueeze(-1).float() * S_aa_neighbours).sum(dim=2)

    # Shannon entropy (naive estimator)
    total = counts.sum(-1, keepdim=True)  # [B, N, 1]
    P = counts / (total + 1e-8)
    H = -(P * torch.log(P + 1e-8)).sum(-1)  # [B, N]

    # Penalize positions where perplexity is below threshold
    perplexity_gap = (torch.exp(H) - np.exp(entropy_min)).clamp(max=0)
    penalty = perplexity_gap.square()  # [B, N]

    # Mask and sum
    penalty = (token_mask.float() * penalty).sum(dim=1)  # [B]

    return penalty
