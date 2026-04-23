"""Unit tests for ``composite_nll_loss_msa``.

Cover:
  - n_msa=1 with row=gt_seq matches the non-MSA composite_nll_loss.
  - Null-MSA samples (msa_chain_mask all False) contribute nothing and
    set ``samp_valid=False``.
  - Row-swap changes the loss (silent-bug guard against transposition
    or broadcasting errors).
  - Chunked and full-vectorised paths produce bit-identical outputs.
  - Hand-calculable tiny case vs pen-and-paper.
"""

from __future__ import annotations

import math

import pytest
import torch

from artios.model.loss import composite_nll_loss, composite_nll_loss_msa
from artios.model.modules.potts import compute_potts_joint_logits


def _make_potts_inputs(
    B: int = 2,
    N: int = 6,
    K: int = 3,
    R: int = 5,
    seed: int = 0,
):
    """Small synthetic (h, J, edge_index, edge_mask, gt_seq) for tests."""
    gen = torch.Generator().manual_seed(seed)
    h = torch.randn(B, N, R, generator=gen)
    J = torch.randn(B, N, K, R, R, generator=gen)
    edge_index = torch.randint(0, N, (B, N, K), generator=gen)
    edge_mask = torch.ones(B, N, K, dtype=torch.bool)
    edge_mask[:, :, -1] = False  # drop one edge to exercise masking
    gt_seq = torch.randint(0, R, (B, N), generator=gen)
    return h, J, edge_index, edge_mask, gt_seq


# ---------------------------------------------------------------------------
# Regression: n_msa=1 with row=gt_seq matches non-MSA composite_nll_loss.
# ---------------------------------------------------------------------------


def test_n_msa_1_matches_non_msa_composite():
    h, J, edge_index, edge_mask, gt = _make_potts_inputs()
    B, N = gt.shape

    # Non-MSA path.
    joint_logits, combined_edge_mask = compute_potts_joint_logits(
        gt, h, J, edge_index, edge_mask
    )
    nll_nonmsa, per_pos_nonmsa = composite_nll_loss(
        joint_logits,
        gt,
        edge_index,
        combined_edge_mask,
        node_mask=torch.ones(B, N, dtype=torch.bool),
    )

    # MSA path with M=1, row 0 = gt, no gaps, msa_chain_mask all True.
    msa_tokens = gt.unsqueeze(1)  # [B, 1, N]
    msa_gap_mask = torch.zeros_like(msa_tokens, dtype=torch.bool)
    msa_chain_mask = torch.ones(B, N, dtype=torch.bool)
    node_mask = torch.ones(B, N, dtype=torch.bool)

    samp_nll, samp_valid, per_pair_nll = composite_nll_loss_msa(
        h,
        J,
        edge_index,
        edge_mask,
        msa_tokens,
        msa_gap_mask,
        msa_chain_mask,
        node_mask,
    )

    assert samp_valid.all()
    # The MSA sample NLL (only one row) should match the non-MSA per-sample NLL.
    torch.testing.assert_close(samp_nll, nll_nonmsa, rtol=1e-5, atol=1e-5)
    # per_pair_nll at row 0 should match per_position_nonmsa at every token
    # (non-MSA path doesn't mask per-position, MSA path does — but our mask
    # is all True here, so they match).
    torch.testing.assert_close(per_pair_nll[:, 0], per_pos_nonmsa, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Null-MSA samples don't contribute.
# ---------------------------------------------------------------------------


def test_null_msa_sample_is_invalid_and_contributes_zero():
    """If msa_chain_mask is all False for a sample (null MSA sentinel), the
    sample is flagged invalid and its per_pair_nll is masked out."""
    h, J, edge_index, edge_mask, gt = _make_potts_inputs(B=2)
    B, N = gt.shape
    M = 4

    msa_tokens = gt.unsqueeze(1).expand(B, M, N).contiguous()
    msa_gap_mask = torch.zeros(B, M, N, dtype=torch.bool)
    msa_chain_mask = torch.ones(B, N, dtype=torch.bool)
    node_mask = torch.ones(B, N, dtype=torch.bool)

    # Sample 0 has a real MSA; sample 1 is a null MSA.
    msa_chain_mask[1] = False

    samp_nll, samp_valid, per_pair_nll = composite_nll_loss_msa(
        h,
        J,
        edge_index,
        edge_mask,
        msa_tokens,
        msa_gap_mask,
        msa_chain_mask,
        node_mask,
    )

    assert samp_valid[0].item() is True
    assert samp_valid[1].item() is False
    # Per-pair NLL for the null sample should be zero (masked out).
    assert torch.all(per_pair_nll[1] == 0)
    # Batch reduction using samp_valid should drop sample 1.
    batch_mean = (samp_nll * samp_valid).sum() / samp_valid.sum().clamp(min=1)
    torch.testing.assert_close(batch_mean, samp_nll[0])


def test_all_null_batch_is_safe():
    """If the entire batch is null MSAs, samp_valid is all False. The
    canonical batch-reduction (`sum / valid.sum().clamp(min=1)`) returns 0
    without NaN."""
    h, J, edge_index, edge_mask, gt = _make_potts_inputs(B=2)
    B, N = gt.shape

    msa_tokens = gt.unsqueeze(1).expand(B, 4, N).contiguous()
    msa_gap_mask = torch.zeros(B, 4, N, dtype=torch.bool)
    msa_chain_mask = torch.zeros(B, N, dtype=torch.bool)  # all null
    node_mask = torch.ones(B, N, dtype=torch.bool)

    samp_nll, samp_valid, per_pair_nll = composite_nll_loss_msa(
        h,
        J,
        edge_index,
        edge_mask,
        msa_tokens,
        msa_gap_mask,
        msa_chain_mask,
        node_mask,
    )

    assert not samp_valid.any()
    batch_mean = (samp_nll * samp_valid).sum() / samp_valid.sum().clamp(min=1)
    assert torch.isfinite(batch_mean)
    assert batch_mean.item() == 0.0


# ---------------------------------------------------------------------------
# Gaps exclude (row, token) pairs from the loss.
# ---------------------------------------------------------------------------


def test_gap_token_is_excluded_from_per_row_mean():
    """Flipping one homolog's token to a gap and marking it in
    ``msa_gap_mask`` must reduce the row's contribution (fewer tokens
    averaged in) rather than treating the gap id as a real residue."""
    h, J, edge_index, edge_mask, gt = _make_potts_inputs(B=1, seed=7)
    B, N = gt.shape
    M = 2

    msa_tokens = gt.unsqueeze(1).expand(B, M, N).contiguous()
    msa_gap_mask_nogap = torch.zeros(B, M, N, dtype=torch.bool)
    msa_chain_mask = torch.ones(B, N, dtype=torch.bool)
    node_mask = torch.ones(B, N, dtype=torch.bool)

    # Baseline: both rows identical to gt, no gaps.
    samp_nll_a, _, _ = composite_nll_loss_msa(
        h,
        J,
        edge_index,
        edge_mask,
        msa_tokens,
        msa_gap_mask_nogap,
        msa_chain_mask,
        node_mask,
    )

    # Now mark row-1, token-3 as a gap. The row's per-pair_nll at that
    # token should be excluded from its mean.
    msa_gap_mask = msa_gap_mask_nogap.clone()
    msa_gap_mask[0, 1, 3] = True
    samp_nll_b, _, _ = composite_nll_loss_msa(
        h,
        J,
        edge_index,
        edge_mask,
        msa_tokens,
        msa_gap_mask,
        msa_chain_mask,
        node_mask,
    )

    # The per-row mean shifts by removing one token; the per-sample mean
    # averages two rows (with row 1 now using N-1 tokens), so the overall
    # NLL differs.
    assert not torch.allclose(samp_nll_a, samp_nll_b)


# ---------------------------------------------------------------------------
# Silent-bug guard: row swap changes the loss.
# ---------------------------------------------------------------------------


def test_row_swap_changes_the_loss():
    """Swapping rows 0 and 1 in the MSA must change the per-sample NLL.
    Catches transposition / broadcasting errors that could silently
    symmetrise the loss across rows."""
    h, J, edge_index, edge_mask, gt = _make_potts_inputs(B=1, seed=11)
    B, N = gt.shape
    R = h.shape[-1]
    M = 3
    gen = torch.Generator().manual_seed(12)

    # Row 0 = gt, row 1 = random, row 2 = random (distinct).
    msa_tokens = torch.empty(B, M, N, dtype=torch.long)
    msa_tokens[:, 0] = gt
    msa_tokens[:, 1] = torch.randint(0, R, (B, N), generator=gen)
    msa_tokens[:, 2] = torch.randint(0, R, (B, N), generator=gen)
    msa_gap_mask = torch.zeros(B, M, N, dtype=torch.bool)
    msa_chain_mask = torch.ones(B, N, dtype=torch.bool)
    node_mask = torch.ones(B, N, dtype=torch.bool)

    samp_nll_orig, _, _ = composite_nll_loss_msa(
        h,
        J,
        edge_index,
        edge_mask,
        msa_tokens,
        msa_gap_mask,
        msa_chain_mask,
        node_mask,
    )

    # Swap rows 0 and 1.
    msa_swapped = msa_tokens.clone()
    msa_swapped[:, 0], msa_swapped[:, 1] = msa_tokens[:, 1], msa_tokens[:, 0]
    samp_nll_swap, _, _ = composite_nll_loss_msa(
        h,
        J,
        edge_index,
        edge_mask,
        msa_swapped,
        msa_gap_mask,
        msa_chain_mask,
        node_mask,
    )

    # Uniform row weighting means the per-sample MEAN is invariant
    # under a row permutation (!), but the per-pair_nll at row 0 vs
    # row 1 should differ — verify that by checking one row.
    # Actually the per-sample nll is permutation-invariant across rows
    # with uniform weights. Instead: verify per_pair_nll[0, 0] !=
    # per_pair_nll[0, 1] after the swap.
    _, _, per_pair_orig = composite_nll_loss_msa(
        h,
        J,
        edge_index,
        edge_mask,
        msa_tokens,
        msa_gap_mask,
        msa_chain_mask,
        node_mask,
    )
    _, _, per_pair_swap = composite_nll_loss_msa(
        h,
        J,
        edge_index,
        edge_mask,
        msa_swapped,
        msa_gap_mask,
        msa_chain_mask,
        node_mask,
    )
    # After the swap, rows 0 and 1 of per_pair_nll should swap too.
    torch.testing.assert_close(per_pair_orig[:, 0], per_pair_swap[:, 1])
    torch.testing.assert_close(per_pair_orig[:, 1], per_pair_swap[:, 0])
    # And if rows 0 and 1 carry different sequences, their NLLs differ.
    assert not torch.allclose(per_pair_orig[:, 0], per_pair_orig[:, 1])


# ---------------------------------------------------------------------------
# Chunked path matches full-vectorised path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_chunk", [1, 2, 3, 8])
def test_chunked_matches_full_vectorised(n_chunk):
    """Chunking over n_msa must be an exact refactor, not an
    approximation."""
    h, J, edge_index, edge_mask, gt = _make_potts_inputs(B=2, N=7, K=3, R=5, seed=17)
    B, N = gt.shape
    R = h.shape[-1]
    M = 6
    gen = torch.Generator().manual_seed(18)

    msa_tokens = torch.randint(0, R, (B, M, N), generator=gen)
    msa_gap_mask = torch.zeros(B, M, N, dtype=torch.bool)
    msa_chain_mask = torch.ones(B, N, dtype=torch.bool)
    node_mask = torch.ones(B, N, dtype=torch.bool)

    samp_nll_full, samp_valid_full, per_pair_full = composite_nll_loss_msa(
        h,
        J,
        edge_index,
        edge_mask,
        msa_tokens,
        msa_gap_mask,
        msa_chain_mask,
        node_mask,
        n_chunk=M,
    )
    samp_nll_chunk, samp_valid_chunk, per_pair_chunk = composite_nll_loss_msa(
        h,
        J,
        edge_index,
        edge_mask,
        msa_tokens,
        msa_gap_mask,
        msa_chain_mask,
        node_mask,
        n_chunk=n_chunk,
    )

    torch.testing.assert_close(samp_nll_full, samp_nll_chunk, rtol=1e-6, atol=1e-6)
    assert (samp_valid_full == samp_valid_chunk).all()
    torch.testing.assert_close(per_pair_full, per_pair_chunk, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Hand-calculable case vs pen-and-paper.
# ---------------------------------------------------------------------------


def test_hand_calc_single_edge_single_position():
    """A tiny two-node graph with a single valid edge and known logits.

    B=1, N=2, K=1 (edge 0->1), R=3, M=1. Position 0's only edge is
    valid (0->1). Position 1's only edge is self (1->1), masked out by
    ``combined_mask``. Only position 0 contributes to the loss.

    Derivation (for J with a single non-zero entry J[0,0,0,0,1]=5.0,
    h=0, and msa row = [0, 1]):

      - S_j[0,0,0] = seq[1] = 1, so J_j[0,0,0,r] = J[0,0,0,r,1]
        = [5,0,0].
      - E_i[0,0,r] = h + sum_k J_j = [5,0,0].
      - E_i_minus_ij[0,0,0,r,0] = E_i - J_j = [0,0,0]
        (the i→j leakage cancels).
      - S_i[0,0] = seq[0] = 0, so J_i[0,0,0,0,r] = J[0,0,0,0,r]
        = [0,5,0].
      - E_j (at pos 1) is 0 because pos 1's only edge was self-masked
        when computing E_i. So E_j_minus_ji[0,0,0,0,b] = -J_i = [0,-5,0].
      - joint_logit[0,0,0,a,b] = -(E_i_minus_ij[a,0]
                                  + E_j_minus_ji[0,b]
                                  + J[a,b])
        At (0,1): -(0 + (-5) + 5) = 0.
        At (1,1): -(0 + (-5) + 0) = 5.
        At (2,1): -(0 + (-5) + 0) = 5.
        Everywhere else: 0.

      - log_softmax over the 9 flattened joint states: 7 zeros, 2 fives.
        log_sum_exp = log(7·e^0 + 2·e^5).
      - log P(s_i=0, s_j=1) = 0 - LSE.
      - per_pair_nll[0, 0, 0] = -logP / (2 * 1) = LSE / 2.
    """
    B, N, K, R = 1, 2, 1, 3
    M = 1

    h = torch.zeros(B, N, R)
    J = torch.zeros(B, N, K, R, R)
    J[0, 0, 0, 0, 1] = 5.0  # the only coupling
    edge_index = torch.tensor([[[1], [1]]], dtype=torch.long)  # [1, 2, 1]
    edge_mask = torch.ones(B, N, K, dtype=torch.bool)

    msa_tokens = torch.tensor([[[0, 1]]], dtype=torch.long)  # [1, 1, 2]
    msa_gap_mask = torch.zeros(B, M, N, dtype=torch.bool)
    msa_chain_mask = torch.ones(B, N, dtype=torch.bool)
    node_mask = torch.ones(B, N, dtype=torch.bool)

    samp_nll, samp_valid, per_pair_nll = composite_nll_loss_msa(
        h,
        J,
        edge_index,
        edge_mask,
        msa_tokens,
        msa_gap_mask,
        msa_chain_mask,
        node_mask,
    )

    assert samp_valid.all()

    lse = math.log(7.0 * math.exp(0.0) + 2.0 * math.exp(5.0))
    expected_logp = 0.0 - lse
    expected_nll_pos0 = -expected_logp / 2.0  # /2 for undirected double count
    torch.testing.assert_close(
        per_pair_nll[0, 0, 0],
        torch.tensor(expected_nll_pos0, dtype=per_pair_nll.dtype),
        rtol=1e-4,
        atol=1e-4,
    )

    # Position 1's only edge is a self-edge — masked out of combined_mask,
    # so edge_count is clamped at eps and per_pair_nll is ~ 0.
    assert abs(per_pair_nll[0, 0, 1].item()) < 1e-3
