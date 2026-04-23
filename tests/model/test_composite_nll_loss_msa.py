"""Unit tests for ``composite_nll_loss_msa``.

The MSA loss now returns a single ``per_position_nll`` tensor [B, N] —
the same shape and semantics as the non-MSA ``composite_nll_loss`` — so
``ArtiosPotts.compute_loss`` can feed it into the exact same
loss-weight / masked-mean / entity-metrics pipeline. Tests cover:

  - n_msa=1 with row=gt_seq matches the non-MSA composite per-position
    NLL.
  - All-gap (null) MSA → per_position_nll is 0 (masked_mean with empty
    mask).
  - One row gap at a specific (row, token) excludes only that cell from
    the per-position row-average.
  - Swapping a row that disagrees with gt changes per_position_nll at
    tokens where the rows differ (silent-bug guard).
  - Chunked and full-vectorised paths are bit-identical.
  - Hand-calculable tiny case vs pen-and-paper.
"""

from __future__ import annotations

import math

import pytest
import torch

from artios.model.loss import composite_nll_loss, composite_nll_loss_msa


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
# Regression: n_msa=1 with row=gt_seq matches non-MSA composite per_position_nll.
# ---------------------------------------------------------------------------


def test_n_msa_1_matches_non_msa_per_position_nll():
    h, J, edge_index, edge_mask, gt = _make_potts_inputs()
    B, N = gt.shape

    _, per_pos_nonmsa = composite_nll_loss(
        h=h,
        J=J,
        edge_index=edge_index,
        edge_mask=edge_mask,
        seq=gt,
        node_mask=torch.ones(B, N, dtype=torch.bool),
    )

    # MSA path with M=1, column 0 = gt, no gaps. MSAContext layout is
    # [B, N, M] — composite_nll_loss_msa handles the transpose.
    msa_tokens = gt.unsqueeze(-1)  # [B, N, 1]
    msa_gap_mask = torch.zeros_like(msa_tokens, dtype=torch.bool)

    per_pos_msa = composite_nll_loss_msa(
        h,
        J,
        edge_index,
        edge_mask,
        msa_tokens,
        msa_gap_mask,
    )

    torch.testing.assert_close(per_pos_msa, per_pos_nonmsa, rtol=1e-5, atol=1e-5)


def test_n_msa_many_all_equal_to_gt_matches_non_msa():
    """If every MSA row is the gt sequence with no gaps, the per-position
    row-average degenerates to the single-sequence composite NLL. Guards
    against a broadcasting bug that would accidentally sum instead of
    average across rows."""
    h, J, edge_index, edge_mask, gt = _make_potts_inputs(seed=3)
    B, N = gt.shape

    _, per_pos_nonmsa = composite_nll_loss(
        h=h,
        J=J,
        edge_index=edge_index,
        edge_mask=edge_mask,
        seq=gt,
        node_mask=torch.ones(B, N, dtype=torch.bool),
    )

    M = 8
    msa_tokens = gt.unsqueeze(-1).expand(B, N, M).contiguous()
    msa_gap_mask = torch.zeros(B, N, M, dtype=torch.bool)

    per_pos_msa = composite_nll_loss_msa(
        h, J, edge_index, edge_mask, msa_tokens, msa_gap_mask
    )

    torch.testing.assert_close(per_pos_msa, per_pos_nonmsa, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Null MSA (all-gap rows) → per_position_nll is 0.
# ---------------------------------------------------------------------------


def test_all_gap_msa_gives_zero_per_position_nll():
    """A batch where every (token, row) pair is masked as a gap must
    produce per_position_nll = 0 — ``masked_mean`` returns 0 when the
    mask sum is zero (its canonical "no valid rows" fallback).

    This is the semantics exercised by test-level null-MSA fixtures that
    bypass the featurizer and construct tokens=gap directly. Note:
    ``compute_potts_joint_logits`` indexes J by the token id, so the
    token id must stay in-bounds even when the row is going to be
    masked out. Production uses R=32 where gap=31 is in-bounds; here
    we use a valid in-range id and rely on ``msa_gap_mask`` alone."""
    h, J, edge_index, edge_mask, gt = _make_potts_inputs(B=2)
    B, N = gt.shape
    R = h.shape[-1]
    M = 4

    # Use an in-range token id — value is masked out anyway.
    msa_tokens = torch.zeros((B, N, M), dtype=torch.long)
    msa_gap_mask = torch.ones(B, N, M, dtype=torch.bool)
    assert msa_tokens.max().item() < R  # sanity: in-bounds for J-gather

    per_pos = composite_nll_loss_msa(
        h, J, edge_index, edge_mask, msa_tokens, msa_gap_mask
    )

    assert torch.isfinite(per_pos).all()
    assert torch.all(per_pos == 0.0)


def test_partial_gap_mask_only_excludes_gapped_rows():
    """Marking row-1 as entirely gapped at every token should reduce the
    per-position row-average to the single-row (row-0) NLL. Guards against
    a bug where gap tokens contribute alongside the gap-mask count."""
    h, J, edge_index, edge_mask, gt = _make_potts_inputs(B=1, seed=5)
    B, N = gt.shape
    R = h.shape[-1]
    M = 2

    gen = torch.Generator().manual_seed(6)
    msa_tokens = torch.empty(B, N, M, dtype=torch.long)
    msa_tokens[..., 0] = gt
    msa_tokens[..., 1] = torch.randint(0, R, (B, N), generator=gen)

    msa_gap_mask = torch.zeros(B, N, M, dtype=torch.bool)
    msa_gap_mask[..., 1] = True  # row 1 entirely gapped

    per_pos_partial = composite_nll_loss_msa(
        h, J, edge_index, edge_mask, msa_tokens, msa_gap_mask
    )

    # Same inputs with row 0 alone.
    per_pos_row0_only = composite_nll_loss_msa(
        h,
        J,
        edge_index,
        edge_mask,
        msa_tokens[..., :1],
        torch.zeros(B, N, 1, dtype=torch.bool),
    )

    torch.testing.assert_close(per_pos_partial, per_pos_row0_only, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Silent-bug guard: row content actually affects the loss.
# ---------------------------------------------------------------------------


def test_row_swap_of_different_content_changes_per_position_nll():
    """Replacing one of the MSA rows with distinct (non-gt) content must
    change per_position_nll at tokens where the replaced row differs.
    A uniform row-mean is permutation-invariant across rows *with the
    same content*, but replacing content is not; this catches a silent
    bug where the m-axis is collapsed before indexing."""
    h, J, edge_index, edge_mask, gt = _make_potts_inputs(B=1, seed=11)
    B, N = gt.shape
    R = h.shape[-1]
    M = 2

    gen = torch.Generator().manual_seed(12)
    msa_a = torch.empty(B, N, M, dtype=torch.long)
    msa_a[..., 0] = gt
    msa_a[..., 1] = torch.randint(0, R, (B, N), generator=gen)

    msa_b = msa_a.clone()
    # Replace row 1 with a different random sequence at all tokens.
    msa_b[..., 1] = torch.randint(0, R, (B, N), generator=gen)

    msa_gap_mask = torch.zeros(B, N, M, dtype=torch.bool)

    per_pos_a = composite_nll_loss_msa(h, J, edge_index, edge_mask, msa_a, msa_gap_mask)
    per_pos_b = composite_nll_loss_msa(h, J, edge_index, edge_mask, msa_b, msa_gap_mask)

    assert not torch.allclose(per_pos_a, per_pos_b)


# ---------------------------------------------------------------------------
# Chunked path matches full-vectorised path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_chunk", [1, 2, 3, 8])
def test_chunked_matches_full_vectorised(n_chunk):
    """Chunking over n_msa must be an exact refactor, not an approximation."""
    h, J, edge_index, edge_mask, gt = _make_potts_inputs(B=2, N=7, K=3, R=5, seed=17)
    B, N = gt.shape
    R = h.shape[-1]
    M = 6
    gen = torch.Generator().manual_seed(18)

    msa_tokens = torch.randint(0, R, (B, N, M), generator=gen)
    msa_gap_mask = torch.zeros(B, N, M, dtype=torch.bool)

    per_pos_full = composite_nll_loss_msa(
        h, J, edge_index, edge_mask, msa_tokens, msa_gap_mask, n_chunk=M
    )
    per_pos_chunk = composite_nll_loss_msa(
        h, J, edge_index, edge_mask, msa_tokens, msa_gap_mask, n_chunk=n_chunk
    )

    torch.testing.assert_close(per_pos_full, per_pos_chunk, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Hand-calculable case vs pen-and-paper.
# ---------------------------------------------------------------------------


def test_hand_calc_single_edge_single_position():
    """A tiny two-node graph with a single valid edge and known logits.

    B=1, N=2, K=1 (edge 0->1), R=3, M=1. Position 0's only edge is
    valid (0->1). Position 1's only edge is self (1->1), masked out by
    ``combined_mask``. Only position 0 contributes.

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
      - per_position_nll[0, 0] = -logP / (2 * 1) = LSE / 2.
    """
    B, N, K, R = 1, 2, 1, 3
    M = 1

    h = torch.zeros(B, N, R)
    J = torch.zeros(B, N, K, R, R)
    J[0, 0, 0, 0, 1] = 5.0  # the only coupling
    edge_index = torch.tensor([[[1], [1]]], dtype=torch.long)  # [1, 2, 1]
    edge_mask = torch.ones(B, N, K, dtype=torch.bool)

    # MSAContext layout: [B, N, M]. Row 0 is the query [0, 1].
    msa_tokens = torch.tensor([[[0], [1]]], dtype=torch.long)  # [1, 2, 1]
    msa_gap_mask = torch.zeros(B, N, M, dtype=torch.bool)

    per_pos = composite_nll_loss_msa(
        h, J, edge_index, edge_mask, msa_tokens, msa_gap_mask
    )

    lse = math.log(7.0 * math.exp(0.0) + 2.0 * math.exp(5.0))
    expected_logp = 0.0 - lse
    expected_nll_pos0 = -expected_logp / 2.0  # /2 for undirected double count
    torch.testing.assert_close(
        per_pos[0, 0],
        torch.tensor(expected_nll_pos0, dtype=per_pos.dtype),
        rtol=1e-4,
        atol=1e-4,
    )

    # Position 1's only edge is a self-edge — masked out, so its
    # per-pair_nll is ~0 and the row-average is too.
    assert abs(per_pos[0, 1].item()) < 1e-3
