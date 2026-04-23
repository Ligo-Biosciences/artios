"""Regression tests for the ``*batch``-dim refactor of the Potts helpers.

These guard that:
  - ``gather_nodes``
  - ``compute_potts_single_logits``
  - ``compute_potts_joint_logits``

produce bit-identical (up to float tolerance) results when called with an
extra leading batch dimension versus calling the function once per slice
along that dimension and stacking. The MSA loss path relies on this
invariant so it can score ``n_msa`` rows in one pass rather than looping.
"""

from __future__ import annotations

import torch

from artios.model.modules.message_passing import gather_nodes
from artios.model.modules.potts import (
    compute_potts_joint_logits,
    compute_potts_single_logits,
)


def _make_potts_inputs(
    B: int = 2,
    N: int = 8,
    K: int = 4,
    R: int = 6,
    device: torch.device | str = "cpu",
    seed: int = 0,
):
    """Small synthetic (seq, h, J, edge_index, edge_mask) for testing."""
    gen = torch.Generator(device=device).manual_seed(seed)
    seq = torch.randint(0, R, (B, N), generator=gen, device=device)
    h = torch.randn(B, N, R, generator=gen, device=device)
    J = torch.randn(B, N, K, R, R, generator=gen, device=device)
    # Edges point into valid token positions; include a few self-edges to
    # exercise the self-edge mask branch.
    edge_index = torch.randint(0, N, (B, N, K), generator=gen, device=device)
    edge_mask = torch.ones(B, N, K, dtype=torch.bool, device=device)
    # Drop a couple of edges to exercise masking.
    edge_mask[:, :, -1] = False
    return seq, h, J, edge_index, edge_mask


# ---------------------------------------------------------------------------
# gather_nodes
# ---------------------------------------------------------------------------


class TestGatherNodesBatchAgnostic:
    def test_4d_input_matches_per_slice_stacked(self):
        """Passing h_nodes=[B, M, N, F] + edge_idx=[B, M, N, K] should equal
        stacking gather_nodes(h_nodes[:, m], edge_idx[:, m]) across m."""
        B, M, N, K, F = 2, 3, 8, 4, 5
        gen = torch.Generator().manual_seed(1)
        h_nodes = torch.randn(B, M, N, F, generator=gen)
        # Each slice has its OWN edge topology so we actually exercise the
        # multi-batch gather (not just broadcasting).
        edge_idx = torch.randint(0, N, (B, M, N, K), generator=gen)

        # Reference: per-slice stacked.
        per_slice = torch.stack(
            [gather_nodes(h_nodes[:, m], edge_idx[:, m]) for m in range(M)],
            dim=1,
        )

        # Vectorised call.
        vectorised = gather_nodes(h_nodes, edge_idx)

        assert vectorised.shape == (B, M, N, K, F)
        torch.testing.assert_close(vectorised, per_slice)

    def test_3d_input_unchanged(self):
        """The original 3D behaviour must be preserved."""
        B, N, K, F = 2, 6, 3, 4
        gen = torch.Generator().manual_seed(2)
        h_nodes = torch.randn(B, N, F, generator=gen)
        edge_idx = torch.randint(0, N, (B, N, K), generator=gen)

        out = gather_nodes(h_nodes, edge_idx)
        assert out.shape == (B, N, K, F)
        # Spot-check by manual gather on batch 0, position 0.
        for k in range(K):
            torch.testing.assert_close(out[0, 0, k], h_nodes[0, edge_idx[0, 0, k]])


# ---------------------------------------------------------------------------
# compute_potts_single_logits
# ---------------------------------------------------------------------------


class TestComputePottsSingleLogitsBatchAgnostic:
    def test_4d_input_matches_per_slice_stacked(self):
        """Extra leading dim matches per-slice stacked computation."""
        B, M, N, K, R = 2, 3, 6, 3, 5
        gen = torch.Generator().manual_seed(3)
        seq = torch.randint(0, R, (B, M, N), generator=gen)
        h = torch.randn(B, M, N, R, generator=gen)
        J = torch.randn(B, M, N, K, R, R, generator=gen)
        edge_index = torch.randint(0, N, (B, M, N, K), generator=gen)
        edge_mask = torch.ones(B, M, N, K, dtype=torch.bool)

        per_slice = torch.stack(
            [
                compute_potts_single_logits(
                    seq[:, m],
                    h[:, m],
                    J[:, m],
                    edge_index[:, m],
                    edge_mask[:, m],
                )
                for m in range(M)
            ],
            dim=1,
        )

        vectorised = compute_potts_single_logits(seq, h, J, edge_index, edge_mask)

        assert vectorised.shape == (B, M, N, R)
        torch.testing.assert_close(vectorised, per_slice, rtol=1e-5, atol=1e-5)

    def test_3d_input_unchanged(self):
        """Existing 3D callers behave identically after the refactor."""
        seq, h, J, edge_index, edge_mask = _make_potts_inputs()
        out = compute_potts_single_logits(seq, h, J, edge_index, edge_mask)
        assert out.shape == (seq.shape[0], seq.shape[1], h.shape[-1])
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# compute_potts_joint_logits
# ---------------------------------------------------------------------------


class TestComputePottsJointLogitsBatchAgnostic:
    def test_4d_input_matches_per_slice_stacked(self):
        """Per-row joint logits computed vectorised over an extra M dim
        must match the stack of per-slice calls."""
        B, M, N, K, R = 2, 3, 6, 3, 4
        gen = torch.Generator().manual_seed(4)
        seq = torch.randint(0, R, (B, M, N), generator=gen)
        h = torch.randn(B, M, N, R, generator=gen)
        J = torch.randn(B, M, N, K, R, R, generator=gen)
        edge_index = torch.randint(0, N, (B, M, N, K), generator=gen)
        edge_mask = torch.ones(B, M, N, K, dtype=torch.bool)
        edge_mask[:, :, :, -1] = False  # drop one edge to exercise masking

        per_slice_logits, per_slice_masks = [], []
        for m in range(M):
            logits, mask = compute_potts_joint_logits(
                seq[:, m],
                h[:, m],
                J[:, m],
                edge_index[:, m],
                edge_mask[:, m],
            )
            per_slice_logits.append(logits)
            per_slice_masks.append(mask)
        expected_logits = torch.stack(per_slice_logits, dim=1)
        expected_mask = torch.stack(per_slice_masks, dim=1)

        vectorised_logits, vectorised_mask = compute_potts_joint_logits(
            seq, h, J, edge_index, edge_mask
        )

        assert vectorised_logits.shape == (B, M, N, K, R, R)
        assert vectorised_mask.shape == (B, M, N, K)
        torch.testing.assert_close(
            vectorised_logits, expected_logits, rtol=1e-5, atol=1e-5
        )
        assert (vectorised_mask == expected_mask).all()

    def test_3d_input_unchanged(self):
        """Existing 3D callers behave identically after the refactor."""
        seq, h, J, edge_index, edge_mask = _make_potts_inputs()
        logits, mask = compute_potts_joint_logits(seq, h, J, edge_index, edge_mask)
        B, N, K = edge_index.shape
        R = h.shape[-1]
        assert logits.shape == (B, N, K, R, R)
        assert mask.shape == (B, N, K)
        assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Broadcast-shared-topology path (what the MSA loss actually calls with)
# ---------------------------------------------------------------------------


class TestBroadcastSharedTopology:
    """The MSA loss broadcasts h/J/edge_index/edge_mask with ``.unsqueeze(1)``
    (shared across rows) and only provides a genuinely per-row ``seq``.
    This mirrors that usage pattern."""

    def test_joint_logits_broadcast_h_J_matches_materialised(self):
        B, M, N, K, R = 2, 3, 5, 3, 4
        gen = torch.Generator().manual_seed(5)
        # Per-row seq but shared h, J, edge_index, edge_mask.
        seq = torch.randint(0, R, (B, M, N), generator=gen)
        h = torch.randn(B, N, R, generator=gen)
        J = torch.randn(B, N, K, R, R, generator=gen)
        edge_index = torch.randint(0, N, (B, N, K), generator=gen)
        edge_mask = torch.ones(B, N, K, dtype=torch.bool)
        edge_mask[:, :, -1] = False

        # Broadcast via unsqueeze+expand (views, no copy).
        h_bcast = h.unsqueeze(1).expand(B, M, N, R)
        J_bcast = J.unsqueeze(1).expand(B, M, N, K, R, R)
        edge_idx_bcast = edge_index.unsqueeze(1).expand(B, M, N, K)
        edge_mask_bcast = edge_mask.unsqueeze(1).expand(B, M, N, K)

        logits_bcast, mask_bcast = compute_potts_joint_logits(
            seq, h_bcast, J_bcast, edge_idx_bcast, edge_mask_bcast
        )

        # Materialised reference (contiguous copy — no broadcast shortcut).
        logits_mat, mask_mat = compute_potts_joint_logits(
            seq,
            h_bcast.contiguous(),
            J_bcast.contiguous(),
            edge_idx_bcast.contiguous(),
            edge_mask_bcast.contiguous(),
        )

        torch.testing.assert_close(logits_bcast, logits_mat, rtol=1e-5, atol=1e-5)
        assert (mask_bcast == mask_mat).all()
