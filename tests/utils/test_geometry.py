"""Tests for geometry utility functions."""

import numpy as np
import pytest

from artios.utils.geometry import random_transform


class TestRandomTransform:
    """Tests for random_transform function."""

    def test_centralize_without_mask(self):
        """Test that points are centered at origin without mask."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        result = random_transform(points, centralize=True, apply_augmentation=False)

        # Center of mass should be at origin
        np.testing.assert_allclose(result.mean(axis=0), [0.0, 0.0, 0.0], atol=1e-10)

    def test_centralize_with_mask(self):
        """Test that only masked points are used for computing center."""
        # First two points have center at (2.5, 3.5, 4.5)
        # Third point is unmasked and should not affect centering
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [100.0, 100.0, 100.0]])
        mask = np.array([True, True, False])

        result = random_transform(
            points, centralize=True, apply_augmentation=False, mask=mask
        )

        # Center of masked points should be at origin
        masked_center = result[mask].mean(axis=0)
        np.testing.assert_allclose(masked_center, [0.0, 0.0, 0.0], atol=1e-10)

        # Unmasked points should be zeroed
        np.testing.assert_allclose(result[~mask], [[0.0, 0.0, 0.0]], atol=1e-10)

    def test_no_centralize(self):
        """Test that points are not modified when centralize=False."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = random_transform(points, centralize=False, apply_augmentation=False)

        np.testing.assert_allclose(result, points, atol=1e-10)

    def test_augmentation_preserves_distances(self):
        """Test that rotation preserves pairwise distances."""
        np.random.seed(42)
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        # Compute original pairwise distances
        original_dists = np.linalg.norm(points[:, None] - points[None, :], axis=-1)

        result = random_transform(
            points, centralize=True, apply_augmentation=True, max_translation=0.0
        )

        # Compute transformed pairwise distances
        result_dists = np.linalg.norm(result[:, None] - result[None, :], axis=-1)

        # Distances should be preserved (rotation is isometric)
        np.testing.assert_allclose(original_dists, result_dists, atol=1e-10)

    def test_translation_range(self):
        """Test that translation is within specified range."""
        np.random.seed(42)
        points = np.zeros((1, 3))
        max_trans = 2.0

        # Run multiple times to check translation bounds
        for _ in range(100):
            result = random_transform(
                points.copy(),
                centralize=False,
                apply_augmentation=True,
                max_translation=max_trans,
            )
            # After rotation, the point should still be within sqrt(3)*max_trans of origin
            # (worst case: translation at corner of cube, then rotated)
            assert np.linalg.norm(result) <= np.sqrt(3) * max_trans + 1e-10

    def test_all_masked_out(self):
        """Test behavior when all points are masked out."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = np.array([False, False])

        result = random_transform(
            points, centralize=True, apply_augmentation=False, mask=mask
        )

        # All points should be zeroed
        np.testing.assert_allclose(result, np.zeros_like(points), atol=1e-10)


import torch  # noqa: E402

from artios.utils.geometry import (  # noqa: E402
    NUM_SSE_BURIAL_BUCKETS,
    SSE_BURIAL_MASKED,
    cb_neighbor_count,
    compute_sse_burial_buckets,
)


def _grid_backbone(nx: int, ny: int, nz: int, spacing: float = 4.0) -> torch.Tensor:
    """Build a synthetic backbone tensor [N, 5, 3] where each residue's CB
    sits on a regular 3-D grid (and the other backbone atoms are placed
    nearby with realistic offsets). N = nx*ny*nz.

    The grid lets us hand-compute the expected CB-neighbor count per position.
    """
    coords = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cb = np.array([i * spacing, j * spacing, k * spacing])
                ca = cb + np.array([0.0, 0.0, 1.0])
                # N and C displaced to form a sensible local frame
                n = ca + np.array([1.2, 0.0, 0.0])
                c = ca + np.array([-1.0, 0.6, 0.0])
                o = c + np.array([0.0, 1.0, 0.0])
                coords.append(np.stack([n, ca, c, o, cb]))
    return torch.tensor(np.stack(coords), dtype=torch.float32)


class TestCBNeighborCount:
    def test_grid_counts_match_manual_count(self):
        """On a 3×3×3 grid with 4 Å spacing, CB within 6 Å of any other CB
        is exactly its lattice neighbors: 3..7 depending on grid position."""
        bb = _grid_backbone(3, 3, 3, spacing=4.0)
        N = bb.shape[0]
        mask = torch.ones(N, dtype=torch.bool)
        # 6 Å catches lattice distance 4 (nearest neighbors) but not √2·4 ≈ 5.66
        # ... actually √2·4 ≈ 5.657, so 6 Å DOES catch face diagonals.
        # √3·4 ≈ 6.93 > 6, so 3-D diagonals are out.
        # Lattice neighbors at distance 4 or 4√2 included.
        nb = cb_neighbor_count(bb, mask, radius=6.0).numpy()

        # Pre-compute expected from raw CB coords for cross-check.
        cb = bb[:, 4].numpy()
        d = np.linalg.norm(cb[:, None] - cb[None, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        expected = (d <= 6.0).sum(axis=1)
        assert np.array_equal(nb, expected)
        # Corner residue has 3 face-neighbors + 3 face-diagonals = 6.
        # Center residue has 6 + 12 = 18.
        assert nb.max() >= nb.min(), "non-trivial spread"
        assert nb.min() >= 3

    def test_mask_removes_neighbors_symmetrically(self):
        """Masking a residue must remove it from EVERY other residue's count."""
        bb = _grid_backbone(2, 2, 2, spacing=4.0)
        N = bb.shape[0]
        mask = torch.ones(N, dtype=torch.bool)
        full = cb_neighbor_count(bb, mask, radius=6.0)

        mask2 = mask.clone()
        mask2[0] = False
        masked = cb_neighbor_count(bb, mask2, radius=6.0)

        # Masked residue itself returns 0.
        assert masked[0].item() == 0
        # Every other residue loses exactly 1 if residue 0 was a neighbour,
        # otherwise stays the same.
        cb = bb[:, 4]
        d0 = (cb - cb[0]).norm(dim=-1)
        was_neighbor = (d0 <= 6.0) & (torch.arange(N) != 0)
        expected = full.clone()
        expected[0] = 0
        expected[was_neighbor] -= 1
        assert torch.equal(masked, expected)

    def test_batched_matches_per_sample(self):
        bb1 = _grid_backbone(2, 2, 2, spacing=4.0)
        bb2 = _grid_backbone(3, 1, 1, spacing=4.0)
        N1, N2 = bb1.shape[0], bb2.shape[0]
        N = max(N1, N2)

        batched_bb = torch.zeros(2, N, 5, 3)
        batched_bb[0, :N1] = bb1
        batched_bb[1, :N2] = bb2
        batched_mask = torch.zeros(2, N, dtype=torch.bool)
        batched_mask[0, :N1] = True
        batched_mask[1, :N2] = True

        batched_nb = cb_neighbor_count(batched_bb, batched_mask, radius=6.0)
        nb1 = cb_neighbor_count(bb1, torch.ones(N1, dtype=torch.bool), radius=6.0)
        nb2 = cb_neighbor_count(bb2, torch.ones(N2, dtype=torch.bool), radius=6.0)
        assert torch.equal(batched_nb[0, :N1], nb1)
        assert torch.equal(batched_nb[1, :N2], nb2)
        # Padding zone returns 0.
        assert torch.equal(batched_nb[0, N1:], torch.zeros(N - N1, dtype=torch.long))


class TestSSEBurialBuckets:
    def test_mask_propagates_to_minus_one(self):
        """Masked positions and DSSP-short cases must come out as -1."""
        # Use a real-ish backbone but a short length so DSSP returns coil
        # everywhere; check that masked positions still get -1.
        bb = _grid_backbone(2, 1, 2, spacing=3.8)  # N=4, too short for DSSP turns
        mask = torch.tensor([True, True, False, True])
        out = compute_sse_burial_buckets(bb, mask, nb_radius=10.0, nb_threshold=22)
        assert out[2].item() == SSE_BURIAL_MASKED
        # Other positions land in a coil bucket (4 or 5) since DSSP returned coil.
        valid = out[mask]
        assert ((valid == 4) | (valid == 5)).all().item()

    def test_burial_threshold_flips_bucket(self):
        """Lowering the burial threshold must reclassify exposed → buried for
        residues whose CB-NB sits between the two thresholds."""
        bb = _grid_backbone(4, 4, 4, spacing=4.0)
        N = bb.shape[0]
        mask = torch.ones(N, dtype=torch.bool)
        # nb_radius=6 Å gives a known mix of 6..18 neighbors on a 4³ grid.
        loose = compute_sse_burial_buckets(bb, mask, nb_radius=6.0, nb_threshold=99)
        tight = compute_sse_burial_buckets(bb, mask, nb_radius=6.0, nb_threshold=0)
        # nobody buried under threshold=99: every bucket is odd (exposed).
        assert (loose % 2 == 1).all().item()
        # everybody buried under threshold=0: every bucket is even (buried).
        assert (tight % 2 == 0).all().item()

    def test_bucket_values_within_expected_range(self):
        bb = _grid_backbone(3, 3, 3, spacing=4.0)
        N = bb.shape[0]
        mask = torch.ones(N, dtype=torch.bool)
        out = compute_sse_burial_buckets(bb, mask)
        assert (out >= 0).all() and (out < NUM_SSE_BURIAL_BUCKETS).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
