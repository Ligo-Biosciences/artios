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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
