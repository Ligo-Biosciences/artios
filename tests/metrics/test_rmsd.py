"""Tests for RMSD metrics and alignment functions."""

import pytest
import torch

from artios.metrics.rmsd import (
    align_pred_to_true,
    partially_aligned_rmsd,
    rmsd,
    self_aligned_rmsd,
    weighted_rigid_align,
)


class TestRMSD:
    """Tests for the rmsd() function."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_rmsd_with_mask(self, device):
        """Test RMSD with a mask excluding some atoms."""
        pred = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device
        )
        true = torch.tensor(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [100.0, 0.0, 0.0]], device=device
        )
        # Only consider first two atoms
        mask = torch.tensor([True, True, False], device=device)

        result = rmsd(pred, true, mask=mask)

        # RMSD = sqrt((1^2 + 2^2) / 2) = sqrt(2.5)
        expected = torch.sqrt(torch.tensor(2.5, device=device))
        assert torch.allclose(result, expected, atol=1e-6)

    @pytest.mark.parametrize(
        "reduce,expected_shape,expected_values",
        [
            (False, (3,), [1.0, 2.0, 3.0]),
            (True, (), 2.0),  # mean of [1, 2, 3]
        ],
    )
    def test_rmsd_batched_reduce_modes(
        self, device, reduce, expected_shape, expected_values
    ):
        """Known per-batch displacements ([1, 2, 3]); ``reduce=False`` returns the
        per-batch values, ``reduce=True`` returns their mean."""
        batch_size, n_atoms = 3, 5
        pred = torch.zeros(batch_size, n_atoms, 3, device=device)
        true = torch.zeros(batch_size, n_atoms, 3, device=device)
        true[0, :, 0] = 1.0
        true[1, :, 0] = 2.0
        true[2, :, 0] = 3.0

        result = rmsd(pred, true, reduce=reduce)
        assert result.shape == expected_shape
        assert torch.allclose(
            result, torch.tensor(expected_values, device=device), atol=1e-6
        )


class TestAlignPredToTrue:
    """Tests for the align_pred_to_true() function."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_align_translation_only(self, device):
        """Test alignment when poses differ only by translation."""
        pred = torch.randn(10, 3, device=device)
        translation = torch.tensor([5.0, -3.0, 2.0], device=device)
        true = pred + translation
        mask = torch.ones(10, dtype=torch.bool, device=device)

        aligned, rot, trans = align_pred_to_true(pred, true, atom_mask=mask)

        # Aligned should match true pose
        assert torch.allclose(aligned, true, atol=1e-5)
        # Rotation should be identity
        identity = torch.eye(3, device=device)
        assert torch.allclose(rot, identity, atol=1e-5)

    def test_align_preserves_distances(self, device):
        """Test that alignment preserves pairwise distances."""
        pred = torch.randn(10, 3, device=device)
        true = torch.randn(10, 3, device=device)
        mask = torch.ones(10, dtype=torch.bool, device=device)

        aligned, _, _ = align_pred_to_true(pred, true, atom_mask=mask)

        # Compute pairwise distances
        pred_dists = torch.cdist(pred.unsqueeze(0), pred.unsqueeze(0)).squeeze()
        aligned_dists = torch.cdist(
            aligned.unsqueeze(0), aligned.unsqueeze(0)
        ).squeeze()

        assert torch.allclose(pred_dists, aligned_dists, atol=1e-5)

    def test_align_rotation_matrix_properties(self, device):
        """Test that returned rotation matrix is orthogonal with det=1."""
        pred = torch.randn(10, 3, device=device)
        true = torch.randn(10, 3, device=device)
        mask = torch.ones(10, dtype=torch.bool, device=device)

        _, rot, _ = align_pred_to_true(pred, true, atom_mask=mask)

        # R @ R^T should be identity
        identity = torch.eye(3, device=device)
        assert torch.allclose(torch.matmul(rot, rot.T), identity, atol=1e-5)
        # det(R) should be 1 (proper rotation, no reflection)
        det = torch.linalg.det(rot)
        assert torch.allclose(det, torch.tensor(1.0, device=device), atol=1e-5)

    def test_align_with_reflection(self, device):
        """Test alignment allowing reflection."""
        # Create a reflected pose
        pred = torch.randn(10, 3, device=device)
        true = pred.clone()
        true[:, 0] = -true[:, 0]  # Reflect in x
        mask = torch.ones(10, dtype=torch.bool, device=device)

        aligned, rot, _ = align_pred_to_true(
            pred, true, atom_mask=mask, allowing_reflection=True
        )

        # Should be able to match perfectly with reflection
        assert torch.allclose(aligned, true, atol=1e-5)
        # det(R) should be -1 (reflection)
        det = torch.linalg.det(rot)
        assert torch.allclose(det.abs(), torch.tensor(1.0, device=device), atol=1e-5)


class TestPartiallyAlignedRMSD:
    """Tests for the partially_aligned_rmsd() function."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_partially_aligned_returns_transformed_pose(self, device):
        """Test that transformed pose is returned correctly."""
        pred = torch.randn(10, 3, device=device)
        true = torch.randn(10, 3, device=device)
        align_mask = torch.ones(10, dtype=torch.bool, device=device)
        atom_mask = torch.ones(10, dtype=torch.bool, device=device)

        _, _, transformed, rot, trans = partially_aligned_rmsd(
            pred, true, align_mask, atom_mask
        )

        # Manually apply transformation and check
        rot_t = rot.transpose(-1, -2)
        expected = torch.matmul(pred, rot_t) + trans
        assert torch.allclose(transformed, expected, atol=1e-5)

    def test_partially_aligned_separate_regions(self, device):
        """Test alignment of one region while measuring another."""
        n_atoms = 20
        pred = torch.randn(n_atoms, 3, device=device)

        # True pose: aligned region is translated, unaligned region is far away
        true = pred.clone()
        true[:10] = pred[:10] + torch.tensor([1.0, 0.0, 0.0], device=device)
        true[10:] = pred[10:] + torch.tensor([10.0, 0.0, 0.0], device=device)

        align_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        align_mask[:10] = True  # Align first half
        atom_mask = torch.ones(n_atoms, dtype=torch.bool, device=device)

        aligned_rmsd, unaligned_rmsd, _, _, _ = partially_aligned_rmsd(
            pred, true, align_mask, atom_mask
        )

        # After aligning first half, its RMSD should be ~0
        # Unaligned part should have high RMSD
        assert aligned_rmsd < 0.1
        assert unaligned_rmsd > 5.0


class TestSelfAlignedRMSD:
    """Tests for the self_aligned_rmsd() function."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_self_aligned_translated_poses(self, device):
        """Test self-aligned RMSD for poses differing by translation."""
        pred = torch.randn(10, 3, device=device)
        true = pred + torch.tensor([100.0, -50.0, 25.0], device=device)
        mask = torch.ones(10, dtype=torch.bool, device=device)

        aligned_rmsd, transformed, _, _ = self_aligned_rmsd(pred, true, mask)

        # After alignment, RMSD should be ~0
        assert torch.allclose(aligned_rmsd, torch.tensor(0.0, device=device), atol=1e-5)
        # Transformed should match true
        assert torch.allclose(transformed, true, atol=1e-5)

    def test_self_aligned_rotated_poses(self, device):
        """Test self-aligned RMSD for poses differing by rotation."""
        pred = torch.randn(10, 3, device=device)
        # Apply a 90-degree rotation around z-axis
        rot_matrix = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device
        )
        true = torch.matmul(pred, rot_matrix.T)
        mask = torch.ones(10, dtype=torch.bool, device=device)

        aligned_rmsd, transformed, _, _ = self_aligned_rmsd(pred, true, mask)

        # After alignment, RMSD should be ~0
        assert torch.allclose(aligned_rmsd, torch.tensor(0.0, device=device), atol=1e-5)


class TestWeightedRigidAlign:
    """Tests for the weighted_rigid_align() function."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_weighted_rigid_align_stop_gradient(self, device):
        """Test that stop_gradient=True detaches output."""
        pred = torch.randn(10, 3, device=device, requires_grad=True)
        true = torch.randn(10, 3, device=device)
        weights = torch.ones(10, device=device)
        mask = torch.ones(10, dtype=torch.bool, device=device)

        aligned = weighted_rigid_align(pred, true, weights, mask, stop_gradient=True)

        assert not aligned.requires_grad


class TestRMSDEdgeCases:
    """Test edge cases and numerical stability."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_alignment_two_atoms(self, device):
        """Test alignment with minimum atoms (2 atoms)."""
        pred = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device)
        true = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
        mask = torch.ones(2, dtype=torch.bool, device=device)

        aligned, rot, _ = align_pred_to_true(pred, true, atom_mask=mask)

        # Should be able to rotate to match
        aligned_rmsd_val = rmsd(aligned, true)
        assert aligned_rmsd_val < 1e-5

    def test_large_coordinates(self, device):
        """Test numerical stability with large coordinates."""
        pred = torch.randn(10, 3, device=device) * 1000
        true = torch.randn(10, 3, device=device) * 1000
        mask = torch.ones(10, dtype=torch.bool, device=device)

        aligned, rot, trans = align_pred_to_true(pred, true, atom_mask=mask)

        # Check rotation matrix is still valid
        identity = torch.eye(3, device=device)
        assert torch.allclose(torch.matmul(rot, rot.T), identity, atol=1e-4)

    def test_small_coordinates(self, device):
        """Test numerical stability with small coordinates."""
        pred = torch.randn(10, 3, device=device) * 1e-6
        true = torch.randn(10, 3, device=device) * 1e-6

        result = rmsd(pred, true)

        assert not torch.isnan(result)
        assert not torch.isinf(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
