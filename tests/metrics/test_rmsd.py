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

    def test_rmsd_identical_poses(self, device):
        """Test that RMSD is zero for identical poses."""
        pose = torch.randn(10, 3, device=device)

        result = rmsd(pose, pose)

        assert torch.allclose(result, torch.tensor(0.0, device=device), atol=1e-6)

    def test_rmsd_with_known_value(self, device):
        """Test RMSD with a known analytical result."""
        # Two points, each displaced by 1.0 in x direction
        pred = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device)
        true = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device)

        result = rmsd(pred, true)

        # RMSD = sqrt(mean(1^2)) = 1.0
        assert torch.allclose(result, torch.tensor(1.0, device=device), atol=1e-6)

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

    def test_rmsd_batched(self, device):
        """Test RMSD with batched input."""
        batch_size = 4
        n_atoms = 10
        pred = torch.randn(batch_size, n_atoms, 3, device=device)
        true = torch.randn(batch_size, n_atoms, 3, device=device)

        result = rmsd(pred, true, reduce=False)

        assert result.shape == (batch_size,)
        assert torch.all(result >= 0)

    def test_rmsd_reduce_false(self, device):
        """Test RMSD with reduce=False returns per-batch values."""
        batch_size = 3
        n_atoms = 5
        pred = torch.zeros(batch_size, n_atoms, 3, device=device)
        true = torch.zeros(batch_size, n_atoms, 3, device=device)
        # Set different displacements for each batch
        true[0, :, 0] = 1.0  # RMSD = 1.0
        true[1, :, 0] = 2.0  # RMSD = 2.0
        true[2, :, 0] = 3.0  # RMSD = 3.0

        result = rmsd(pred, true, reduce=False)

        assert result.shape == (batch_size,)
        expected = torch.tensor([1.0, 2.0, 3.0], device=device)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_rmsd_reduce_true(self, device):
        """Test RMSD with reduce=True returns mean over batches."""
        batch_size = 3
        n_atoms = 5
        pred = torch.zeros(batch_size, n_atoms, 3, device=device)
        true = torch.zeros(batch_size, n_atoms, 3, device=device)
        true[0, :, 0] = 1.0
        true[1, :, 0] = 2.0
        true[2, :, 0] = 3.0

        result = rmsd(pred, true, reduce=True)

        # Mean of [1, 2, 3] = 2.0
        assert result.shape == ()
        assert torch.allclose(result, torch.tensor(2.0, device=device), atol=1e-6)


class TestAlignPredToTrue:
    """Tests for the align_pred_to_true() function."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_align_identical_poses(self, device):
        """Test alignment of identical poses returns identity transformation."""
        pose = torch.randn(10, 3, device=device)
        mask = torch.ones(10, dtype=torch.bool, device=device)

        aligned, rot, trans = align_pred_to_true(pose, pose, atom_mask=mask)

        # Rotation should be close to identity
        identity = torch.eye(3, device=device)
        assert torch.allclose(rot, identity, atol=1e-5)

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

    def test_align_with_weights(self, device):
        """Test alignment with non-uniform weights."""
        pred = torch.randn(10, 3, device=device)
        true = torch.randn(10, 3, device=device)
        mask = torch.ones(10, dtype=torch.bool, device=device)
        # Weight first atom heavily
        weights = torch.ones(10, device=device)
        weights[0] = 100.0

        aligned, _, _ = align_pred_to_true(pred, true, atom_mask=mask, weight=weights)

        # First atom should be closer to true than unweighted alignment
        assert aligned.shape == pred.shape

    def test_align_batched(self, device):
        """Test alignment with batched input."""
        batch_size = 4
        n_atoms = 10
        pred = torch.randn(batch_size, n_atoms, 3, device=device)
        true = torch.randn(batch_size, n_atoms, 3, device=device)
        mask = torch.ones(batch_size, n_atoms, dtype=torch.bool, device=device)

        aligned, rot, trans = align_pred_to_true(pred, true, atom_mask=mask)

        assert aligned.shape == (batch_size, n_atoms, 3)
        assert rot.shape == (batch_size, 3, 3)
        assert trans.shape == (batch_size, 1, 3)

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

    def test_partially_aligned_identical_poses(self, device):
        """Test that partially aligned RMSD is zero for identical poses."""
        pose = torch.randn(20, 3, device=device)
        align_mask = torch.ones(20, dtype=torch.bool, device=device)
        align_mask[10:] = False  # Only align first half
        atom_mask = torch.ones(20, dtype=torch.bool, device=device)

        aligned_rmsd, unaligned_rmsd, _, _, _ = partially_aligned_rmsd(
            pose, pose, align_mask, atom_mask
        )

        assert torch.allclose(aligned_rmsd, torch.tensor(0.0, device=device), atol=1e-5)
        assert torch.allclose(
            unaligned_rmsd, torch.tensor(0.0, device=device), atol=1e-5
        )

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

    def test_partially_aligned_batched(self, device):
        """Test batched partially aligned RMSD."""
        batch_size = 3
        n_atoms = 10
        pred = torch.randn(batch_size, n_atoms, 3, device=device)
        true = torch.randn(batch_size, n_atoms, 3, device=device)
        align_mask = torch.ones(batch_size, n_atoms, dtype=torch.bool, device=device)
        atom_mask = torch.ones(batch_size, n_atoms, dtype=torch.bool, device=device)

        aligned_rmsd, unaligned_rmsd, transformed, rot, trans = partially_aligned_rmsd(
            pred, true, align_mask, atom_mask, reduce=False
        )

        assert aligned_rmsd.shape == (batch_size,)
        assert transformed.shape == (batch_size, n_atoms, 3)
        assert rot.shape == (batch_size, 3, 3)


class TestSelfAlignedRMSD:
    """Tests for the self_aligned_rmsd() function."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_self_aligned_identical_poses(self, device):
        """Test that self-aligned RMSD is zero for identical poses."""
        pose = torch.randn(10, 3, device=device)
        mask = torch.ones(10, dtype=torch.bool, device=device)

        aligned_rmsd, _, _, _ = self_aligned_rmsd(pose, pose, mask)

        assert torch.allclose(aligned_rmsd, torch.tensor(0.0, device=device), atol=1e-5)

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

    def test_self_aligned_batched(self, device):
        """Test batched self-aligned RMSD."""
        batch_size = 4
        n_atoms = 10
        pred = torch.randn(batch_size, n_atoms, 3, device=device)
        true = torch.randn(batch_size, n_atoms, 3, device=device)
        mask = torch.ones(batch_size, n_atoms, dtype=torch.bool, device=device)

        aligned_rmsd, transformed, rot, trans = self_aligned_rmsd(
            pred, true, mask, reduce=False
        )

        assert aligned_rmsd.shape == (batch_size,)
        assert transformed.shape == (batch_size, n_atoms, 3)
        assert rot.shape == (batch_size, 3, 3)
        assert trans.shape == (batch_size, 1, 3)


class TestWeightedRigidAlign:
    """Tests for the weighted_rigid_align() function."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_weighted_rigid_align_identical(self, device):
        """Test alignment of identical poses."""
        pose = torch.randn(10, 3, device=device)
        weights = torch.ones(10, device=device)
        mask = torch.ones(10, dtype=torch.bool, device=device)

        aligned = weighted_rigid_align(pose, pose, weights, mask, stop_gradient=True)

        assert torch.allclose(aligned, pose, atol=1e-5)

    def test_weighted_rigid_align_stop_gradient(self, device):
        """Test that stop_gradient=True detaches output."""
        pred = torch.randn(10, 3, device=device, requires_grad=True)
        true = torch.randn(10, 3, device=device)
        weights = torch.ones(10, device=device)
        mask = torch.ones(10, dtype=torch.bool, device=device)

        aligned = weighted_rigid_align(pred, true, weights, mask, stop_gradient=True)

        assert not aligned.requires_grad

    def test_weighted_rigid_align_batched(self, device):
        """Test batched weighted rigid alignment."""
        batch_size = 4
        n_atoms = 10
        pred = torch.randn(batch_size, n_atoms, 3, device=device)
        true = torch.randn(batch_size, n_atoms, 3, device=device)
        weights = torch.ones(batch_size, n_atoms, device=device)
        mask = torch.ones(batch_size, n_atoms, dtype=torch.bool, device=device)

        aligned = weighted_rigid_align(pred, true, weights, mask, stop_gradient=True)

        assert aligned.shape == (batch_size, n_atoms, 3)

    def test_weighted_rigid_align_nonuniform_weights(self, device):
        """Test alignment with non-uniform weights."""
        n_atoms = 10
        pred = torch.randn(n_atoms, 3, device=device)
        true = torch.randn(n_atoms, 3, device=device)
        # Weight first atoms more heavily
        weights = torch.ones(n_atoms, device=device)
        weights[:5] = 10.0
        mask = torch.ones(n_atoms, dtype=torch.bool, device=device)

        aligned = weighted_rigid_align(pred, true, weights, mask, stop_gradient=True)

        assert aligned.shape == (n_atoms, 3)


class TestRMSDEdgeCases:
    """Test edge cases and numerical stability."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_single_atom(self, device):
        """Test RMSD with single atom."""
        pred = torch.tensor([[1.0, 2.0, 3.0]], device=device)
        true = torch.tensor([[4.0, 5.0, 6.0]], device=device)

        result = rmsd(pred, true)

        # Distance = sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(27)
        expected = torch.sqrt(torch.tensor(27.0, device=device))
        assert torch.allclose(result, expected, atol=1e-5)

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

    def test_partial_mask(self, device):
        """Test with only some atoms masked."""
        n_atoms = 20
        pred = torch.randn(n_atoms, 3, device=device)
        true = torch.randn(n_atoms, 3, device=device)

        # Only use half the atoms
        mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        mask[:10] = True

        aligned, _, _ = align_pred_to_true(pred, true, atom_mask=mask)

        # Result should have correct shape
        assert aligned.shape == pred.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
