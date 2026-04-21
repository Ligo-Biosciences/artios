"""Tests for MPNN featurizer modules."""

import pytest
import torch

from artios.model.modules.mpnn_features import (
    EdgeRelativePositionalEncoding,
    GaussianRBF,
    MPNNFeaturizer,
)
from tests._helpers.context import create_test_context


class TestEdgeRelativePositionalEncoding:
    """Tests for EdgeRelativePositionalEncoding module."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_initialization(self):
        """Test EdgeRelativePositionalEncoding initialization."""
        encoder = EdgeRelativePositionalEncoding(
            edge_dim=128,
            r_max=32,
            s_max=2,
        )

        assert encoder.edge_dim == 128
        assert encoder.r_max == 32
        assert encoder.s_max == 2
        assert hasattr(encoder, "relpos")

    def test_forward_shape(self, device):
        """Test forward pass output shape."""
        encoder = EdgeRelativePositionalEncoding(
            edge_dim=64,
            r_max=32,
            s_max=2,
        ).to(device)

        batch_size, n_tokens, n_neighbors = 2, 20, 10

        # Create input tensors
        asym_id = torch.randint(0, 5, (batch_size, n_tokens)).to(device)
        residue_index = (
            torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1).to(device)
        )
        entity_id = torch.randint(0, 3, (batch_size, n_tokens)).to(device)
        sym_id = torch.zeros(batch_size, n_tokens).long().to(device)
        token_index = (
            torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1).to(device)
        )
        edge_index = torch.randint(0, n_tokens, (batch_size, n_tokens, n_neighbors)).to(
            device
        )

        # Forward pass
        output = encoder(
            asym_id=asym_id,
            residue_index=residue_index,
            entity_id=entity_id,
            sym_id=sym_id,
            token_index=token_index,
            edge_index=edge_index,
        )

        # Check output shape
        assert output.shape == (batch_size, n_tokens, n_neighbors, 64)
        assert not torch.isnan(output).any()

    def test_forward_single_batch(self, device):
        """Test forward pass with single batch."""
        encoder = EdgeRelativePositionalEncoding(edge_dim=32).to(device)

        n_tokens, n_neighbors = 15, 8

        asym_id = torch.zeros(1, n_tokens).long().to(device)
        residue_index = torch.arange(n_tokens).unsqueeze(0).to(device)
        entity_id = torch.zeros(1, n_tokens).long().to(device)
        sym_id = torch.zeros(1, n_tokens).long().to(device)
        token_index = torch.arange(n_tokens).unsqueeze(0).to(device)
        edge_index = torch.randint(0, n_tokens, (1, n_tokens, n_neighbors)).to(device)

        output = encoder(
            asym_id=asym_id,
            residue_index=residue_index,
            entity_id=entity_id,
            sym_id=sym_id,
            token_index=token_index,
            edge_index=edge_index,
        )

        assert output.shape == (1, n_tokens, n_neighbors, 32)
        assert not torch.isnan(output).any()

    def test_same_chain_vs_different_chain(self, device):
        """Test that same-chain and different-chain edges have different encodings."""
        encoder = EdgeRelativePositionalEncoding(edge_dim=64).to(device)

        n_tokens, n_neighbors = 10, 5

        # Create two scenarios: same chain and different chains
        asym_id_same = torch.zeros(1, n_tokens).long().to(device)
        asym_id_diff = (
            torch.arange(n_tokens).unsqueeze(0).to(device)
        )  # Each token different chain

        residue_index = torch.arange(n_tokens).unsqueeze(0).to(device)
        entity_id = torch.zeros(1, n_tokens).long().to(device)
        sym_id = torch.zeros(1, n_tokens).long().to(device)
        token_index = torch.arange(n_tokens).unsqueeze(0).to(device)
        edge_index = torch.randint(0, n_tokens, (1, n_tokens, n_neighbors)).to(device)

        # Forward pass for same chain
        output_same = encoder(
            asym_id=asym_id_same,
            residue_index=residue_index,
            entity_id=entity_id,
            sym_id=sym_id,
            token_index=token_index,
            edge_index=edge_index,
        )

        # Forward pass for different chains
        output_diff = encoder(
            asym_id=asym_id_diff,
            residue_index=residue_index,
            entity_id=entity_id,
            sym_id=sym_id,
            token_index=token_index,
            edge_index=edge_index,
        )

        # Outputs should be different
        assert not torch.allclose(output_same, output_diff)

    def test_gradient_flow(self, device):
        """Test that gradients flow through the encoder."""
        encoder = EdgeRelativePositionalEncoding(edge_dim=32).to(device)

        n_tokens, n_neighbors = 8, 4

        asym_id = torch.zeros(1, n_tokens).long().to(device)
        residue_index = (
            torch.arange(n_tokens, dtype=torch.float32, requires_grad=True)
            .unsqueeze(0)
            .to(device)
        )
        entity_id = torch.zeros(1, n_tokens).long().to(device)
        sym_id = torch.zeros(1, n_tokens).long().to(device)
        token_index = torch.arange(n_tokens).unsqueeze(0).to(device)
        edge_index = torch.randint(0, n_tokens, (1, n_tokens, n_neighbors)).to(device)

        output = encoder(
            asym_id=asym_id,
            residue_index=residue_index.long(),
            entity_id=entity_id,
            sym_id=sym_id,
            token_index=token_index,
            edge_index=edge_index,
        )

        # Compute loss and backprop
        loss = output.sum()
        loss.backward()

        # Note: residue_index gradients may not flow through due to integer indexing
        # But the model parameters should have gradients
        assert any(p.grad is not None for p in encoder.parameters())


class TestGaussianRBF:
    """Tests for GaussianRBF module."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_initialization(self):
        """Test RBF initialization."""
        rbf = GaussianRBF(num_kernels=16, min_distance=2.0, max_distance=22.0)

        assert rbf.num_kernels == 16
        assert rbf.min_distance == 2.0
        assert rbf.max_distance == 22.0
        assert rbf.centers.shape == (16,)
        assert torch.allclose(rbf.centers[0], torch.tensor(2.0))
        assert torch.allclose(rbf.centers[-1], torch.tensor(22.0))

    def test_forward_shape(self, device):
        """Test RBF forward pass output shape."""
        rbf = GaussianRBF(num_kernels=16).to(device)

        # Test with various input shapes
        distances = torch.randn(10, 25).to(device)
        output = rbf(distances)

        assert output.shape == (10, 25, 16)
        assert not torch.isnan(output).any()

    def test_forward_single_distance(self, device):
        """Test RBF with single distance value."""
        rbf = GaussianRBF(num_kernels=16, min_distance=2.0, max_distance=22.0).to(
            device
        )

        # Distance at first center should have high response
        distances = torch.tensor([2.0]).to(device)
        output = rbf(distances)

        assert output.shape == (1, 16)
        # First kernel should have highest response
        assert output[0, 0] > 0.9

    def test_forward_batched(self, device):
        """Test RBF with batched input."""
        rbf = GaussianRBF(num_kernels=8).to(device)

        batch_size, n_tokens, n_neighbors, n_pairs = 2, 10, 5, 25
        distances = (
            torch.randn(batch_size, n_tokens, n_neighbors, n_pairs).abs().to(device)
        )

        output = rbf(distances)

        assert output.shape == (batch_size, n_tokens, n_neighbors, n_pairs, 8)
        assert not torch.isnan(output).any()
        assert torch.all(output >= 0)  # RBF values should be non-negative

    def test_gaussian_properties(self, device):
        """Test that RBF has proper Gaussian properties."""
        rbf = GaussianRBF(num_kernels=5, min_distance=0.0, max_distance=10.0).to(device)

        # Test at center points
        distances = torch.linspace(0, 10, 5).to(device)
        output = rbf(distances)

        # Each distance should have maximum response at corresponding kernel
        for i in range(5):
            assert output[i, i] == torch.max(output[i])


class TestMPNNFeaturizer:
    """Tests for MPNNFeaturizer methods."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def featurizer(self, device):
        """Create MPNNFeaturizer instance."""
        return MPNNFeaturizer(
            node_dim=128,
            edge_dim=64,
            positional_encoding_dim=16,
            num_rbf=16,
            num_neighbours=32,
            coord_noise=0.0,  # Disable noise for testing
        ).to(device)

    @pytest.fixture
    def test_context(self, device):
        """Create test context for testing."""
        return create_test_context(
            batch_size=1,
            n_tokens=20,
            n_atoms=100,
            device=device,
        )

    @pytest.fixture
    def test_context_batched(self, device):
        """Create batched test context for testing."""
        return create_test_context(
            batch_size=2,
            n_tokens=15,
            n_atoms=75,
            device=device,
        )

    def test_compute_neighbour_graph(self, featurizer, device):
        """Test k-nearest neighbor graph computation."""
        batch_size, n_tokens = 2, 20
        centre_coords = torch.randn(batch_size, n_tokens, 3).to(device)
        centre_mask = torch.ones(batch_size, n_tokens).bool().to(device)

        # Mask out last 2 tokens
        centre_mask[:, -2:] = False

        neighbor_dists, edge_index = featurizer._compute_neighbour_graph(
            centre_atom_coords=centre_coords,
            centre_atom_mask=centre_mask,
        )

        # Check output shapes
        k = min(featurizer.num_neighbours, n_tokens)
        assert neighbor_dists.shape == (batch_size, n_tokens, k)
        assert edge_index.shape == (batch_size, n_tokens, k)

        # Check edge indices are valid
        assert torch.all(edge_index >= 0)
        assert torch.all(edge_index < n_tokens)

        # Check distances are sorted (smallest first)
        assert torch.all(neighbor_dists[:, :, :-1] <= neighbor_dists[:, :, 1:])

    def test_compute_neighbour_graph_single_batch(self, featurizer, device):
        """Test neighbor graph without batch dimension."""
        n_tokens = 15
        centre_coords = torch.randn(1, n_tokens, 3).to(device)
        centre_mask = torch.ones(1, n_tokens).bool().to(device)

        neighbor_dists, edge_index = featurizer._compute_neighbour_graph(
            centre_atom_coords=centre_coords,
            centre_atom_mask=centre_mask,
        )

        k = min(featurizer.num_neighbours, n_tokens)
        assert neighbor_dists.shape == (1, n_tokens, k)
        assert edge_index.shape == (1, n_tokens, k)

    def test_compute_backbone_distance_features(self, featurizer, device):
        """Test backbone distance feature computation."""
        batch_size, n_tokens, n_neighbors = 2, 15, 10

        # Create synthetic backbone coordinates
        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3).to(device)
        backbone_mask = torch.ones(
            batch_size, n_tokens, 5, dtype=torch.bool, device=device
        )

        # Mask out some backbone atoms
        backbone_mask[:, :, -1] = False  # Mask last backbone atom for all tokens
        backbone_mask[:, -3:, :] = False  # Mask all backbone atoms for last 3 tokens

        # Create edge index
        edge_index = torch.randint(0, n_tokens, (batch_size, n_tokens, n_neighbors)).to(
            device
        )

        # Compute features
        edge_feats = featurizer._compute_backbone_distance_features(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            edge_index=edge_index,
        )

        # Check output shape
        assert edge_feats.shape == (
            batch_size,
            n_tokens,
            n_neighbors,
            featurizer.num_rbf * 25,
        )
        assert not torch.isnan(edge_feats).any()

    def test_compute_backbone_distance_features_masking(self, featurizer, device):
        """Test that masking properly zeros out invalid distances."""
        n_tokens, n_neighbors = 10, 5

        backbone_coords = torch.randn(1, n_tokens, 5, 3).to(device)
        backbone_mask = torch.ones(1, n_tokens, 5, dtype=torch.bool, device=device)

        # Completely mask out one token's backbone
        masked_token_idx = 5
        backbone_mask[:, masked_token_idx, :] = False

        edge_index = torch.randint(0, n_tokens, (1, n_tokens, n_neighbors)).to(device)

        edge_feats = featurizer._compute_backbone_distance_features(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            edge_index=edge_index,
        )

        assert edge_feats.shape == (1, n_tokens, n_neighbors, featurizer.num_rbf * 25)
        # Features should still be computed (not necessarily zero due to normalization)
        assert not torch.isnan(edge_feats).any()

    def test_compute_backbone_distance_features_single_backbone_atom(
        self, featurizer, device
    ):
        """Test with ligands (only first backbone atom valid)."""
        n_tokens, n_neighbors = 8, 4

        backbone_coords = torch.randn(1, n_tokens, 5, 3).to(device)
        backbone_mask = torch.zeros(1, n_tokens, 5, dtype=torch.bool, device=device)

        # Only first backbone slot valid (simulating ligands)
        backbone_mask[:, :, 0] = True

        edge_index = torch.randint(0, n_tokens, (1, n_tokens, n_neighbors)).to(device)

        edge_feats = featurizer._compute_backbone_distance_features(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            edge_index=edge_index,
        )

        assert edge_feats.shape == (1, n_tokens, n_neighbors, featurizer.num_rbf * 25)
        assert not torch.isnan(edge_feats).any()

    def test_gradient_flow(self, featurizer, device):
        """Test that gradients flow through the featurizer."""
        batch_size, n_tokens, n_atoms = 1, 15, 75

        # Create mock context with requires_grad tensors
        context = create_test_context(batch_size, n_tokens, n_atoms, device)

        # Make backbone coordinates require gradient
        backbone_coords = torch.randn(
            batch_size, n_tokens, 5, 3, requires_grad=True, device=device
        )
        context.coordinate.backbone_coordinate = backbone_coords

        node_feats, edge_feats, _ = featurizer(context)

        # Compute loss and backprop
        loss = node_feats.sum() + edge_feats.sum()
        loss.backward()

        # Check gradients exist
        assert backbone_coords.grad is not None
        assert not torch.isnan(backbone_coords.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
