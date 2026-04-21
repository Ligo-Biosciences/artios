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


class TestGaussianRBF:
    """Tests for GaussianRBF module."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_rbf_properties(self, device):
        """Pin the three non-trivial RBF invariants:
        - centers span [min_distance, max_distance] exactly
        - output is non-negative everywhere (Gaussian kernel)
        - each kernel is maximal when the distance equals its own center
        """
        n_kernels, min_d, max_d = 5, 0.0, 10.0
        rbf = GaussianRBF(
            num_kernels=n_kernels, min_distance=min_d, max_distance=max_d
        ).to(device)

        assert rbf.centers[0] == min_d
        assert rbf.centers[-1] == max_d

        # Evaluate at each kernel's center: out[i, j] is kernel j's response when
        # the input distance equals center i.
        distances = torch.linspace(min_d, max_d, n_kernels, device=device)
        out = rbf(distances)
        assert (out >= 0).all(), "RBF output must be non-negative"
        for i in range(n_kernels):
            assert (
                out[i, i] == out[i].max()
            ), f"kernel {i} should be maximal at its own center"


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
