"""Tests for MPNN layers."""

import pytest
import torch

from artios.model.modules.mpnn_layers import (
    MPNNDecoderLayer,
    MPNNEncoderLayer,
)


@pytest.fixture
def batch_size():
    """Batch size for tests."""
    return 2


@pytest.fixture
def num_nodes():
    """Number of nodes in the graph."""
    return 10


@pytest.fixture
def num_neighbors():
    """Number of neighbors per node."""
    return 5


@pytest.fixture
def node_dim():
    """Node feature dimension."""
    return 32


@pytest.fixture
def edge_dim():
    """Edge feature dimension."""
    return 16


@pytest.fixture
def hidden_dim():
    """Hidden dimension for layers."""
    return 64


@pytest.fixture
def node_features(batch_size, num_nodes, node_dim):
    """Create random node features."""
    return torch.randn(batch_size, num_nodes, node_dim)


@pytest.fixture
def edge_features(batch_size, num_nodes, num_neighbors, edge_dim):
    """Create random edge features."""
    return torch.randn(batch_size, num_nodes, num_neighbors, edge_dim)


@pytest.fixture
def edge_index(batch_size, num_nodes, num_neighbors):
    """Create random edge indices."""
    # Create valid edge indices (ensuring they point to valid nodes)
    return torch.randint(0, num_nodes, (batch_size, num_nodes, num_neighbors))


@pytest.fixture
def node_mask(batch_size, num_nodes):
    """Create node mask (some nodes are valid, some are padding)."""
    mask = torch.ones(batch_size, num_nodes)
    # Mask out last 2 nodes in each batch
    mask[:, -2:] = 0.0
    return mask


@pytest.fixture
def edge_mask(batch_size, num_nodes, num_neighbors):
    """Create edge mask (some edges are valid, some are padding)."""
    mask = torch.ones(batch_size, num_nodes, num_neighbors)
    # Mask out last neighbor for each node
    mask[:, :, -1] = 0.0
    return mask


class TestBaseMPNNLayer:
    """Tests for BaseMPNNLayer."""

    def test_gather_and_concat_neighbourhood(
        self,
        batch_size,
        num_nodes,
        num_neighbors,
        node_dim,
        edge_dim,
        hidden_dim,
    ):
        """Test gather_and_concat_neighbourhood method."""
        # Create a concrete implementation for testing
        layer = MPNNEncoderLayer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            # hidden_dim=hidden_dim,
        )

        node_emb = torch.randn(batch_size, num_nodes, node_dim)
        edge_emb = torch.randn(batch_size, num_nodes, num_neighbors, edge_dim)
        edge_index = torch.randint(0, num_nodes, (batch_size, num_nodes, num_neighbors))

        result = layer.gather_and_concat_neighbourhood(
            node_emb=node_emb,
            edge_emb=edge_emb,
            edge_index=edge_index,
        )

        # Check output shape: should be [..., nodes, neighbors, 2*node_dim + edge_dim]
        expected_shape = (batch_size, num_nodes, num_neighbors, 2 * node_dim + edge_dim)
        assert result.shape == expected_shape


class TestMPNNEncoderLayer:
    """Tests for MPNNEncoderLayer."""

    def test_masking_zeros_out_invalid_nodes(
        self,
        node_features,
        edge_features,
        edge_index,
        node_mask,
        edge_mask,
        node_dim,
        edge_dim,
        hidden_dim,
    ):
        """Test that masking properly zeros out invalid nodes."""
        layer = MPNNEncoderLayer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            # hidden_dim=hidden_dim,
        )

        node_out, edge_out = layer(
            node_emb=node_features,
            edge_emb=edge_features,
            edge_index=edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        # Check that masked nodes are zero
        # node_mask[:, -2:] = 0, so those nodes should be zeroed
        assert torch.allclose(
            node_out[:, -2:, :], torch.zeros_like(node_out[:, -2:, :])
        )

        # Check that masked edges are zero
        # edge_mask[:, :, -1] = 0, so those edges should be zeroed
        assert torch.allclose(
            edge_out[:, :, -1, :], torch.zeros_like(edge_out[:, :, -1, :])
        )


class TestMPNNDecoderLayer:
    """Tests for MPNNDecoderLayer."""

    def test_masking_zeros_out_invalid_nodes(
        self,
        batch_size,
        num_nodes,
        num_neighbors,
        hidden_dim,
    ):
        """Test that masking properly zeros out invalid nodes."""
        layer = MPNNDecoderLayer(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            edge_input_dim=hidden_dim,
        )

        node_features = torch.randn(batch_size, num_nodes, hidden_dim)
        edge_features = torch.randn(batch_size, num_nodes, num_neighbors, hidden_dim)
        node_mask = torch.ones(batch_size, num_nodes)
        node_mask[:, -2:] = 0.0  # Mask out last 2 nodes
        edge_mask = torch.ones(batch_size, num_nodes, num_neighbors)

        node_out = layer(
            node_emb=node_features,
            edge_emb=edge_features,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        # Check that masked nodes are zero
        assert torch.allclose(
            node_out[:, -2:, :], torch.zeros_like(node_out[:, -2:, :])
        )

    def test_edge_mask_effect(
        self,
        batch_size,
        num_nodes,
        num_neighbors,
        hidden_dim,
    ):
        """Test that edge masking affects the output."""
        layer = MPNNDecoderLayer(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            edge_input_dim=hidden_dim,
        )

        node_features = torch.randn(batch_size, num_nodes, hidden_dim)
        edge_features = torch.randn(batch_size, num_nodes, num_neighbors, hidden_dim)

        # Run with full edge mask
        edge_mask_full = torch.ones(batch_size, num_nodes, num_neighbors)
        node_out_full = layer(
            node_emb=node_features,
            edge_emb=edge_features,
            edge_mask=edge_mask_full,
        )

        # Run with partial edge mask
        edge_mask_partial = edge_mask_full.clone()
        edge_mask_partial[:, :, -1] = 0.0  # Mask out last neighbor
        node_out_partial = layer(
            node_emb=node_features,
            edge_emb=edge_features,
            edge_mask=edge_mask_partial,
        )

        # Outputs should be different
        assert not torch.allclose(node_out_full, node_out_partial)
