"""Tests for MPNN layers."""

import pytest
import torch
from torch import nn

from artios.model.modules.mpnn_layers import (
    BaseMPNNLayer,
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

    def test_abstract_class(self):
        """Test that BaseMPNNLayer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMPNNLayer()

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

    def test_gather_and_concat_neighbourhood_no_batch(
        self,
        num_nodes,
        num_neighbors,
        node_dim,
        edge_dim,
        hidden_dim,
    ):
        """Test gather_and_concat_neighbourhood without batch dimension.

        Note: The underlying gather_nodes function requires a batch dimension,
        so we add it and then remove it to test the single-sample case.
        """
        layer = MPNNEncoderLayer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            # hidden_dim=hidden_dim,
        )

        # Add batch dimension of 1 since gather_nodes requires it
        node_emb = torch.randn(1, num_nodes, node_dim)
        edge_emb = torch.randn(1, num_nodes, num_neighbors, edge_dim)
        edge_index = torch.randint(0, num_nodes, (1, num_nodes, num_neighbors))

        result = layer.gather_and_concat_neighbourhood(
            node_emb=node_emb,
            edge_emb=edge_emb,
            edge_index=edge_index,
        )

        expected_shape = (1, num_nodes, num_neighbors, 2 * node_dim + edge_dim)
        assert result.shape == expected_shape


class TestMPNNEncoderLayer:
    """Tests for MPNNEncoderLayer."""

    def test_initialization(self, node_dim, edge_dim, hidden_dim):
        """Test layer initialization."""
        layer = MPNNEncoderLayer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            dropout=0.1,
        )

        assert layer.node_dim == node_dim
        assert layer.edge_dim == edge_dim
        assert isinstance(layer.dropout1, nn.Dropout)
        assert layer.dropout1.p == 0.1

    def test_forward_pass(
        self,
        node_features,
        edge_features,
        edge_index,
        node_mask,
        edge_mask,
        node_dim,
        edge_dim,
        hidden_dim,
        batch_size,
        num_nodes,
        num_neighbors,
    ):
        """Test forward pass with all inputs."""
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

        # Check output shapes
        assert node_out.shape == (batch_size, num_nodes, node_dim)
        assert edge_out.shape == (batch_size, num_nodes, num_neighbors, edge_dim)

    def test_forward_pass_no_mask(
        self,
        node_features,
        edge_features,
        edge_index,
        node_dim,
        edge_dim,
        hidden_dim,
        batch_size,
        num_nodes,
        num_neighbors,
    ):
        """Test forward pass without masks."""
        layer = MPNNEncoderLayer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            # hidden_dim=hidden_dim,
        )

        node_out, edge_out = layer(
            node_emb=node_features,
            edge_emb=edge_features,
            edge_index=edge_index,
            node_mask=None,
            edge_mask=None,
        )

        # Check output shapes
        assert node_out.shape == (batch_size, num_nodes, node_dim)
        assert edge_out.shape == (batch_size, num_nodes, num_neighbors, edge_dim)

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

    def test_residual_connections(
        self,
        node_features,
        edge_features,
        edge_index,
        node_dim,
        edge_dim,
        hidden_dim,
    ):
        """Test that residual connections are working."""
        layer = MPNNEncoderLayer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            # hidden_dim=hidden_dim,
        )

        # With zero dropout, the output should be different from input
        node_out, edge_out = layer(
            node_emb=node_features,
            edge_emb=edge_features,
            edge_index=edge_index,
        )

        # Outputs should not be identical to inputs (due to residual + updates)
        assert not torch.allclose(node_out, node_features)
        assert not torch.allclose(edge_out, edge_features)

    def test_single_batch_dimension(
        self,
        num_nodes,
        num_neighbors,
        node_dim,
        edge_dim,
        hidden_dim,
    ):
        """Test forward pass with single batch (batch_size=1).

        Note: The underlying gather_nodes function requires a batch dimension,
        so we test with batch_size=1 instead of no batch dimension.
        """
        layer = MPNNEncoderLayer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            # hidden_dim=hidden_dim,
        )

        node_features = torch.randn(1, num_nodes, node_dim)
        edge_features = torch.randn(1, num_nodes, num_neighbors, edge_dim)
        edge_index = torch.randint(0, num_nodes, (1, num_nodes, num_neighbors))

        node_out, edge_out = layer(
            node_emb=node_features,
            edge_emb=edge_features,
            edge_index=edge_index,
        )

        assert node_out.shape == (1, num_nodes, node_dim)
        assert edge_out.shape == (1, num_nodes, num_neighbors, edge_dim)

    def test_gradient_flow(
        self,
        node_features,
        edge_features,
        edge_index,
        node_dim,
        edge_dim,
        hidden_dim,
    ):
        """Test that gradients flow through the layer."""
        layer = MPNNEncoderLayer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            # hidden_dim=hidden_dim,
        )

        node_features = node_features.clone().requires_grad_(True)
        edge_features = edge_features.clone().requires_grad_(True)

        node_out, edge_out = layer(
            node_emb=node_features,
            edge_emb=edge_features,
            edge_index=edge_index,
        )

        # Compute a simple loss
        loss = node_out.sum() + edge_out.sum()
        loss.backward()

        # Check that gradients exist
        assert node_features.grad is not None
        assert edge_features.grad is not None
        assert not torch.allclose(
            node_features.grad, torch.zeros_like(node_features.grad)
        )
        assert not torch.allclose(
            edge_features.grad, torch.zeros_like(edge_features.grad)
        )


class TestMPNNDecoderLayer:
    """Tests for MPNNDecoderLayer."""

    def test_initialization(self, hidden_dim):
        """Test layer initialization."""
        layer = MPNNDecoderLayer(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            edge_input_dim=hidden_dim * 2,
            dropout=0.1,
        )

        assert layer.node_dim == hidden_dim
        assert isinstance(layer.dropout1, nn.Dropout)
        assert layer.dropout1.p == 0.1

    def test_forward_pass(
        self,
        batch_size,
        num_nodes,
        num_neighbors,
        hidden_dim,
    ):
        """Test forward pass with all inputs."""
        edge_input_dim = hidden_dim * 3
        layer = MPNNDecoderLayer(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            edge_input_dim=edge_input_dim,
        )

        # Decoder uses hidden_dim for nodes
        node_features = torch.randn(batch_size, num_nodes, hidden_dim)
        # Edge features have edge_input_dim dimension
        edge_features = torch.randn(
            batch_size, num_nodes, num_neighbors, edge_input_dim
        )
        node_mask = torch.ones(batch_size, num_nodes)
        edge_mask = torch.ones(batch_size, num_nodes, num_neighbors)

        node_out = layer(
            node_emb=node_features,
            edge_emb=edge_features,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        # Check output shape - only returns updated nodes
        assert node_out.shape == (batch_size, num_nodes, hidden_dim)

    def test_forward_pass_no_mask(
        self,
        batch_size,
        num_nodes,
        num_neighbors,
        hidden_dim,
    ):
        """Test forward pass without masks."""
        layer = MPNNDecoderLayer(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            edge_input_dim=hidden_dim,
        )

        node_features = torch.randn(batch_size, num_nodes, hidden_dim)
        edge_features = torch.randn(batch_size, num_nodes, num_neighbors, hidden_dim)

        node_out = layer(
            node_emb=node_features,
            edge_emb=edge_features,
            node_mask=None,
            edge_mask=None,
        )

        assert node_out.shape == (batch_size, num_nodes, hidden_dim)

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

    def test_single_batch_dimension(
        self,
        num_nodes,
        num_neighbors,
        hidden_dim,
    ):
        """Test forward pass with single batch (batch_size=1)."""
        layer = MPNNDecoderLayer(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            edge_input_dim=hidden_dim,
        )

        node_features = torch.randn(1, num_nodes, hidden_dim)
        edge_features = torch.randn(1, num_nodes, num_neighbors, hidden_dim)

        node_out = layer(
            node_emb=node_features,
            edge_emb=edge_features,
        )

        assert node_out.shape == (1, num_nodes, hidden_dim)

    def test_gradient_flow(
        self,
        batch_size,
        num_nodes,
        num_neighbors,
        hidden_dim,
    ):
        """Test that gradients flow through the layer."""
        layer = MPNNDecoderLayer(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            edge_input_dim=hidden_dim,
        )

        node_features = torch.randn(
            batch_size, num_nodes, hidden_dim, requires_grad=True
        )
        edge_features = torch.randn(
            batch_size, num_nodes, num_neighbors, hidden_dim, requires_grad=True
        )

        node_out = layer(
            node_emb=node_features,
            edge_emb=edge_features,
        )

        # Compute a simple loss
        loss = node_out.sum()
        loss.backward()

        # Check that node gradients exist and are non-zero
        assert node_features.grad is not None
        assert not torch.allclose(
            node_features.grad, torch.zeros_like(node_features.grad)
        )

    def test_residual_connections(
        self,
        batch_size,
        num_nodes,
        num_neighbors,
        hidden_dim,
    ):
        """Test that residual connections are working."""
        layer = MPNNDecoderLayer(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            edge_input_dim=hidden_dim,
        )

        node_features = torch.randn(batch_size, num_nodes, hidden_dim)
        edge_features = torch.randn(batch_size, num_nodes, num_neighbors, hidden_dim)

        node_out = layer(
            node_emb=node_features,
            edge_emb=edge_features,
        )

        # Output should be different from input (residual + updates)
        assert not torch.allclose(node_out, node_features)


@pytest.mark.parametrize("dropout", [0.0, 0.1, 0.5])
def test_dropout_effect(
    dropout, batch_size, num_nodes, num_neighbors, node_dim, edge_dim
):
    """Test that dropout has an effect during training."""
    layer = MPNNEncoderLayer(
        node_dim=node_dim,
        edge_dim=edge_dim,
        dropout=dropout,
    )

    node_features = torch.randn(batch_size, num_nodes, node_dim)
    edge_features = torch.randn(batch_size, num_nodes, num_neighbors, edge_dim)
    edge_index = torch.randint(0, num_nodes, (batch_size, num_nodes, num_neighbors))

    # Training mode
    layer.train()
    node_out_1, edge_out_1 = layer(node_features, edge_features, edge_index)
    node_out_2, edge_out_2 = layer(node_features, edge_features, edge_index)

    # With dropout > 0, outputs should be different in training mode
    if dropout > 0:
        assert not torch.allclose(node_out_1, node_out_2)
        assert not torch.allclose(edge_out_1, edge_out_2)

    # Eval mode - outputs should be deterministic
    layer.eval()
    node_out_3, edge_out_3 = layer(node_features, edge_features, edge_index)
    node_out_4, edge_out_4 = layer(node_features, edge_features, edge_index)

    assert torch.allclose(node_out_3, node_out_4)
    assert torch.allclose(edge_out_3, edge_out_4)
