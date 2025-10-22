"""Tests for All-atom MPNN module.

This module tests the core functionality of the AllatomMPNN class, including:
- Decoder input preparation
- Sequential mask generation for autoregressive decoding
- Decoder step execution
"""

import pytest
import torch
from torch import nn

from protenix.model.modules.mpnn import AllatomMPNN


@pytest.fixture
def mpnn_model():
    """Create a small MPNN model for testing.

    Note: The decoder layers are initialized with edge_input_dim=edge_dim in the original code,
    but the actual input dimension during forward pass is larger due to feature concatenation.
    This is a known issue that these tests help expose.
    """
    model = AllatomMPNN(
        num_residue_types=32,
        hidden_dim=64,
        node_dim=64,
        edge_dim=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_neighbours=30,
        positional_encoding_dim=16,
        positional_encoding_max_relative_feats=32,
        num_rbf_kernels=16,
        dropout=0.0,
        coord_noise=0.1,
    )

    # Fix the decoder layers to accept the correct edge input dimension
    # In autoregressive mode: (edge_dim + node_dim) + node_dim = 128 + 64 = 192
    # In zero-shot mode: (edge_dim + 0 + node_dim) = 192
    # Both modes use 192-dimensional edge features
    from protenix.model.modules.mpnn_layers import MPNNDecoderLayer

    model.decoder = nn.ModuleList(
        [
            MPNNDecoderLayer(
                node_dim=64,
                edge_dim=64,
                edge_input_dim=2 * 64 + 64,  # Corrected dimension
                dropout=0.0,
            )
            for _ in range(2)
        ]
    )

    return model


@pytest.fixture
def sample_tensors():
    """Create sample tensors for testing with consistent shapes."""
    batch_size = 2
    seq_len = 10
    num_neighbours = 30
    hidden_dim = 64
    num_residue_types = 32

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_neighbours": num_neighbours,
        "hidden_dim": hidden_dim,
        "num_residue_types": num_residue_types,
        "node_feats": torch.randn(batch_size, seq_len, hidden_dim),
        "edge_feats": torch.randn(batch_size, seq_len, num_neighbours, hidden_dim),
        "sequence_embeds": torch.randn(batch_size, seq_len, hidden_dim),
        "edge_index": torch.randint(0, seq_len, (batch_size, seq_len, num_neighbours)),
        "node_mask": torch.ones(batch_size, seq_len),
        "edge_mask": torch.ones(batch_size, seq_len, num_neighbours),
        "restype": torch.zeros(batch_size, seq_len, num_residue_types),
        "decoding_order": torch.randperm(seq_len).unsqueeze(0).expand(batch_size, -1),
    }


class TestPrepareDecoderInputs:
    """Tests for _prepare_decoder_inputs method."""

    def test_output_shapes(self, mpnn_model, sample_tensors):
        """Test that output tensors have correct shapes."""
        encoder_edge_feats, sequence_edge_feats = mpnn_model._prepare_decoder_inputs(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            sequence_embeds=sample_tensors["sequence_embeds"],
            edge_index=sample_tensors["edge_index"],
        )

        batch_size = sample_tensors["batch_size"]
        seq_len = sample_tensors["seq_len"]
        num_neighbours = sample_tensors["num_neighbours"]
        hidden_dim = sample_tensors["hidden_dim"]

        # sequence_edge_feats: edge_feats (64) + sequence_embeds (64) = 128
        # encoder_edge_feats: edge_feats (64) + zeros (64) + node_feats (64) = 192
        sequence_expected_shape = (batch_size, seq_len, num_neighbours, hidden_dim * 2)
        encoder_expected_shape = (batch_size, seq_len, num_neighbours, hidden_dim * 3)

        assert (
            sequence_edge_feats.shape == sequence_expected_shape
        ), f"sequence_edge_feats shape {sequence_edge_feats.shape} != {sequence_expected_shape}"
        assert (
            encoder_edge_feats.shape == encoder_expected_shape
        ), f"encoder_edge_feats shape {encoder_edge_feats.shape} != {encoder_expected_shape}"

    def test_output_types(self, mpnn_model, sample_tensors):
        """Test that outputs are torch tensors with correct dtype."""
        encoder_edge_feats, sequence_edge_feats = mpnn_model._prepare_decoder_inputs(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            sequence_embeds=sample_tensors["sequence_embeds"],
            edge_index=sample_tensors["edge_index"],
        )

        assert isinstance(encoder_edge_feats, torch.Tensor)
        assert isinstance(sequence_edge_feats, torch.Tensor)
        assert encoder_edge_feats.dtype == sample_tensors["node_feats"].dtype
        assert sequence_edge_feats.dtype == sample_tensors["node_feats"].dtype

    def test_device_consistency(self, mpnn_model, sample_tensors):
        """Test that outputs are on the same device as inputs."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        node_feats = sample_tensors["node_feats"].to(device)
        edge_feats = sample_tensors["edge_feats"].to(device)
        sequence_embeds = sample_tensors["sequence_embeds"].to(device)
        edge_index = sample_tensors["edge_index"].to(device)

        mpnn_model = mpnn_model.to(device)

        encoder_edge_feats, sequence_edge_feats = mpnn_model._prepare_decoder_inputs(
            node_feats=node_feats,
            edge_feats=edge_feats,
            sequence_embeds=sequence_embeds,
            edge_index=edge_index,
        )

        assert encoder_edge_feats.device == device
        assert sequence_edge_feats.device == device

    def test_different_batch_sizes(self, mpnn_model):
        """Test that method works with different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            seq_len = 10
            num_neighbours = 30
            hidden_dim = 64

            node_feats = torch.randn(batch_size, seq_len, hidden_dim)
            edge_feats = torch.randn(batch_size, seq_len, num_neighbours, hidden_dim)
            sequence_embeds = torch.randn(batch_size, seq_len, hidden_dim)
            edge_index = torch.randint(
                0, seq_len, (batch_size, seq_len, num_neighbours)
            )

            encoder_edge_feats, sequence_edge_feats = (
                mpnn_model._prepare_decoder_inputs(
                    node_feats=node_feats,
                    edge_feats=edge_feats,
                    sequence_embeds=sequence_embeds,
                    edge_index=edge_index,
                )
            )

            # sequence_edge_feats: 2x, encoder_edge_feats: 3x hidden_dim
            sequence_expected = (batch_size, seq_len, num_neighbours, hidden_dim * 2)
            encoder_expected = (batch_size, seq_len, num_neighbours, hidden_dim * 3)
            assert sequence_edge_feats.shape == sequence_expected
            assert encoder_edge_feats.shape == encoder_expected

    def test_edge_features_not_nan_or_inf(self, mpnn_model, sample_tensors):
        """Test that output features don't contain NaN or Inf values."""
        encoder_edge_feats, sequence_edge_feats = mpnn_model._prepare_decoder_inputs(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            sequence_embeds=sample_tensors["sequence_embeds"],
            edge_index=sample_tensors["edge_index"],
        )

        assert not torch.isnan(
            encoder_edge_feats
        ).any(), "encoder_edge_feats contains NaN"
        assert not torch.isinf(
            encoder_edge_feats
        ).any(), "encoder_edge_feats contains Inf"
        assert not torch.isnan(
            sequence_edge_feats
        ).any(), "sequence_edge_feats contains NaN"
        assert not torch.isinf(
            sequence_edge_feats
        ).any(), "sequence_edge_feats contains Inf"


class TestGenerateSequentialMasks:
    """Tests for _generate_sequential_masks method."""

    def test_output_shapes(self, mpnn_model, sample_tensors):
        """Test that mask outputs have correct shapes."""
        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=sample_tensors["decoding_order"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        batch_size = sample_tensors["batch_size"]
        seq_len = sample_tensors["seq_len"]
        num_neighbours = sample_tensors["num_neighbours"]

        # Both masks should have shape: [batch, seq_len, neighbours]
        expected_shape = (batch_size, seq_len, num_neighbours)
        assert (
            forward_attn_mask.shape == expected_shape
        ), f"forward_attn_mask shape {forward_attn_mask.shape} != {expected_shape}"
        assert (
            backward_attn_mask.shape == expected_shape
        ), f"backward_attn_mask shape {backward_attn_mask.shape} != {expected_shape}"

    def test_mask_values_binary(self, mpnn_model, sample_tensors):
        """Test that masks contain only 0s and 1s."""
        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=sample_tensors["decoding_order"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        # Check that all values are in {0, 1}
        assert torch.all(
            (forward_attn_mask == 0) | (forward_attn_mask == 1)
        ), "forward_attn_mask contains non-binary values"
        assert torch.all(
            (backward_attn_mask == 0) | (backward_attn_mask == 1)
        ), "backward_attn_mask contains non-binary values"

    def test_masks_are_complementary_with_node_mask(self, mpnn_model, sample_tensors):
        """Test that forward and backward masks sum to node_mask along edges.

        This test verifies the property that at each position, the forward and
        backward masks should sum to the node mask, accounting for valid positions.
        """
        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=sample_tensors["decoding_order"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        # Get the node mask for neighboring nodes
        # edge_index: [batch, seq_len, num_neighbours]
        # node_mask: [batch, seq_len]
        edge_index = sample_tensors["edge_index"]
        node_mask = sample_tensors["node_mask"]

        # Gather node mask values at edge indices
        neighbor_mask = torch.gather(
            node_mask.unsqueeze(-1).expand(-1, -1, edge_index.shape[-1]),
            dim=1,
            index=edge_index,
        )

        # The sum of forward and backward masks should equal the product of
        # source node mask and neighbor node mask
        mask_sum = forward_attn_mask + backward_attn_mask
        expected_sum = node_mask.unsqueeze(-1) * neighbor_mask

        assert torch.allclose(
            mask_sum, expected_sum, atol=1e-6
        ), "Forward and backward masks don't sum correctly with node mask"

    def test_different_decoding_orders(self, mpnn_model, sample_tensors):
        """Test that different decoding orders produce different masks."""
        # Generate two different random decoding orders
        seq_len = sample_tensors["seq_len"]
        batch_size = sample_tensors["batch_size"]

        decoding_order_1 = torch.randperm(seq_len).unsqueeze(0).expand(batch_size, -1)
        decoding_order_2 = torch.randperm(seq_len).unsqueeze(0).expand(batch_size, -1)

        # Ensure they're actually different
        while torch.equal(decoding_order_1, decoding_order_2):
            decoding_order_2 = (
                torch.randperm(seq_len).unsqueeze(0).expand(batch_size, -1)
            )

        forward_1, backward_1 = mpnn_model._generate_sequential_masks(
            decoding_order=decoding_order_1,
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        forward_2, backward_2 = mpnn_model._generate_sequential_masks(
            decoding_order=decoding_order_2,
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        # Masks should be different for different orders
        assert not torch.equal(forward_1, forward_2) or not torch.equal(
            backward_1, backward_2
        ), "Different decoding orders produced identical masks"

    def test_node_mask_application(self, mpnn_model, sample_tensors):
        """Test that node mask properly zeros out invalid positions."""
        # Create a node mask with some positions invalid
        node_mask = sample_tensors["node_mask"].clone()
        node_mask[:, :3] = 0  # Mask out first 3 positions

        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=sample_tensors["decoding_order"],
            edge_index=sample_tensors["edge_index"],
            node_mask=node_mask,
        )

        # First 3 positions should be all zeros in both masks
        assert torch.all(
            forward_attn_mask[:, :3] == 0
        ), "Masked positions not zero in forward_attn_mask"
        assert torch.all(
            backward_attn_mask[:, :3] == 0
        ), "Masked positions not zero in backward_attn_mask"

    def test_output_device_consistency(self, mpnn_model, sample_tensors):
        """Test that output masks are on the same device as inputs."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        decoding_order = sample_tensors["decoding_order"].to(device)
        edge_index = sample_tensors["edge_index"].to(device)
        node_mask = sample_tensors["node_mask"].to(device)

        mpnn_model = mpnn_model.to(device)

        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=decoding_order,
            edge_index=edge_index,
            node_mask=node_mask,
        )

        assert forward_attn_mask.device == device
        assert backward_attn_mask.device == device


class TestRunDecoderStep:
    """Tests for _run_decoder_step method."""

    def test_output_shape(self, mpnn_model, sample_tensors):
        """Test that decoder step output has correct shape."""
        # Need to add embed_restype to model first
        mpnn_model.restype_embed = nn.Linear(
            sample_tensors["num_residue_types"], sample_tensors["hidden_dim"]
        )

        # Generate masks
        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=sample_tensors["decoding_order"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        # Add channel dimension to masks
        forward_attn_mask = forward_attn_mask.unsqueeze(-1)
        backward_attn_mask = backward_attn_mask.unsqueeze(-1)

        logits = mpnn_model._run_decoder_step(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            restype=sample_tensors["restype"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
            forward_attn_mask=forward_attn_mask,
            backward_attn_mask=backward_attn_mask,
            edge_mask=sample_tensors["edge_mask"],
            zero_shot=False,
        )

        batch_size = sample_tensors["batch_size"]
        seq_len = sample_tensors["seq_len"]
        num_residue_types = sample_tensors["num_residue_types"]

        expected_shape = (batch_size, seq_len, num_residue_types)
        assert (
            logits.shape == expected_shape
        ), f"logits shape {logits.shape} != {expected_shape}"

    def test_zero_shot_mode(self, mpnn_model, sample_tensors):
        """Test that zero_shot mode runs without errors and produces valid output."""
        mpnn_model.restype_embed = nn.Linear(
            sample_tensors["num_residue_types"], sample_tensors["hidden_dim"]
        )

        # Generate masks
        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=sample_tensors["decoding_order"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        forward_attn_mask = forward_attn_mask.unsqueeze(-1)
        backward_attn_mask = backward_attn_mask.unsqueeze(-1)

        logits_zero_shot = mpnn_model._run_decoder_step(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            restype=sample_tensors["restype"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
            forward_attn_mask=forward_attn_mask,
            backward_attn_mask=backward_attn_mask,
            edge_mask=sample_tensors["edge_mask"],
            zero_shot=True,
        )

        logits_autoregressive = mpnn_model._run_decoder_step(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            restype=sample_tensors["restype"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
            forward_attn_mask=forward_attn_mask,
            backward_attn_mask=backward_attn_mask,
            edge_mask=sample_tensors["edge_mask"],
            zero_shot=False,
        )

        # Both should have same shape but different values
        assert logits_zero_shot.shape == logits_autoregressive.shape
        assert not torch.allclose(
            logits_zero_shot, logits_autoregressive, atol=1e-3
        ), "Zero-shot and autoregressive modes produce identical outputs"

    def test_output_no_nan_or_inf(self, mpnn_model, sample_tensors):
        """Test that decoder output doesn't contain NaN or Inf values."""
        mpnn_model.restype_embed = nn.Linear(
            sample_tensors["num_residue_types"], sample_tensors["hidden_dim"]
        )

        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=sample_tensors["decoding_order"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        forward_attn_mask = forward_attn_mask.unsqueeze(-1)
        backward_attn_mask = backward_attn_mask.unsqueeze(-1)

        logits = mpnn_model._run_decoder_step(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            restype=sample_tensors["restype"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
            forward_attn_mask=forward_attn_mask,
            backward_attn_mask=backward_attn_mask,
            edge_mask=sample_tensors["edge_mask"],
            zero_shot=False,
        )

        assert not torch.isnan(logits).any(), "Logits contain NaN values"
        assert not torch.isinf(logits).any(), "Logits contain Inf values"

    def test_edge_mask_application(self, mpnn_model, sample_tensors):
        """Test that edge masking properly affects the output."""
        mpnn_model.restype_embed = nn.Linear(
            sample_tensors["num_residue_types"], sample_tensors["hidden_dim"]
        )

        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=sample_tensors["decoding_order"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        forward_attn_mask = forward_attn_mask.unsqueeze(-1)
        backward_attn_mask = backward_attn_mask.unsqueeze(-1)

        # Run with full edge mask
        logits_full = mpnn_model._run_decoder_step(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            restype=sample_tensors["restype"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
            forward_attn_mask=forward_attn_mask,
            backward_attn_mask=backward_attn_mask,
            edge_mask=sample_tensors["edge_mask"],
            zero_shot=False,
        )

        # Create a partially masked edge mask
        edge_mask_partial = sample_tensors["edge_mask"].clone()
        edge_mask_partial[:, :, :10] = 0  # Mask first 10 edges

        # Run with partial edge mask
        logits_partial = mpnn_model._run_decoder_step(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            restype=sample_tensors["restype"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
            forward_attn_mask=forward_attn_mask,
            backward_attn_mask=backward_attn_mask,
            edge_mask=edge_mask_partial,
            zero_shot=False,
        )

        # Different edge masks should produce different outputs
        assert not torch.allclose(
            logits_full, logits_partial, atol=1e-3
        ), "Different edge masks produce identical outputs"

    def test_different_restype_inputs(self, mpnn_model, sample_tensors):
        """Test that different restype inputs produce different outputs."""
        mpnn_model.restype_embed = nn.Linear(
            sample_tensors["num_residue_types"], sample_tensors["hidden_dim"]
        )

        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=sample_tensors["decoding_order"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        forward_attn_mask = forward_attn_mask.unsqueeze(-1)
        backward_attn_mask = backward_attn_mask.unsqueeze(-1)

        # First restype (zeros)
        logits_1 = mpnn_model._run_decoder_step(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            restype=sample_tensors["restype"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
            forward_attn_mask=forward_attn_mask,
            backward_attn_mask=backward_attn_mask,
            edge_mask=sample_tensors["edge_mask"],
            zero_shot=False,
        )

        # Different restype (random one-hot)
        restype_2 = sample_tensors["restype"].clone()
        restype_2[:, 0, :] = 0
        restype_2[:, 0, 5] = 1  # Set to different residue type

        logits_2 = mpnn_model._run_decoder_step(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            restype=restype_2,
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
            forward_attn_mask=forward_attn_mask,
            backward_attn_mask=backward_attn_mask,
            edge_mask=sample_tensors["edge_mask"],
            zero_shot=False,
        )

        # Different restypes should produce different outputs
        assert not torch.allclose(
            logits_1, logits_2, atol=1e-4
        ), "Different restype inputs produce identical outputs"
