"""Tests for the refactored MPNN module.

This module tests the core functionality of the MPNN class, including:
- Encoding (structure → node/edge features)
- Decoding mask generation
- Decoder step execution
- Forward pass (training with teacher forcing)
- Sampling (autoregressive inference)
"""

import pytest
import torch

from protenix.model.modules.mpnn import MPNN, DecodingMasks
from tests.context_utils import create_test_context


@pytest.fixture
def mpnn_model():
    """Create a small MPNN model for testing."""
    return MPNN(
        num_residue_types=32,
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


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def full_context(device) -> "FullContext":
    """Create batched FullContext for testing."""
    return create_test_context(
        batch_size=2,
        n_tokens=32,
        n_atoms=64,
        device=device,
    )


@pytest.fixture
def single_context(device) -> "FullContext":
    """Create single-sample FullContext for testing."""
    return create_test_context(
        batch_size=1,
        n_tokens=32,
        n_atoms=64,
        device=device,
    )


class TestEncoderOutput:
    """Tests for the EncoderOutput dataclass."""

    def test_encoder_output_fields(self, mpnn_model, single_context):
        """Test that EncoderOutput has all expected fields."""
        encoder_output = mpnn_model.encode(single_context)

        assert hasattr(encoder_output, "node_feats")
        assert hasattr(encoder_output, "edge_feats")
        assert hasattr(encoder_output, "edge_index")
        assert hasattr(encoder_output, "edge_mask")

    def test_encoder_output_types(self, mpnn_model, single_context):
        """Test that EncoderOutput fields are tensors."""
        encoder_output = mpnn_model.encode(single_context)

        assert isinstance(encoder_output.node_feats, torch.Tensor)
        assert isinstance(encoder_output.edge_feats, torch.Tensor)
        assert isinstance(encoder_output.edge_index, torch.Tensor)
        assert isinstance(encoder_output.edge_mask, torch.Tensor)


class TestEncode:
    """Tests for the encode() method."""

    def test_output_shapes(self, mpnn_model, single_context):
        """Test that encoder outputs have correct shapes."""
        n_tokens = single_context.token.token_index.shape[-1]
        num_neighbours = mpnn_model.num_neighbours
        node_dim = mpnn_model.node_dim
        edge_dim = mpnn_model.edge_dim
        batch_size = 1

        encoder_output = mpnn_model.encode(single_context)

        # node_feats: [batch, n_token, node_dim]
        assert encoder_output.node_feats.shape == (batch_size, n_tokens, node_dim)

        # edge_feats: [batch, n_token, neighbours, edge_dim]
        assert encoder_output.edge_feats.shape == (
            batch_size,
            n_tokens,
            num_neighbours,
            edge_dim,
        )

        # edge_index: [batch, n_token, neighbours]
        assert encoder_output.edge_index.shape == (
            batch_size,
            n_tokens,
            num_neighbours,
        )

        # edge_mask: [batch, n_token, neighbours]
        assert encoder_output.edge_mask.shape == (
            batch_size,
            n_tokens,
            num_neighbours,
        )

    def test_output_no_nan_or_inf(self, mpnn_model, single_context):
        """Test that encoder outputs don't contain NaN or Inf values."""
        encoder_output = mpnn_model.encode(single_context)

        assert not torch.isnan(
            encoder_output.node_feats
        ).any(), "node_feats contains NaN"
        assert not torch.isinf(
            encoder_output.node_feats
        ).any(), "node_feats contains Inf"
        assert not torch.isnan(
            encoder_output.edge_feats
        ).any(), "edge_feats contains NaN"
        assert not torch.isinf(
            encoder_output.edge_feats
        ).any(), "edge_feats contains Inf"

    def test_edge_index_validity(self, mpnn_model, single_context):
        """Test that edge indices are valid (in range [0, n_token))."""
        n_tokens = single_context.token.token_index.shape[-1]
        encoder_output = mpnn_model.encode(single_context)

        assert torch.all(
            encoder_output.edge_index >= 0
        ), "edge_index contains negative values"
        assert torch.all(
            encoder_output.edge_index < n_tokens
        ), f"edge_index contains values >= n_tokens ({n_tokens})"

    def test_edge_mask_reflects_token_mask(self, mpnn_model, single_context):
        """Test that edge mask properly reflects token masking."""
        encoder_output = mpnn_model.encode(single_context)
        token_mask = single_context.mask.token_mask

        # Edges from masked tokens should be masked
        masked_token_indices = torch.where(~token_mask[0])[0]
        if len(masked_token_indices) > 0:
            for idx in masked_token_indices[:5]:  # Check first 5 masked tokens
                assert torch.all(
                    encoder_output.edge_mask[0, idx] == 0
                ), f"Edge mask not zero for masked token {idx}"

    def test_output_values_reasonable(self, mpnn_model, single_context):
        """Test that output values are within reasonable ranges.

        This helps catch masking bugs that could produce large values.
        """
        encoder_output = mpnn_model.encode(single_context)

        assert torch.all(
            torch.abs(encoder_output.node_feats) < 1e6
        ), "node_feats contains unreasonably large values - possible masking bug"
        assert torch.all(
            torch.abs(encoder_output.edge_feats) < 1e6
        ), "edge_feats contains unreasonably large values - possible masking bug"

    def test_device_consistency(self, mpnn_model, single_context, device):
        """Test that outputs are on the same device as inputs."""
        mpnn_model = mpnn_model.to(device)

        encoder_output = mpnn_model.encode(single_context)

        assert encoder_output.node_feats.device == device
        assert encoder_output.edge_feats.device == device
        assert encoder_output.edge_index.device == device
        assert encoder_output.edge_mask.device == device


class TestGenerateDecodingOrder:
    """Tests for the generate_decoding_order() method."""

    def test_output_shape(self, mpnn_model, single_context):
        """Test that output has correct shape."""
        n_tokens = single_context.token.token_index.shape[-1]
        batch_size = 1

        decoding_order = mpnn_model.generate_decoding_order(
            centre_atom_coords=single_context.coordinate.centre_coordinate,
            centre_atom_mask=single_context.mask.token_mask,
            seed=42,
        )

        assert decoding_order.shape == (batch_size, n_tokens)

    def test_output_is_valid_permutation(self, mpnn_model, single_context):
        """Test that output is a valid permutation of indices."""
        n_tokens = single_context.token.token_index.shape[-1]

        decoding_order = mpnn_model.generate_decoding_order(
            centre_atom_coords=single_context.coordinate.centre_coordinate,
            centre_atom_mask=single_context.mask.token_mask,
            seed=42,
        )

        # Check that it contains all indices from 0 to n_tokens-1
        sorted_indices = torch.sort(decoding_order[0])[0]
        expected_indices = torch.arange(n_tokens, device=decoding_order.device)
        assert torch.equal(
            sorted_indices, expected_indices
        ), "decoding_order is not a valid permutation"

    def test_deterministic_with_seed(self, mpnn_model, single_context):
        """Test that same seed produces same decoding order."""
        decoding_order_1 = mpnn_model.generate_decoding_order(
            centre_atom_coords=single_context.coordinate.centre_coordinate,
            centre_atom_mask=single_context.mask.token_mask,
            seed=42,
            smoothing_factor=0.5,
        )

        decoding_order_2 = mpnn_model.generate_decoding_order(
            centre_atom_coords=single_context.coordinate.centre_coordinate,
            centre_atom_mask=single_context.mask.token_mask,
            seed=42,
            smoothing_factor=0.5,
        )

        assert torch.equal(
            decoding_order_1, decoding_order_2
        ), "Same seed should produce same decoding order"

    def test_different_seeds_produce_different_orders(self, mpnn_model, single_context):
        """Test that different seeds produce different decoding orders."""
        decoding_order_1 = mpnn_model.generate_decoding_order(
            centre_atom_coords=single_context.coordinate.centre_coordinate,
            centre_atom_mask=single_context.mask.token_mask,
            seed=42,
            smoothing_factor=0.5,
        )

        decoding_order_2 = mpnn_model.generate_decoding_order(
            centre_atom_coords=single_context.coordinate.centre_coordinate,
            centre_atom_mask=single_context.mask.token_mask,
            seed=123,
            smoothing_factor=0.5,
        )

        assert not torch.equal(
            decoding_order_1, decoding_order_2
        ), "Different seeds should produce different decoding orders"

    def test_smoothing_factors(self, mpnn_model, single_context):
        """Test that different smoothing factors produce different orders."""
        # Fully spatial (mu=1.0)
        decoding_order_spatial = mpnn_model.generate_decoding_order(
            centre_atom_coords=single_context.coordinate.centre_coordinate,
            centre_atom_mask=single_context.mask.token_mask,
            seed=42,
            smoothing_factor=1.0,
        )

        # Fully random (mu=0.0)
        decoding_order_random = mpnn_model.generate_decoding_order(
            centre_atom_coords=single_context.coordinate.centre_coordinate,
            centre_atom_mask=single_context.mask.token_mask,
            seed=42,
            smoothing_factor=0.0,
        )

        # Both should be valid permutations but likely different
        assert not torch.equal(
            decoding_order_spatial, decoding_order_random
        ), "Fully spatial and fully random should produce different orders"


class TestGenerateDecodingMasks:
    """Tests for the generate_decoding_masks() method."""

    def test_output_type(self, mpnn_model, single_context):
        """Test that output is a DecodingMasks instance."""
        encoder_output = mpnn_model.encode(single_context)
        decoding_order = mpnn_model.generate_decoding_order(
            centre_atom_coords=single_context.coordinate.centre_coordinate,
            centre_atom_mask=single_context.mask.token_mask,
            seed=42,
        )

        masks = mpnn_model.generate_decoding_masks(
            decoding_order=decoding_order,
            edge_index=encoder_output.edge_index,
            node_mask=single_context.mask.token_mask,
        )

        assert isinstance(masks, DecodingMasks)
        assert hasattr(masks, "forward_mask")
        assert hasattr(masks, "backward_mask")

    def test_output_shapes(self, mpnn_model, single_context):
        """Test that mask outputs have correct shapes."""
        n_tokens = single_context.token.token_index.shape[-1]
        num_neighbours = mpnn_model.num_neighbours
        batch_size = 1

        encoder_output = mpnn_model.encode(single_context)
        decoding_order = mpnn_model.generate_decoding_order(
            centre_atom_coords=single_context.coordinate.centre_coordinate,
            centre_atom_mask=single_context.mask.token_mask,
            seed=42,
        )

        masks = mpnn_model.generate_decoding_masks(
            decoding_order=decoding_order,
            edge_index=encoder_output.edge_index,
            node_mask=single_context.mask.token_mask,
        )

        # Masks should have shape [batch, n_token, neighbours, 1]
        expected_shape = (batch_size, n_tokens, num_neighbours, 1)
        assert masks.forward_mask.shape == expected_shape
        assert masks.backward_mask.shape == expected_shape

    def test_mask_values_binary(self, mpnn_model, single_context):
        """Test that masks contain only 0s and 1s."""
        encoder_output = mpnn_model.encode(single_context)
        decoding_order = mpnn_model.generate_decoding_order(
            centre_atom_coords=single_context.coordinate.centre_coordinate,
            centre_atom_mask=single_context.mask.token_mask,
            seed=42,
        )

        masks = mpnn_model.generate_decoding_masks(
            decoding_order=decoding_order,
            edge_index=encoder_output.edge_index,
            node_mask=single_context.mask.token_mask,
        )

        assert torch.all(
            (masks.forward_mask == 0) | (masks.forward_mask == 1)
        ), "forward_mask contains non-binary values"
        assert torch.all(
            (masks.backward_mask == 0) | (masks.backward_mask == 1)
        ), "backward_mask contains non-binary values"

    def test_masks_sum_correctly(self, mpnn_model, single_context):
        """Test that forward and backward masks are complementary."""
        encoder_output = mpnn_model.encode(single_context)
        decoding_order = mpnn_model.generate_decoding_order(
            centre_atom_coords=single_context.coordinate.centre_coordinate,
            centre_atom_mask=single_context.mask.token_mask,
            seed=42,
        )
        node_mask = single_context.mask.token_mask

        masks = mpnn_model.generate_decoding_masks(
            decoding_order=decoding_order,
            edge_index=encoder_output.edge_index,
            node_mask=node_mask,
        )

        # Forward + backward should equal 1 for valid positions
        mask_sum = masks.forward_mask + masks.backward_mask

        # For valid tokens, the sum should be node_mask value (0 or 1)
        expected_sum = node_mask[..., None, None]  # [batch, n_token, 1, 1]
        # Broadcast to [batch, n_token, neighbours, 1]
        assert torch.allclose(
            mask_sum, expected_sum.to(mask_sum.dtype).expand_as(mask_sum), atol=1e-6
        ), "Forward and backward masks don't sum correctly"


class TestDecode:
    """Tests for the decode() method."""

    def test_output_shape(self, mpnn_model, single_context):
        """Test that decoder output has correct shape."""
        n_tokens = single_context.token.token_index.shape[-1]
        num_residue_types = mpnn_model.num_residue_types
        batch_size = 1

        encoder_output = mpnn_model.encode(single_context)
        decoding_order = mpnn_model.generate_decoding_order(
            centre_atom_coords=single_context.coordinate.centre_coordinate,
            centre_atom_mask=single_context.mask.token_mask,
            seed=42,
        )
        masks = mpnn_model.generate_decoding_masks(
            decoding_order=decoding_order,
            edge_index=encoder_output.edge_index,
            node_mask=single_context.mask.token_mask,
        )

        logits = mpnn_model.decode(
            encoder_output=encoder_output,
            restype=single_context.token.restype,
            masks=masks,
            node_mask=single_context.mask.token_mask,
            zero_shot=False,
        )

        expected_shape = (batch_size, n_tokens, num_residue_types)
        assert logits.shape == expected_shape

    def test_zero_shot_vs_autoregressive(self, mpnn_model, single_context):
        """Test that zero_shot and autoregressive modes produce different outputs."""
        encoder_output = mpnn_model.encode(single_context)
        decoding_order = mpnn_model.generate_decoding_order(
            centre_atom_coords=single_context.coordinate.centre_coordinate,
            centre_atom_mask=single_context.mask.token_mask,
            seed=42,
        )
        masks = mpnn_model.generate_decoding_masks(
            decoding_order=decoding_order,
            edge_index=encoder_output.edge_index,
            node_mask=single_context.mask.token_mask,
        )

        logits_zero_shot = mpnn_model.decode(
            encoder_output=encoder_output,
            restype=single_context.token.restype,
            masks=masks,
            node_mask=single_context.mask.token_mask,
            zero_shot=True,
        )

        logits_autoregressive = mpnn_model.decode(
            encoder_output=encoder_output,
            restype=single_context.token.restype,
            masks=masks,
            node_mask=single_context.mask.token_mask,
            zero_shot=False,
        )

        assert not torch.allclose(
            logits_zero_shot, logits_autoregressive, atol=1e-3
        ), "Zero-shot and autoregressive modes produce identical outputs"

    def test_output_no_nan_or_inf(self, mpnn_model, single_context):
        """Test that decoder output doesn't contain NaN or Inf values."""
        encoder_output = mpnn_model.encode(single_context)
        decoding_order = mpnn_model.generate_decoding_order(
            centre_atom_coords=single_context.coordinate.centre_coordinate,
            centre_atom_mask=single_context.mask.token_mask,
            seed=42,
        )
        masks = mpnn_model.generate_decoding_masks(
            decoding_order=decoding_order,
            edge_index=encoder_output.edge_index,
            node_mask=single_context.mask.token_mask,
        )

        logits = mpnn_model.decode(
            encoder_output=encoder_output,
            restype=single_context.token.restype,
            masks=masks,
            node_mask=single_context.mask.token_mask,
            zero_shot=False,
        )

        assert not torch.isnan(logits).any(), "Logits contain NaN values"
        assert not torch.isinf(logits).any(), "Logits contain Inf values"

    def test_different_restype_inputs(self, mpnn_model, single_context):
        """Test that different restype inputs produce different outputs."""
        encoder_output = mpnn_model.encode(single_context)
        decoding_order = mpnn_model.generate_decoding_order(
            centre_atom_coords=single_context.coordinate.centre_coordinate,
            centre_atom_mask=single_context.mask.token_mask,
            seed=42,
        )
        masks = mpnn_model.generate_decoding_masks(
            decoding_order=decoding_order,
            edge_index=encoder_output.edge_index,
            node_mask=single_context.mask.token_mask,
        )

        # First restype
        restype_1 = single_context.token.restype
        logits_1 = mpnn_model.decode(
            encoder_output=encoder_output,
            restype=restype_1,
            masks=masks,
            node_mask=single_context.mask.token_mask,
            zero_shot=False,
        )

        # Different restype
        restype_2 = restype_1.clone()
        restype_2[:, 0, :] = 0
        restype_2[:, 0, 5] = 1  # Set to different residue type

        logits_2 = mpnn_model.decode(
            encoder_output=encoder_output,
            restype=restype_2,
            masks=masks,
            node_mask=single_context.mask.token_mask,
            zero_shot=False,
        )

        assert not torch.allclose(
            logits_1, logits_2, atol=1e-4
        ), "Different restype inputs produce identical outputs"


class TestForward:
    """Tests for the forward() method (training pass)."""

    def test_output_structure(self, mpnn_model, single_context):
        """Test that forward returns correct output structure."""
        result = mpnn_model.forward(single_context)

        assert isinstance(result, tuple), "Forward should return a tuple"
        assert len(result) == 3, "Forward should return 3 values"

        logits, node_feats, edge_feats = result
        assert isinstance(logits, torch.Tensor)
        assert isinstance(node_feats, torch.Tensor)
        assert isinstance(edge_feats, torch.Tensor)

    def test_logits_shape(self, mpnn_model, single_context):
        """Test that forward produces logits with correct shape."""
        n_tokens = single_context.token.token_index.shape[-1]
        num_residue_types = mpnn_model.num_residue_types
        batch_size = 1

        logits, _, _ = mpnn_model.forward(single_context)

        expected_shape = (batch_size, n_tokens, num_residue_types)
        assert logits.shape == expected_shape

    def test_with_custom_decoding_order(self, mpnn_model, single_context):
        """Test forward with custom decoding order."""
        n_tokens = single_context.token.token_index.shape[-1]
        decoding_order = torch.randperm(n_tokens).unsqueeze(0)

        logits, _, _ = mpnn_model.forward(single_context, decoding_order=decoding_order)

        assert not torch.isnan(logits).any()

    def test_zero_shot_mode(self, mpnn_model, single_context):
        """Test forward in zero-shot mode."""
        logits_zero_shot, _, _ = mpnn_model.forward(single_context, zero_shot=True)
        logits_autoregressive, _, _ = mpnn_model.forward(
            single_context, zero_shot=False
        )

        assert not torch.allclose(
            logits_zero_shot, logits_autoregressive, atol=1e-3
        ), "Zero-shot and autoregressive modes produce identical outputs"

    def test_no_nan_or_inf(self, mpnn_model, single_context):
        """Test that forward doesn't produce NaN or Inf values."""
        logits, node_feats, edge_feats = mpnn_model.forward(single_context)

        assert not torch.isnan(logits).any(), "Forward logits contain NaN"
        assert not torch.isinf(logits).any(), "Forward logits contain Inf"
        assert not torch.isnan(node_feats).any(), "node_feats contain NaN"
        assert not torch.isinf(node_feats).any(), "node_feats contain Inf"
        assert not torch.isnan(edge_feats).any(), "edge_feats contain NaN"
        assert not torch.isinf(edge_feats).any(), "edge_feats contain Inf"


@pytest.mark.slow
class TestSample:
    """Tests for the sample() method (autoregressive inference)."""

    def test_output_structure(self, mpnn_model, single_context):
        """Test that sample returns correct output structure."""
        result = mpnn_model.sample(single_context, temperature=1.0)

        assert isinstance(result, tuple), "Sample should return a tuple"
        assert len(result) == 2, "Sample should return 2 values"

        sampled_sequence, node_feats = result
        assert isinstance(sampled_sequence, torch.Tensor)
        assert isinstance(node_feats, torch.Tensor)

    def test_output_shape(self, mpnn_model, single_context):
        """Test that sample returns correct output shape."""
        n_tokens = single_context.token.token_index.shape[-1]
        num_residue_types = mpnn_model.num_residue_types
        batch_size = 1

        sampled_sequence, node_feats = mpnn_model.sample(
            single_context, temperature=1.0
        )

        expected_seq_shape = (batch_size, n_tokens, num_residue_types)
        expected_node_shape = (batch_size, n_tokens, mpnn_model.node_dim)

        assert sampled_sequence.shape == expected_seq_shape
        assert node_feats.shape == expected_node_shape

    def test_returns_one_hot(self, mpnn_model, single_context):
        """Test that sample returns valid one-hot encoded sequences."""
        sampled_sequence, _ = mpnn_model.sample(single_context, temperature=1.0)

        # Check that each position sums to 1
        position_sums = sampled_sequence.sum(dim=-1)
        assert torch.allclose(
            position_sums, torch.ones_like(position_sums), atol=1e-5
        ), "Sampled sequence is not valid one-hot encoding"

        # Check that all values are 0 or 1
        assert torch.all(
            (sampled_sequence == 0) | (sampled_sequence == 1)
        ), "Sampled sequence contains non-binary values"

    def test_greedy_sampling(self, mpnn_model, single_context):
        """Test greedy sampling (temperature=0)."""
        sampled_sequence, _ = mpnn_model.sample(single_context, temperature=0.0)

        # Should still be valid one-hot
        position_sums = sampled_sequence.sum(dim=-1)
        assert torch.allclose(position_sums, torch.ones_like(position_sums), atol=1e-5)

    def test_different_temperatures(self, mpnn_model, single_context):
        """Test sampling with different temperature values."""
        for temperature in [0.0, 0.5, 1.0, 1.5]:
            sampled_sequence, _ = mpnn_model.sample(
                single_context, temperature=temperature
            )

            assert not torch.isnan(sampled_sequence).any()
            assert sampled_sequence.shape[-1] == mpnn_model.num_residue_types

    def test_with_custom_decoding_order(self, mpnn_model, single_context):
        """Test sampling with custom decoding order."""
        n_tokens = single_context.token.token_index.shape[-1]
        decoding_order = torch.randperm(n_tokens).unsqueeze(0)

        sampled_sequence, _ = mpnn_model.sample(
            single_context,
            decoding_order=decoding_order,
            temperature=1.0,
        )

        assert sampled_sequence.shape[-1] == mpnn_model.num_residue_types

    def test_no_nan_or_inf(self, mpnn_model, single_context):
        """Test that sample doesn't produce NaN or Inf values."""
        sampled_sequence, node_feats = mpnn_model.sample(
            single_context, temperature=1.0
        )

        assert not torch.isnan(sampled_sequence).any(), "Sampled sequence contains NaN"
        assert not torch.isinf(sampled_sequence).any(), "Sampled sequence contains Inf"
        assert not torch.isnan(node_feats).any(), "node_feats contains NaN"
        assert not torch.isinf(node_feats).any(), "node_feats contains Inf"


class TestBatchProcessing:
    """Tests for batch processing capabilities."""

    def test_encode_batch(self, mpnn_model, full_context):
        """Test that encode works with batched inputs."""
        batch_size = full_context.token.token_index.shape[0]
        n_tokens = full_context.token.token_index.shape[-1]

        encoder_output = mpnn_model.encode(full_context)

        assert encoder_output.node_feats.shape[0] == batch_size
        assert encoder_output.node_feats.shape[1] == n_tokens

    def test_forward_batch(self, mpnn_model, full_context):
        """Test that forward works with batched inputs."""
        batch_size = full_context.token.token_index.shape[0]
        n_tokens = full_context.token.token_index.shape[-1]

        logits, node_feats, edge_feats = mpnn_model.forward(full_context)

        assert logits.shape[0] == batch_size
        assert logits.shape[1] == n_tokens
        assert logits.shape[2] == mpnn_model.num_residue_types


class TestGradients:
    """Tests for gradient flow."""

    def test_forward_gradients(self, mpnn_model, single_context):
        """Test that gradients flow through forward pass."""
        mpnn_model.train()

        logits, _, _ = mpnn_model.forward(single_context)

        # Create a simple loss
        loss = logits.sum()
        loss.backward()

        # Check that gradients are computed for model parameters
        for name, param in mpnn_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_decode_gradients(self, mpnn_model, single_context):
        """Test that gradients flow through decode."""
        mpnn_model.train()

        encoder_output = mpnn_model.encode(single_context)
        decoding_order = mpnn_model.generate_decoding_order(
            centre_atom_coords=single_context.coordinate.centre_coordinate,
            centre_atom_mask=single_context.mask.token_mask,
            seed=42,
        )
        masks = mpnn_model.generate_decoding_masks(
            decoding_order=decoding_order,
            edge_index=encoder_output.edge_index,
            node_mask=single_context.mask.token_mask,
        )

        logits = mpnn_model.decode(
            encoder_output=encoder_output,
            restype=single_context.token.restype,
            masks=masks,
            node_mask=single_context.mask.token_mask,
            zero_shot=False,
        )

        loss = logits.sum()
        loss.backward()

        # Check decoder has gradients
        for name, param in mpnn_model.decoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for decoder.{name}"
