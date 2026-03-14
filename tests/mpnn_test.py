"""Tests for the discrete MPNN with masked diffusion.

This module tests the core functionality of the MPNN class, including:
- Encoding (structure → node/edge features)
- Masking (clean restypes → masked restypes)
- Decoding (masked restypes → logits)
- Forward pass (training with random masking)
"""

import pytest
import torch

from artios.model.modules.mpnn import MPNN, DecoderOutput, MPNNOutput
from tests.context_utils import FullContext, create_test_context


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

        assert encoder_output.node_feats.shape == (batch_size, n_tokens, node_dim)
        assert encoder_output.edge_feats.shape == (
            batch_size,
            n_tokens,
            num_neighbours,
            edge_dim,
        )
        assert encoder_output.edge_index.shape == (
            batch_size,
            n_tokens,
            num_neighbours,
        )
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

        masked_token_indices = torch.where(~token_mask[0])[0]
        if len(masked_token_indices) > 0:
            for idx in masked_token_indices[:5]:
                assert torch.all(
                    encoder_output.edge_mask[0, idx] == 0
                ), f"Edge mask not zero for masked token {idx}"

    def test_output_values_reasonable(self, mpnn_model, single_context):
        """Test that output values are within reasonable ranges."""
        encoder_output = mpnn_model.encode(single_context)

        assert torch.all(
            torch.abs(encoder_output.node_feats) < 1e6
        ), "node_feats contains unreasonably large values"
        assert torch.all(
            torch.abs(encoder_output.edge_feats) < 1e6
        ), "edge_feats contains unreasonably large values"

    def test_device_consistency(self, mpnn_model, single_context, device):
        """Test that outputs are on the same device as inputs."""
        mpnn_model = mpnn_model.to(device)

        encoder_output = mpnn_model.encode(single_context)

        assert encoder_output.node_feats.device == device
        assert encoder_output.edge_feats.device == device
        assert encoder_output.edge_index.device == device
        assert encoder_output.edge_mask.device == device


class TestMaskRestypes:
    """Tests for the mask_restypes() method."""

    def test_output_shapes(self, mpnn_model):
        """Test that masked restypes have correct shapes."""
        batch, n_token, num_rt = 2, 16, 32
        restypes = torch.zeros(batch, n_token, num_rt, dtype=torch.long)
        restypes[..., 0] = 1
        mask_rate = torch.tensor([0.5, 0.5])

        masked, is_masked = mpnn_model.mask_restypes(restypes, mask_rate)

        assert masked.shape == (batch, n_token, num_rt + 1)
        assert masked.dtype == torch.long
        assert is_masked.shape == (batch, n_token)

    def test_masked_positions_have_mask_token(self, mpnn_model):
        """Test that masked positions contain the mask token one-hot."""
        batch, n_token, num_rt = 1, 16, 32
        restypes = torch.zeros(batch, n_token, num_rt, dtype=torch.long)
        restypes[..., 0] = 1
        mask_rate = torch.tensor([1.0])

        masked, is_masked = mpnn_model.mask_restypes(restypes, mask_rate)

        assert is_masked.all(), "All positions should be masked at rate 1.0"
        assert (masked[..., -1] == 1).all(), "Masked positions should have mask token"
        assert (
            masked[..., :-1] == 0
        ).all(), "Masked positions should have zero for real types"

    def test_unmasked_positions_preserve_restypes(self, mpnn_model):
        """Test that unmasked positions keep original restypes."""
        batch, n_token, num_rt = 1, 16, 32
        restypes = torch.zeros(batch, n_token, num_rt, dtype=torch.long)
        restypes[..., 0] = 1
        mask_rate = torch.tensor([0.0])

        masked, is_masked = mpnn_model.mask_restypes(restypes, mask_rate)

        assert not is_masked.any(), "No positions should be masked at rate 0.0"
        assert (
            masked[..., :num_rt] == restypes
        ).all(), "Unmasked should preserve restypes"
        assert (masked[..., -1] == 0).all(), "Unmasked should have zero mask dim"

    def test_token_mask_prevents_masking_invalid(self, mpnn_model):
        """Test that invalid positions (token_mask=False) are never masked."""
        batch, n_token, num_rt = 1, 16, 32
        restypes = torch.zeros(batch, n_token, num_rt, dtype=torch.long)
        restypes[..., 0] = 1
        mask_rate = torch.tensor([1.0])
        token_mask = torch.zeros(batch, n_token, dtype=torch.bool)
        token_mask[0, :8] = True

        _, is_masked = mpnn_model.mask_restypes(restypes, mask_rate, token_mask)

        assert not is_masked[0, 8:].any(), "Invalid positions should not be masked"
        assert is_masked[0, :8].all(), "Valid positions should be masked at rate 1.0"

    def test_per_batch_mask_rate(self, mpnn_model):
        """Test that mask_rate is applied independently per batch element."""
        batch, n_token, num_rt = 2, 1000, 32
        restypes = torch.zeros(batch, n_token, num_rt, dtype=torch.long)
        restypes[..., 0] = 1
        mask_rate = torch.tensor([0.0, 1.0])

        _, is_masked = mpnn_model.mask_restypes(restypes, mask_rate)

        assert not is_masked[0].any(), "Batch 0 should have no masks at rate 0.0"
        assert is_masked[1].all(), "Batch 1 should be fully masked at rate 1.0"

    def test_output_is_valid_one_hot(self, mpnn_model):
        """Test that each output position is a valid one-hot vector."""
        batch, n_token, num_rt = 2, 16, 32
        restypes = torch.zeros(batch, n_token, num_rt, dtype=torch.long)
        indices = torch.randint(0, num_rt, (batch, n_token))
        restypes.scatter_(-1, indices.unsqueeze(-1), 1)
        mask_rate = torch.tensor([0.5, 0.5])

        masked, _ = mpnn_model.mask_restypes(restypes, mask_rate)

        sums = masked.sum(dim=-1)
        assert (sums == 1).all(), "Each position should sum to 1"


class TestDecode:
    """Tests for the decode() method."""

    def test_output_shape(self, mpnn_model, single_context):
        """Test that decoder output has correct shape."""
        n_tokens = single_context.token.token_index.shape[-1]
        num_residue_types = mpnn_model.num_residue_types
        batch_size = 1

        encoder_output = mpnn_model.encode(single_context)

        # Create fully masked input (int one-hot)
        masked_restypes = torch.zeros(
            batch_size, n_tokens, num_residue_types + 1, dtype=torch.long
        )
        masked_restypes[..., -1] = 1

        decoder_output = mpnn_model.decode(
            encoder_output=encoder_output,
            restypes=masked_restypes,
            node_mask=single_context.mask.token_mask,
        )

        assert isinstance(decoder_output, DecoderOutput)
        expected_shape = (batch_size, n_tokens, num_residue_types)
        assert decoder_output.logits.shape == expected_shape

    def test_masked_vs_unmasked_differ(self, mpnn_model, single_context):
        """Test that fully masked and unmasked inputs produce different logits."""
        n_tokens = single_context.token.token_index.shape[-1]
        num_rt = mpnn_model.num_residue_types
        batch_size = 1

        encoder_output = mpnn_model.encode(single_context)
        node_mask = single_context.mask.token_mask

        # Fully masked
        masked_input = torch.zeros(batch_size, n_tokens, num_rt + 1, dtype=torch.long)
        masked_input[..., -1] = 1

        # Unmasked (all ALA, padded)
        unmasked_input = torch.zeros(batch_size, n_tokens, num_rt + 1, dtype=torch.long)
        unmasked_input[..., 0] = 1

        dec_masked = mpnn_model.decode(encoder_output, masked_input, node_mask)
        dec_unmasked = mpnn_model.decode(encoder_output, unmasked_input, node_mask)

        assert not torch.allclose(
            dec_masked.logits, dec_unmasked.logits, atol=1e-3
        ), "Masked and unmasked inputs should produce different logits"

    def test_output_no_nan_or_inf(self, mpnn_model, single_context):
        """Test that decoder output doesn't contain NaN or Inf values."""
        n_tokens = single_context.token.token_index.shape[-1]
        num_rt = mpnn_model.num_residue_types
        batch_size = 1

        encoder_output = mpnn_model.encode(single_context)

        masked_restypes = torch.zeros(
            batch_size, n_tokens, num_rt + 1, dtype=torch.long
        )
        masked_restypes[..., -1] = 1

        decoder_output = mpnn_model.decode(
            encoder_output=encoder_output,
            restypes=masked_restypes,
            node_mask=single_context.mask.token_mask,
        )

        assert not torch.isnan(decoder_output.logits).any(), "Logits contain NaN values"
        assert not torch.isinf(decoder_output.logits).any(), "Logits contain Inf values"


class TestForward:
    """Tests for the forward() method (training pass)."""

    def test_output_structure(self, mpnn_model, single_context):
        """Test that forward returns an MPNNOutput dataclass."""
        mask_rate = torch.tensor([0.5])
        result = mpnn_model.forward(single_context, mask_rate)

        assert isinstance(result, MPNNOutput), "Forward should return an MPNNOutput"
        assert isinstance(result.logits, torch.Tensor)
        assert isinstance(result.is_masked, torch.Tensor)
        assert isinstance(result.node_feats, torch.Tensor)
        assert isinstance(result.edge_feats, torch.Tensor)
        assert isinstance(result.edge_index, torch.Tensor)
        assert isinstance(result.edge_mask, torch.Tensor)

    def test_logits_shape(self, mpnn_model, single_context):
        """Test that forward produces logits with correct shape."""
        n_tokens = single_context.token.token_index.shape[-1]
        num_residue_types = mpnn_model.num_residue_types
        batch_size = 1

        mask_rate = torch.tensor([0.5])
        mpnn_output = mpnn_model.forward(single_context, mask_rate)

        expected_shape = (batch_size, n_tokens, num_residue_types)
        assert mpnn_output.logits.shape == expected_shape

    def test_is_masked_shape(self, mpnn_model, single_context):
        """Test that is_masked has correct shape."""
        n_tokens = single_context.token.token_index.shape[-1]
        batch_size = 1

        mask_rate = torch.tensor([0.5])
        mpnn_output = mpnn_model.forward(single_context, mask_rate)

        assert mpnn_output.is_masked.shape == (batch_size, n_tokens)
        assert mpnn_output.is_masked.dtype == torch.bool

    def test_no_nan_or_inf(self, mpnn_model, single_context):
        """Test that forward doesn't produce NaN or Inf values."""
        mask_rate = torch.tensor([0.5])
        mpnn_output = mpnn_model.forward(single_context, mask_rate)

        assert not torch.isnan(mpnn_output.logits).any(), "Forward logits contain NaN"
        assert not torch.isinf(mpnn_output.logits).any(), "Forward logits contain Inf"
        assert not torch.isnan(mpnn_output.node_feats).any(), "node_feats contain NaN"
        assert not torch.isinf(mpnn_output.node_feats).any(), "node_feats contain Inf"
        assert not torch.isnan(mpnn_output.edge_feats).any(), "edge_feats contain NaN"
        assert not torch.isinf(mpnn_output.edge_feats).any(), "edge_feats contain Inf"


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

        mask_rate = torch.full((batch_size,), 0.5)
        mpnn_output = mpnn_model.forward(full_context, mask_rate)

        assert mpnn_output.logits.shape[0] == batch_size
        assert mpnn_output.logits.shape[1] == n_tokens
        assert mpnn_output.logits.shape[2] == mpnn_model.num_residue_types
        assert mpnn_output.is_masked.shape == (batch_size, n_tokens)


class TestGradients:
    """Tests for gradient flow."""

    def test_forward_gradients(self, mpnn_model, single_context):
        """Test that gradients flow through forward pass."""
        mpnn_model.train()

        mask_rate = torch.tensor([0.5])
        mpnn_output = mpnn_model.forward(single_context, mask_rate)

        loss = mpnn_output.logits.sum()
        loss.backward()

        for name, param in mpnn_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_decode_gradients(self, mpnn_model, single_context):
        """Test that gradients flow through decode."""
        mpnn_model.train()

        n_tokens = single_context.token.token_index.shape[-1]
        num_rt = mpnn_model.num_residue_types

        encoder_output = mpnn_model.encode(single_context)

        masked_restypes = torch.zeros(1, n_tokens, num_rt + 1, dtype=torch.long)
        masked_restypes[..., -1] = 1

        decoder_output = mpnn_model.decode(
            encoder_output=encoder_output,
            restypes=masked_restypes,
            node_mask=single_context.mask.token_mask,
        )

        loss = decoder_output.logits.sum()
        loss.backward()

        for name, param in mpnn_model.decoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for decoder.{name}"


class TestAsPredictor:
    """Tests for the as_predictor() method."""

    def test_returns_callable(self, mpnn_model, single_context):
        """Test that as_predictor returns a callable."""
        predictor = mpnn_model.as_predictor(single_context)
        assert callable(predictor)

    def test_predictor_output_shape(self, mpnn_model, single_context):
        """Test that predictor returns logits with correct shape."""
        predictor = mpnn_model.as_predictor(single_context)

        batch_size = 1
        n_tokens = single_context.token.token_index.shape[-1]
        num_rt = mpnn_model.num_residue_types

        # All-masked input
        restype = torch.zeros(batch_size, n_tokens, num_rt + 1, dtype=torch.long)
        restype[..., -1] = 1

        logits = predictor(restype)

        assert logits.shape == (batch_size, n_tokens, num_rt)

    def test_predictor_finite_values(self, mpnn_model, single_context):
        """Test that predictor returns finite values."""
        predictor = mpnn_model.as_predictor(single_context)

        batch_size = 1
        n_tokens = single_context.token.token_index.shape[-1]
        num_rt = mpnn_model.num_residue_types

        restype = torch.zeros(batch_size, n_tokens, num_rt + 1, dtype=torch.long)
        restype[..., -1] = 1

        logits = predictor(restype)

        assert torch.isfinite(logits).all(), "Predictor returned non-finite values"

    def test_predictor_different_inputs_different_outputs(
        self, mpnn_model, single_context
    ):
        """Test that different restype inputs produce different outputs."""
        predictor = mpnn_model.as_predictor(single_context)

        batch_size = 1
        n_tokens = single_context.token.token_index.shape[-1]
        num_rt = mpnn_model.num_residue_types

        # All masked
        masked = torch.zeros(batch_size, n_tokens, num_rt + 1, dtype=torch.long)
        masked[..., -1] = 1

        # All type 0 (unmasked)
        unmasked = torch.zeros(batch_size, n_tokens, num_rt + 1, dtype=torch.long)
        unmasked[..., 0] = 1

        logits_masked = predictor(masked)
        logits_unmasked = predictor(unmasked)

        assert not torch.allclose(logits_masked, logits_unmasked, atol=1e-3)

    def test_predictor_batched(self, mpnn_model, full_context):
        """Test that predictor works with batched inputs."""
        predictor = mpnn_model.as_predictor(full_context)

        batch_size = full_context.token.token_index.shape[0]
        n_tokens = full_context.token.token_index.shape[-1]
        num_rt = mpnn_model.num_residue_types

        restype = torch.zeros(batch_size, n_tokens, num_rt + 1, dtype=torch.long)
        restype[..., -1] = 1

        logits = predictor(restype)

        assert logits.shape == (batch_size, n_tokens, num_rt)


class TestSample:
    """Tests for the sample() method (discrete diffusion inference)."""

    def test_output_shape(self, mpnn_model, single_context):
        """Test that sample returns logits with correct shape."""
        from artios.model.generator import CosineScheduler, LinearMaskingTimestepSampler

        mpnn_model.eval()
        with torch.no_grad():
            logits = mpnn_model.sample(
                context=single_context,
                timestep_scheduler=CosineScheduler(),
                masking_timestep_sampler=LinearMaskingTimestepSampler(),
                n_steps=5,
                seed=42,
            )

        n_tokens = single_context.token.token_index.shape[-1]
        assert logits.shape == (1, n_tokens, mpnn_model.num_residue_types)

    def test_finite_values(self, mpnn_model, single_context):
        """Test that sample returns finite values."""
        from artios.model.generator import CosineScheduler, LinearMaskingTimestepSampler

        mpnn_model.eval()
        with torch.no_grad():
            logits = mpnn_model.sample(
                context=single_context,
                timestep_scheduler=CosineScheduler(),
                masking_timestep_sampler=LinearMaskingTimestepSampler(),
                n_steps=5,
                seed=42,
            )

        assert torch.isfinite(logits).all(), "sample returned non-finite values"

    def test_batched(self, mpnn_model, full_context):
        """Test that sample works with batched inputs."""
        from artios.model.generator import CosineScheduler, LinearMaskingTimestepSampler

        mpnn_model.eval()
        batch_size = full_context.token.token_index.shape[0]
        n_tokens = full_context.token.token_index.shape[-1]

        with torch.no_grad():
            logits = mpnn_model.sample(
                context=full_context,
                timestep_scheduler=CosineScheduler(),
                masking_timestep_sampler=LinearMaskingTimestepSampler(),
                n_steps=5,
                seed=42,
            )

        assert logits.shape == (batch_size, n_tokens, mpnn_model.num_residue_types)

    def test_reproducibility(self, mpnn_model, single_context):
        """Test that same seed produces identical results."""
        from artios.model.generator import CosineScheduler, LinearMaskingTimestepSampler

        mpnn_model.eval()
        kwargs = dict(
            context=single_context,
            timestep_scheduler=CosineScheduler(),
            masking_timestep_sampler=LinearMaskingTimestepSampler(),
            n_steps=5,
        )

        with torch.no_grad():
            result1 = mpnn_model.sample(**kwargs, seed=42)
            result2 = mpnn_model.sample(**kwargs, seed=42)

        assert torch.allclose(result1, result2), "Same seed should give same results"

    def test_different_seeds_differ(self, mpnn_model, single_context):
        """Test that different seeds produce different results."""
        from artios.model.generator import CosineScheduler, LinearMaskingTimestepSampler

        mpnn_model.eval()
        kwargs = dict(
            context=single_context,
            timestep_scheduler=CosineScheduler(),
            masking_timestep_sampler=LinearMaskingTimestepSampler(),
            n_steps=5,
        )

        with torch.no_grad():
            result1 = mpnn_model.sample(**kwargs, seed=42)
            result2 = mpnn_model.sample(**kwargs, seed=123)

        assert not torch.allclose(result1, result2), "Different seeds should differ"
