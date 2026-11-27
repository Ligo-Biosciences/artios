"""Tests for the discrete MPNN with masked diffusion.

This module tests the core functionality of the MPNN class, including:
- Encoding (structure → node/edge features)
- Masking (clean restypes → masked restypes)
- Decoding (masked restypes → logits)
- Forward pass (training with random masking)
"""

import pytest
import torch

from artios.model.modules.mpnn import MPNN, DecoderOutput
from tests._helpers.context import FullContext, create_test_context


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


class TestEncode:
    """Tests for the encode() method."""

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
