#!/usr/bin/env python3

import math

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from artios.model.modules.layout import gather_trunk_pair_embedding
from artios.model.modules.primitives import (
    Linear,
    attention,
)


class TestLinearPrecision:
    """Test Linear layer precision handling, especially bfloat16 to higher precision casting."""

    @pytest.fixture
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_linear_precision_casting_bfloat16(self, device: torch.device) -> None:
        """Test that bfloat16 inputs get cast to higher precision during matmul."""
        # if device == torch.device("cpu"):
        #    pytest.skip("Skipping bfloat16 test on CPU")

        in_features, out_features = 64, 32
        batch_size = 4

        # Create Linear layer with high_precision=True
        linear_high_precision = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            high_precision=True,
        ).to(device)

        # Create regular Linear layer for comparison
        linear_regular = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            high_precision=False,
        ).to(device)

        # Copy weights to ensure same parameters
        linear_regular.weight.data.copy_(linear_high_precision.weight.data)
        linear_regular.bias.data.copy_(linear_high_precision.bias.data)

        linear_regular.weight = nn.Parameter(linear_regular.weight.to(torch.bfloat16))
        linear_regular.bias = nn.Parameter(linear_regular.bias.to(torch.bfloat16))

        # Create bfloat16 input
        input_tensor = torch.randn(
            batch_size, in_features, device=device, dtype=torch.bfloat16
        )

        # Forward pass with high precision (no manual autocast needed)
        output_high_precision = linear_high_precision(input_tensor)

        # Forward pass with regular precision
        output_regular = linear_regular(input_tensor)

        # The outputs should have different precision characteristics
        assert (
            output_high_precision.dtype == torch.bfloat16
        )  # Output dtype matches input
        assert output_regular.dtype == torch.bfloat16

        # Test that high precision version is actually computing with higher precision
        # by using float32 reference computation
        input_f32 = input_tensor.to(torch.float32)
        weight_f32 = linear_high_precision.weight.to(torch.float32)
        bias_f32 = linear_high_precision.bias.to(torch.float32)
        reference_output = F.linear(input_f32, weight_f32, bias_f32).to(torch.bfloat16)

        # High precision output should be closer to reference than regular output
        error_high_precision = torch.abs(
            output_high_precision - reference_output
        ).mean()
        error_regular = torch.abs(output_regular - reference_output).mean()

        # Allow some tolerance but high precision should be better or equal
        assert error_high_precision <= error_regular
        assert error_regular - error_high_precision > 1e-4

    def test_linear_precision_casting_float16(self, device: torch.device) -> None:
        """Test that float16 inputs get cast to higher precision during matmul."""
        in_features, out_features = 64, 32
        batch_size = 4

        # Create Linear layer with high_precision=True
        linear_high_precision = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            high_precision=True,
        ).to(device)

        # Create regular Linear layer for comparison
        linear_regular = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            high_precision=False,
        ).to(device)

        # Copy weights to ensure same parameters
        linear_regular.weight.data.copy_(linear_high_precision.weight.data)
        linear_regular.bias.data.copy_(linear_high_precision.bias.data)

        linear_regular.weight = nn.Parameter(linear_regular.weight.to(torch.float16))
        linear_regular.bias = nn.Parameter(linear_regular.bias.to(torch.float16))

        # Create float16 input
        input_tensor = torch.randn(
            batch_size, in_features, device=device, dtype=torch.float16
        )

        # Forward pass with high precision (no manual autocast needed)
        output_high_precision = linear_high_precision(input_tensor)

        # Forward pass with regular precision
        output_regular = linear_regular(input_tensor)

        # The outputs should have different precision characteristics
        assert (
            output_high_precision.dtype == torch.float16
        )  # Output dtype matches input
        assert output_regular.dtype == torch.float16

        # Test that high precision version is actually computing with higher precision
        input_f32 = input_tensor.to(torch.float32)
        weight_f32 = linear_high_precision.weight.to(torch.float32)
        bias_f32 = linear_high_precision.bias.to(torch.float32)
        reference_output = F.linear(input_f32, weight_f32, bias_f32).to(torch.float16)

        # High precision output should be closer to reference than regular output
        error_high_precision = torch.abs(
            output_high_precision - reference_output
        ).mean()
        error_regular = torch.abs(output_regular - reference_output).mean()

        # Allow some tolerance but high precision should be better or equal
        assert error_high_precision <= error_regular
        assert error_regular - error_high_precision > 1e-4


class TestAttentionFunction:
    """Test attention function with manual QK computation comparison and F.sdpa tests."""

    @pytest.fixture
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_attention_manual_vs_implementation(self, device: torch.device) -> None:
        """Test attention function against manual QK computation."""
        batch_size, n_q, n_kv, d = 2, 16, 20, 64

        # Create test tensors
        q = torch.randn(batch_size, n_q, d, device=device)
        k = torch.randn(batch_size, n_kv, d, device=device)
        v = torch.randn(batch_size, n_kv, d, device=device)
        pair_mask = torch.ones(batch_size, n_q, n_kv, device=device, dtype=torch.bool)

        # Manual attention computation
        # [..., n_kv, d] -> [..., d, n_kv]
        k_transposed = k.transpose(-1, -2)

        # [..., n_q, d], [..., d, n_kv] -> [..., n_q, n_kv]
        attn_weights_manual = q @ k_transposed / (q.shape[-1] ** 0.5)

        # Apply mask (fill False positions with -inf)
        inf = 1e10
        attn_weights_manual = torch.masked_fill(attn_weights_manual, ~pair_mask, -inf)

        # [..., n_q, n_kv]
        attn_weights_manual = F.softmax(attn_weights_manual, dim=-1)

        # [..., n_q, n_kv], [..., n_kv, d] -> [..., n_q, d]
        manual_output = attn_weights_manual @ v

        # Implementation output
        impl_output = attention(q, k, v, pair_mask, inf=inf)

        # Compare outputs
        torch.testing.assert_close(manual_output, impl_output, rtol=1e-5, atol=1e-6)

    def test_attention_with_bias(self, device: torch.device) -> None:
        """Test attention function with attention bias."""
        batch_size, n_q, n_kv, d = 2, 16, 20, 64

        # Create test tensors
        q = torch.randn(batch_size, n_q, d, device=device)
        k = torch.randn(batch_size, n_kv, d, device=device)
        v = torch.randn(batch_size, n_kv, d, device=device)
        pair_mask = torch.ones(batch_size, n_q, n_kv, device=device, dtype=torch.bool)
        attn_bias = torch.randn(batch_size, n_q, n_kv, device=device)

        # Manual attention computation with bias
        k_transposed = k.transpose(-1, -2)
        attn_weights_manual = q @ k_transposed / (q.shape[-1] ** 0.5)

        # Add bias
        attn_weights_manual = attn_weights_manual + attn_bias

        # Apply mask
        inf = 1e10
        attn_weights_manual = torch.masked_fill(attn_weights_manual, ~pair_mask, -inf)
        attn_weights_manual = F.softmax(attn_weights_manual, dim=-1)
        manual_output = attn_weights_manual @ v

        # Implementation output
        impl_output = attention(q, k, v, pair_mask, attn_bias=attn_bias, inf=inf)

        # Compare outputs
        torch.testing.assert_close(manual_output, impl_output, rtol=1e-5, atol=1e-6)

    def test_attention_sdpa_without_bias(self, device: torch.device) -> None:
        """Test F.scaled_dot_product_attention without attention bias."""

        batch_size, n_heads, n_q, n_kv, d = 2, 8, 32, 40, 64

        # Create test tensors in multi-head format
        q = torch.randn(batch_size, n_heads, n_q, d, device=device)
        k = torch.randn(batch_size, n_heads, n_kv, d, device=device)
        v = torch.randn(batch_size, n_heads, n_kv, d, device=device)

        # Create attention mask (True for valid positions)
        attn_mask = torch.ones(
            batch_size, n_heads, n_q, n_kv, device=device, dtype=torch.bool
        )
        attn_mask[:, :, :, -10:] = False  # Mask last 10 key positions

        # Use F.scaled_dot_product_attention
        sdpa_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

        # Manual computation for comparison
        scale = 1.0 / math.sqrt(d)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply mask
        attn_weights = torch.masked_fill(attn_weights, ~attn_mask, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        manual_output = torch.matmul(attn_weights, v)

        # Compare outputs (allow some tolerance due to optimizations)
        torch.testing.assert_close(sdpa_output, manual_output, rtol=1e-4, atol=1e-5)

    def test_attention_with_partial_mask(self, device: torch.device) -> None:
        """Test attention function with partial masking."""
        batch_size, n_q, n_kv, d = 2, 16, 20, 64

        # Create test tensors
        q = torch.randn(batch_size, n_q, d, device=device)
        k = torch.randn(batch_size, n_kv, d, device=device)
        v = torch.randn(batch_size, n_kv, d, device=device)

        # Create partial mask (mask out some positions)
        pair_mask = torch.ones(batch_size, n_q, n_kv, device=device, dtype=torch.bool)
        pair_mask[:, :, -5:] = False  # Mask last 5 key positions
        pair_mask[:, -3:, :] = False  # Mask last 3 query positions

        # Manual computation
        k_transposed = k.transpose(-1, -2)
        attn_weights_manual = q @ k_transposed / (q.shape[-1] ** 0.5)

        inf = 1e10
        attn_weights_manual = torch.masked_fill(attn_weights_manual, ~pair_mask, -inf)
        attn_weights_manual = F.softmax(attn_weights_manual, dim=-1)
        manual_output = attn_weights_manual @ v

        # Implementation output
        impl_output = attention(q, k, v, pair_mask, inf=inf)

        # Compare outputs
        torch.testing.assert_close(manual_output, impl_output, rtol=1e-5, atol=1e-6)


def _gather_pair_embedding_in_dense_trunk(x: Tensor, idx_q: Tensor, idx_k: Tensor):
    """
    Selectively gather elements from a tensor using two sets of indices.

        x: [..., N_token, N_token, d]
        idx_q: [N_b, N_q]
        idx_k: [N_b, N_k]

    Return:
        y: [..., N_b, N_q, N_k, d]
            where y[..., b, i, j, :] = x[..., idx_q[b, i], idx_k[b, j], :]

    Note: this function will be correct as long as there is not a mapping like
    real atom -> masked token in the atom_to_token_idx. This should be safe
    since real atoms only map to real tokens.
    """
    idx_q = idx_q.long()
    idx_k = idx_k.long()
    assert len(idx_q.shape) == len(idx_k.shape) == 2

    # Get the shape parameters
    N_b, N_q = idx_q.shape
    N_k = idx_k.shape[1]

    # Expand idx_q and idx_k to match the shape required for advanced indexing
    idx_q_expanded = idx_q.unsqueeze(-1).expand(-1, -1, N_k)
    idx_k_expanded = idx_k.unsqueeze(1).expand(-1, N_q, -1)

    # Use advanced indexing to gather the desired elements
    y = x[..., idx_q_expanded, idx_k_expanded, :]

    return y


class TestGatherPairEmbedding:
    """Test gather_pair_embedding_in_dense_trunk and its batched version."""

    @pytest.fixture
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_gather_equivalence_no_batch_dims(self, device: torch.device) -> None:
        """Test equivalence when no batch dimensions are present."""
        N_token, d = 50, 64
        N_b, N_q, N_k = 8, 16, 128

        # Create test tensors
        x = torch.randn(N_token, N_token, d, device=device)
        idx_q = torch.randint(0, N_token, (N_b, N_q), device=device)
        idx_k = torch.randint(0, N_token, (N_b, N_k), device=device)

        # Test unbatched function
        result_unbatched = _gather_pair_embedding_in_dense_trunk(x, idx_q, idx_k)

        # Test batched function with no batch dims
        result_batched = gather_trunk_pair_embedding(x, idx_q, idx_k)

        # Results should be identical
        torch.testing.assert_close(
            result_unbatched, result_batched, rtol=1e-10, atol=1e-10
        )

        # Check output shape
        expected_shape = (N_b, N_q, N_k, d)
        assert result_unbatched.shape == expected_shape
        assert result_batched.shape == expected_shape

    def test_gather_equivalence_with_batch_dims(self, device: torch.device) -> None:
        """Test batched function vs loop over unbatched function."""
        batch_size = 4
        N_token, d = 30, 32
        N_b, N_q, N_k = 5, 10, 128

        # Create test tensors with batch dimension
        x_batched = torch.randn(batch_size, N_token, N_token, d, device=device)
        idx_q_batched = torch.randint(0, N_token, (batch_size, N_b, N_q), device=device)
        idx_k_batched = torch.randint(0, N_token, (batch_size, N_b, N_k), device=device)

        # Method 1: Use batched function
        result_batched = gather_trunk_pair_embedding(
            x_batched, idx_q_batched, idx_k_batched
        )

        # Method 2: Loop over batch dimension and stack results
        results_list = []
        for i in range(batch_size):
            result_i = _gather_pair_embedding_in_dense_trunk(
                x_batched[i], idx_q_batched[i], idx_k_batched[i]
            )
            results_list.append(result_i)
        result_stacked = torch.stack(results_list, dim=0)

        # Results should be identical
        torch.testing.assert_close(
            result_batched, result_stacked, rtol=1e-10, atol=1e-10
        )

        # Check output shape
        expected_shape = (batch_size, N_b, N_q, N_k, d)
        assert result_batched.shape == expected_shape
        assert result_stacked.shape == expected_shape

    def test_gather_with_multiple_batch_dims(self, device: torch.device) -> None:
        """Test batched function with multiple batch dimensions."""
        batch_size1, batch_size2 = 3, 2
        N_token, d = 25, 16
        N_b, N_q, N_k = 4, 6, 5

        # Create test tensors with multiple batch dimensions
        x_batched = torch.randn(
            batch_size1, batch_size2, N_token, N_token, d, device=device
        )
        idx_q_batched = torch.randint(
            0, N_token, (batch_size1, batch_size2, N_b, N_q), device=device
        )
        idx_k_batched = torch.randint(
            0, N_token, (batch_size1, batch_size2, N_b, N_k), device=device
        )

        # Method 1: Use batched function
        result_batched = gather_trunk_pair_embedding(
            x_batched, idx_q_batched, idx_k_batched
        )

        # Method 2: Double loop and stack results
        results_list = []
        for i in range(batch_size1):
            results_inner = []
            for j in range(batch_size2):
                result_ij = gather_trunk_pair_embedding(
                    x_batched[i, j], idx_q_batched[i, j], idx_k_batched[i, j]
                )
                results_inner.append(result_ij)
            results_list.append(torch.stack(results_inner, dim=0))
        result_stacked = torch.stack(results_list, dim=0)

        # Results should be identical
        torch.testing.assert_close(
            result_batched, result_stacked, rtol=1e-10, atol=1e-10
        )

        # Check output shape
        expected_shape = (batch_size1, batch_size2, N_b, N_q, N_k, d)
        assert result_batched.shape == expected_shape
        assert result_stacked.shape == expected_shape
