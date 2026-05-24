#!/usr/bin/env python3

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from artios.model.modules.layout import gather_trunk_pair_embedding
from artios.model.modules.primitives import (
    Linear,
    attention,
    attention_with_bias,
)


class TestLinearPrecision:
    """Test Linear layer precision handling, especially bfloat16 to higher precision casting."""

    @pytest.fixture
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.mark.parametrize("low_dtype", [torch.bfloat16, torch.float16])
    def test_high_precision_beats_low_precision(
        self, device: torch.device, low_dtype: torch.dtype
    ) -> None:
        """``Linear(high_precision=True)`` must upcast matmul to fp32 so the
        result is strictly closer to an fp32 reference than a fully-low-precision
        Linear would be. Pins the 'upcast during matmul, cast back after'
        contract against a regression that silently drops the upcast."""
        in_features, out_features, batch_size = 64, 32, 4

        linear_high = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            high_precision=True,
        ).to(device)
        linear_regular = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            high_precision=False,
        ).to(device)

        # Match weights, then cast the regular layer to low precision.
        linear_regular.weight.data.copy_(linear_high.weight.data)
        linear_regular.bias.data.copy_(linear_high.bias.data)
        linear_regular.weight = nn.Parameter(linear_regular.weight.to(low_dtype))
        linear_regular.bias = nn.Parameter(linear_regular.bias.to(low_dtype))

        input_tensor = torch.randn(
            batch_size, in_features, device=device, dtype=low_dtype
        )
        output_high = linear_high(input_tensor)
        output_regular = linear_regular(input_tensor)

        # Output dtype matches input dtype regardless of internal precision.
        assert output_high.dtype == low_dtype
        assert output_regular.dtype == low_dtype

        # fp32 reference.
        reference_output = F.linear(
            input_tensor.to(torch.float32),
            linear_high.weight.to(torch.float32),
            linear_high.bias.to(torch.float32),
        ).to(low_dtype)

        error_high = (output_high - reference_output).abs().mean()
        error_regular = (output_regular - reference_output).abs().mean()
        assert error_high <= error_regular
        assert error_regular - error_high > 1e-4


class TestAttentionFunction:
    """Test attention function with manual QK computation comparison and F.sdpa tests."""

    @pytest.fixture
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def test_attention_with_bias_broadcasts_batch_dims(
        self, device: torch.device
    ) -> None:
        """Biased attention must match the literal formula with broadcasted bias."""
        batch_size, n_sample, num_heads, n_q, n_kv, d = 2, 3, 4, 8, 10, 16
        q = torch.randn(batch_size, n_sample, num_heads, n_q, d, device=device)
        k = torch.randn(batch_size, n_sample, num_heads, n_kv, d, device=device)
        v = torch.randn(batch_size, n_sample, num_heads, n_kv, d, device=device)
        pair_mask = torch.ones(
            batch_size, 1, 1, n_q, n_kv, device=device, dtype=torch.bool
        )
        pair_mask[..., -2:] = False
        attn_bias = torch.randn(batch_size, 1, num_heads, n_q, n_kv, device=device)
        inf = 1e9

        # Manual formula: pair_bias + scale * (q @ k.T).
        attn_weights_manual = q @ k.transpose(-1, -2) / (d**0.5)
        attn_weights_manual = attn_weights_manual + attn_bias
        attn_weights_manual = torch.masked_fill(attn_weights_manual, ~pair_mask, -inf)
        attn_weights_manual = F.softmax(attn_weights_manual, dim=-1)
        manual_output = attn_weights_manual @ v

        impl_output = attention_with_bias(
            q=q, k=k, v=v, pair_mask=pair_mask, attn_bias=attn_bias, inf=inf
        )

        torch.testing.assert_close(manual_output, impl_output, rtol=1e-5, atol=1e-6)

    def test_attention_with_bias_backward_matches_literal_formula(
        self, device: torch.device
    ) -> None:
        """The baddbmm path should preserve gradients of the literal formula."""
        batch_size, n_sample, num_heads, n_q, n_kv, d = 2, 2, 3, 4, 5, 8
        q = torch.randn(
            batch_size, n_sample, num_heads, n_q, d, device=device, requires_grad=True
        )
        k = torch.randn(
            batch_size, n_sample, num_heads, n_kv, d, device=device, requires_grad=True
        )
        v = torch.randn(
            batch_size, n_sample, num_heads, n_kv, d, device=device, requires_grad=True
        )
        attn_bias = torch.randn(
            batch_size, 1, num_heads, n_q, n_kv, device=device, requires_grad=True
        )
        pair_mask = torch.ones(
            batch_size, 1, 1, n_q, n_kv, device=device, dtype=torch.bool
        )
        pair_mask[..., -1:] = False
        upstream = torch.randn(batch_size, n_sample, num_heads, n_q, d, device=device)
        inf = 1e9

        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)
        attn_bias_ref = attn_bias.detach().clone().requires_grad_(True)

        impl_output = attention_with_bias(
            q=q, k=k, v=v, pair_mask=pair_mask, attn_bias=attn_bias, inf=inf
        )
        impl_loss = (impl_output * upstream).sum()
        impl_grads = torch.autograd.grad(impl_loss, (q, k, v, attn_bias))

        # Manual formula: pair_bias + scale * (q @ k.T).
        attn_weights_manual = q_ref @ k_ref.transpose(-1, -2) / (d**0.5)
        attn_weights_manual = attn_weights_manual + attn_bias_ref
        attn_weights_manual = torch.masked_fill(attn_weights_manual, ~pair_mask, -inf)
        attn_weights_manual = F.softmax(attn_weights_manual, dim=-1)
        manual_output = attn_weights_manual @ v_ref
        manual_loss = (manual_output * upstream).sum()
        manual_grads = torch.autograd.grad(
            manual_loss, (q_ref, k_ref, v_ref, attn_bias_ref)
        )

        for impl_grad, manual_grad in zip(impl_grads, manual_grads):
            torch.testing.assert_close(impl_grad, manual_grad, rtol=1e-5, atol=1e-6)

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
