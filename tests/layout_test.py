"""Tests for layout module - rearrange, gather, and broadcast operations."""

import pytest
import torch

from protenix.model.modules.layout import (
    broadcast_token_to_local_atom_pair,
    gather_trunk_pair_embedding,
    gather_trunk_pair_embedding_legacy,
    rearrange_to_key_subset,
    rearrange_to_key_subset_legacy,
    rearrange_to_query_subset,
)


class TestRearrangeToKeySubset:
    """Test the unfold-based rearrange_to_key_subset implementation."""

    # Standard test parameters
    NUM_QUERIES = 32
    NUM_KEYS = 128

    def test_output_shape_basic(self):
        """Output shape should be [..., n_blocks, num_keys, ...]."""
        n_atom = 256
        c = 16
        k_x = torch.randn(n_atom, c)

        result = rearrange_to_key_subset(
            k_x,
            rearrange_dim=-2,
            num_keys=self.NUM_KEYS,
            num_queries=self.NUM_QUERIES,
        )

        n_blocks = n_atom // self.NUM_QUERIES
        assert result.shape == (n_blocks, self.NUM_KEYS, c)

    def test_output_shape_with_batch_dims(self):
        """Shape should be correct with leading batch dimensions."""
        batch, n_atom, c = 4, 128, 32
        k_x = torch.randn(batch, n_atom, c)

        result = rearrange_to_key_subset(
            k_x,
            rearrange_dim=-2,
            num_keys=self.NUM_KEYS,
            num_queries=self.NUM_QUERIES,
        )

        n_blocks = n_atom // self.NUM_QUERIES
        assert result.shape == (batch, n_blocks, self.NUM_KEYS, c)

    def test_output_shape_with_trailing_dims(self):
        """Shape should be correct with trailing dimensions after rearrange_dim."""
        n_atom, c1, c2 = 256, 16, 8
        k_x = torch.randn(n_atom, c1, c2)

        result = rearrange_to_key_subset(
            k_x,
            rearrange_dim=0,
            num_keys=self.NUM_KEYS,
            num_queries=self.NUM_QUERIES,
        )

        n_blocks = n_atom // self.NUM_QUERIES
        assert result.shape == (n_blocks, self.NUM_KEYS, c1, c2)

    def test_negative_dim_equivalence(self):
        """Negative and positive rearrange_dim should produce same results."""
        batch, n_atom, c = 2, 128, 16
        k_x = torch.randn(batch, n_atom, c)

        result_neg = rearrange_to_key_subset(
            k_x, rearrange_dim=-2, num_keys=self.NUM_KEYS, num_queries=self.NUM_QUERIES
        )
        result_pos = rearrange_to_key_subset(
            k_x, rearrange_dim=1, num_keys=self.NUM_KEYS, num_queries=self.NUM_QUERIES
        )

        torch.testing.assert_close(result_neg, result_pos)

    def test_interior_block_values_match_direct_slicing(self):
        """For interior blocks (not at boundaries), values should match direct slicing."""
        n_atom = 256
        c = 16
        # Use sequential values for easy verification
        k_x = torch.arange(n_atom * c, dtype=torch.float32).reshape(n_atom, c)

        result = rearrange_to_key_subset(
            k_x,
            rearrange_dim=-2,
            num_keys=self.NUM_KEYS,
            num_queries=self.NUM_QUERIES,
        )

        # For interior blocks, the window should directly correspond to slices of the input
        # Block 2 is interior (not affected by boundary padding)
        # Block center is at: 2 * NUM_QUERIES + NUM_QUERIES // 2 = 80
        # Window covers: [80 - 64, 80 + 64) = [16, 144)
        block_idx = 2
        pad_size = (self.NUM_KEYS - self.NUM_QUERIES) // 2
        window_start = block_idx * self.NUM_QUERIES
        # With unfold, after padding, the window extracts positions:
        # [block_idx * NUM_QUERIES, block_idx * NUM_QUERIES + NUM_KEYS) in padded coords
        # which is [block_idx * NUM_QUERIES - pad_size, block_idx * NUM_QUERIES + NUM_KEYS - pad_size) in original coords
        original_start = window_start - pad_size
        original_end = original_start + self.NUM_KEYS

        expected = k_x[original_start:original_end]
        actual = result[block_idx]

        torch.testing.assert_close(actual, expected)

    def test_boundary_blocks_have_padding(self):
        """Boundary blocks should have zero-padded regions."""
        n_atom = 128
        c = 8
        # Fill with non-zero values
        k_x = torch.ones(n_atom, c)

        result = rearrange_to_key_subset(
            k_x,
            rearrange_dim=-2,
            num_keys=self.NUM_KEYS,
            num_queries=self.NUM_QUERIES,
        )

        # First block: window is centered on position NUM_QUERIES // 2 = 16
        # Window covers [-48, 80) in original coords -> first 48 positions are padding
        pad_size = (self.NUM_KEYS - self.NUM_QUERIES) // 2  # = 48

        # The first pad_size elements of block 0 should be zeros (from padding)
        first_block = result[0]
        assert torch.all(
            first_block[:pad_size] == 0
        ), "First block should have leading zeros from padding"

        # The remaining elements should be ones (from the original tensor)
        assert torch.all(
            first_block[pad_size:] == 1
        ), "First block should have ones after padding region"

    def test_last_block_boundary_padding(self):
        """Last block should have trailing zero-padding."""
        n_atom = 128
        c = 8
        k_x = torch.ones(n_atom, c)

        result = rearrange_to_key_subset(
            k_x,
            rearrange_dim=-2,
            num_keys=self.NUM_KEYS,
            num_queries=self.NUM_QUERIES,
        )

        pad_size = (self.NUM_KEYS - self.NUM_QUERIES) // 2
        n_blocks = n_atom // self.NUM_QUERIES

        # Last block: window centered on n_atom - NUM_QUERIES // 2
        # Window extends beyond original tensor, so should have trailing zeros
        last_block = result[n_blocks - 1]
        assert torch.all(
            last_block[-pad_size:] == 0
        ), "Last block should have trailing zeros from padding"
        assert torch.all(
            last_block[:-pad_size] == 1
        ), "Last block should have ones before padding region"

    def test_consistency_with_query_subset_num_blocks(self):
        """Number of blocks should match between query and key subsets."""
        n_atom = 256
        c = 16
        x = torch.randn(n_atom, c)

        query_result = rearrange_to_query_subset(
            x, rearrange_dim=-2, num_queries=self.NUM_QUERIES
        )
        key_result = rearrange_to_key_subset(
            x,
            rearrange_dim=-2,
            num_keys=self.NUM_KEYS,
            num_queries=self.NUM_QUERIES,
        )

        assert (
            query_result.shape[0] == key_result.shape[0]
        ), "Number of blocks should match"

    def test_window_centering_property(self):
        """Windows should be centered on query blocks.

        The center of key window i should align with the center of query block i.
        With padding, this means key_window_center = query_block_center.
        """
        n_atom = 256
        # Create position-encoded tensor: value at position i is i
        k_x = torch.arange(n_atom, dtype=torch.float32).unsqueeze(-1)

        result = rearrange_to_key_subset(
            k_x,
            rearrange_dim=-2,
            num_keys=self.NUM_KEYS,
            num_queries=self.NUM_QUERIES,
        )

        # For each block, verify the center of the key window
        n_blocks = n_atom // self.NUM_QUERIES
        key_center_idx = self.NUM_KEYS // 2
        query_center_idx = self.NUM_QUERIES // 2

        for block_idx in range(n_blocks):
            query_center_pos = block_idx * self.NUM_QUERIES + query_center_idx
            # The key window center should correspond to the query center position
            # (accounting for padding at boundaries)
            key_center_value = result[block_idx, key_center_idx, 0].item()

            if block_idx == 0:
                # First block: padded, so center value is query_center_pos
                assert key_center_value == query_center_pos
            elif block_idx == n_blocks - 1:
                # Last block might be affected by boundary
                # Still check centering is maintained
                assert key_center_value == query_center_pos
            else:
                # Interior blocks: exact centering
                assert key_center_value == query_center_pos

    def test_gradient_flow(self):
        """Gradients should flow through the unfold operation."""
        n_atom = 128
        c = 16
        k_x = torch.randn(n_atom, c, requires_grad=True)

        result = rearrange_to_key_subset(
            k_x,
            rearrange_dim=-2,
            num_keys=self.NUM_KEYS,
            num_queries=self.NUM_QUERIES,
        )

        loss = result.sum()
        loss.backward()

        assert k_x.grad is not None
        assert k_x.grad.shape == k_x.shape
        # Gradient should be non-zero (each element appears in some window)
        assert torch.any(k_x.grad != 0)

    def test_different_num_keys_num_queries(self):
        """Test with different num_keys/num_queries ratios."""
        n_atom = 128
        c = 8

        test_cases = [
            (64, 32),  # 2:1 ratio
            (96, 32),  # 3:1 ratio
            (128, 32),  # 4:1 ratio
            (64, 16),  # different base
        ]

        for num_keys, num_queries in test_cases:
            k_x = torch.randn(n_atom, c)
            result = rearrange_to_key_subset(
                k_x,
                rearrange_dim=-2,
                num_keys=num_keys,
                num_queries=num_queries,
            )
            n_blocks = n_atom // num_queries
            assert result.shape == (
                n_blocks,
                num_keys,
                c,
            ), f"Failed for num_keys={num_keys}, num_queries={num_queries}"

    def test_1d_input(self):
        """Test with 1D input tensor (just the atom dimension)."""
        n_atom = 256
        k_x = torch.randn(n_atom)

        result = rearrange_to_key_subset(
            k_x,
            rearrange_dim=0,
            num_keys=self.NUM_KEYS,
            num_queries=self.NUM_QUERIES,
        )

        n_blocks = n_atom // self.NUM_QUERIES
        assert result.shape == (n_blocks, self.NUM_KEYS)

    def test_multi_batch_dims(self):
        """Test with multiple batch dimensions."""
        b1, b2, n_atom, c = 2, 3, 128, 16
        k_x = torch.randn(b1, b2, n_atom, c)

        result = rearrange_to_key_subset(
            k_x,
            rearrange_dim=-2,
            num_keys=self.NUM_KEYS,
            num_queries=self.NUM_QUERIES,
        )

        n_blocks = n_atom // self.NUM_QUERIES
        assert result.shape == (b1, b2, n_blocks, self.NUM_KEYS, c)


class TestRearrangeToKeySubsetVsLegacy:
    """Compare unfold implementation against legacy index-based implementation."""

    NUM_QUERIES = 32
    NUM_KEYS = 128

    def test_interior_blocks_match_legacy(self):
        """Interior blocks (not at boundaries) should match legacy implementation."""
        n_atom = 256
        c = 16
        k_x = torch.randn(n_atom, c)

        new_result = rearrange_to_key_subset(
            k_x,
            rearrange_dim=-2,
            num_keys=self.NUM_KEYS,
            num_queries=self.NUM_QUERIES,
        )
        legacy_result = rearrange_to_key_subset_legacy(
            k_x,
            rearrange_dim=-2,
            num_keys=self.NUM_KEYS,
            num_queries=self.NUM_QUERIES,
        )

        # Interior blocks should match exactly
        # Blocks 2, 3, 4, 5 are interior for n_atom=256, num_keys=128, num_queries=32
        n_blocks = n_atom // self.NUM_QUERIES
        pad_size = (self.NUM_KEYS - self.NUM_QUERIES) // 2

        # A block is interior if its window doesn't extend beyond [0, n_atom)
        for block_idx in range(n_blocks):
            window_start_original = block_idx * self.NUM_QUERIES - pad_size
            window_end_original = window_start_original + self.NUM_KEYS

            if window_start_original >= 0 and window_end_original <= n_atom:
                # Interior block - should match exactly
                torch.testing.assert_close(
                    new_result[block_idx],
                    legacy_result[block_idx],
                    msg=f"Interior block {block_idx} should match legacy",
                )

    def test_boundary_behavior_difference_documented(self):
        """Document the intentional difference at boundaries.

        Legacy: shifts window to stay in bounds
        New: pads with zeros, maintains centering
        """
        n_atom = 128
        c = 8
        k_x = torch.arange(n_atom * c, dtype=torch.float32).reshape(n_atom, c)

        new_result = rearrange_to_key_subset(
            k_x,
            rearrange_dim=-2,
            num_keys=self.NUM_KEYS,
            num_queries=self.NUM_QUERIES,
        )
        legacy_result = rearrange_to_key_subset_legacy(
            k_x,
            rearrange_dim=-2,
            num_keys=self.NUM_KEYS,
            num_queries=self.NUM_QUERIES,
        )

        # First block should differ
        # New: has zero padding at the start
        # Legacy: window is shifted right to start at 0
        assert not torch.allclose(
            new_result[0], legacy_result[0]
        ), "First block should differ between new and legacy (padding vs shift)"

        # Verify the specific difference
        pad_size = (self.NUM_KEYS - self.NUM_QUERIES) // 2
        # New impl: first pad_size values are 0
        assert torch.all(new_result[0, :pad_size] == 0)
        # Legacy impl: first value corresponds to k_x[0]
        torch.testing.assert_close(legacy_result[0, 0], k_x[0])


class TestAssertions:
    """Test that assertions catch invalid inputs."""

    def test_num_keys_must_exceed_num_queries(self):
        """num_keys must be greater than num_queries."""
        k_x = torch.randn(128, 16)
        with pytest.raises(AssertionError):
            rearrange_to_key_subset(k_x, rearrange_dim=-2, num_keys=32, num_queries=32)

    def test_num_keys_must_be_even(self):
        """num_keys must be even."""
        k_x = torch.randn(128, 16)
        with pytest.raises(AssertionError):
            rearrange_to_key_subset(k_x, rearrange_dim=-2, num_keys=127, num_queries=32)

    def test_num_queries_must_be_even(self):
        """num_queries must be even."""
        k_x = torch.randn(128, 16)
        with pytest.raises(AssertionError):
            rearrange_to_key_subset(k_x, rearrange_dim=-2, num_keys=128, num_queries=31)

    def test_n_atom_must_be_divisible_by_num_queries(self):
        """n_atom must be divisible by num_queries."""
        k_x = torch.randn(100, 16)  # 100 is not divisible by 32
        with pytest.raises(AssertionError):
            rearrange_to_key_subset(k_x, rearrange_dim=-2, num_keys=128, num_queries=32)


class TestDeviceAndDtype:
    """Test device and dtype handling."""

    def test_preserves_dtype(self):
        """Output dtype should match input dtype."""
        for dtype in [torch.float32, torch.float64, torch.float16]:
            k_x = torch.randn(128, 16, dtype=dtype)
            result = rearrange_to_key_subset(
                k_x, rearrange_dim=-2, num_keys=128, num_queries=32
            )
            assert result.dtype == dtype

    @pytest.mark.gpu
    def test_works_on_cuda(self):
        """Should work on CUDA device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        k_x = torch.randn(128, 16, device="cuda")
        result = rearrange_to_key_subset(
            k_x, rearrange_dim=-2, num_keys=128, num_queries=32
        )
        assert result.device.type == "cuda"

    @pytest.mark.gpu
    def test_gradient_flow_cuda(self):
        """Gradients should flow correctly on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        k_x = torch.randn(128, 16, device="cuda", requires_grad=True)
        result = rearrange_to_key_subset(
            k_x, rearrange_dim=-2, num_keys=128, num_queries=32
        )
        loss = result.sum()
        loss.backward()

        assert k_x.grad is not None
        assert k_x.grad.device.type == "cuda"


class TestGatherTrunkPairEmbedding:
    """Test the torch.gather-based gather_trunk_pair_embedding implementation."""

    def test_output_shape_basic(self):
        """Output shape should be [..., n_b, n_q, n_k, d]."""
        n_token, d = 16, 32
        n_b, n_q, n_k = 4, 8, 16

        x = torch.randn(n_token, n_token, d)
        idx_q = torch.randint(0, n_token, (n_b, n_q))
        idx_k = torch.randint(0, n_token, (n_b, n_k))

        result = gather_trunk_pair_embedding(x, idx_q, idx_k)

        assert result.shape == (n_b, n_q, n_k, d)

    def test_output_shape_with_batch(self):
        """Shape should be correct with batch dimensions."""
        batch, n_token, d = 2, 16, 32
        n_b, n_q, n_k = 4, 8, 16

        x = torch.randn(batch, n_token, n_token, d)
        idx_q = torch.randint(0, n_token, (batch, n_b, n_q))
        idx_k = torch.randint(0, n_token, (batch, n_b, n_k))

        result = gather_trunk_pair_embedding(x, idx_q, idx_k)

        assert result.shape == (batch, n_b, n_q, n_k, d)

    def test_output_shape_multi_batch(self):
        """Shape should be correct with multiple batch dimensions."""
        b1, b2, n_token, d = 2, 3, 16, 32
        n_b, n_q, n_k = 4, 8, 16

        x = torch.randn(b1, b2, n_token, n_token, d)
        idx_q = torch.randint(0, n_token, (b1, b2, n_b, n_q))
        idx_k = torch.randint(0, n_token, (b1, b2, n_b, n_k))

        result = gather_trunk_pair_embedding(x, idx_q, idx_k)

        assert result.shape == (b1, b2, n_b, n_q, n_k, d)

    def test_specific_values_match_direct_indexing(self):
        """Verify specific gathered values match direct indexing."""
        n_token, d = 8, 4
        n_b, n_q, n_k = 2, 3, 4

        # Create tensor with unique values for easy verification
        x = torch.arange(n_token * n_token * d, dtype=torch.float32).reshape(
            n_token, n_token, d
        )
        idx_q = torch.tensor([[0, 2, 4], [1, 3, 5]])  # [n_b, n_q]
        idx_k = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])  # [n_b, n_k]

        result = gather_trunk_pair_embedding(x, idx_q, idx_k)

        # Verify specific values
        # result[b, i, j, :] should equal x[idx_q[b, i], idx_k[b, j], :]
        for b in range(n_b):
            for i in range(n_q):
                for j in range(n_k):
                    expected = x[idx_q[b, i], idx_k[b, j], :]
                    actual = result[b, i, j, :]
                    torch.testing.assert_close(
                        actual,
                        expected,
                        msg=f"Mismatch at b={b}, i={i}, j={j}",
                    )

    def test_matches_legacy_implementation(self):
        """New implementation should match legacy for all inputs."""
        n_token, d = 16, 32
        n_b, n_q, n_k = 4, 8, 16

        x = torch.randn(n_token, n_token, d)
        idx_q = torch.randint(0, n_token, (n_b, n_q))
        idx_k = torch.randint(0, n_token, (n_b, n_k))

        new_result = gather_trunk_pair_embedding(x, idx_q, idx_k)
        legacy_result = gather_trunk_pair_embedding_legacy(x, idx_q, idx_k)

        torch.testing.assert_close(new_result, legacy_result)

    def test_matches_legacy_with_batch(self):
        """New implementation should match legacy with batch dimensions."""
        batch, n_token, d = 3, 16, 32
        n_b, n_q, n_k = 4, 8, 16

        x = torch.randn(batch, n_token, n_token, d)
        idx_q = torch.randint(0, n_token, (batch, n_b, n_q))
        idx_k = torch.randint(0, n_token, (batch, n_b, n_k))

        new_result = gather_trunk_pair_embedding(x, idx_q, idx_k)
        legacy_result = gather_trunk_pair_embedding_legacy(x, idx_q, idx_k)

        torch.testing.assert_close(new_result, legacy_result)

    def test_matches_legacy_multi_batch(self):
        """New implementation should match legacy with multiple batch dims."""
        b1, b2, n_token, d = 2, 3, 12, 16
        n_b, n_q, n_k = 3, 6, 10

        x = torch.randn(b1, b2, n_token, n_token, d)
        idx_q = torch.randint(0, n_token, (b1, b2, n_b, n_q))
        idx_k = torch.randint(0, n_token, (b1, b2, n_b, n_k))

        new_result = gather_trunk_pair_embedding(x, idx_q, idx_k)
        legacy_result = gather_trunk_pair_embedding_legacy(x, idx_q, idx_k)

        torch.testing.assert_close(new_result, legacy_result)

    def test_gradient_flow(self):
        """Gradients should flow through gather operation."""
        n_token, d = 16, 32
        n_b, n_q, n_k = 4, 8, 16

        x = torch.randn(n_token, n_token, d, requires_grad=True)
        idx_q = torch.randint(0, n_token, (n_b, n_q))
        idx_k = torch.randint(0, n_token, (n_b, n_k))

        result = gather_trunk_pair_embedding(x, idx_q, idx_k)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_matches_legacy(self):
        """Gradients should match between new and legacy implementations."""
        n_token, d = 16, 32
        n_b, n_q, n_k = 4, 8, 16

        # Same random seed for both
        torch.manual_seed(42)
        x_new = torch.randn(n_token, n_token, d, requires_grad=True)
        torch.manual_seed(42)
        x_legacy = torch.randn(n_token, n_token, d, requires_grad=True)

        idx_q = torch.randint(0, n_token, (n_b, n_q))
        idx_k = torch.randint(0, n_token, (n_b, n_k))

        # Forward and backward for new
        result_new = gather_trunk_pair_embedding(x_new, idx_q, idx_k)
        result_new.sum().backward()

        # Forward and backward for legacy
        result_legacy = gather_trunk_pair_embedding_legacy(x_legacy, idx_q, idx_k)
        result_legacy.sum().backward()

        torch.testing.assert_close(x_new.grad, x_legacy.grad)

    def test_handles_repeated_indices(self):
        """Should correctly handle repeated indices (same token accessed multiple times)."""
        n_token, d = 8, 4

        x = torch.randn(n_token, n_token, d)
        # Use same index multiple times
        idx_q = torch.tensor([[0, 0, 1], [2, 2, 2]])
        idx_k = torch.tensor([[0, 0, 0], [1, 1, 1]])

        new_result = gather_trunk_pair_embedding(x, idx_q, idx_k)
        legacy_result = gather_trunk_pair_embedding_legacy(x, idx_q, idx_k)

        torch.testing.assert_close(new_result, legacy_result)

    def test_preserves_dtype(self):
        """Output dtype should match input dtype."""
        for dtype in [torch.float32, torch.float64, torch.float16]:
            n_token, d = 8, 4
            n_b, n_q, n_k = 2, 3, 4

            x = torch.randn(n_token, n_token, d, dtype=dtype)
            idx_q = torch.randint(0, n_token, (n_b, n_q))
            idx_k = torch.randint(0, n_token, (n_b, n_k))

            result = gather_trunk_pair_embedding(x, idx_q, idx_k)
            assert result.dtype == dtype

    @pytest.mark.gpu
    def test_works_on_cuda(self):
        """Should work on CUDA device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        n_token, d = 16, 32
        n_b, n_q, n_k = 4, 8, 16

        x = torch.randn(n_token, n_token, d, device="cuda")
        idx_q = torch.randint(0, n_token, (n_b, n_q), device="cuda")
        idx_k = torch.randint(0, n_token, (n_b, n_k), device="cuda")

        result = gather_trunk_pair_embedding(x, idx_q, idx_k)

        assert result.device.type == "cuda"
        assert result.shape == (n_b, n_q, n_k, d)

    @pytest.mark.gpu
    def test_matches_legacy_on_cuda(self):
        """New implementation should match legacy on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        n_token, d = 16, 32
        n_b, n_q, n_k = 4, 8, 16

        x = torch.randn(n_token, n_token, d, device="cuda")
        idx_q = torch.randint(0, n_token, (n_b, n_q), device="cuda")
        idx_k = torch.randint(0, n_token, (n_b, n_k), device="cuda")

        new_result = gather_trunk_pair_embedding(x, idx_q, idx_k)
        legacy_result = gather_trunk_pair_embedding_legacy(x, idx_q, idx_k)

        torch.testing.assert_close(new_result, legacy_result)


class TestBroadcastTokenToLocalAtomPair:
    """Test broadcast_token_to_local_atom_pair function."""

    num_queries = 32
    num_keys = 64

    def test_output_shape_basic(self):
        """Output shape should be [..., n_blocks, num_queries, num_keys, d]."""
        n_token, n_atom, d = 8, 64, 16

        z_token = torch.randn(n_token, n_token, d)
        # Each token has 8 atoms (64 atoms / 8 tokens)
        atom_to_token_idx = torch.repeat_interleave(
            torch.arange(n_token), n_atom // n_token
        )

        result = broadcast_token_to_local_atom_pair(
            z_token=z_token,
            atom_to_token_idx=atom_to_token_idx,
            num_queries=self.num_queries,
            num_keys=self.num_keys,
        )

        n_blocks = n_atom // self.num_queries
        assert result.shape == (n_blocks, self.num_queries, self.num_keys, d)

    def test_output_shape_with_batch(self):
        """Shape should be correct with batch dimension."""
        batch, n_token, n_atom, d = 2, 8, 64, 16

        z_token = torch.randn(batch, n_token, n_token, d)
        atom_to_token_idx = (
            torch.repeat_interleave(torch.arange(n_token), n_atom // n_token)
            .unsqueeze(0)
            .expand(batch, -1)
        )

        result = broadcast_token_to_local_atom_pair(
            z_token=z_token,
            atom_to_token_idx=atom_to_token_idx,
            num_queries=self.num_queries,
            num_keys=self.num_keys,
        )

        n_blocks = n_atom // self.num_queries
        assert result.shape == (batch, n_blocks, self.num_queries, self.num_keys, d)

    def test_correct_token_pairs_gathered(self):
        """Verify that correct token pairs are gathered for each atom pair."""
        n_token, n_atom, d = 4, 32, 8
        num_queries, num_keys = 16, 32

        # Create unique values for each token pair
        z_token = torch.arange(n_token * n_token * d, dtype=torch.float32).reshape(
            n_token, n_token, d
        )
        # 8 atoms per token
        atom_to_token_idx = torch.repeat_interleave(
            torch.arange(n_token), n_atom // n_token
        )

        result = broadcast_token_to_local_atom_pair(
            z_token=z_token,
            atom_to_token_idx=atom_to_token_idx,
            num_queries=num_queries,
            num_keys=num_keys,
        )

        # For block 0, query atoms are [0, 16), key atoms are a window centered on queries
        # Query atom 0 belongs to token 0, Query atom 8 belongs to token 1
        # Check that result[0, 0, :, :] corresponds to z_token[token_of_atom_0, token_of_key_atom, :]
        n_blocks = n_atom // num_queries

        # Get windowed key indices using the same function
        key_indices = rearrange_to_key_subset(
            atom_to_token_idx,
            rearrange_dim=-1,
            num_keys=num_keys,
            num_queries=num_queries,
        )
        query_indices = rearrange_to_query_subset(
            atom_to_token_idx, rearrange_dim=-1, num_queries=num_queries
        )

        # Check a few specific positions
        for block_idx in range(n_blocks):
            for q_idx in [0, num_queries // 2, num_queries - 1]:
                for k_idx in [0, num_keys // 2, num_keys - 1]:
                    query_token = query_indices[block_idx, q_idx].item()
                    key_token = key_indices[block_idx, k_idx].item()

                    expected = z_token[int(query_token), int(key_token), :]
                    actual = result[block_idx, q_idx, k_idx, :]

                    torch.testing.assert_close(
                        actual,
                        expected,
                        msg=f"Mismatch at block={block_idx}, q={q_idx}, k={k_idx}",
                    )

    def test_gradient_flow(self):
        """Gradients should flow through the broadcast operation."""
        n_token, n_atom, d = 8, 64, 16

        z_token = torch.randn(n_token, n_token, d, requires_grad=True)
        atom_to_token_idx = torch.repeat_interleave(
            torch.arange(n_token), n_atom // n_token
        )

        result = broadcast_token_to_local_atom_pair(
            z_token=z_token,
            atom_to_token_idx=atom_to_token_idx,
            num_queries=self.num_queries,
            num_keys=self.num_keys,
        )

        loss = result.sum()
        loss.backward()

        assert z_token.grad is not None
        assert z_token.grad.shape == z_token.shape

    def test_boundary_blocks_use_padding_indices(self):
        """At boundaries, unfold-based key indices will be 0 for padding positions.

        This test verifies that boundary blocks gather from token 0 for padded positions.
        In practice, these positions get masked out by trunked_pair_mask.
        """
        n_token, n_atom, d = 4, 32, 8
        num_queries, num_keys = 16, 32

        # Fill z_token with token indices for easy verification
        z_token = torch.zeros(n_token, n_token, d)
        for i in range(n_token):
            for j in range(n_token):
                z_token[i, j, :] = i * 10 + j  # Encode (i, j) in the value

        atom_to_token_idx = torch.repeat_interleave(
            torch.arange(n_token), n_atom // n_token
        )

        result = broadcast_token_to_local_atom_pair(
            z_token=z_token,
            atom_to_token_idx=atom_to_token_idx,
            num_queries=num_queries,
            num_keys=num_keys,
        )

        # For the first block, padded key positions should have gathered from token 0
        # (because unfold pads with 0s, and those 0s become indices)
        pad_size = (num_keys - num_queries) // 2
        first_block = result[0]  # [num_queries, num_keys, d]

        # First pad_size key positions are from padding
        # They should have gathered from token 0 (the padded index value)
        padded_region = first_block[:, :pad_size, :]  # [num_queries, pad_size, d]

        # All padded positions should have j=0 in the encoded value (mod 10)
        # because they indexed into token 0 for the key dimension
        assert torch.all(
            padded_region % 10 == 0
        ), "Padded key positions should gather from token 0"

    def test_interior_blocks_no_padding_effect(self):
        """Interior blocks should not be affected by padding."""
        n_token, n_atom, d = 8, 128, 16
        num_queries, num_keys = 32, 64

        z_token = torch.randn(n_token, n_token, d)
        atom_to_token_idx = torch.repeat_interleave(
            torch.arange(n_token), n_atom // n_token
        )

        result = broadcast_token_to_local_atom_pair(
            z_token=z_token,
            atom_to_token_idx=atom_to_token_idx,
            num_queries=num_queries,
            num_keys=num_keys,
        )

        # For interior blocks, all values should come from valid token pairs
        # (no zeros from padding in the mask region)
        n_blocks = n_atom // num_queries

        # Block 1 should be interior for this configuration
        if n_blocks > 2:
            interior_block = result[1]
            # Check that it has valid data throughout (non-uniform, from randn)
            assert (
                interior_block.std() > 0.1
            ), "Interior block should have varied values"

    def test_preserves_dtype(self):
        """Output dtype should match input dtype."""
        for dtype in [torch.float32, torch.float64]:
            n_token, n_atom, d = 4, 32, 8

            z_token = torch.randn(n_token, n_token, d, dtype=dtype)
            atom_to_token_idx = torch.repeat_interleave(
                torch.arange(n_token), n_atom // n_token
            )

            result = broadcast_token_to_local_atom_pair(
                z_token=z_token,
                atom_to_token_idx=atom_to_token_idx,
                num_queries=16,
                num_keys=32,
            )

            assert result.dtype == dtype

    @pytest.mark.gpu
    def test_works_on_cuda(self):
        """Should work on CUDA device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        n_token, n_atom, d = 8, 64, 16

        z_token = torch.randn(n_token, n_token, d, device="cuda")
        atom_to_token_idx = torch.repeat_interleave(
            torch.arange(n_token, device="cuda"), n_atom // n_token
        )

        result = broadcast_token_to_local_atom_pair(
            z_token=z_token,
            atom_to_token_idx=atom_to_token_idx,
            num_queries=self.num_queries,
            num_keys=self.num_keys,
        )

        assert result.device.type == "cuda"


class TestMaskingIntegration:
    """Test that mask properly handles padding from unfold-based operations.

    These tests verify that when broadcast_token_to_local_atom_pair is used
    with trunked_pair_mask (as in atom_attention.py), padded positions are
    correctly zeroed out.
    """

    def test_mask_zeros_out_padded_positions(self):
        """Verify that mask correctly identifies and zeros padding positions."""
        n_token, n_atom, d = 4, 32, 8
        num_queries, num_keys = 16, 32

        # Create z_token with non-zero values
        z_token = torch.ones(n_token, n_token, d)
        atom_to_token_idx = torch.repeat_interleave(
            torch.arange(n_token), n_atom // n_token
        )

        # Create atom mask (all valid)
        atom_mask = torch.ones(n_atom, dtype=torch.bool)

        # Compute masks using same functions as atom_attention.py
        queries_mask = rearrange_to_query_subset(
            atom_mask, rearrange_dim=-1, num_queries=num_queries
        )
        keys_mask = rearrange_to_key_subset(
            atom_mask, rearrange_dim=-1, num_keys=num_keys, num_queries=num_queries
        )
        trunked_pair_mask = queries_mask[..., :, None] & keys_mask[..., None, :]

        # Get broadcast result
        z_local = broadcast_token_to_local_atom_pair(
            z_token=z_token,
            atom_to_token_idx=atom_to_token_idx,
            num_queries=num_queries,
            num_keys=num_keys,
        )

        # Apply mask (as done in atom_attention.py)
        z_local_masked = z_local * trunked_pair_mask[..., None]

        # At boundary blocks, padded positions should be zeroed by the mask
        pad_size = (num_keys - num_queries) // 2

        # First block: first pad_size key positions should be zero
        first_block_masked = z_local_masked[0]
        assert torch.all(
            first_block_masked[:, :pad_size, :] == 0
        ), "Padded key positions in first block should be zeroed by mask"

        # Non-padded positions should retain their values
        assert torch.all(
            first_block_masked[:, pad_size:, :] == 1
        ), "Non-padded positions should retain values"

    def test_mask_alignment_with_key_indices(self):
        """Verify mask and key indices are aligned (both use same windowing)."""
        n_atom = 64
        num_queries, num_keys = 16, 32

        # Create a mask with some invalid atoms
        atom_mask = torch.ones(n_atom, dtype=torch.bool)
        atom_mask[10:15] = False  # Some atoms are invalid

        # Window the mask
        keys_mask = rearrange_to_key_subset(
            atom_mask, rearrange_dim=-1, num_keys=num_keys, num_queries=num_queries
        )

        # Window some indices
        indices = torch.arange(n_atom)
        keys_indices = rearrange_to_key_subset(
            indices, rearrange_dim=-1, num_keys=num_keys, num_queries=num_queries
        )

        # Verify alignment: where mask is False (padding), indices should be 0
        pad_size = (num_keys - num_queries) // 2

        # At boundaries, padding positions in mask and indices should align
        first_block_mask = keys_mask[0]  # [num_keys]
        first_block_idx = keys_indices[0]  # [num_keys]

        # First pad_size positions are padding (mask=False, idx=0)
        assert torch.all(
            ~first_block_mask[:pad_size]
        ), "Padding positions should have False mask"
        assert torch.all(
            first_block_idx[:pad_size] == 0
        ), "Padding positions should have index 0"
