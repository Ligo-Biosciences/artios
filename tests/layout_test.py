"""Tests for layout module - specifically the unfold-based rearrange_to_key_subset."""

import pytest
import torch

from protenix.model.modules.layout import (
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
