"""Tests for layout module - rearrange, gather, and broadcast operations."""

import pytest
import torch

from artios.model.modules.layout import (
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

    @pytest.mark.parametrize(
        "n_atom,num_keys,num_queries",
        [
            (256, 128, 32),  # Standard config
            (512, 128, 32),  # More atoms
            (1024, 128, 32),  # Large atom count
            (2048, 128, 32),  # Very large
            (256, 64, 32),  # Smaller key window
            (
                512,
                256,
                32,
            ),  # Larger key window (need n_atom > num_keys for interior blocks)
            (512, 128, 64),  # Different query size
            (1024, 256, 64),  # Large windows
        ],
    )
    def test_all_interior_blocks_match_direct_slicing(
        self, n_atom: int, num_keys: int, num_queries: int
    ):
        """All interior blocks should match direct slicing for various configurations.

        An interior block is one where the key window [start, end) falls entirely
        within the original tensor bounds [0, n_atom).

        For block i:
        - window_start_in_original = i * num_queries - pad_size
        - window_end_in_original = window_start + num_keys

        Block is interior if window_start >= 0 AND window_end <= n_atom.
        """
        c = 16
        # Sequential values make verification straightforward
        k_x = torch.arange(n_atom * c, dtype=torch.float32).reshape(n_atom, c)

        result = rearrange_to_key_subset(
            k_x,
            rearrange_dim=-2,
            num_keys=num_keys,
            num_queries=num_queries,
        )

        n_blocks = n_atom // num_queries
        pad_size = (num_keys - num_queries) // 2

        interior_blocks_found = 0

        for block_idx in range(n_blocks):
            window_start = block_idx * num_queries - pad_size
            window_end = window_start + num_keys

            # Check if this block is interior (window fully within bounds)
            if window_start >= 0 and window_end <= n_atom:
                interior_blocks_found += 1
                expected = k_x[window_start:window_end]
                actual = result[block_idx]

                torch.testing.assert_close(
                    actual,
                    expected,
                    msg=f"Block {block_idx} mismatch for n_atom={n_atom}, "
                    f"num_keys={num_keys}, num_queries={num_queries}. "
                    f"Window [{window_start}, {window_end})",
                )

        # Verify we actually tested some interior blocks
        assert interior_blocks_found > 0, (
            f"No interior blocks found for n_atom={n_atom}, "
            f"num_keys={num_keys}, num_queries={num_queries}"
        )

    def test_interior_blocks_with_batch_dims(self):
        """Interior block slicing should work correctly with batch dimensions."""
        batch = 4
        n_atom = 512
        c = 16
        num_keys, num_queries = self.NUM_KEYS, self.NUM_QUERIES

        # Create batched sequential tensor
        k_x = torch.arange(batch * n_atom * c, dtype=torch.float32).reshape(
            batch, n_atom, c
        )

        result = rearrange_to_key_subset(
            k_x,
            rearrange_dim=-2,
            num_keys=num_keys,
            num_queries=num_queries,
        )

        n_blocks = n_atom // num_queries
        pad_size = (num_keys - num_queries) // 2

        # Check all interior blocks for each batch element
        for b in range(batch):
            for block_idx in range(n_blocks):
                window_start = block_idx * num_queries - pad_size
                window_end = window_start + num_keys

                if window_start >= 0 and window_end <= n_atom:
                    expected = k_x[b, window_start:window_end]
                    actual = result[b, block_idx]

                    torch.testing.assert_close(
                        actual,
                        expected,
                        msg=f"Batch {b}, Block {block_idx} mismatch",
                    )

    def test_interior_blocks_window_overlap_correctness(self):
        """Verify overlapping windows correctly share elements.

        Adjacent interior blocks should share (num_keys - num_queries) elements.
        This tests that the overlap pattern is correct.
        """
        n_atom = 512
        c = 8
        num_keys, num_queries = self.NUM_KEYS, self.NUM_QUERIES

        k_x = torch.arange(n_atom * c, dtype=torch.float32).reshape(n_atom, c)

        result = rearrange_to_key_subset(
            k_x,
            rearrange_dim=-2,
            num_keys=num_keys,
            num_queries=num_queries,
        )

        n_blocks = n_atom // num_queries
        pad_size = (num_keys - num_queries) // 2
        overlap_size = num_keys - num_queries

        for block_idx in range(n_blocks - 1):
            # Check if both this block and next block are interior
            window_start_curr = block_idx * num_queries - pad_size
            window_end_curr = window_start_curr + num_keys
            window_start_next = (block_idx + 1) * num_queries - pad_size
            window_end_next = window_start_next + num_keys

            if (
                window_start_curr >= 0
                and window_end_curr <= n_atom
                and window_start_next >= 0
                and window_end_next <= n_atom
            ):
                # The last overlap_size elements of block i should equal
                # the first overlap_size elements of block i+1
                curr_tail = result[block_idx, -overlap_size:]
                next_head = result[block_idx + 1, :overlap_size]

                torch.testing.assert_close(
                    curr_tail,
                    next_head,
                    msg=f"Overlap mismatch between blocks {block_idx} and {block_idx + 1}",
                )

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
    """rearrange_to_key_subset enforces a set of size/parity preconditions."""

    @pytest.mark.parametrize(
        "n_atom, num_keys, num_queries, reason",
        [
            (128, 32, 32, "num_keys must exceed num_queries"),
            (128, 127, 32, "num_keys must be even"),
            (128, 128, 31, "num_queries must be even"),
            (100, 128, 32, "n_atom must be divisible by num_queries"),
        ],
    )
    def test_invalid_config_raises(
        self, n_atom: int, num_keys: int, num_queries: int, reason: str
    ):
        k_x = torch.randn(n_atom, 16)
        with pytest.raises(AssertionError):
            rearrange_to_key_subset(
                k_x, rearrange_dim=-2, num_keys=num_keys, num_queries=num_queries
            )


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

    @pytest.mark.parametrize(
        "batch_shape, n_token, d, n_b, n_q, n_k",
        [
            ((), 16, 32, 4, 8, 16),  # no batch dim
            ((3,), 16, 32, 4, 8, 16),  # single batch dim
            ((2, 3), 12, 16, 3, 6, 10),  # multi batch dim
        ],
    )
    def test_matches_legacy_implementation(
        self, batch_shape, n_token, d, n_b, n_q, n_k
    ):
        """New implementation must match legacy across batch-dim ranks."""
        x = torch.randn(*batch_shape, n_token, n_token, d)
        idx_q = torch.randint(0, n_token, (*batch_shape, n_b, n_q))
        idx_k = torch.randint(0, n_token, (*batch_shape, n_b, n_k))

        new_result = gather_trunk_pair_embedding(x, idx_q, idx_k)
        legacy_result = gather_trunk_pair_embedding_legacy(x, idx_q, idx_k)

        torch.testing.assert_close(new_result, legacy_result)

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


class TestBroadcastTokenToLocalAtomPair:
    """Test broadcast_token_to_local_atom_pair function."""

    num_queries = 32
    num_keys = 64

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
