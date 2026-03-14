import torch

from artios.model.modules.layout import (
    _get_key_trunk_indices,
    rearrange_to_key_subset,
    rearrange_to_query_subset,
)
from artios.utils.tensor_utils import expand_at_dim


class TestGetKeyTrunkIndices:
    """Test the _get_key_trunk_indices function."""

    def test_basic_functionality(self):
        """Test basic key trunk indices generation."""
        n = 16
        n_queries = 4
        n_keys = 12

        key_indices = _get_key_trunk_indices(n, n_queries, n_keys)

        n_trunks = n // n_queries  # 4 trunks
        assert key_indices.shape == (n_trunks, n_keys)

        # Check that indices are within bounds
        assert torch.all(key_indices >= 0)
        assert torch.all(key_indices < n)

    def test_trunk_centers_positioning(self):
        """Test that trunk centers are correctly positioned."""
        n = 16
        n_queries = 4
        n_keys = 12

        key_indices = _get_key_trunk_indices(n, n_queries, n_keys)

        # First trunk center should be at n_queries // 2 = 8
        # Key window should be centered around this
        first_trunk_center_idx = n_keys // 2 - 1  # Middle index of first trunk
        print(key_indices)
        assert key_indices[0, first_trunk_center_idx] == first_trunk_center_idx

    def test_boundary_handling_left(self):
        """Test handling of left boundary when keys extend beyond start."""
        n = 32
        n_queries = 8
        n_keys = 16

        key_indices = _get_key_trunk_indices(n, n_queries, n_keys)

        # First trunk center is at 4, with keys from -4 to 11
        # Should be shifted to start at 0
        assert key_indices[0, 0] == 0

    def test_boundary_handling_right(self):
        """Test handling of right boundary when keys extend beyond end."""
        n = 32
        n_queries = 8
        n_keys = 16

        key_indices = _get_key_trunk_indices(n, n_queries, n_keys)

        # Last trunk center is at 28, with keys from 20 to 35
        # Should be shifted to end at 31
        assert key_indices[-1, -1] == n - 1

    def test_caching(self):
        """Test that the function is cached correctly."""
        import time

        n = 3000
        n_queries = 8
        n_keys = 16

        # Time the first call
        start_time = time.perf_counter()
        result1 = _get_key_trunk_indices(n, n_queries, n_keys)
        first_call_time = time.perf_counter() - start_time

        # Time the second call with same parameters
        start_time = time.perf_counter()
        result2 = _get_key_trunk_indices(n, n_queries, n_keys)
        second_call_time = time.perf_counter() - start_time

        # Should be the exact same object due to caching
        assert result1 is result2

        # Second call should be significantly faster (at least 10x faster)
        print("first call time: ", first_call_time)
        print("second call time: ", second_call_time)
        assert (
            second_call_time < first_call_time / 10
        ), f"Caching not working: first call took {first_call_time:.6f}s, second call took {second_call_time:.6f}s"

    def test_different_parameters_different_cache(self):
        """Test that different parameters create different cached results."""
        result1 = _get_key_trunk_indices(32, 8, 16)
        result2 = _get_key_trunk_indices(64, 8, 16)

        # Should be different objects
        assert result1 is not result2
        assert not torch.equal(result1, result2)


class TestRearrangeQKToDenseTrunk:
    """Test the query and key rearrangement functions together."""

    def test_basic_single_tensor(self):
        """Test basic functionality with single tensors."""
        batch_size = 1
        seq_len = 16
        hidden_dim = 4
        n_queries = 4
        n_keys = 12

        atoms = expand_at_dim(
            expand_at_dim(torch.arange(seq_len), 0, batch_size), -1, hidden_dim
        )

        q_trunked = rearrange_to_query_subset(
            atoms, rearrange_dim=1, num_queries=n_queries
        )
        k_trunked = rearrange_to_key_subset(
            atoms, rearrange_dim=1, num_keys=n_keys, num_queries=n_queries
        )

        n_trunks = seq_len // n_queries

        # Check output shapes
        print("q_trunked: ", q_trunked)
        print("k_trunked:", k_trunked)
        assert q_trunked.shape == (batch_size, n_trunks, n_queries, hidden_dim)
        assert k_trunked.shape == (batch_size, n_trunks, n_keys, hidden_dim)

    def test_mask_tensor(self):
        """Test functionality without a channel dimension as would hold when a mask is given."""
        batch_size = 1
        seq_len = 16
        n_queries = 4
        n_keys = 12

        atoms = expand_at_dim(torch.arange(seq_len), 0, batch_size)

        q_trunked = rearrange_to_query_subset(
            atoms, rearrange_dim=1, num_queries=n_queries
        )
        k_trunked = rearrange_to_key_subset(
            atoms, rearrange_dim=1, num_keys=n_keys, num_queries=n_queries
        )

        n_trunks = seq_len // n_queries

        # Check output shapes
        print("q_trunked: ", q_trunked)
        print("k_trunked:", k_trunked)
        assert q_trunked.shape == (batch_size, n_trunks, n_queries)
        assert k_trunked.shape == (batch_size, n_trunks, n_keys)

    def test_different_dimensions(self):
        """Test with different target dimensions."""
        seq_len = 32
        batch_size = 2
        hidden_dim = 64
        n_queries = 8
        n_keys = 16

        # Test with dim=0
        q = torch.randn(seq_len, batch_size, hidden_dim)
        k = torch.randn(seq_len, batch_size, hidden_dim)

        q_trunked = rearrange_to_query_subset(q, rearrange_dim=0, num_queries=n_queries)
        k_trunked = rearrange_to_key_subset(
            k, rearrange_dim=0, num_keys=n_keys, num_queries=n_queries
        )

        n_trunks = seq_len // n_queries

        assert q_trunked.shape == (n_trunks, n_queries, batch_size, hidden_dim)
        assert k_trunked.shape == (n_trunks, n_keys, batch_size, hidden_dim)

    def test_negative_dimensions(self):
        """Test with negative dimension indices."""
        batch_size = 2
        seq_len = 32
        hidden_dim = 64
        n_queries = 8
        n_keys = 16

        q = torch.randn(batch_size, seq_len, hidden_dim)
        k = torch.randn(batch_size, seq_len, hidden_dim)

        # Use -2 to refer to sequence dimension
        q_trunked = rearrange_to_query_subset(
            q, rearrange_dim=-2, num_queries=n_queries
        )
        k_trunked = rearrange_to_key_subset(
            k, rearrange_dim=-2, num_keys=n_keys, num_queries=n_queries
        )

        n_trunks = seq_len // n_queries

        assert q_trunked.shape == (batch_size, n_trunks, n_queries, hidden_dim)
        assert k_trunked.shape == (batch_size, n_trunks, n_keys, hidden_dim)

    def test_preserve_data_content(self):
        """Test that the data content is preserved correctly in trunking."""
        seq_len = 16
        hidden_dim = 8
        n_queries = 4
        n_keys = 8

        # Create simple sequential data for easy verification
        q = torch.arange(seq_len * hidden_dim, dtype=torch.float32).view(
            1, seq_len, hidden_dim
        )
        k = (
            torch.arange(seq_len * hidden_dim, dtype=torch.float32).view(
                1, seq_len, hidden_dim
            )
            * 2
        )

        q_trunked = rearrange_to_query_subset(q, rearrange_dim=1, num_queries=n_queries)
        k_trunked = rearrange_to_key_subset(
            k, rearrange_dim=1, num_keys=n_keys, num_queries=n_queries
        )

        # Check that first trunk's first query matches original first n_queries elements
        expected_q_first_trunk = q[0, :n_queries, :]  # First 4 sequence elements
        assert torch.equal(q_trunked[0, 0, :, :], expected_q_first_trunk)

        # Check that key trunking follows the key indices pattern
        # First trunk should contain keys around the first trunk center
        key_indices = _get_key_trunk_indices(seq_len, n_queries, n_keys)
        expected_k_first_trunk = k[0, key_indices[0], :]
        assert torch.equal(k_trunked[0, 0, :, :], expected_k_first_trunk)
