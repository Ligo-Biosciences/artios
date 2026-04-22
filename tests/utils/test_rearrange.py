from artios.model.modules.layout import _get_key_trunk_indices


class TestGetKeyTrunkIndices:
    """Test the _get_key_trunk_indices function."""

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
