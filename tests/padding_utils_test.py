import pytest
import torch

from artios.data.collate.padding import pad_tensor


class TestPadTensor:
    """Test pad_tensor function for Context tensor padding."""

    # Typical dimensions during training
    N_TOKEN = 128
    N_ATOM = 512
    BATCH_SIZE = 4
    PAD_TOKEN = 16
    PAD_ATOM = 32

    # ==========================================================================
    # Unbatched tests (num_batch_dims=0)
    # ==========================================================================

    def test_pad_token_1d_unbatched(self):
        """Test padding 1D token tensor [n_token] without batch."""
        tensor = torch.randn(self.N_TOKEN)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_TOKEN,), num_batch_dims=0)

        assert padded.shape == (self.N_TOKEN + self.PAD_TOKEN,)
        assert torch.equal(padded[: self.N_TOKEN], tensor)
        assert torch.all(padded[self.N_TOKEN :] == 0)

    def test_pad_token_2d_unbatched(self):
        """Test padding 2D token tensor [n_token, 32] (restype) without batch."""
        tensor = torch.randn(self.N_TOKEN, 32)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_TOKEN,), num_batch_dims=0)

        assert padded.shape == (self.N_TOKEN + self.PAD_TOKEN, 32)
        assert torch.equal(padded[: self.N_TOKEN], tensor)
        assert torch.all(padded[self.N_TOKEN :] == 0)

    def test_pad_token_pair_unbatched(self):
        """Test padding token pair tensor [n_token, n_token] (token_bonds) without batch."""
        tensor = torch.randn(self.N_TOKEN, self.N_TOKEN)
        padded = pad_tensor(
            tensor, pad_sizes=(self.PAD_TOKEN, self.PAD_TOKEN), num_batch_dims=0
        )

        expected_size = self.N_TOKEN + self.PAD_TOKEN
        assert padded.shape == (expected_size, expected_size)
        assert torch.equal(padded[: self.N_TOKEN, : self.N_TOKEN], tensor)
        # Check padding regions are zeros
        assert torch.all(padded[self.N_TOKEN :, :] == 0)
        assert torch.all(padded[:, self.N_TOKEN :] == 0)

    def test_pad_atom_1d_unbatched(self):
        """Test padding 1D atom tensor [n_atom] without batch."""
        tensor = torch.randn(self.N_ATOM)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_ATOM,), num_batch_dims=0)

        assert padded.shape == (self.N_ATOM + self.PAD_ATOM,)
        assert torch.equal(padded[: self.N_ATOM], tensor)
        assert torch.all(padded[self.N_ATOM :] == 0)

    def test_pad_atom_2d_unbatched(self):
        """Test padding 2D atom tensor [n_atom, 3] (ref_pos, coordinate) without batch."""
        tensor = torch.randn(self.N_ATOM, 3)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_ATOM,), num_batch_dims=0)

        assert padded.shape == (self.N_ATOM + self.PAD_ATOM, 3)
        assert torch.equal(padded[: self.N_ATOM], tensor)
        assert torch.all(padded[self.N_ATOM :] == 0)

    def test_pad_atom_3d_unbatched(self):
        """Test padding 3D atom tensor [n_atom, 4, 64] (ref_atom_name_chars) without batch."""
        tensor = torch.randn(self.N_ATOM, 4, 64)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_ATOM,), num_batch_dims=0)

        assert padded.shape == (self.N_ATOM + self.PAD_ATOM, 4, 64)
        assert torch.equal(padded[: self.N_ATOM], tensor)
        assert torch.all(padded[self.N_ATOM :] == 0)

    def test_pad_backbone_coords_unbatched(self):
        """Test padding backbone coords [n_token, 5, 3] without batch."""
        tensor = torch.randn(self.N_TOKEN, 5, 3)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_TOKEN,), num_batch_dims=0)

        assert padded.shape == (self.N_TOKEN + self.PAD_TOKEN, 5, 3)
        assert torch.equal(padded[: self.N_TOKEN], tensor)
        assert torch.all(padded[self.N_TOKEN :] == 0)

    # ==========================================================================
    # Batched tests (num_batch_dims=1)
    # ==========================================================================

    def test_pad_token_1d_batched(self):
        """Test padding 1D token tensor [batch, n_token] with batch dim."""
        tensor = torch.randn(self.BATCH_SIZE, self.N_TOKEN)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_TOKEN,), num_batch_dims=1)

        assert padded.shape == (self.BATCH_SIZE, self.N_TOKEN + self.PAD_TOKEN)
        assert torch.equal(padded[:, : self.N_TOKEN], tensor)
        assert torch.all(padded[:, self.N_TOKEN :] == 0)

    def test_pad_token_2d_batched(self):
        """Test padding 2D token tensor [batch, n_token, 32] (restype) with batch dim."""
        tensor = torch.randn(self.BATCH_SIZE, self.N_TOKEN, 32)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_TOKEN,), num_batch_dims=1)

        assert padded.shape == (self.BATCH_SIZE, self.N_TOKEN + self.PAD_TOKEN, 32)
        assert torch.equal(padded[:, : self.N_TOKEN, :], tensor)
        assert torch.all(padded[:, self.N_TOKEN :, :] == 0)

    def test_pad_token_pair_batched(self):
        """Test padding token pair [batch, n_token, n_token] (token_bonds) with batch dim."""
        tensor = torch.randn(self.BATCH_SIZE, self.N_TOKEN, self.N_TOKEN)
        padded = pad_tensor(
            tensor, pad_sizes=(self.PAD_TOKEN, self.PAD_TOKEN), num_batch_dims=1
        )

        expected_size = self.N_TOKEN + self.PAD_TOKEN
        assert padded.shape == (self.BATCH_SIZE, expected_size, expected_size)
        assert torch.equal(padded[:, : self.N_TOKEN, : self.N_TOKEN], tensor)
        assert torch.all(padded[:, self.N_TOKEN :, :] == 0)
        assert torch.all(padded[:, :, self.N_TOKEN :] == 0)

    def test_pad_atom_1d_batched(self):
        """Test padding 1D atom tensor [batch, n_atom] with batch dim."""
        tensor = torch.randn(self.BATCH_SIZE, self.N_ATOM)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_ATOM,), num_batch_dims=1)

        assert padded.shape == (self.BATCH_SIZE, self.N_ATOM + self.PAD_ATOM)
        assert torch.equal(padded[:, : self.N_ATOM], tensor)
        assert torch.all(padded[:, self.N_ATOM :] == 0)

    def test_pad_atom_2d_batched(self):
        """Test padding 2D atom tensor [batch, n_atom, 3] (ref_pos) with batch dim."""
        tensor = torch.randn(self.BATCH_SIZE, self.N_ATOM, 3)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_ATOM,), num_batch_dims=1)

        assert padded.shape == (self.BATCH_SIZE, self.N_ATOM + self.PAD_ATOM, 3)
        assert torch.equal(padded[:, : self.N_ATOM, :], tensor)
        assert torch.all(padded[:, self.N_ATOM :, :] == 0)

    def test_pad_atom_3d_batched(self):
        """Test padding 3D atom tensor [batch, n_atom, 4, 64] (ref_atom_name_chars) with batch dim."""
        tensor = torch.randn(self.BATCH_SIZE, self.N_ATOM, 4, 64)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_ATOM,), num_batch_dims=1)

        assert padded.shape == (self.BATCH_SIZE, self.N_ATOM + self.PAD_ATOM, 4, 64)
        assert torch.equal(padded[:, : self.N_ATOM, :, :], tensor)
        assert torch.all(padded[:, self.N_ATOM :, :, :] == 0)

    def test_pad_backbone_coords_batched(self):
        """Test padding backbone coords [batch, n_token, 5, 3] with batch dim."""
        tensor = torch.randn(self.BATCH_SIZE, self.N_TOKEN, 5, 3)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_TOKEN,), num_batch_dims=1)

        assert padded.shape == (self.BATCH_SIZE, self.N_TOKEN + self.PAD_TOKEN, 5, 3)
        assert torch.equal(padded[:, : self.N_TOKEN, :, :], tensor)
        assert torch.all(padded[:, self.N_TOKEN :, :, :] == 0)

    # ==========================================================================
    # Edge cases
    # ==========================================================================

    def test_pad_zero_padding(self):
        """Test that zero padding returns original tensor."""
        tensor = torch.randn(self.N_TOKEN, 32)
        padded = pad_tensor(tensor, pad_sizes=(0,), num_batch_dims=0)

        assert padded.shape == tensor.shape
        assert torch.equal(padded, tensor)

    def test_pad_empty_pad_sizes(self):
        """Test that empty pad_sizes returns original tensor."""
        tensor = torch.randn(self.N_TOKEN, 32)
        padded = pad_tensor(tensor, pad_sizes=(), num_batch_dims=0)

        assert padded.shape == tensor.shape
        assert torch.equal(padded, tensor)

    def test_pad_preserves_dtype(self):
        """Test that padding preserves tensor dtype."""
        for dtype in [torch.float32, torch.float64, torch.int64, torch.bool]:
            if dtype == torch.bool:
                tensor = torch.randint(0, 2, (self.N_TOKEN,)).bool()
            elif dtype in [torch.int64]:
                tensor = torch.randint(0, 100, (self.N_TOKEN,), dtype=dtype)
            else:
                tensor = torch.randn(self.N_TOKEN, dtype=dtype)

            padded = pad_tensor(tensor, pad_sizes=(self.PAD_TOKEN,), num_batch_dims=0)
            assert padded.dtype == dtype, f"dtype mismatch for {dtype}"

    def test_pad_preserves_device(self):
        """Test that padding preserves tensor device."""
        tensor = torch.randn(self.N_TOKEN)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_TOKEN,), num_batch_dims=0)
        assert padded.device == tensor.device

    def test_pad_assertion_on_insufficient_dims(self):
        """Test that assertion fires when pad_sizes exceeds available dims."""
        tensor = torch.randn(self.BATCH_SIZE, self.N_TOKEN)  # 2D tensor

        with pytest.raises(AssertionError, match="pad_sizes .* exceed available"):
            # Trying to pad 3 dims but only 1 non-batch dim available
            pad_tensor(tensor, pad_sizes=(10, 10, 10), num_batch_dims=1)

    def test_pad_ref_element_128_unbatched(self):
        """Test padding ref_element [n_atom, 128] without batch."""
        tensor = torch.randn(self.N_ATOM, 128)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_ATOM,), num_batch_dims=0)

        assert padded.shape == (self.N_ATOM + self.PAD_ATOM, 128)
        assert torch.equal(padded[: self.N_ATOM], tensor)

    def test_pad_ref_element_128_batched(self):
        """Test padding ref_element [batch, n_atom, 128] with batch dim."""
        tensor = torch.randn(self.BATCH_SIZE, self.N_ATOM, 128)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_ATOM,), num_batch_dims=1)

        assert padded.shape == (self.BATCH_SIZE, self.N_ATOM + self.PAD_ATOM, 128)
        assert torch.equal(padded[:, : self.N_ATOM, :], tensor)


if __name__ == "__main__":
    pytest.main([__file__])
