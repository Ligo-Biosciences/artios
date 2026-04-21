import pytest
import torch

from artios.data.collate.padding import pad_tensor

# Covers: rank 1-4 tensors, batched and unbatched, single-dim and two-dim pad.
_SHAPE_CASES = [
    # Unbatched (num_batch_dims=0)
    ((128,), (16,), 0),  # 1D token
    ((128, 32), (16,), 0),  # 2D token, restype-shaped
    ((128, 128), (16, 16), 0),  # Token pair (two-dim pad)
    ((512,), (32,), 0),  # 1D atom
    ((512, 3), (32,), 0),  # 2D atom, coord-shaped
    ((512, 4, 64), (32,), 0),  # 3D atom, ref_atom_name_chars
    ((128, 5, 3), (16,), 0),  # Backbone coords
    ((512, 128), (32,), 0),  # ref_element-shaped
    # Batched (num_batch_dims=1)
    ((4, 128), (16,), 1),
    ((4, 128, 32), (16,), 1),
    ((4, 128, 128), (16, 16), 1),
    ((4, 512), (32,), 1),
    ((4, 512, 3), (32,), 1),
    ((4, 512, 4, 64), (32,), 1),
    ((4, 128, 5, 3), (16,), 1),
    ((4, 512, 128), (32,), 1),
]


@pytest.mark.parametrize("shape, pad_sizes, num_batch_dims", _SHAPE_CASES)
def test_pad_tensor_preserves_values_and_zero_fills(
    shape: tuple[int, ...], pad_sizes: tuple[int, ...], num_batch_dims: int
) -> None:
    """pad_tensor must: (1) grow each target dim by its pad amount, (2) leave
    the original slice untouched, (3) fill the pad region with zeros."""
    tensor = torch.randn(*shape)
    padded = pad_tensor(tensor, pad_sizes=pad_sizes, num_batch_dims=num_batch_dims)

    # (1) shape: only the `len(pad_sizes)` dims after the batch dims grow.
    expected_shape = list(shape)
    for i, pad in enumerate(pad_sizes):
        expected_shape[num_batch_dims + i] += pad
    assert padded.shape == tuple(expected_shape)

    # (2) original slice equals input.
    slicer = [slice(None)] * padded.ndim
    for i, _ in enumerate(pad_sizes):
        slicer[num_batch_dims + i] = slice(0, shape[num_batch_dims + i])
    assert torch.equal(padded[tuple(slicer)], tensor)

    # (3) each pad region is zero-filled.
    for i, _ in enumerate(pad_sizes):
        pad_dim = num_batch_dims + i
        pad_slicer = [slice(None)] * padded.ndim
        pad_slicer[pad_dim] = slice(shape[pad_dim], None)
        assert torch.all(
            padded[tuple(pad_slicer)] == 0
        ), f"pad region along dim {pad_dim} was not zero-filled"


def test_pad_tensor_raises_when_pad_sizes_exceed_available_dims() -> None:
    """Passing more pad_sizes than non-batch dims must fail loudly at the
    assertion rather than silently truncating."""
    tensor = torch.randn(4, 128)  # 2D: one batch dim, one non-batch dim
    with pytest.raises(AssertionError, match="pad_sizes .* exceed available"):
        pad_tensor(tensor, pad_sizes=(10, 10, 10), num_batch_dims=1)
