"""Padding utils."""

import torch
import torch.nn.functional as F

from artios.utils.logger import get_logger

logger = get_logger(__name__)


def get_padding_length(num: int, bin_ends: list[int]) -> int:
    """
    Given an input number of atoms or tokes as well as a list of bin ends, returns the bin
    the input lies in.

    Args
        num (int): Input number of atoms or tokens.
        bin_ends (list[int]): List of bin ends.

    Returns:
        int: Bin end for the given input number.
    """
    for end in bin_ends:
        if num <= end:
            return end
    # Fallback: if num exceeds all bins, pad to num itself so we never
    # return a padding target smaller than the actual size.
    logger.warning(
        f"Input size {num} exceeds all bin ends {bin_ends}. "
        "Padding to exact size — this batch will not benefit from bin-aligned padding."
    )
    return num


def pad_tensor(
    tensor: torch.Tensor,
    pad_sizes: tuple[int, ...],
    num_batch_dims: int = 0,
) -> torch.Tensor:
    """Pad tensor along non-batch dimensions, with batch dimensions handled.

    Padding is applied to the dimensions immediately following the batch dimensions.
    Each entry in `pad_sizes` corresponds to a non-batch dimension (starting at dim=num_batch_dims).

    Args:
        tensor (torch.Tensor): Input tensor to pad. Shape: [*batch, D0, D1, ...].
        pad_sizes (tuple[int, ...]): Amounts to pad each non-batch dimension (in order, dim=num_batch_dims, num_batch_dims+1, ...).
            E.g., pad_sizes=(5,) with num_batch_dims=0 pads dim 0 by 5; pad_sizes=(5,2) with num_batch_dims=1 pads dims 1 and 2.
        num_batch_dims (int, optional): Number of leading batch dimensions (not padded). Defaults to 0.

    Returns:
        torch.Tensor: Tensor padded on specified non-batch dims, shape [*batch, padded_D...].

    Raises:
        AssertionError: if batch dimensions are not present or pad_sizes do not match tensor dims after batch.
    """
    ndim = tensor.ndim
    non_batch_dims = ndim - num_batch_dims
    assert non_batch_dims >= len(pad_sizes), (
        f"pad_sizes {pad_sizes} exceed available ({non_batch_dims}) non-batch dims "
        f"for tensor shape {tensor.shape} and num_batch_dims={num_batch_dims}"
    )

    # torch.nn.functional.pad expects: (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
    # Only pad the right side of each non-batch dim, starting from the dim after the batch dims
    padding = [0] * (2 * ndim)

    for i, pad_size in enumerate(pad_sizes):
        # For the i-th non-batch dim, its index in tensor is num_batch_dims + i
        dim = num_batch_dims + i
        # Position in padding tuple: -2 * (ndim - dim)
        pos = 2 * (ndim - 1 - dim)
        padding[pos + 1] = pad_size  # pad to the right only

    return F.pad(tensor, padding)
