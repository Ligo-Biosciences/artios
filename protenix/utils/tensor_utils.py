"""Tensor utility functions."""

from typing import Union

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from protenix.utils.typing import typecheck


def expand_at_dim(x: Tensor, dim: int, n: int) -> Tensor:
    """Expand a tensor at specific dim by n times.

    Args:
        x: Input tensor.
        dim: Dimension to expand.
        n: Expand size.

    Returns:
        Expanded tensor of shape [..., n, ...].
    """
    x = x.unsqueeze(dim=dim)
    if dim < 0:
        dim = x.dim() + dim
    before_shape = x.shape[:dim]
    after_shape = x.shape[dim + 1 :]
    return x.expand(*before_shape, n, *after_shape)


def pad_at_dim(
    x: Tensor,
    dim: int,
    pad_length: Union[tuple[int, int], list[int]],
    value: float = 0,
) -> Tensor:
    """Pad input x at dimension dim.

    Args:
        x: Input tensor.
        dim: Padding dimension.
        pad_length: Length to pad to the beginning and end.
        value: Padding value.

    Returns:
        Padded tensor.
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    pad = (pad_length[0], pad_length[1])
    if pad == (0, 0):
        return x
    k = n_dim - (dim + 1)
    if k > 0:
        pad_skip = (0, 0) * k
        pad = (*pad_skip, *pad)
    return nn.functional.pad(x, pad=pad, value=value)


def reshape_at_dim(
    x: Tensor, dim: int, target_shape: Union[tuple[int, ...], list[int]]
) -> Tensor:
    """Reshape dimension dim of x to target_shape.

    Args:
        x: Input tensor.
        dim: Dimension to reshape.
        target_shape: Target shape of dim.

    Returns:
        Reshaped tensor.
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    target_shape = tuple(target_shape)
    target_shape = (*x.shape[:dim], *target_shape)
    if dim + 1 < n_dim:
        target_shape = (*target_shape, *x.shape[dim + 1 :])
    return x.reshape(target_shape)


def index_at_dim(x: Tensor, dim: int, idx: Tensor) -> Tensor:
    """Return x indexed by idx along `dim`, preserving all other dims.

    If idx is [T, K], the result inserts those two dims at `dim`.
    """
    if dim < 0:
        dim += x.ndim
    index = [slice(None)] * x.ndim
    index[dim] = idx
    return x[tuple(index)]


_CDIST_THRESHOLD = 2147400000


@typecheck
def cdist(
    x: Float[Tensor, "... p m"],
    y: Float[Tensor, "... r m"] | None = None,
    p: float = 2.0,
) -> Float[Tensor, "... p r"]:
    """Compute pairwise distances between sets of points.

    Args:
        x: First set of points with shape (..., p, m).
        y: Second set of points with shape (..., r, m). If None, uses x.
        p: The p-norm to use for distance computation. Default is 2.0 (Euclidean).

    Returns:
        Pairwise distances with shape (..., p, r).
    """
    y = y if y is not None else x
    assert x.ndim == y.ndim

    n, m = x.shape[-2], y.shape[-2]
    flat_size = torch.prod(torch.tensor(x.shape[:-2])) * n * m

    if x.is_cuda and flat_size > _CDIST_THRESHOLD:
        # Torch cdist without mm fails when the total number of
        # points is > _threshold (in dimension 3)
        # or 8192 points for batch size 32.
        # To preserve accuracy, we fallback to naive distances
        return _naive_pairwise_distances(x, y)

    return torch.cdist(x1=x, x2=y, compute_mode="donot_use_mm_for_euclid_dist", p=p)


@typecheck
def _naive_pairwise_distances(
    x: Float[Tensor, "... p m"],
    y: Float[Tensor, "... r m"] | None = None,
    eps: float = 1e-10,
) -> Float[Tensor, "... p r"]:
    """Compute pairwise distances using naive method (memory intensive but accurate).

    Args:
        x: First set of points with shape (..., p, m).
        y: Second set of points with shape (..., r, m). If None, uses x.
        eps: Small epsilon for numerical stability.

    Returns:
        Pairwise distances with shape (..., p, r).
    """
    y = y if y is not None else x
    diff = x.unsqueeze(-2) - y.unsqueeze(-3)
    return diff.pow_(2).sum(dim=-1).add_(eps).sqrt_()
