# import functools
# TODO: re-write this stuff for efficiency. Currently, indexing at a certain dimension and backproping
# is very slow for large number of atoms.

import torch
from einops import rearrange, repeat
from torch import Tensor

from artios.utils.tensor_utils import index_at_dim, pad_at_dim, reshape_at_dim


# @functools.lru_cache(maxsize=10)
def _get_key_trunk_indices(n, n_queries, n_keys):
    # Key windows centered around the trunk centers
    trunk_centers = torch.arange(n_queries // 2, n, n_queries)
    key_indices = (
        trunk_centers[:, None] + torch.arange(-n_keys // 2, n_keys // 2)[None, :]
    ).long()

    # Shift key indices to be within bounds
    for row in range(key_indices.shape[0]):
        if key_indices[row, 0] < 0:
            key_indices[row, :] -= key_indices[row, 0].item()
        elif key_indices[row, -1] > n - 1:
            overflow = key_indices[row, -1] - (n - 1)
            key_indices[row, :] -= overflow
    return key_indices


# @profile_function("rearrange_to_query_subset")
@torch.compiler.disable  # ty: ignore[possibly-missing-attribute]
def rearrange_to_query_subset(
    q_x: Tensor, rearrange_dim: int, num_queries: int
) -> Tensor:
    """Rearrange q_x to a query subset.

    Args:
        q_x (Tensor): query tensor. [..., n_q, ...] (n_q is at dimension rearrange_dim)
        rearrange_dim (int): dimension to rearrange.
        n_queries (int): number of queries in the subset.

    Returns:
        Tensor: query subset. [..., n_q // n_queries, n_queries, ...]
    """
    n_q = q_x.size(rearrange_dim)

    assert n_q % num_queries == 0, f"n_q: {n_q}"
    assert num_queries % 2 == 0, f"num_queries: {num_queries}"

    n_trunks = n_q // num_queries
    return reshape_at_dim(q_x, dim=rearrange_dim, target_shape=(n_trunks, num_queries))


##@profile_function("rearrange_to_key_subset")
@torch.compiler.disable  # ty: ignore[possibly-missing-attribute]
def rearrange_to_key_subset_legacy(
    k_x: Tensor, rearrange_dim: int, num_keys: int, num_queries: int
) -> Tensor:
    """Rearrange k_x to a key subset (legacy index-based implementation).

    Args:
        k_x (Tensor): key tensor. [..., n_k, ...] (n_k is at dimension rearrange_dim)
        rearrange_dim (int): dimension to rearrange.
        n_keys (int): number of keys in the subset.
        n_queries (int): number of queries in the subset.

    Returns:
        Tensor: key subset. [..., n_k // n_keys, n_keys, ...]
    """
    n_atom = k_x.size(rearrange_dim)

    assert num_keys > num_queries, f"num_keys: {num_keys}, num_queries: {num_queries}"
    assert num_keys % 2 == 0, f"num_keys: {num_keys}"
    assert num_queries % 2 == 0, f"num_queries: {num_queries}"
    assert n_atom % num_queries == 0, f"n_atom: {n_atom}"

    key_indices = _get_key_trunk_indices(n_atom, num_queries, num_keys)
    key_indices = key_indices.to(k_x.device)
    return index_at_dim(k_x, rearrange_dim, key_indices)


@torch.compiler.disable  # ty: ignore[possibly-missing-attribute]
def rearrange_to_key_subset(
    k_x: Tensor, rearrange_dim: int, num_keys: int, num_queries: int
) -> Tensor:
    """Rearrange k_x to overlapping key windows using unfold.

    Creates windows of size num_keys with stride num_queries, centered on each
    query block. Boundary windows are zero-padded to maintain centering (unlike
    the legacy implementation which shifts windows to stay in bounds).

    Args:
        k_x: Key tensor with shape [..., n_atom, ...] where n_atom is at rearrange_dim.
        rearrange_dim: Dimension containing the atom/sequence axis to window.
        num_keys: Size of each key window.
        num_queries: Stride between windows (also the query block size).

    Returns:
        Tensor with shape [..., n_blocks, num_keys, ...] where n_blocks = n_atom // num_queries.
    """
    n_atom = k_x.size(rearrange_dim)

    assert num_keys > num_queries, f"num_keys: {num_keys}, num_queries: {num_queries}"
    assert num_keys % 2 == 0, f"num_keys: {num_keys}"
    assert num_queries % 2 == 0, f"num_queries: {num_queries}"
    assert n_atom % num_queries == 0, f"n_atom: {n_atom}"

    # Normalize negative dimension index
    if rearrange_dim < 0:
        rearrange_dim = k_x.ndim + rearrange_dim

    # Pad symmetrically so windows are centered on query blocks
    # Window center should align with query block center (at n_queries // 2 within each block)
    pad_each_side = (num_keys - num_queries) // 2
    k_x_padded = pad_at_dim(
        k_x, dim=rearrange_dim, pad_length=(pad_each_side, pad_each_side)
    )

    # unfold(dim, size, step) extracts sliding windows
    # Output puts the new window dimension at the end
    k_x_unfolded = k_x_padded.unfold(rearrange_dim, num_keys, num_queries)

    # Move window dim from end to right after the block dim
    # Shape: [..., n_blocks, <rest>, num_keys] -> [..., n_blocks, num_keys, <rest>]
    k_x_unfolded = k_x_unfolded.movedim(-1, rearrange_dim + 1)

    return k_x_unfolded


##@profile_function("gather_trunk_pair_embedding")
@torch.compiler.disable  # ty: ignore[possibly-missing-attribute]
def gather_trunk_pair_embedding_legacy(
    x: Tensor, idx_q: Tensor, idx_k: Tensor
) -> Tensor:
    """Legacy implementation using advanced indexing (kept for testing/comparison).

    Args:
        x: Pair embedding tensor [..., n_token, n_token, d].
        idx_q: Query indices [..., n_b, n_q].
        idx_k: Key indices [..., n_b, n_k].

    Returns:
        Gathered pairs [..., n_b, n_q, n_k, d] where
        y[..., b, i, j, :] = x[..., idx_q[..., b, i], idx_k[..., b, j], :].
    """
    assert x.shape[:-3] == idx_q.shape[:-2]
    assert x.shape[:-3] == idx_k.shape[:-2]
    assert idx_q.shape[:-1] == idx_k.shape[:-1]
    idx_q = idx_q.long()
    idx_k = idx_k.long()

    # Grab dim data
    batch_dims = x.shape[:-3]
    no_batch_dims = len(batch_dims)
    dim_q = -3
    dim_k = -2

    # Create torch.arange indices for all the batch dims
    ranges = []
    for i, s in enumerate(batch_dims):
        r = torch.arange(s)
        pre_index_dims = (1,) * i
        post_index_dims = (1,) * (len(idx_q.shape) - i)
        r = r.view(*pre_index_dims, -1, *post_index_dims)
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(x.shape) - no_batch_dims)]
    ranges.extend(remaining_dims)
    ranges[dim_q] = idx_q[..., :, None]
    ranges[dim_k] = idx_k[..., None, :]
    return x[tuple(ranges)]


@torch.compiler.disable  # ty: ignore[possibly-missing-attribute]
def gather_trunk_pair_embedding(x: Tensor, idx_q: Tensor, idx_k: Tensor) -> Tensor:
    """Gather token pairs into local attention blocks using linear indexing.

    More efficient than advanced indexing: flattens the token pair dimensions,
    computes linear indices, and uses a single torch.gather operation.

    Args:
        x: Pair embedding tensor [..., n_token, n_token, d].
        idx_q: Query token indices [..., n_b, n_q].
        idx_k: Key token indices [..., n_b, n_k].

    Returns:
        Gathered pairs [..., n_b, n_q, n_k, d] where
        y[..., b, i, j, :] = x[..., idx_q[..., b, i], idx_k[..., b, j], :].
    """
    assert (
        x.shape[:-3] == idx_q.shape[:-2]
    ), f"Batch shape mismatch: {x.shape[:-3]} vs {idx_q.shape[:-2]}"
    assert (
        x.shape[:-3] == idx_k.shape[:-2]
    ), f"Batch shape mismatch: {x.shape[:-3]} vs {idx_k.shape[:-2]}"
    assert (
        idx_q.shape[:-1] == idx_k.shape[:-1]
    ), f"Block shape mismatch: {idx_q.shape[:-1]} vs {idx_k.shape[:-1]}"

    batch_shape = x.shape[:-3]
    n_token, d = x.shape[-2:]
    n_b, n_q = idx_q.shape[-2:]
    n_k = idx_k.shape[-1]

    # Flatten token pair dims
    x_flat = rearrange(x, "... i j d -> ... (i j) d")

    # Compute 2D -> 1D linear indices: [..., n_b, n_q, n_k]
    idx_q_exp = rearrange(idx_q.long(), "... b q -> ... b q 1")
    idx_k_exp = rearrange(idx_k.long(), "... b k -> ... b 1 k")
    linear_idx = idx_q_exp * n_token + idx_k_exp

    linear_idx_flat = rearrange(linear_idx, "... b q k -> ... (b q k)")
    linear_idx_exp = repeat(linear_idx_flat, "... bqk -> ... bqk d", d=d)

    # Single gather operation: [..., n_token * n_token, d] -> [..., n_b * n_q * n_k, d]
    result_flat = torch.gather(x_flat, dim=-2, index=linear_idx_exp)

    return result_flat.reshape(*batch_shape, n_b, n_q, n_k, d)


# @profile_function("broadcast_token_to_local_atom_pair")
@torch.compiler.disable  # ty: ignore[possibly-missing-attribute]
def broadcast_token_to_local_atom_pair(
    z_token: Tensor,
    atom_to_token_idx: Tensor,
    num_queries: int,
    num_keys: int,
) -> Tensor:
    """Broadcast token pair embedding to local atom pair windows.

    For each local attention block, gathers the token pair embeddings corresponding
    to the (query_atom, key_atom) pairs based on atom-to-token mapping.

    Note: The key indices use unfold-based windowing which zero-pads at boundaries.
    Padding positions will have index 0, gathering from token 0's pairs. This is
    safe because the caller must apply trunked_pair_mask after adding this result
    to p_lm, which zeros out padding positions.

    Args:
        z_token: Token pair embedding [..., n_token, n_token, d].
        atom_to_token_idx: Maps each atom to its parent token [..., n_atom].
        num_queries: Number of query atoms per block.
        num_keys: Number of key atoms per block (window size).

    Returns:
        Local atom pair embedding [..., n_blocks, num_queries, num_keys, d].
    """
    assert atom_to_token_idx.shape[:-1] == z_token.shape[:-3]

    # Window atom-to-token indices into local attention blocks
    # Query subset: simple reshape (no overlap)
    # Key subset: overlapping windows with zero-padding at boundaries
    atom_to_token_idx_q = rearrange_to_query_subset(
        atom_to_token_idx, rearrange_dim=-1, num_queries=num_queries
    )
    atom_to_token_idx_k = rearrange_to_key_subset(
        atom_to_token_idx, rearrange_dim=-1, num_keys=num_keys, num_queries=num_queries
    )

    # Gather token pairs for each local attention block
    z_gathered_blocked = gather_trunk_pair_embedding(
        z_token, idx_q=atom_to_token_idx_q, idx_k=atom_to_token_idx_k
    )

    return z_gathered_blocked
