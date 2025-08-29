import functools

import torch
from torch import Tensor

from protenix.model.utils import index_at_dim, reshape_at_dim


@functools.lru_cache(maxsize=10)
def _get_key_trunk_indices(n, n_queries, n_keys):
    # Key windows centered around the trunk centers
    trunk_centers = torch.arange(n_queries // 2, n, n_queries)
    key_indices = (
        trunk_centers[:, None] + torch.arange(-n_keys // 2, n_keys // 2)[None, :]
    ).to(dtype=int)

    # Shift key indices to be within bounds
    for row in range(key_indices.shape[0]):
        if key_indices[row, 0] < 0:
            key_indices[row, :] -= key_indices[row, 0].item()
        elif key_indices[row, -1] > n - 1:
            overflow = key_indices[row, -1] - (n - 1)
            key_indices[row, :] -= overflow
    return key_indices


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

    assert n_q % num_queries == 0
    assert num_queries % 2 == 0

    n_trunks = n_q // num_queries
    return reshape_at_dim(q_x, dim=rearrange_dim, target_shape=(n_trunks, num_queries))


def rearrange_to_key_subset(
    k_x: Tensor, rearrange_dim: int, num_keys: int, num_queries: int
) -> Tensor:
    """Rearrange k_x to a key subset.

    Args:
        k_x (Tensor): key tensor. [..., n_k, ...] (n_k is at dimension rearrange_dim)
        rearrange_dim (int): dimension to rearrange.
        n_keys (int): number of keys in the subset.
        n_queries (int): number of queries in the subset.

    Returns:
        Tensor: key subset. [..., n_k // n_keys, n_keys, ...]
    """
    n_atom = k_x.size(rearrange_dim)

    assert num_keys > num_queries
    assert num_keys % 2 == 0
    assert num_queries % 2 == 0
    assert n_atom % num_queries == 0

    key_indices = _get_key_trunk_indices(n_atom, num_queries, num_keys)
    key_indices = key_indices.to(k_x.device)
    return index_at_dim(k_x, rearrange_dim, key_indices)


def gather_trunk_pair_embedding(x: Tensor, idx_q: Tensor, idx_k: Tensor):
    """
    Selectively gather elements from a tensor using two sets of indices.
    Batched version of gather_pair_embedding_in_dense_trunk

    Args:
        x (Tensor): [..., N_token, N_token, d]
        idx_q (Tensor): [..., N_b, N_q]
        idx_k (Tensor): [..., N_b, N_k]

    Returns:
        y (Tensor): [..., N_b, N_q, N_k, d]
            where y[..., b, i, j, :] = x[..., idx_q[..., b, i], idx_k[..., b, j], :]
    """
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
    return x[ranges]


def broadcast_token_to_local_atom_pair(
    z_token: Tensor,
    atom_to_token_idx: Tensor,
    n_queries: int,
    n_keys: int,
) -> Tensor:
    """Broadcast token pair embedding to atom pair embedding

    Args:
        z_token (Tensor): token pair embedding
            [..., N_token, N_token, d]
        atom_to_token_idx (Tensor): map atom idx to token idx
            [..., N_atom]
        n_queries (int): number of queries
        n_keys (int): number of keys

    Returns:
        z_gathered_blocked (Tensor): atom pair embedding, with local blocked shape
            [..., n_trunks, n_queries, n_keys, d]
    """

    # [..., N_atom] -> [..., n_trunks, n_queries] and [..., n_trunks, n_keys]
    atom_to_token_idx_q = rearrange_to_query_subset(
        atom_to_token_idx, rearrange_dim=-1, num_queries=n_queries
    )
    atom_to_token_idx_k = rearrange_to_key_subset(
        atom_to_token_idx, rearrange_dim=-1, num_keys=n_keys, num_queries=n_queries
    )

    z_gathered_blocked = gather_trunk_pair_embedding(
        z_token, idx_q=atom_to_token_idx_q, idx_k=atom_to_token_idx_k
    )

    return z_gathered_blocked
