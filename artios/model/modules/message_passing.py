"""Message-passing utilities."""

import torch
from einops import rearrange
from torch import Tensor

from artios.utils.tensor_utils import expand_at_dim
from artios.utils.typing import Float, Int, Shaped, typecheck


@typecheck
def gather_nodes(
    h_nodes: Shaped[Tensor, "batch nodes features"],
    edge_idx: Shaped[Tensor, "batch nodes neighbours"],
) -> Shaped[Tensor, "batch nodes neighbours features"]:
    """Gather node features at specified neighbor indices.

    Args:
        h_nodes: Node features
        edge_idx: Neighbor indices

    Returns:
        Neighbor node features
    """
    # Flatten neighbor indices
    edges_flat = rearrange(edge_idx, "b n k -> b (n k)")

    # Gather
    edges_flat = expand_at_dim(edges_flat, dim=-1, n=h_nodes.shape[-1])
    neighbors = torch.gather(h_nodes, 1, edges_flat)

    # Split
    neighbors = rearrange(neighbors, "b (n k) f -> b n k f", k=edge_idx.size(-1))
    return neighbors


@typecheck
def gather_edges(
    pairwise_edges: Float[Tensor, "batch nodes nodes features"],
    edge_idx: Int[Tensor, "batch nodes neighbours"],
) -> Float[Tensor, "batch nodes neighbours features"]:
    """Gather edge features at specified neighbor indices.

    Takes pairwise edge features [B, N, N, C] and gathers them at neighbor indices [B, N, K]
    to produce neighbor edge features [B, N, K, C].

    Args:
        pairwise_edges: Pairwise edge features for all node pairs
        edge_idx: Neighbor indices for each node

    Returns:
        Neighbor edge features gathered at specified indices
    """
    edges_flat = expand_at_dim(edge_idx, dim=-1, n=pairwise_edges.shape[-1])
    edge_features = torch.gather(pairwise_edges, 2, edges_flat)

    return edge_features


@typecheck
def augment_edges_with_nb_feats(
    edge_feats: Float[Tensor, "batch nodes neighbours edge_features"],
    features: Float[Tensor, "batch nodes node_features"],
    edge_idx: Int[Tensor, "batch nodes neighbours"],
) -> Float[Tensor, "batch nodes neighbours edge_features+node_features"]:
    """Augment edge features with neighbor node features.

    Args:
        edge_feats: Edge features for each neighbor connection
        features: Node features to gather for neighbors
        edge_idx: Neighbor indices for each node

    Returns:
        Float[Tensor, "batch nodes neighbours edge_features+node_features"]:
            Edge features concatenated with neighbor node features
    """
    neighbor_feats = gather_nodes(
        features, edge_idx
    )  # [..., neighbours, node_features]
    edge_feats = torch.cat(
        [edge_feats, neighbor_feats], dim=-1
    )  # [..., edge_features + node_features]

    return edge_feats
