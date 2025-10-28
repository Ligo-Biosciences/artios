"""

TODO: re-write this logic to make it look cleaner.

"""

import torch
from jaxtyping import Float, Int
from torch import Tensor


def gather_nodes(
    h_nodes: Float[Tensor, "batch nodes features"],
    edge_idx: Int[Tensor, "batch nodes neighbours"],
) -> Float[Tensor, "batch nodes neighbours features"]:
    """Gather node features at specified neighbor indices.

    Args:
        h_nodes (Float[Tensor, "batch nodes features"]): Node features
        edge_idx (Int[Tensor, "batch nodes k"]): Neighbor indices

    Returns:
        Float[Tensor, "batch nodes k features"]: Neighbor node features
    """
    edges_flat = edge_idx.view((edge_idx.shape[0], -1))
    edges_flat = edges_flat.unsqueeze(-1).expand(-1, -1, h_nodes.size(2))

    neighbors = torch.gather(h_nodes, 1, edges_flat)
    neighbors = neighbors.view(*edge_idx.shape[:3], -1)

    return neighbors


def gather_edges(
    pairwise_edges: Float[Tensor, "batch nodes nodes features"],
    edge_idx: Int[Tensor, "batch nodes neighbours"],
) -> Float[Tensor, "batch nodes neighbours features"]:
    """Gather edge features at specified neighbor indices.

    Args:
        h_edges (Float[Tensor, "batch nodes neighbours features"]): Edge features
        edge_idx (Int[Tensor, "batch nodes neighbours"]): Neighbor indices

    Returns:
        Float[Tensor, "batch nodes neighbours features"]: Neighbor edge features
    """
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    edges_flat = edge_idx.unsqueeze(-1).expand(-1, -1, -1, pairwise_edges.size(-1))
    edge_features = torch.gather(pairwise_edges, 2, edges_flat)

    return edge_features


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
