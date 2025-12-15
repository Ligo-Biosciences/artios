"""Message passing neural network layers."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

import protenix.model.modules.message_passing as mp
from protenix.model.modules.primitives import LayerNorm, LinearNoBias
from protenix.model.utils import expand_at_dim


class BaseMPNNLayer(nn.Module, ABC):
    """Base class for message passing neural net layers that process node and edge features in graph neural networks."""

    @abstractmethod
    def forward(
        self,
        h_nodes: Float[Tensor, "... nodes node_dim"],
        h_edges: Float[Tensor, "... nodes neighbors edge_dim"],
        edge_index: Int[Tensor, "... nodes neighbors"],
        node_mask: Float[Tensor, "... nodes"],
        edge_mask: Float[Tensor, "... nodes neighbors"],
    ) -> tuple[
        Float[Tensor, "... nodes node_dim"],
        Float[Tensor, "... nodes neighbors edge_dim"],
    ]:
        """Forward pass for the encoder layer.

        Args:
            h_nodes: Node features
            h_edges: Edge features
            edge_index: Edge indices
            node_mask: Node mask for valid nodes
            edge_mask: Edge mask for valid edges

        Returns:
            Tuple[Tensor, Tensor]: Updated node and edge features
                [..., nodes, node_dim], [..., nodes, neighbors, edge_dim]
        """
        pass

    def gather_and_concat_neighbourhood(
        self,
        node_emb: Float[Tensor, "... nodes node_dim"],
        edge_emb: Float[Tensor, "... nodes neighbors edge_dim"],
        edge_index: Int[Tensor, "... nodes neighbors"],
    ) -> Float[Tensor, "... nodes neighbors concat_dim"]:
        """Gather neighbour nodes and concatenate with self nodes and edges.

        Args:
            node_emb: Node features
            edge_emb: Edge features
            edge_index: Edge indices

        Returns:
            Concatenated features [..., nodes, neighbors, 2*node_dim + edge_dim]
        """
        n_neighbours = edge_emb.shape[-2]
        h_neighbors = mp.gather_nodes(
            node_emb, edge_index
        )  # [..., nodes, neighbors, node_dim]
        h_nodes_self = expand_at_dim(
            node_emb, dim=-2, n=n_neighbours
        )  # [..., nodes, node_dim] -> [..., nodes, neighbors, node_dim]
        h_nn = torch.cat([h_nodes_self, edge_emb, h_neighbors], dim=-1)
        return h_nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class SwiGLU_MLP(nn.Module):
    def __init__(self, num_in: int, num_hidden, n: int = 2) -> None:
        super().__init__()
        self.W_in = LinearNoBias(num_in, num_hidden * n * 2)
        self.W_out = LinearNoBias(num_hidden * n, num_hidden)
        self.act = nn.SiLU()

    def forward(self, x):
        a, b = self.W_in(x).chunk(2, dim=-1)
        return self.W_out(a * self.act(b))


class MPNNEncoderLayer(BaseMPNNLayer):
    """Message Passing Neural Network encoder layer with neighborhood aggregation."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        dropout: float = 0.0,
        scale: float = 30.0,
    ):
        """Initialize the MPNN encoder layer.

        Args:
            node_dim: Node feature dimensionality
            edge_dim: Edge feature dimensionality
            dropout: Dropout probability. Defaults to 0.0.
            scale: Scaling factor for message aggregation. Defaults to 30.0.
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.scale = scale

        # Node message - 3-layer MLP
        # self.W1 = nn.Linear(2 * node_dim + edge_dim, node_dim, bias=True)
        # self.W2 = nn.Linear(node_dim, node_dim, bias=True)
        # self.W3 = nn.Linear(node_dim, node_dim, bias=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(node_dim)
        self.node_message_mlp = SwiGLU_MLP(
            num_in=2 * node_dim + edge_dim, num_hidden=node_dim, n=1
        )

        # Node feedforward
        # self.dense = PositionWiseFeedForward(
        #     node_dim, node_dim * 4
        # )  # Transition(c_in=node_dim, n=4)
        self.dense = SwiGLU_MLP(num_in=node_dim, num_hidden=node_dim, n=2)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(node_dim)

        # Edge update - 3-layer MLP
        # self.W11 = nn.Linear(2 * node_dim + edge_dim, edge_dim, bias=True)
        # self.W12 = nn.Linear(edge_dim, edge_dim, bias=True)
        # self.W13 = nn.Linear(edge_dim, edge_dim, bias=True)
        self.edge_update_mlp = SwiGLU_MLP(
            num_in=2 * node_dim + edge_dim, num_hidden=edge_dim, n=1
        )
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = LayerNorm(edge_dim)

        self.act = nn.GELU()

    # @torch.compile
    def forward(
        self,
        node_emb: Float[Tensor, "... nodes node_dim"],
        edge_emb: Float[Tensor, "... nodes neighbors edge_dim"],
        edge_index: Int[Tensor, "... nodes neighbors"],
        node_mask: Optional[Float[Tensor, "... nodes"]] = None,
        edge_mask: Optional[Float[Tensor, "... nodes neighbors"]] = None,
    ) -> tuple[
        Float[Tensor, "... nodes node_dim"],
        Float[Tensor, "... nodes neighbors edge_dim"],
    ]:
        """Forward pass through the MPNN encoder layer.

        Args:
            node_emb: Node features
            edge_emb: Edge features
            edge_index: Edge indices
            node_mask: Node mask for valid nodes
            edge_mask: Edge mask for valid edges

        Returns:
            Updated node and edge features:
                [..., nodes, node_dim], [..., nodes, neighbors, edge_dim]
        """
        if node_mask is None:
            node_mask = torch.ones_like(node_emb[..., 0])
        if edge_mask is None:
            edge_mask = torch.ones_like(edge_emb[..., 0])

        # Node message computation
        h_nn = self.gather_and_concat_neighbourhood(
            node_emb=node_emb,
            edge_emb=edge_emb,
            edge_index=edge_index,
        )  # [..., nodes, neighbors, 2*node_dim + edge_dim]

        h_message = self.node_message_mlp(h_nn)
        # self.W3(self.act(self.W2(self.act(self.W1(h_nn)))))
        h_message = h_message * edge_mask[..., None]
        dh = torch.sum(h_message, dim=-2) / self.scale
        node_emb = self.norm1(node_emb + self.dropout1(dh))

        # Node feedforward update
        dh = self.dense(node_emb)
        node_emb = self.norm2(node_emb + self.dropout2(dh))
        node_emb = node_emb * node_mask[..., None]

        # Edge update
        h_nn = self.gather_and_concat_neighbourhood(
            node_emb, edge_emb, edge_index
        )  # [..., nodes, neighbors, 2*node_dim + edge_dim]

        h_message = self.edge_update_mlp(h_nn)
        # self.W13(self.act(self.W12(self.act(self.W11(h_nn)))))
        edge_emb = self.norm3(edge_emb + self.dropout3(h_message))
        edge_emb = edge_emb * edge_mask[..., None]

        return node_emb, edge_emb


class MPNNDecoderLayer(BaseMPNNLayer):
    """Protein Message Passing Neural Network decoder layer for node feature updates."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        edge_input_dim: int,
        dropout: float = 0.0,
        scale: float = 30.0,
    ):
        """Initialize the ProteinMPNN decoder layer.

        Args:
            node_dim: Node feature dimension
            edge_dim: Edge hidden dimension (kept for interface compatibility)
            edge_input_dim: Input edge feature dimension
            dropout: Dropout probability. Defaults to 0.0.
            scale: Scaling factor for message aggregation. Defaults to 30.0.
        """
        super().__init__()

        self.node_dim = node_dim
        self.scale = scale

        # Node message from edges - 3-layer MLP
        # self.W1 = nn.Linear(node_dim + edge_input_dim, node_dim, bias=True)
        # self.W2 = nn.Linear(node_dim, node_dim, bias=True)
        # self.W3 = nn.Linear(node_dim, node_dim, bias=True)
        self.node_message_mlp = SwiGLU_MLP(
            num_in=node_dim + edge_input_dim, num_hidden=node_dim, n=1
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(node_dim)

        # Node feedforward
        # self.dense = PositionWiseFeedForward(
        #     node_dim, node_dim * 4
        # )  # Transition(c_in=node_dim, n=4)
        self.dense = SwiGLU_MLP(num_in=node_dim, num_hidden=node_dim, n=2)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(node_dim)

        self.act = nn.GELU()

    # @torch.compile
    def forward(
        self,
        node_emb: Float[Tensor, "... nodes node_dim"],
        edge_emb: Float[Tensor, "... nodes neighbors edge_feat_dim"],
        node_mask: Optional[Float[Tensor, "... nodes"]] = None,
        edge_mask: Optional[Float[Tensor, "... nodes neighbors"]] = None,
    ) -> Float[Tensor, "... nodes node_dim"]:
        """Forward pass through the ProteinMPNN decoder layer.

        Args:
            node_emb: Node features
            edge_emb: Edge features
            node_mask: Node mask for valid nodes
            edge_mask: Edge mask for valid edges

        Returns:
            Updated node features
        """
        n_neighbours = edge_emb.shape[-2]

        if node_mask is None:
            node_mask = torch.ones_like(node_emb[..., 0])
        if edge_mask is None:
            edge_mask = torch.ones_like(edge_emb[..., 0])

        # Concatenate h_V_i to h_E_ij
        nodes_self = expand_at_dim(
            node_emb, dim=-2, n=n_neighbours
        )  # [..., nodes, neighbors, node_dim]
        h_EV = torch.cat([nodes_self, edge_emb], dim=-1)

        # Node message computation
        h_message = self.node_message_mlp(h_EV)
        # self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        h_message = h_message * edge_mask[..., None]
        dh = torch.sum(h_message, dim=-2) / self.scale
        node_emb = self.norm1(node_emb + self.dropout1(dh))

        # Node feedforward update
        dh = self.dense(node_emb)
        node_emb = self.norm2(node_emb + self.dropout2(dh))
        node_emb = node_emb * node_mask[..., None]

        return node_emb
