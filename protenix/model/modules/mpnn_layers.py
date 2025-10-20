"""Message passing neural network layers."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

import protenix.model.modules.message_passing as mp
from protenix.model import utils
from protenix.model.modules.components import MLP, Transition
from protenix.model.modules.primitives import LayerNorm
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
    ) -> Tuple[
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


class MPNNEncoderLayer(BaseMPNNLayer):
    """Message Passing Neural Network encoder layer with neighborhood aggregation."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        """Initialize the PMPNN encoder layer.

        Args:
            node_dim (int): node dimensionality
            edge_dim (int): edge dimensionality
            hidden_dim (int): Hidden dimension for internal transformations
            output_dim (int): Output dimension for the MLP transformation
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Node message
        self.pre_message_node_norm = LayerNorm(node_dim)
        self.pre_message_edge_norm = LayerNorm(edge_dim)
        self.node_message_mlp = MLP(
            c_in=2 * node_dim + edge_dim, c_out=node_dim, n=2
        )  # TODO: zero-init these and see if that helps

        # Node update
        self.node_transition = Transition(
            c_in=node_dim, n=2, output_linear_init="default"
        )

        # Edge update
        self.pre_edge_update_node_norm = LayerNorm(node_dim)
        self.pre_edge_update_edge_norm = LayerNorm(edge_dim)
        self.edge_update_mlp = MLP(c_in=2 * node_dim + edge_dim, c_out=edge_dim, n=2)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_emb: Float[Tensor, "... nodes node_dim"],
        edge_emb: Float[Tensor, "... nodes neighbors edge_dim"],
        edge_index: Int[Tensor, "... nodes neighbors"],
        node_mask: Optional[Float[Tensor, "... nodes"]] = None,
        edge_mask: Optional[Float[Tensor, "... nodes neighbors"]] = None,
    ) -> Tuple[
        Float[Tensor, "... nodes node_dim"],
        Float[Tensor, "... nodes neighbors edge_dim"],
    ]:
        """Forward pass through the MPNN encoder layer.

        Args:
            node_emb: Node features
            edge_emb: Edge features
            edge_index: Edge indices
            node_mask: Node mask for valid nodes
            edge_mask: Attention mask for valid edges

        Returns:
            Updated node and edge features
                [..., nodes, node_dim], [..., nodes, neighbors, edge_dim]
        """

        if node_mask is None:
            node_mask = torch.ones_like(node_emb[..., 0])
        if edge_mask is None:
            edge_mask = torch.ones_like(edge_emb[..., 0])

        # Node Message
        h_nn = self.gather_and_concat_neighbourhood(
            node_emb=self.pre_message_node_norm(node_emb),
            edge_emb=self.pre_message_edge_norm(edge_emb),
            edge_index=edge_index,
        )
        messages = self.dropout(self.node_message_mlp(h_nn))
        messages = (
            messages * edge_mask[..., None]
        )  # [..., nodes, neighbors, output_dim]
        node_message = utils.masked_mean(messages, mask=edge_mask[..., None], axes=-2)

        node_emb = node_emb + node_message

        # Node MLP update
        node_emb = node_emb + self.dropout(self.node_transition(node_emb))
        node_emb = node_emb * node_mask[..., None]

        # Edge MLP update
        nodes_emb_norm = self.pre_edge_update_node_norm(node_emb)
        edges_emb_norm = self.pre_edge_update_edge_norm(edge_emb)
        h_nn = self.gather_and_concat_neighbourhood(
            nodes_emb_norm, edges_emb_norm, edge_index
        )

        edge_emb = edge_emb + self.dropout(
            self.edge_update_mlp(h_nn)
        )  # [..., nodes, neighbors, edge_dim]
        edge_emb = edge_emb * edge_mask[..., None]

        return node_emb, edge_emb


class MPNNDecoderLayer(BaseMPNNLayer):
    """Protein Message Passing Neural Network decoder layer for node feature updates."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        """Initialize the ProteinMPNN decoder layer.
        TODO: the dimensionality in encoder and decoder layers is not quite right most likely, can be changed during integration.
        Args:
            input_dim (int): Input dimension for node+edge concatenation
            hidden_dim (int): Hidden dimension for internal transformations
            output_dim (int): Output dimension for the MLP transformation
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        self.pre_message_node_norm = LayerNorm(hidden_dim)
        self.pre_message_edge_norm = LayerNorm(hidden_dim)
        self.edge_mlp = MLP(c_in=2 * hidden_dim, c_out=hidden_dim, n=2)

        self.node_transition = Transition(c_in=hidden_dim, n=4)

        self.dropout = nn.Dropout(dropout)

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
            edge_mask: Attention mask for valid edges

        Returns:
            Updated node features
        """
        n_neighbours = edge_emb.shape[-2]

        if node_mask is None:
            node_mask = torch.ones_like(node_emb[..., 0])
        if edge_mask is None:
            edge_mask = torch.ones_like(edge_emb[..., 0])

        # Node message from edges
        nodes_emb_norm = self.pre_message_node_norm(node_emb)
        edges_emb_norm = self.pre_message_edge_norm(edge_emb)
        nodes_self = expand_at_dim(
            nodes_emb_norm, dim=-2, n=n_neighbours
        )  # [..., nodes, neighbors, node_dim+seq_dim]
        h_nn = torch.cat([nodes_self, edges_emb_norm], dim=-1)

        messages = self.dropout(
            self.edge_mlp(h_nn)
        )  # [..., nodes, neighbors, output_dim]
        messages = messages * edge_mask[..., None]
        node_message = utils.masked_mean(messages, mask=edge_mask[..., None], axes=-2)
        node_emb = node_emb + node_message

        # Node update
        node_emb = node_emb + self.dropout(self.node_transition(node_emb))
        node_emb = node_emb * node_mask[..., None]

        return node_emb
