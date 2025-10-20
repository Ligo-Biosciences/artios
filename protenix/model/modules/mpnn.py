from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from protenix.model.modules.mpnn_layers import (
    MPNNDecoderLayer,
    MPNNEncoderLayer,
)


class AllatomMPNN(nn.Module):
    """All-atom message passing neural network for sequence design.

    Args:
        num_residue_types (int): Number of residue types (32 for all biomolecules)
        hidden_dim (int): Hidden dimension for all network layers
        num_encoder_layers (int): Number of encoder layers
        num_decoder_layers (int): Number of decoder layers
        num_neighbours (int): Maximum number of neighbors per residue
        positional_encoding_dim (int, optional): Positional encoding dimension. Defaults to 16.
        positional_encoding_max_relative_feats (int, optional): Max relative features. Defaults to 32.
        num_rbf_kernels (int, optional): Number of RBF kernels for distance encoding. Defaults to 16.
        dropout (float): Dropout probability for regularization. Defaults to 0.0
        coord_noise (float): Noise scale for coordinate perturbation during training. Defaults to 0.1
    """

    def __init__(
        self,
        num_residue_types: int,
        hidden_dim: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_neighbours: int,
        positional_encoding_dim: int = 16,
        positional_encoding_max_relative_feats: int = 32,
        num_rbf_kernels: int = 16,
        dropout: float = 0.0,
        coord_noise: float = 0.1,
    ):
        super().__init__()

        self.num_residue_types = num_residue_types
        self.hidden_dim = hidden_dim
        self.num_neighbours = num_neighbours
        self.positional_encoding_dim = positional_encoding_dim
        self.num_rbf_kernels = num_rbf_kernels
        self.dropout = dropout
        self.coord_noise = coord_noise

        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        self.featurizer = None

        self.encoder = nn.ModuleList(
            [MPNNEncoderLayer() for _ in range(num_encoder_layers)]
        )
        self.decoder = nn.ModuleList(
            [MPNNDecoderLayer() for _ in range(num_decoder_layers)]
        )

        # Projection layers
        self.sequence_ids_to_hidden = nn.Embedding()
        self.edge_to_hidden = nn.Linear()
        self.linear_out = nn.Linear()

    def _generate_random_decoding_order(
        self,
        node_feats: Float[Tensor, "batch seqlen hidden_dim"],
    ) -> Float[Tensor, "batch seq_len"]:
        """Generate a random decoding order.

        NOTE: we can add spatial smoothing to this later like the one in the Chroma paper.
        """
        return torch.argsort(
            torch.randn(node_feats.shape[:-1], device=node_feats.device), dim=-1
        )

    def _generate_sequential_masks(
        self,
        decoding_order: Float[Tensor, "batch seqlen"],
        edge_index: Float[Tensor, "batch seqlen neighbours"],
        node_mask: Float[Tensor, "batch seqlen"],
    ) -> Tuple[
        Float[Tensor, "batch seqlen neighbours"],  # forward attention mask
        Float[Tensor, "batch seqlen neighbours"],  # backward attention mask
    ]:
        """Generate autoregressive attention masks based on decoding order.

        Creates forward and backward attention masks to enforce autoregressive generation order.
        Forward mask allows attention to future positions, backward mask allows attention to
        past positions only.

        Args:
            decoding_order: Permutation defining generation order
            edge_index: Graph edge connectivity matrix
            node_mask: Mask for valid sequence positions

        Returns:
            Tuple of forward and backward atention masks for autoregressive generation
        """
        mask_size = edge_index.shape[-2]
        permutation_matrix_reverse = F.one_hot(
            decoding_order, num_classes=mask_size
        ).float()

        # TODO: I don't understand this einsum as well as I should, it comes from ProteinMPNN repo
        # We should re-write this to be more readable.
        edge_space_backward_mask = torch.einsum(
            "ij, biq, bjp -> bqp",
            (
                1
                - torch.triu(torch.ones(mask_size, mask_size, device=edge_index.device))
            ),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )

        # map back to kNN adjacency space and apply node masking
        backward_attn_mask = torch.gather(
            edge_space_backward_mask, dim=-1, index=edge_index
        )
        mask_1D = node_mask[..., None]

        backward_attn_mask = mask_1D * backward_attn_mask
        forward_attn_mask = mask_1D * (1.0 - backward_attn_mask)

        return forward_attn_mask, backward_attn_mask

    def _run_decoder_step(
        self,
        node_feats: Float[Tensor, "batch seqlen node_dim"],
        edge_feats: Float[Tensor, "batch seqlen neighbours edge_dim"],
        sequence_ids: Float[Tensor, "batch seqlen"],
        edge_index: Float[Tensor, "batch seqlen neighbours"],
        node_mask: Float[Tensor, "batch seqlen"],
        forward_attn_mask: Float[Tensor, "batch seqlen neighbours"],
        backward_attn_mask: Float[Tensor, "batch seqlen neighbours"],
        edge_mask: Float[Tensor, "batch seqlen neighbours"],
        zero_shot: bool = False,
    ) -> Float[Tensor, "batch seqlen num_residue_types"]:
        """Run a single decoder step with attention masking.

        Args:
            node_feats: Node features from encoder
            edge_feats: Base edge features
            sequence_ids: Current sequence state (residue indices)
            edge_index: Graph edge connectivity
            node_mask: Mask indicating valid sequence positions
            forward_attn_mask: Forward attention mask
            backward_attn_mask: Backward attention mask
            zero_shot: Whether to use zero-shot mode. Defaults to False.

        Returns:
            Raw logits over residue types.
        """
        pass

    def run_encoder_trunk(
        self,
        backbone_coords: Float[Tensor, "batch seqlen 4 3"],
        node_mask: Float[Tensor, "batch seqlen"],
        residue_idx: Float[Tensor, "batch seqlen"],
    ) -> Dict[str, Tensor]:
        """Runs the encoder trunk of the model. The encoder's output is cached between decoder steps.

        Args:
            context_coords: context atom coordinates
            node_mask: Mask indicating valid sequence positions
            residue_idx: Residue position indices for positional encoding

        Returns:
            dict: Dictionary containing node_feats, edge_feats, edge_index, and edge_mask
        """
        # Initialize features

        # Make attention masks

        # Run encoder
        return {
            # node features
            # edge features
            # edge index
            # edge mask
        }

    def forward(
        self,
        context_coords: Float[Tensor, "batch seqlen 4 3"],
        sequence_ids: Float[Tensor, "batch seqlen"],
        node_mask: Float[Tensor, "batch seqlen"],
        residue_idx: Float[Tensor, "batch seqlen"],
        decoding_order: Optional[Float[Tensor, "batch seqlen"]] = None,
        zero_shot: bool = False,
    ) -> Tuple[dict, dict]:
        """Forward pass for autoregressive protein sequence generation.

        Encodes protein structure into node representations, then generates sequence
        probabilities using masked attention.

        Args:
            context_coords
            sequence_ids
            node_mask
            residue_idx
            decoding_order
            zero_shot

        Returns:
            Tuple[dict, dict]: Dictionary with logits and stats dictionary
        """

        # Run encoder trunk

        # Select decoding order

        # Generate masks

        # Run decoder step
        stats_dict = {}
        return {}, stats_dict

    def sample():
        """Sample protein sequence autoregressively from the model."""
        pass
