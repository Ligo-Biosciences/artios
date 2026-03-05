"""Discrete MPNN with masked diffusion for sequence design.

This module provides the MPNN with masked diffusion modeling:
- `encode()`: Structure encoding (cached during sampling)
- `decode()`: Single decoder step returning logits from masked input
- `forward()`: Training pass with random masking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from artios.data.features import FullContext
from artios.model.modules import message_passing as mp
from artios.model.modules.mpnn_features import MPNNFeaturizer
from artios.model.modules.mpnn_layers import (
    MPNNDecoderLayer,
    MPNNEncoderLayer,
)
from artios.model.modules.primitives import LinearNoBias
from artios.utils.typing import Bool, Num, typecheck


@typecheck
@dataclass
class EncoderOutput:
    """Output from the MPNN encoder.

    Attributes:
        node_feats: Encoded node features
        edge_feats: Encoded edge features
        edge_index: Graph connectivity
        edge_mask: Valid edge mask
    """

    node_feats: Float[Tensor, "... n_token node_dim"]
    edge_feats: Float[Tensor, "... n_token neighbours edge_dim"]
    edge_index: Int[Tensor, "... n_token neighbours"]
    edge_mask: Bool[Tensor, "... n_token neighbours"]


class MPNN(nn.Module):
    """Message passing neural network with masked diffusion for sequence design.

    Encodes protein structure into node/edge representations, then decodes
    masked sequences to predict original tokens. During training, random positions
    are replaced with a learned mask token and the model predicts the original
    residue types at those positions.

    The mask token is added internally at index `num_residue_types`, so the
    input embedding has size `num_residue_types + 1` while the output projection
    predicts over the original `num_residue_types` classes only.

    Args:
        num_residue_types: Number of residue types (32 for all biomolecules)
        node_dim: Hidden dimension for node features
        edge_dim: Hidden dimension for edge features
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        num_neighbours: Maximum number of neighbors per residue
        positional_encoding_dim: Positional encoding dimension. Defaults to 16.
        positional_encoding_max_relative_feats: Max relative features. Defaults to 32.
        num_rbf_kernels: Number of RBF kernels for distance encoding. Defaults to 16.
        dropout: Dropout probability for regularization. Defaults to 0.0
        coord_noise: Noise scale for coordinate perturbation during training. Defaults to 0.1
    """

    def __init__(
        self,
        num_residue_types: int,
        node_dim: int,
        edge_dim: int,
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
        self.mask_token_idx = num_residue_types
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_neighbours = num_neighbours
        self.positional_encoding_dim = positional_encoding_dim
        self.num_rbf_kernels = num_rbf_kernels
        self.dropout = dropout
        self.coord_noise = coord_noise
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        # Featurizer
        self.featurizer = MPNNFeaturizer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            positional_encoding_dim=positional_encoding_dim,
            num_rbf=num_rbf_kernels,
            num_neighbours=num_neighbours,
            coord_noise=coord_noise,
        )

        # Encoder layers
        self.encoder = nn.ModuleList(
            [
                MPNNEncoderLayer(
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    scale=num_neighbours,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Decoder layers
        self.decoder = nn.ModuleList(
            [
                MPNNDecoderLayer(
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    edge_input_dim=2 * node_dim + edge_dim,
                    dropout=dropout,
                    scale=num_neighbours,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        # Input embedding includes mask token (+1)
        self.embed_restype = LinearNoBias(num_residue_types + 1, node_dim)
        self.linear_out = LinearNoBias(node_dim, num_residue_types)

    def encode(self, context: FullContext) -> EncoderOutput:
        """Encode protein structure into node and edge representations.

        Args:
            context: Full context containing structural features.

        Returns:
            EncoderOutput containing node features, edge features, and masks.
        """
        centre_atom_mask = context.mask.token_mask

        node_feats, edge_feats, edge_index = self.featurizer(context)

        edge_mask = mp.gather_nodes(
            h_nodes=centre_atom_mask[..., None], edge_idx=edge_index
        ).squeeze(-1)
        edge_mask = centre_atom_mask[..., None] * edge_mask

        for layer in self.encoder:
            node_feats, edge_feats = checkpoint(
                layer,
                node_feats,
                edge_feats,
                edge_index,
                centre_atom_mask,
                edge_mask,
                use_reentrant=False,
            )

        return EncoderOutput(
            node_feats=node_feats,
            edge_feats=edge_feats,
            edge_index=edge_index,
            edge_mask=edge_mask,
        )

    @typecheck
    def mask_restypes(
        self,
        restypes: Int[Tensor, "batch n_token num_residue_types"],
        mask_rate: Float[Tensor, "batch"],
        token_mask: Optional[Bool[Tensor, "batch n_token"]] = None,
    ) -> tuple[
        Int[Tensor, "batch n_token num_residue_types_plus_1"],
        Bool[Tensor, "batch n_token"],
    ]:
        """Randomly mask residue types by replacing with the mask token.

        Each valid position is independently masked with probability `mask_rate`.

        Args:
            restypes: One-hot encoded residue types (int) [B, N, num_residue_types]
            mask_rate: Per-batch probability of masking each position.
            token_mask: Mask for valid positions. Invalid positions are never masked.

        Returns:
            Tuple of (masked_restypes, is_masked):
                - masked_restypes: One-hot (int) with appended mask token dimension [B, N, num_residue_types + 1]
                - is_masked: Boolean indicating which positions were masked [B, N]
        """
        batch, n_token, _ = restypes.shape
        device = restypes.device

        rand = torch.rand(batch, n_token, device=device)
        is_masked = rand < mask_rate[:, None]

        if token_mask is not None:
            is_masked = is_masked & token_mask

        # Pad restypes with zero for mask dimension: [B, N, 32] -> [B, N, 33]
        padded = F.pad(restypes, (0, 1), value=0)

        # Create mask token one-hot: [33] with 1 at mask_token_idx
        mask_one_hot = torch.zeros(
            self.num_residue_types + 1, device=device, dtype=restypes.dtype
        )
        mask_one_hot[self.mask_token_idx] = 1

        masked_restypes = torch.where(is_masked[..., None], mask_one_hot, padded)

        return masked_restypes, is_masked

    @typecheck
    def decode(
        self,
        encoder_output: EncoderOutput,
        restypes: Num[Tensor, "batch n_token vocab_with_mask"],
        node_mask: Bool[Tensor, "batch n_token"],
    ) -> Float[Tensor, "batch n_token num_residue_types"]:
        """Decode masked sequence to predict original residue types.

        Takes masked restypes (where some positions contain the mask token) and
        produces logits over the original residue vocabulary. Edge features are
        always augmented with sequence embeddings — the mask token embedding
        naturally represents "unknown identity" at masked positions.

        Args:
            encoder_output: Cached encoder output
            restypes: Masked residue types (one-hot including mask token dim)
            node_mask: Mask for valid positions

        Returns:
            Logits over residue types [B, N, num_residue_types]
        """
        node_feats = encoder_output.node_feats
        edge_feats = encoder_output.edge_feats
        edge_index = encoder_output.edge_index
        edge_mask = encoder_output.edge_mask
        dtype = node_feats.dtype

        sequence_embeds = self.embed_restype(restypes.to(dtype))

        # Base edge features augmented with neighbor sequence embeddings (fixed)
        sequence_edge_feats = mp.augment_edges_with_nb_feats(
            edge_feats=edge_feats, features=sequence_embeds, edge_idx=edge_index
        )  # [B, N, K, edge_dim + node_dim]

        for layer in self.decoder:
            # Augment with current node features (updated each layer)
            decoder_edge_feats = mp.augment_edges_with_nb_feats(
                edge_feats=sequence_edge_feats, features=node_feats, edge_idx=edge_index
            )  # [B, N, K, edge_dim + 2*node_dim]

            node_feats = checkpoint(
                layer,
                node_feats,
                decoder_edge_feats,
                node_mask,
                edge_mask,
                use_reentrant=False,
            )

        logits = self.linear_out(node_feats)
        return logits

    @typecheck
    def forward(
        self,
        context: FullContext,
        mask_rate: Float[Tensor, "batch"],
    ) -> tuple[
        Float[Tensor, "batch n_token num_residue_types"],
        Bool[Tensor, "batch n_token"],
        Float[Tensor, "batch n_token node_dim"],
        Float[Tensor, "batch n_token neighbours edge_dim"],
    ]:
        """Training forward pass with random masking.

        Encodes structure, masks input sequence at the given rate, and decodes
        to predict original tokens. Loss should be computed on masked positions only.

        Args:
            context: Full context containing structure and sequence.
            mask_rate: Per-batch fraction of positions to mask.

        Returns:
            Tuple of (logits, is_masked, node_feats, edge_feats)
        """
        token_mask = context.mask.token_mask

        encoder_output = self.encode(context)

        masked_restypes, is_masked = self.mask_restypes(
            context.token.restype, mask_rate, token_mask
        )

        logits = self.decode(
            encoder_output=encoder_output,
            restypes=masked_restypes,
            node_mask=token_mask,
        )

        return logits, is_masked, encoder_output.node_feats, encoder_output.edge_feats
