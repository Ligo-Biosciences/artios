"""Discrete MPNN with masked diffusion for sequence design.

This module provides the MPNN with masked diffusion modeling:
- `encode()`: Structure encoding (cached during sampling)
- `decode()`: Single decoder step returning logits from masked input
- `forward()`: Training pass with random masking
"""

from __future__ import annotations

from dataclasses import dataclass

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


@typecheck
@dataclass
class DecoderOutput:
    """Output from the MPNN decoder.

    Attributes:
        logits: Predicted logits over residue types
        node_feats: Decoded node features
        edge_feats: Decoded edge features
    """

    logits: Float[Tensor, "... n_token num_residue_types"]
    node_feats: Float[Tensor, "... n_token node_dim"]
    edge_feats: Float[Tensor, "... n_token neighbours edge_dim"]


@typecheck
@dataclass
class MPNNOutput:
    """Output from the MPNN forward pass.

    Attributes:
        logits: Predicted logits over residue types
        is_masked: Boolean mask indicating which positions were masked
        dec_node_feats: Decoder node features
        dec_edge_feats: Decoder edge features
        edge_index: Graph connectivity
        edge_mask: Valid edge mask
    """

    logits: Float[Tensor, "... n_token num_residue_types"]
    is_masked: Bool[Tensor, "... n_token"]
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
        coord_noise_near_ligand: Noise scale applied to non-ligand token backbone
            coordinates within ``coord_noise_distance_cutoff`` of any ligand token.
        coord_noise_away_from_ligand: Noise scale applied to backbone coordinates
            further than ``coord_noise_distance_cutoff`` from any ligand token.
        coord_noise_distance_cutoff: Distance (angstroms) separating "near" from
            "away" for coordinate noise scaling.
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
        coord_noise_near_ligand: float = 0.1,
        coord_noise_away_from_ligand: float = 0.1,
        coord_noise_distance_cutoff: float = 8.0,
        checkpoint_grads: bool = True,
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
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.checkpoint_grads = checkpoint_grads

        # Featurizer
        self.featurizer = MPNNFeaturizer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            positional_encoding_dim=positional_encoding_dim,
            num_rbf=num_rbf_kernels,
            num_neighbours=num_neighbours,
            coord_noise_near_ligand=coord_noise_near_ligand,
            coord_noise_away_from_ligand=coord_noise_away_from_ligand,
            coord_noise_distance_cutoff=coord_noise_distance_cutoff,
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
            if self.checkpoint_grads:
                node_feats, edge_feats = checkpoint(
                    layer,
                    node_feats,
                    edge_feats,
                    edge_index,
                    centre_atom_mask,
                    edge_mask,
                    use_reentrant=False,
                )
            else:
                node_feats, edge_feats = layer(
                    node_feats, edge_feats, edge_index, centre_atom_mask, edge_mask
                )

        return EncoderOutput(
            node_feats=node_feats,
            edge_feats=edge_feats,
            edge_index=edge_index,
            edge_mask=edge_mask,
        )

    @typecheck
    def pad_and_mask_restypes(
        self,
        restypes: Num[Tensor, "batch n_token num_residue_types"],
        is_masked: Bool[Tensor, "batch n_token"],
    ) -> Num[Tensor, "batch n_token num_residue_types_plus_1"]:
        """Pad restypes with the mask token dimension and apply the mask token.

        Args:
            restypes: One-hot residue types [B, N, R]. Values at masked positions are ignored.
            is_masked: Which positions to replace with the mask token.

        Returns:
            Padded one-hot tensor [B, N, R+1] with mask token at masked positions.
        """
        padded = F.pad(restypes, (0, 1), value=0)  # [B, N, R+1]
        mask_one_hot = torch.zeros(
            self.num_residue_types + 1, device=restypes.device, dtype=restypes.dtype
        )  # [R+1]
        mask_one_hot[self.mask_token_idx] = 1
        return torch.where(is_masked[..., None], mask_one_hot, padded)

    @typecheck
    def decode(
        self,
        encoder_output: EncoderOutput,
        restypes: Num[Tensor, "batch n_token vocab_with_mask"],
        node_mask: Bool[Tensor, "batch n_token"],
    ) -> DecoderOutput:
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
            DecoderOutput containing logits, node features, and edge features.
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

            if self.checkpoint_grads:
                node_feats = checkpoint(
                    layer,
                    node_feats,
                    decoder_edge_feats,
                    node_mask,
                    edge_mask,
                    use_reentrant=False,
                )
            else:
                node_feats = layer(node_feats, decoder_edge_feats, node_mask, edge_mask)

        logits = self.linear_out(node_feats)

        return DecoderOutput(
            logits=logits,
            node_feats=node_feats,
            edge_feats=decoder_edge_feats,
        )

    @typecheck
    def forward(
        self,
        context: FullContext,
        masked_restypes: Num[Tensor, "batch n_token vocab_with_mask"],
        is_masked: Bool[Tensor, "batch n_token"],
    ) -> MPNNOutput:
        """Forward pass with pre-masked sequence input.

        Encodes structure, then decodes masked sequence to predict original tokens.

        Args:
            context: Full context containing structure and sequence.
            masked_restypes: Pre-masked residue types (one-hot including mask token dim).
            is_masked: Boolean mask indicating which positions were masked.

        Returns:
            MPNNOutput containing logits, masking info, and encoder/decoder features.
        """
        encoder_output = self.encode(context)

        decoder_output = self.decode(
            encoder_output=encoder_output,
            restypes=masked_restypes,
            node_mask=context.mask.token_mask,
        )

        return MPNNOutput(
            logits=decoder_output.logits,
            is_masked=is_masked,
            node_feats=decoder_output.node_feats,
            edge_feats=decoder_output.edge_feats,
            edge_index=encoder_output.edge_index,
            edge_mask=encoder_output.edge_mask,
        )
