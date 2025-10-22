from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn

from protenix.model.modules import message_passing as mp
from protenix.model.modules.mpnn_features import MPNNFeaturizer
from protenix.model.modules.mpnn_layers import (
    MPNNDecoderLayer,
    MPNNEncoderLayer,
)
from protenix.model.modules.primitives import LinearNoBias


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
        self.hidden_dim = hidden_dim
        self.num_neighbours = num_neighbours
        self.positional_encoding_dim = positional_encoding_dim
        self.num_rbf_kernels = num_rbf_kernels
        self.dropout = dropout
        self.coord_noise = coord_noise

        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        self.featurizer = MPNNFeaturizer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            positional_encoding_dim=positional_encoding_dim,
            num_rbf=num_rbf_kernels,
            num_neighbours=num_neighbours,
            coord_noise=0.1,
        )

        self.encoder = nn.ModuleList(
            [
                MPNNEncoderLayer(node_dim=node_dim, edge_dim=edge_dim, dropout=dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                MPNNDecoderLayer(
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    edge_input_dim=edge_dim,  # TODO: this should be different
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        # Projection layers
        self.embed_restype = LinearNoBias(num_residue_types, edge_dim)
        self.linear_out = LinearNoBias()

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

    def _prepare_decoder_inputs(
        self,
        node_feats: Float[Tensor, "batch n_token hidden"],
        edge_feats: Float[Tensor, "batch n_token neighbours hidden"],
        sequence_embeds: Float[Tensor, "batch n_token hidden"],
        edge_index: Int[Tensor, "batch n_token neighbours"],
    ) -> Tuple[
        Float[Tensor, "batch n_token neighbours hidden"],
        Float[Tensor, "batch n_token neighbours hidden"],
    ]:
        """Prepare edge features for decoder with sequence and encoder information.

        Creates two types of edge features: one augmented with sequence embeddings for
        autoregressive decoding, and one with encoder features for context.

        Args:
            node_feats: node features from encoder
            edge_feats: base edge features
            sequence_embeds: Embedded sequence information
            edge_index: Graph edge connectivity

        Returns:
            Tuple[Tensor, Tensor]: (encoder_edge_feats, sequence_edge_feats)
        """
        sequence_edge_feats = mp.augment_edges_with_nb_feats(
            edge_feats=edge_feats, features=sequence_embeds, edge_idx=edge_index
        )

        # create encoder context: zero-sequence + encoder node features
        aug_encoder_feats = mp.augment_edges_with_nb_feats(
            edge_feats=edge_feats,
            features=torch.zeros_like(sequence_embeds),
            edge_idx=edge_index,
        )
        encoder_edge_feats = mp.augment_edges_with_nb_feats(
            edge_feats=aug_encoder_feats, features=node_feats, edge_idx=edge_index
        )
        return encoder_edge_feats, sequence_edge_feats

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
        restype: Float[Tensor, "batch seqlen 32"],
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
        sequence_embeds = self.restype_embed(restype)

        encoder_edge_feats, sequence_edge_feats = self._prepare_decoder_inputs(
            node_feats=node_feats,
            edge_feats=edge_feats,
            sequence_embeds=sequence_embeds,
            edge_index=edge_index,
        )

        # decode
        for layer in self.decoder:
            if zero_shot:
                combined_input_edge_feats = encoder_edge_feats * (
                    forward_attn_mask + backward_attn_mask
                )
            else:
                decoder_edge_feats = mp.augment_edges_with_nb_feats(
                    edge_feats=sequence_edge_feats,
                    features=node_feats,
                    edge_idx=edge_index,
                )
                combined_input_edge_feats = (
                    decoder_edge_feats * backward_attn_mask
                    + encoder_edge_feats * forward_attn_mask
                )

            node_feats = layer(
                node_emb=node_feats,
                edge_emb=combined_input_edge_feats,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )

        logits = self.linear_out(node_feats)
        return logits

    def run_encoder_trunk(
        self,
        backbone_coords: Float[Tensor, "batch n_token 5 3"],
        backbone_mask: Float[Tensor, "batch n_token 5"],
        centre_atom_coord: Float[Tensor, "batch n_token 3"],
        centre_atom_mask: Float[Tensor, "batch n_token"],
        input_feature_dict: Dict[str, Tensor],
    ) -> Tuple[
        Float[Tensor, "batch n_token node_dim"],
        Float[Tensor, "batch n_token neighbours edge_dim"],
        Int[Tensor, "batch n_token neighbours"],
        Float[Tensor, "batch n_token neighbours"],
    ]:
        """Runs the encoder trunk of the model. The encoder's output is cached between decoder steps.

        Args:
            backbone_coords: Backbone atom coordinates per token
            backbone_mask: Mask for valid backbone atoms
            centre_atom_coord: Centre atom coordinates per token
            centre_atom_mask: Mask for valid centre atoms
            input_feature_dict: Input feature dict with the following tensors:
                - asym_id, residue_index, entity_id, sym_id, etc.

        Returns:
            Tuple containing:
                node_feats: Node features [..., n_token, node_dim]
                edge_feats: Edge features [..., n_token, neighbours, edge_dim]
                edge_index: Graph edge connectivity [..., n_token, neighbours]
                edge_mask: Edge mask for valid edges [..., n_token, neighbours]
        """
        # Initialize features
        node_feats, edge_feats, edge_index = self.featurizer(
            input_feature_dict=input_feature_dict,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
        )

        # Make attention masks
        edge_mask = mp.gather_nodes(
            h_nodes=centre_atom_mask[..., None], edge_idx=edge_index
        ).squeeze(-1)
        edge_mask = centre_atom_mask[..., None] * edge_mask
        # [..., n_token, neighbours]

        # Encode
        for layer in self.encoder:
            node_feats, edge_feats = layer(
                node_emb=node_feats,
                edge_emb=edge_feats,
                edge_index=edge_index,
                node_mask=centre_atom_mask,
                edge_mask=edge_mask,
            )

        return node_feats, edge_feats, edge_index, edge_mask

    def forward(
        self,
        backbone_coords: Float[Tensor, "batch n_token 5 3"],
        backbone_mask: Float[Tensor, "batch n_token 5"],
        centre_atom_coord: Float[Tensor, "batch n_token 3"],
        centre_atom_mask: Float[Tensor, "batch n_token"],
        restype: Float[Tensor, "batch seqlen 32"],
        input_feature_dict: Dict[str, Tensor],
        decoding_order: Optional[Float[Tensor, "batch seqlen"]] = None,
        zero_shot: bool = False,
    ) -> Tuple[dict, dict]:
        """Forward pass for autoregressive protein sequence generation.

        Encodes protein structure into node representations, then generates sequence
        probabilities using masked attention.

        Args:
            backbone_coords: backbone coordinates for each token
            backbone_mask: backbone mask for each token (which backbone atoms are valid)
            centre_atom_coord: centre atom coordinate for each token
            centre_atom_mask: centre atom mask is resolved mask
            restype: residue types
            input_feature_dict: Input feature dict with the following tensors:
                - asym_id, residue_index, entity_id, sym_id, etc.
            decoding_order: Permutation defining generation order
            zero_shot: whether to use zero-shot mode. Defaults to False

        Returns:
            Tuple[dict, dict]: Dictionary with logits and stats dictionary
        """

        # Run encoder trunk
        node_feats, edge_feats, edge_idx, edge_mask = self.run_encoder_trunk(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
        )

        # Optionally prepare random decoding order
        if decoding_order is None:
            decoding_order = self._generate_random_decoding_order(node_feats)

        # Generate masks
        forward_attn_mask, backward_attn_mask = self._generate_sequential_masks(
            decoding_order=decoding_order,
            edge_index=edge_idx,
            node_mask=centre_atom_mask,
        )

        forward_attn_mask = forward_attn_mask[..., None]
        backward_attn_mask = backward_attn_mask[..., None]

        logits = self._run_decoder_step(
            node_feats=node_feats,
            edge_feats=edge_feats,
            restypes=restype,
            edge_index=edge_idx,
            node_mask=centre_atom_mask,
            forward_attn_mask=forward_attn_mask,
            backward_attn_mask=backward_attn_mask,
            edge_mask=edge_mask,
            zero_shot=zero_shot,
        )

        # Run decoder step
        stats_dict = {}
        return {"logits": logits}, stats_dict

    def _sample_from_logits(
        self,
        logits: Float[Tensor, "batch n_token num_residue_types"],
        temperature: float = 1.0,
    ) -> Int[Tensor, "batch n_token"]:
        """
        Sample amino acid indices from logits using temperature-controlled sampling.

        Args:
            logits: Raw model logits over amino acid types
            temperature: Sampling temperature (0 = greedy, >0 = stochastic sampling)

        """
        assert 0.0 <= temperature <= 1.0
        if temperature == 0:
            return torch.argmax(logits, dim=-1)
        else:
            probits = F.softmax(logits / temperature, dim=-1)
            batch_size, seq_len, num_classes = probits.shape
            probits_2d = probits.view(-1, num_classes)
            samples = torch.multinomial(probits_2d, num_samples=1).squeeze(-1)

            return samples.view(batch_size, seq_len)
