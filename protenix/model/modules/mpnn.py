"""Refactored Message Passing Neural Network for sequence design.

This module provides a cleaner API for the MPNN with:
- `encode()`: Structure encoding (cached during sampling)
- `decode()`: Single decoder step returning logits
- `forward()`: Training pass with teacher forcing
- `sample()`: Autoregressive inference
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from protenix.data.features import FullContext
from protenix.model import utils
from protenix.model.modules import message_passing as mp
from protenix.model.modules.mpnn_features import MPNNFeaturizer
from protenix.model.modules.mpnn_layers import (
    MPNNDecoderLayer,
    MPNNEncoderLayer,
)
from protenix.model.modules.primitives import LinearNoBias
from protenix.utils.typing import Bool, Num, typecheck


@dataclass
class EncoderOutput:
    """Output from the MPNN encoder.

    Attributes:
        node_feats: Encoded node features [..., n_token, node_dim]
        edge_feats: Encoded edge features [..., n_token, neighbours, edge_dim]
        edge_index: Graph connectivity [..., n_token, neighbours]
        edge_mask: Valid edge mask [..., n_token, neighbours]
    """

    node_feats: Float[Tensor, "... n_token node_dim"]
    edge_feats: Float[Tensor, "... n_token neighbours edge_dim"]
    edge_index: Int[Tensor, "... n_token neighbours"]
    edge_mask: Float[Tensor, "... n_token neighbours"]


@dataclass
class DecodingMasks:
    """Attention masks for autoregressive decoding.

    Attributes:
        forward_mask: Mask for attending to future positions [..., n_token, neighbours, 1]
        backward_mask: Mask for attending to past positions [..., n_token, neighbours, 1]
    """

    forward_mask: Float[Tensor, "... n_token neighbours 1"]
    backward_mask: Float[Tensor, "... n_token neighbours 1"]


class MPNN(nn.Module):
    """Message passing neural network for sequence design.

    This network encodes protein structure into node/edge representations,
    then autoregressively decodes a sequence. During training, teacher forcing
    is used. During inference, tokens are sampled one-by-one.

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
                MPNNEncoderLayer(node_dim=node_dim, edge_dim=edge_dim, dropout=dropout)
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
                )
                for _ in range(num_decoder_layers)
            ]
        )

        # Projection layers
        self.embed_restype = LinearNoBias(num_residue_types, node_dim)
        self.linear_out = LinearNoBias(node_dim, num_residue_types)

    def generate_decoding_order(
        self,
        centre_atom_coords: Float[Tensor, "batch n_token 3"],
        centre_atom_mask: Float[Tensor, "batch n_token"],
        seed: Optional[int] = None,
        smoothing_factor: Optional[float] = None,
        seed_token_index: Optional[int] = None,
    ) -> Int[Tensor, "batch n_token"]:
        """Generate a spatially-smooth random decoding order.

        Interpolates between random and spatial ordering. In spatial ordering,
        a random token is selected as seed and decoding expands outward by proximity.

        Args:
            centre_atom_coords: Centre atom coordinates
            centre_atom_mask: Mask for valid tokens
            seed: Random seed for reproducibility
            smoothing_factor: Interpolation factor (0=random, 1=spatial). If None, sampled.
            seed_token_index: Fixed seed token index. If None, randomly selected.

        Returns:
            Decoding order
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(centre_atom_mask.device).manual_seed(seed)

        # Select seed residue
        seed_token_rand = torch.rand(
            size=centre_atom_coords.shape[:-1],
            device=centre_atom_coords.device,
            generator=generator,
        )  # [..., n_token]
        seed_token_rand = seed_token_rand + centre_atom_mask - 1  # mask out invalid
        seed_token_indices = torch.argmax(seed_token_rand, dim=-1, keepdim=True)

        if seed_token_index is not None:
            seed_token_indices = torch.full_like(
                seed_token_indices, fill_value=seed_token_index
            )  # [..., 1]

        # Compute distances from seed residue
        spatial_seed_coord = utils.batched_gather(
            data=centre_atom_coords,
            inds=seed_token_indices,
            dim=-2,
            no_batch_dims=len(centre_atom_coords.shape[:-2]),
        )  # [..., 1, 3]

        dists = torch.sqrt(
            torch.sum((spatial_seed_coord - centre_atom_coords) ** 2, dim=-1)
        )  # [..., n_token]

        # Normalized distances for interpolation
        norm_dist = dists / torch.sum(dists, dim=-1, keepdim=True)

        # Sample random noise
        noise = torch.rand(
            centre_atom_coords.shape[:-1],
            device=centre_atom_coords.device,
            generator=generator,
        )
        norm_noise = noise / torch.sum(noise, dim=-1, keepdim=True)

        # Interpolation factor
        mu = torch.rand(
            centre_atom_mask.shape[:-1],
            device=centre_atom_coords.device,
            generator=generator,
        )
        if smoothing_factor is not None:
            mu = torch.full_like(mu, fill_value=smoothing_factor)

        # Interpolate and compute order
        mu = mu[..., None]  # [..., 1]
        smoothed = mu * norm_dist + (1 - mu) * norm_noise
        decoding_order = torch.argsort(smoothed, dim=-1)

        return decoding_order

    def encode(self, context: FullContext) -> EncoderOutput:
        """Encode protein structure into node and edge representations.

        Args:
            context: Full context containing structural features.

        Returns:
            EncoderOutput containing node features, edge features, and masks.
        """
        centre_atom_mask = context.mask.token_mask  # TODO: bool!!

        node_feats, edge_feats, edge_index = self.featurizer(context)

        edge_mask = mp.gather_nodes(
            h_nodes=centre_atom_mask[..., None], edge_idx=edge_index
        ).squeeze(-1)
        edge_mask = centre_atom_mask[..., None] * edge_mask
        # [..., n_token, neighbours]

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

    def generate_decoding_masks(
        self,
        decoding_order: Float[Tensor, "batch n_token"],
        edge_index: Float[Tensor, "batch n_token neighbours"],
        node_mask: Float[Tensor, "batch n_token"],
    ) -> DecodingMasks:
        """Generate autoregressive attention masks from decoding order.

        Creates forward and backward masks that enforce the autoregressive property:
        - Forward mask: attend to positions not yet decoded
        - Backward mask: attend to already-decoded positions

        Args:
            decoding_order: Permutation defining generation order
            edge_index: Graph edge connectivity
            node_mask: Mask for valid sequence positions

        Returns:
            DecodingMasks with forward and backward attention masks.
        """
        mask_size = edge_index.shape[-2]
        permutation_matrix_reverse = F.one_hot(
            decoding_order, num_classes=mask_size
        ).float()

        # Compute backward mask in edge space
        # TODO: Simplify this einsum for readability
        edge_space_backward_mask = torch.einsum(
            "ij, biq, bjp -> bqp",
            (
                1
                - torch.triu(torch.ones(mask_size, mask_size, device=edge_index.device))
            ),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )

        # Map to kNN adjacency space and apply node masking
        backward_attn_mask = torch.gather(
            edge_space_backward_mask, dim=-1, index=edge_index
        )
        mask_1D = node_mask[..., None]

        backward_attn_mask = mask_1D * backward_attn_mask
        forward_attn_mask = mask_1D * (1.0 - backward_attn_mask)

        return DecodingMasks(
            forward_mask=forward_attn_mask[..., None],
            backward_mask=backward_attn_mask[..., None],
        )

    def _prepare_decoder_edge_features(
        self,
        encoder_output: EncoderOutput,
        sequence_embeds: Float[Tensor, "batch n_token node_dim"],
    ) -> Tuple[
        Float[Tensor, "batch n_token neighbours hidden"],
        Float[Tensor, "batch n_token neighbours hidden"],
    ]:
        """Prepare edge features for decoder with sequence and encoder information.

        Creates two edge feature types:
        - encoder_edge_feats: For context from encoder (zero sequence + encoder nodes)
        - sequence_edge_feats: For autoregressive context (sequence embeds)

        Args:
            encoder_output: Output from encode()
            sequence_embeds: Embedded sequence information

        Returns:
            Tuple of (encoder_edge_feats, sequence_edge_feats)
        """
        edge_feats = encoder_output.edge_feats
        edge_index = encoder_output.edge_index
        node_feats = encoder_output.node_feats

        # Sequence edge features
        sequence_edge_feats = mp.augment_edges_with_nb_feats(
            edge_feats=edge_feats, features=sequence_embeds, edge_idx=edge_index
        )

        # Encoder context: zero-sequence + encoder node features
        aug_encoder_feats = mp.augment_edges_with_nb_feats(
            edge_feats=edge_feats,
            features=torch.zeros_like(sequence_embeds),
            edge_idx=edge_index,
        )
        encoder_edge_feats = mp.augment_edges_with_nb_feats(
            edge_feats=aug_encoder_feats, features=node_feats, edge_idx=edge_index
        )

        return encoder_edge_feats, sequence_edge_feats

    @typecheck
    def decode(
        self,
        encoder_output: EncoderOutput,
        restype: Num[Tensor, "batch n_token num_residue_types"],
        masks: DecodingMasks,
        node_mask: Bool[Tensor, "batch n_token"],
        zero_shot: bool = False,
    ) -> Float[Tensor, "batch n_token num_residue_types"]:
        """Run a single decoder pass to produce logits.

        Args:
            encoder_output: Cached encoder output
            restype: Current sequence state (one-hot or soft labels)
            masks: Forward and backward attention masks
            node_mask: Mask for valid positions
            zero_shot: If True, ignore sequence context (structure-only prediction)

        Returns:
            Logits over residue types [..., n_token, num_residue_types]
        """
        node_feats = encoder_output.node_feats
        edge_index = encoder_output.edge_index
        edge_mask = encoder_output.edge_mask
        dtype = node_feats.dtype

        sequence_embeds = self.embed_restype(restype.to(dtype))

        encoder_edge_feats, sequence_edge_feats = self._prepare_decoder_edge_features(
            encoder_output=encoder_output,
            sequence_embeds=sequence_embeds,
        )

        # Run decoder layers
        for layer in self.decoder:
            if zero_shot:
                # Structure-only: use encoder features for all positions
                combined_edge_feats = encoder_edge_feats * (
                    masks.forward_mask + masks.backward_mask
                )
            else:
                # Autoregressive: sequence for past, encoder for future
                decoder_edge_feats = mp.augment_edges_with_nb_feats(
                    edge_feats=sequence_edge_feats,
                    features=node_feats,
                    edge_idx=edge_index,
                )
                combined_edge_feats = (
                    decoder_edge_feats * masks.backward_mask
                    + encoder_edge_feats * masks.forward_mask
                )

            node_feats = checkpoint(
                layer,
                node_feats,
                combined_edge_feats,
                node_mask,
                edge_mask,
                use_reentrant=False,
            )

        logits = self.linear_out(node_feats)
        return logits

    def _sample_from_logits(
        self,
        logits: Float[Tensor, "batch n_token num_residue_types"],
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> Float[Tensor, "batch n_token num_residue_types"]:
        """Sample residue types from logits using temperature-controlled sampling.

        Args:
            logits: Raw model logits [..., n_token, num_residue_types]
            temperature: Sampling temperature (0=greedy, >0=stochastic)
            seed: Seed for deterministic sampling

        Returns:
            One-hot encoded sampled residue types
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(logits.device).manual_seed(seed)

        if temperature == 0:
            samples = torch.argmax(logits, dim=-1)
        else:
            probits = F.softmax(logits / temperature, dim=-1)
            batch_size, seq_len, num_classes = probits.shape
            probits_2d = probits.view(-1, num_classes)
            samples = torch.multinomial(
                probits_2d, num_samples=1, generator=generator
            ).squeeze(-1)
            samples = samples.view(batch_size, seq_len)

        one_hot = F.one_hot(samples, num_classes=self.num_residue_types).float()
        return one_hot

    def sample(
        self,
        context: FullContext,
        decoding_order: Optional[Float[Tensor, "batch n_token"]] = None,
        temperature: float = 1.0,
        zero_shot: bool = False,
        smoothing_factor: float = 0.5,
        show_progress: bool = False,
        seed: Optional[int] = None,
    ) -> Tuple[
        Float[Tensor, "batch n_token num_residue_types"],
        Float[Tensor, "batch n_token node_dim"],
    ]:
        """Autoregressive sequence sampling.

        Encodes structure once, then iteratively samples tokens according to
        the decoding order. Each step conditions on previously sampled tokens.

        Args:
            context: Full context containing structure.
            decoding_order: Permutation for generation order. If None, generated randomly.
            temperature: Sampling temperature (0=greedy, >0=stochastic).
            zero_shot: If True, predict without sequence context.
            smoothing_factor: Smoothing factor for decoding order generation.
            show_progress: Show tqdm progress bar.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (sampled_sequence, node_feats)
                - sampled_sequence: One-hot encoded [..., n_token, num_residue_types]
                - node_feats: Final node features [..., n_token, node_dim]
        """
        assert temperature >= 0, "Temperature must be non-negative"

        centre_atom_coord = context.coordinate.centre_coordinate
        centre_atom_mask = context.mask.token_mask
        restype_shape = context.token.restype.shape

        device = centre_atom_coord.device
        dtype = centre_atom_coord.dtype

        encoder_output = self.encode(context)

        if decoding_order is None:
            decoding_order = self.generate_decoding_order(
                centre_atom_coords=centre_atom_coord,
                centre_atom_mask=centre_atom_mask,
                smoothing_factor=smoothing_factor,
                seed=seed,
            )

        masks = self.generate_decoding_masks(
            decoding_order=decoding_order,
            edge_index=encoder_output.edge_index,
            node_mask=centre_atom_mask,
        )

        restype = torch.zeros(
            restype_shape,
            device=device,
            dtype=dtype,
        )
        logits_at_timestep = torch.zeros(
            restype_shape,
            device=device,
            dtype=dtype,
        )

        timestep_range = range(decoding_order.shape[-1])
        if show_progress:
            timestep_range = tqdm(timestep_range, desc="Sampling")

        for timestep in timestep_range:
            logits = self.decode(
                encoder_output=encoder_output,
                restype=restype,
                masks=masks,
                node_mask=centre_atom_mask,
                zero_shot=zero_shot,
            )

            next_pos_idx = decoding_order[:, timestep]
            batch_idx = torch.arange(logits.shape[0])

            pred_vals = self._sample_from_logits(
                logits, temperature=temperature, seed=seed
            )
            restype[batch_idx, next_pos_idx] = pred_vals[batch_idx, next_pos_idx]
            logits_at_timestep[batch_idx, timestep] = logits[batch_idx, next_pos_idx]

        return restype, encoder_output.node_feats

    def forward(
        self,
        context: FullContext,
        decoding_order: Optional[Float[Tensor, "batch n_token"]] = None,
        zero_shot: bool = False,
        seed: Optional[int] = None,
    ) -> Tuple[
        Float[Tensor, "batch n_token num_residue_types"],
        Float[Tensor, "batch n_token node_dim"],
        Float[Tensor, "batch n_token neighbours edge_dim"],
    ]:
        """Training forward pass with teacher forcing.

        Encodes structure and decodes with full sequence visibility (teacher forcing).
        Used during training to compute loss over all positions in parallel.

        Args:
            context: Full context containing structure and sequence.
            decoding_order: Permutation for mask generation. If None, generated randomly.
            zero_shot: If True, predict without sequence context.
            seed: Random seed for decoding order generation.

        Returns:
            Tuple of (logits, node_feats, edge_feats)
        """
        centre_atom_coord = context.coordinate.centre_coordinate
        centre_atom_mask = context.mask.token_mask
        restype = context.token.restype

        encoder_output: EncoderOutput = self.encode(context)

        if decoding_order is None:
            decoding_order = self.generate_decoding_order(
                centre_atom_coords=centre_atom_coord,
                centre_atom_mask=centre_atom_mask,
                seed=seed,
            )

        masks: DecodingMasks = self.generate_decoding_masks(
            decoding_order=decoding_order,
            edge_index=encoder_output.edge_index,
            node_mask=centre_atom_mask,
        )

        logits = self.decode(
            encoder_output=encoder_output,
            restype=restype,
            masks=masks,
            node_mask=centre_atom_mask,
            zero_shot=zero_shot,
        )

        return logits, encoder_output.node_feats, encoder_output.edge_feats
