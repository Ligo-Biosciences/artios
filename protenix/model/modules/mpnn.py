from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from protenix.model import utils
from protenix.model.modules import message_passing as mp
from protenix.model.modules.mpnn_features import MPNNFeaturizer
from protenix.model.modules.mpnn_layers import (
    MPNNDecoderLayer,
    MPNNEncoderLayer,
)
from protenix.model.modules.primitives import LinearNoBias


class MPNN(nn.Module):
    """Message passing neural network for sequence design.

    Args:
        num_residue_types (int): Number of residue types (32 for all biomolecules)
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
            coord_noise=coord_noise,
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
                    edge_input_dim=2 * node_dim + edge_dim,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        # Projection layers
        self.embed_restype = LinearNoBias(num_residue_types, node_dim)
        self.linear_out = LinearNoBias(node_dim, num_residue_types)

    def _generate_random_decoding_order(
        self,
        node_feats: Float[Tensor, "batch n_token hidden_dim"],
        seed: Optional[int] = None,
    ) -> Int[Tensor, "batch n_token"]:
        """Generate a random decoding order."""
        generator = None
        if seed is not None:
            generator = torch.Generator(node_feats.device).manual_seed(seed)

        return torch.argsort(
            torch.randn(
                size=node_feats.shape[:-1],
                device=node_feats.device,
                generator=generator,
            ),
            dim=-1,
        )

    def _generate_random_spatially_smooth_decoding_order(
        self,
        centre_atom_coords: Float[Tensor, "batch n_token 3"],
        centre_atom_mask: Float[Tensor, "batch n_token"],
        seed: Optional[int] = None,
        smoothing_factor: Optional[float] = None,
        seed_token_index: Optional[int] = None,
    ) -> Int[Tensor, "batch n_token"]:
        """Generate a random decoding order with spatial smoothing.

        The algorithm interpolates between ProteinMPNN-style random decoding order and
        fully spatial decoding order. In the fully spatial decoding order, a random token
        is selected as the centre and expands outward based on token centre atom proximity.

        The two methods are interpolated between using intermediate normalized noise and
        normalized distances, followed by argsort.

        Args:
            centre_atom_coords: centre atom coordinates
            centre_atom_mask: centre atom mask
            seed: random seed for reproducibility
            smoothing_factor: pre-determined smoothing factor
            seed_token_index: token index for spatial smoothing

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
        seed_token_rand += centre_atom_mask - 1  # mask out from index selection
        seed_token_indices = torch.argmax(seed_token_rand, dim=-1, keepdim=True)

        if seed_token_index is not None:
            seed_token_indices = torch.full_like(
                seed_token_indices,
                fill_value=seed_token_index,  # same index across batches
            )  # [..., 1]

        # Compute distances from the seed residue
        spatial_seed_coord = utils.batched_gather(
            data=centre_atom_coords,
            inds=seed_token_indices,
            dim=-2,
            no_batch_dims=len(centre_atom_coords.shape[:-2]),
        )  # [..., 1, 3]

        dists = torch.sqrt(
            torch.sum((spatial_seed_coord - centre_atom_coords) ** 2, axis=-1)
        )  # [..., n_token]

        # Compute normalized distances for interpolation
        norm_dist = dists / torch.sum(dists, axis=-1, keepdim=True)

        # Sample random uniform noise
        noise = torch.rand(
            centre_atom_coords.shape[:-1],
            device=centre_atom_coords.device,
            generator=generator,
        )
        norm_noise = noise / torch.sum(noise, dim=-1, keepdim=True)

        # Interpolate between normalized noise and normalized distances
        mu = torch.rand(
            centre_atom_mask.shape[:-1],
            device=centre_atom_coords.device,
            generator=generator,
        )
        if smoothing_factor is not None:
            mu = torch.full_like(mu, fill_value=smoothing_factor)

        # Interpolate and get decoding order
        mu = mu[..., None]  # [..., 1]
        smoothed = mu * norm_dist + (1 - mu) * norm_noise
        decoding_order = torch.argsort(smoothed, dim=-1)

        return decoding_order

    def _prepare_decoder_inputs(
        self,
        node_feats: Float[Tensor, "batch n_token node_dim"],
        edge_feats: Float[Tensor, "batch n_token neighbours edge_dim"],
        sequence_embeds: Float[Tensor, "batch n_token node_dim"],
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
        decoding_order: Float[Tensor, "batch n_token"],
        edge_index: Float[Tensor, "batch n_token neighbours"],
        node_mask: Float[Tensor, "batch n_token"],
    ) -> Tuple[
        Float[Tensor, "batch n_token neighbours"],  # forward attention mask
        Float[Tensor, "batch n_token neighbours"],  # backward attention mask
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
        node_feats: Float[Tensor, "batch n_token node_dim"],
        edge_feats: Float[Tensor, "batch n_token neighbours edge_dim"],
        restype: Float[Tensor, "batch n_token num_residue_types"],
        edge_index: Float[Tensor, "batch n_token neighbours"],
        node_mask: Float[Tensor, "batch n_token"],
        forward_attn_mask: Float[Tensor, "batch n_token neighbours"],
        backward_attn_mask: Float[Tensor, "batch n_token neighbours"],
        edge_mask: Float[Tensor, "batch n_token neighbours"],
        zero_shot: bool = False,
    ) -> Float[Tensor, "batch n_token num_residue_types"]:
        """Run a single decoder step with attention masking.

        Args:
            node_feats: Node features from encoder
            edge_feats: Base edge features
            restype: Current sequence state as one-hot vectors
            edge_index: Graph edge connectivity
            node_mask: Mask indicating valid sequence positions
            forward_attn_mask: Forward attention mask
            backward_attn_mask: Backward attention mask
            zero_shot: Whether to use zero-shot mode. Defaults to False.

        Returns:
            Raw logits over residue types.
        """
        sequence_embeds = self.embed_restype(restype)

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

            node_feats = checkpoint(
                layer,
                node_feats,
                combined_input_edge_feats,
                node_mask,
                edge_mask,
                use_reentrant=False,
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
            node_feats, edge_feats = checkpoint(
                layer,
                node_feats,
                edge_feats,
                edge_index,
                centre_atom_mask,
                edge_mask,
                use_reentrant=False,
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
        seed: Optional[int] = None,
    ) -> Tuple[
        Float[Tensor, "batch n_token num_residue_types"],
        Float[Tensor, "batch n_token node_dim"],
        Float[Tensor, "batch n_token edge_dim"],
    ]:
        """Forward pass for autoregressive sequence generation.

        Encodes protein structure into node representations, then generates sequence
        probabilities using masked attention.

        Args:
            backbone_coords: backbone coordinates for each token
            backbone_mask: backbone mask for each token (which backbone atoms are valid)
            centre_atom_coord: centre atom coordinate for each token
            centre_atom_mask: centre atom mask is resolved mask
            restype: one-hot encoded residue types
            input_feature_dict: Input feature dict with the following tensors:
                - asym_id, residue_index, entity_id, sym_id, etc.
            decoding_order: Permutation defining generation order
            zero_shot: whether to use zero-shot mode. Defaults to False
            seed: seed for deterministic sampling

        Returns:
            Output dictionary
        """

        # Run encoder trunk
        node_feats, edge_feats, edge_idx, edge_mask = self.run_encoder_trunk(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            input_feature_dict=input_feature_dict,
        )

        # Optionally prepare random decoding order
        if decoding_order is None:
            # decoding_order = self._generate_random_decoding_order(
            #     node_feats=node_feats, seed=seed
            # )
            decoding_order = self._generate_random_spatially_smooth_decoding_order(
                centre_atom_coords=centre_atom_coord,
                centre_atom_mask=centre_atom_mask,
                seed=seed,
            )

        # Generate masks
        forward_attn_mask, backward_attn_mask = self._generate_sequential_masks(
            decoding_order=decoding_order,
            edge_index=edge_idx,
            node_mask=centre_atom_mask,
        )

        forward_attn_mask = forward_attn_mask[..., None]
        backward_attn_mask = backward_attn_mask[..., None]

        # Run decoder step
        logits = self._run_decoder_step(
            node_feats=node_feats,
            edge_feats=edge_feats,
            restype=restype,
            edge_index=edge_idx,
            node_mask=centre_atom_mask,
            forward_attn_mask=forward_attn_mask,
            backward_attn_mask=backward_attn_mask,
            edge_mask=edge_mask,
            zero_shot=zero_shot,
        )

        return logits, node_feats, edge_feats

    def _sample_from_logits(
        self,
        logits: Float[Tensor, "batch n_token num_residue_types"],
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> Float[Tensor, "batch n_token num_residue_types"]:
        """
        Sample amino acid indices from logits using temperature-controlled sampling.

        Args:
            logits: Raw model logits over amino acid types
                [..., n_token, num_residue_types]
            temperature: Sampling temperature (0 = greedy, >0 = stochastic sampling)
            seed: seed for deterministic sampling

        Returns:
            one_hot: One-hot encoded sampled residue types
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(logits.device).manual_seed(seed)

        if temperature == 0:
            # Greedy sampling: take argmax
            samples = torch.argmax(logits, dim=-1)
        else:
            # Stochastic sampling with temperature
            probits = F.softmax(logits / temperature, dim=-1)
            batch_size, seq_len, num_classes = probits.shape
            probits_2d = probits.view(-1, num_classes)
            samples = torch.multinomial(
                probits_2d, num_samples=1, generator=generator
            ).squeeze(-1)
            samples = samples.view(batch_size, seq_len)

        # One-hot encode the sampled indices
        num_residue_types = logits.shape[-1]
        one_hot = F.one_hot(samples, num_classes=num_residue_types).float()

        return one_hot

    def sample(
        self,
        backbone_coords: Float[Tensor, "batch n_token 5 3"],
        backbone_mask: Float[Tensor, "batch n_token 5"],
        centre_atom_coord: Float[Tensor, "batch n_token 3"],
        centre_atom_mask: Float[Tensor, "batch n_token"],
        restype: Float[Tensor, "batch n_token num_residue_types"],
        input_feature_dict: Dict[str, Tensor],
        decoding_order: Optional[Float[Tensor, "batch n_token"]] = None,
        temperature: float = 1.0,
        show_progress: bool = False,
        return_logits: bool = False,
        seed: Optional[int] = None,
    ) -> Union[
        Float[Tensor, "batch n_token num_residue_types"],
        Tuple[
            Float[Tensor, "batch n_token num_residue_types"],
            Float[Tensor, "batch n_token num_residue_types"],
        ],
    ]:
        """Sample protein sequence autoregressively from the model.

        Args:
            backbone_coords: backbone coordinates for each token
            backbone_mask: backbone mask for each token (which backbone atoms are valid)
            centre_atom_coord: centre atom coordinate for each token
            centre_atom_mask: centre atom mask is resolved mask
            restype: one-hot encoded residue types
            input_feature_dict: Input feature dict with the following tensors:
                - asym_id, residue_index, entity_id, sym_id, etc.
            decoding_order: Permutation defining generation order
            zero_shot: whether to use zero-shot mode. Defaults to False
            temperature: sampling temperature
            show_progress: show progress during sampling
            return_logits: whether to return the logits or not
            seed: seed for deterministic sampling

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]:
            If return_logits:
                sequence_ids: sampled sequence as one-hot encoded tensor
                logits_at_timestep: sequence probabilities for all residue types
            else:
                sequence_ids: sampled sequence as one-hot encoded tensor
        """
        assert temperature >= 0, "Temperature must be non-negative"

        # Run encoder trunk
        node_feats, edge_feats, edge_idx, edge_mask = self.run_encoder_trunk(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            input_feature_dict=input_feature_dict,
        )

        # Optionally prepare random decoding order
        if decoding_order is None:
            decoding_order = self._generate_random_decoding_order(node_feats, seed)

        # Generate masks
        forward_attn_mask, backward_attn_mask = self._generate_sequential_masks(
            decoding_order=decoding_order,
            edge_index=edge_idx,
            node_mask=centre_atom_mask,
        )

        forward_attn_mask = forward_attn_mask[..., None]
        backward_attn_mask = backward_attn_mask[..., None]

        sequence_ids = torch.zeros(
            restype.shape,
            device=backbone_coords.device,
            dtype=restype.dtype,
        )
        logits_at_timestep = torch.zeros(
            restype.shape,
            device=backbone_coords.device,
            dtype=backbone_coords.dtype,
        )

        timestep_range = range(decoding_order.shape[-1])
        if show_progress:
            timestep_range = tqdm(timestep_range, desc="Sampling")

        for timestep in timestep_range:
            logits = self._run_decoder_step(
                node_feats=node_feats,
                edge_feats=edge_feats,
                restype=restype,
                edge_index=edge_idx,
                node_mask=centre_atom_mask,
                forward_attn_mask=forward_attn_mask,
                backward_attn_mask=backward_attn_mask,
                edge_mask=edge_mask,
                zero_shot=False,
            )

            # sample from logits and update
            next_pos_idx = decoding_order[:, timestep]
            batch_idx = torch.arange(logits.shape[0])

            pred_vals = self._sample_from_logits(
                logits, temperature=temperature, seed=seed
            )
            sequence_ids[batch_idx, next_pos_idx] = pred_vals[batch_idx, next_pos_idx]
            logits_at_timestep[batch_idx, timestep] = logits[batch_idx, next_pos_idx]

        if return_logits:
            return sequence_ids, node_feats, logits_at_timestep
        else:
            return sequence_ids, node_feats
