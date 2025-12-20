""" """

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

import protenix.model.modules.message_passing as mp
from protenix.data.features import FullContext
from protenix.model.modules.embedders import RelativePositionEncoding
from protenix.model.modules.primitives import LayerNorm, LinearNoBias
from protenix.model.utils import average_atom_to_token
from protenix.utils.tensor_utils import cdist
from protenix.utils.typing import typecheck


class EdgeRelativePositionalEncoding(nn.Module):
    def __init__(self, edge_dim: int = 128, r_max: int = 32, s_max: int = 2):
        """
        Args:
            r_max (int, optional): Relative position indices clip value. Defaults to 32.
            s_max (int, optional): Relative chain indices clip value. Defaults to 2.
            c_edge_dim (int, optional): hidden dim for edge embedding. Defaults to 128.
        """
        super(EdgeRelativePositionalEncoding, self).__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.edge_dim = edge_dim
        self.relpos = RelativePositionEncoding(r_max=r_max, s_max=s_max, c_z=edge_dim)

    def forward(
        self,
        *,
        asym_id: Int[Tensor, "batch n_token"],
        residue_index: Int[Tensor, "batch n_token"],
        entity_id: Int[Tensor, "batch n_token"],
        sym_id: Int[Tensor, "batch n_token"],
        token_index: Int[Tensor, "batch n_token"],
        edge_index: Int[Tensor, "batch n_token neighbours"],
    ) -> Float[Tensor, "batch n_token neighbours c_z"]:
        """
        Args:
            asym_id: asymmetric unit IDs
            residue_index: residue indices
            entity_id: entity IDs
            sym_id: symmetry IDs
            token_index: token indices

        Returns:
            relative position encoding embeddings
        """
        p = self.relpos(
            asym_id=asym_id,
            residue_index=residue_index,
            entity_id=entity_id,
            sym_id=sym_id,
            token_index=token_index,
        )
        return mp.gather_edges(p, edge_idx=edge_index)


class GaussianRBF(nn.Module):
    """Gaussian Radial Basis Function for distance encoding.

    Encodes distances using Gaussian kernels centered at regular intervals.
    """

    gamma: Tensor
    centers: Tensor

    def __init__(
        self,
        num_kernels: int = 16,
        min_distance: float = 2.0,
        max_distance: float = 22.0,
    ):
        """Initialize Gaussian RBF.

        Args:
            num_kernels: Number of Gaussian kernels
            min_distance: Minimum distance for RBF centers (Angstroms)
            max_distance: Maximum distance for RBF centers (Angstroms)
        """
        super().__init__()

        self.num_kernels = num_kernels
        self.min_distance = min_distance
        self.max_distance = max_distance

        # Centers are evenly spaced between min and max distance
        centers = torch.linspace(min_distance, max_distance, num_kernels)
        self.register_buffer("centers", centers)

        # Gamma controls the width of the Gaussians
        # Use spacing between centers to determine width
        spacing = (max_distance - min_distance) / (num_kernels - 1)
        # gamma_ =
        self.register_buffer("gamma", torch.tensor(1.0 / (spacing**2)))

    def forward(
        self,
        distances: Float[Tensor, "... distance_pairs"],
    ) -> Float[Tensor, "... distance_pairs num_kernels"]:
        """Apply RBF encoding to distances.

        Args:
            distances: Pairwise distances

        Returns:
            RBF encoded distances
        """
        rbf = torch.exp(-self.gamma * (distances[..., None] - self.centers) ** 2)
        return rbf


class BaseEdgeFeaturizer(nn.Module, ABC):
    """Base class for all edge featurizers.

    Defines the interface that all featurizers must implement for converting
    raw input data into feature representations suitable for neural networks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, context: FullContext) -> tuple[Tensor, Tensor]:
        """Transform input data into feature representation.

        Args:
            context: Full context containing all features.

        Returns:
            Tensor: Feature representation of input data edges
            Tensor: Edge indices
        """
        pass


class MPNNFeaturizer(nn.Module):
    """
    Featurizer for MPNN that converts input feature dict into node and edge features.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        positional_encoding_dim: int,
        num_rbf: int,
        num_neighbours: int,
        coord_noise: float = 0.1,
    ):
        """MPNN featurizer for general biomolecular graphs."""
        super().__init__()

        assert coord_noise >= 0, f"coord_noise must be non-negative, got {coord_noise}"

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.positional_encoding_dim = positional_encoding_dim
        self.num_rbf = num_rbf
        self.num_neighbours = num_neighbours
        self.coord_noise = coord_noise

        # Entity features
        self.entity_features = {
            "is_protein_token": 1,
            "is_ligand_token": 1,
            "is_dna_token": 1,
            "is_rna_token": 1,
        }
        self.entity_type_embed = LinearNoBias(
            sum(self.entity_features.values()), node_dim
        )
        self.non_protein_restype_embed = LinearNoBias(32, node_dim)

        # Ligand input features
        self.ligand_input_feature = {
            "ref_charge": 1,
            "ref_mask": 1,
            "ref_element": 128,
        }
        self.ligand_condition_embed = LinearNoBias(
            sum(self.ligand_input_feature.values()), node_dim
        )

        self.positional_encoder = EdgeRelativePositionalEncoding(
            edge_dim=positional_encoding_dim, r_max=32, s_max=2
        )

        # RBF encoding for backbone distances
        self.rbf = GaussianRBF(num_kernels=num_rbf, min_distance=2.0, max_distance=22.0)

        # Project RBF features to edge dimension
        # 25 backbone pairs * num_rbf kernels -> edge_dim
        self.embed_edge = LinearNoBias(25 * num_rbf + positional_encoding_dim, edge_dim)

        self.edge_norm = LayerNorm(edge_dim)
        self.node_norm = LayerNorm(node_dim)

    @typecheck
    def _compute_neighbour_graph(
        self,
        centre_atom_coords: Float[Tensor, "batch n_token 3"],
        centre_atom_mask: Bool[Tensor, "batch n_token"],
    ) -> tuple[Float[Tensor, "batch n_token k"], Int[Tensor, "batch n_token k"]]:
        """Compute k-nearest neighbour distances and edge indices based on centre atom coordinates.

        Finds the k-nearest token centre atoms for each token.
        Args:
            centre_atom_coords: centre atom coordinates per token
                (CA for proteins, C1' for nucleotides, ligand atom for ligands)
            centre_atom_mask: mask for valid centre atoms
            eps: small constant to avoid division by zero

        Returns:
            neighbour_dists: distances to k-nearest neighbour tokens
            edge_index: indices of the k-nearest neighbour tokens
        """
        mask_i = rearrange(centre_atom_mask, "... n -> ... n 1")
        mask_j = rearrange(centre_atom_mask, "... n -> ... 1 n")
        mask_2d = (mask_i & mask_j).float()

        pairwise_dist = cdist(centre_atom_coords, centre_atom_coords) * mask_2d

        # compute distances robustly wrt masked nodes
        d_max, _ = pairwise_dist.max(dim=-1, keepdim=True)  # [..., n_token, 1]
        d_adjust = pairwise_dist + (1.0 - mask_2d) * d_max  # [..., n_token, n_token]

        neighbor_dists, edge_index = torch.topk(
            d_adjust,
            min(self.num_neighbours, centre_atom_coords.shape[-2]),
            dim=-1,
            largest=False,
        )  # [..., n_tokens, k], [..., n_tokens, k]

        return neighbor_dists, edge_index

    @typecheck
    def _compute_node_features(
        self, context: FullContext
    ) -> Float[Tensor, "batch n_token node_dim"]:
        """Compute node features with ligand and entity features.

        Args:
            context: Full context containing all features.

        Returns:
            Node features tensor
        """
        dtype = context.reference.ref_pos.dtype
        n_token = context.mask.token_mask.shape[-1]

        # Compute node features
        is_ligand = context.entity.is_ligand.unsqueeze(-1)  # [..., n_atom, 1]

        ligand_input = torch.cat(
            [
                rearrange(context.reference.ref_charge, "... n -> ... n 1").to(dtype),
                rearrange(context.reference.ref_mask, "... n -> ... n 1").to(dtype),
                context.reference.ref_element.to(dtype),
            ],
            dim=-1,
        )
        ligand_feats = self.ligand_condition_embed(ligand_input) * is_ligand
        # [..., n_atom, node_dim]

        ligand_feats = average_atom_to_token(
            x_atom=ligand_feats,
            atom_to_token_idx=context.index.atom_to_token_idx,
            atom_mask=context.mask.atom_mask,
            n_token=n_token,
        )  # [..., n_token, node_dim]

        # Embed entity flags
        entity_feats = self.entity_type_embed(
            torch.stack(
                [
                    getattr(context.entity, name).to(dtype)
                    for name in self.entity_features
                ],
                dim=-1,
            )
        )  # [..., n_token, node_dim]

        # Embed non-protein entity restypes
        not_protein = rearrange(
            1 - context.entity.is_protein_token.to(dtype), "... n -> ... n 1"
        )
        non_protein_restypes = self.non_protein_restype_embed(
            context.token.restype * not_protein
        )

        node_feats = ligand_feats + entity_feats + non_protein_restypes
        return self.node_norm(node_feats)

    @typecheck
    def _compute_backbone_distance_features(
        self,
        backbone_coords: Float[Tensor, "... n_token 5 3"],
        backbone_mask: Bool[Tensor, "... n_token 5"],
        edge_index: Int[Tensor, "... n_token neighbors"],
    ) -> Float[Tensor, "... n_token neighbors edge_dim"]:
        """Compute edge features from backbone atom distances.

        Args:
            backbone_coords: Backbone atom coordinates
            backbone_mask: Mask for valid backbone atoms
            edge_index: Neighbor indices

        Returns:
            Edge features from backbone distances
        """
        # Gather neighbor backbone coordinates
        backbone_coords_flat = rearrange(backbone_coords, "... n b c -> ... n (b c)")
        neighbor_backbone_flat = mp.gather_nodes(backbone_coords_flat, edge_index)
        neighbor_backbone_coords = rearrange(
            neighbor_backbone_flat, "... n k (b c) -> ... n k b c", b=5, c=3
        )  # [..., n_token, neighbors, 5, 3]

        neighbor_backbone_mask = mp.gather_nodes(backbone_mask, edge_index)
        # [..., n_token, neighbors, 5]

        # Compute pairwise distances between all backbone atom pairs
        bb_self = rearrange(backbone_coords, "... n b c -> ... n 1 b 1 c")
        bb_neighbor = rearrange(
            neighbor_backbone_coords, "... n k b c -> ... n k 1 b c"
        )
        distances = torch.norm(bb_self - bb_neighbor, dim=-1)
        # [..., n_token, neighbors, 5, 5]

        # Create pairwise mask
        mask_self = rearrange(backbone_mask, "... n b -> ... n 1 b 1")
        mask_neighbor = rearrange(neighbor_backbone_mask, "... n k b -> ... n k 1 b")
        distance_mask = mask_self * mask_neighbor
        # [..., n_token, neighbors, 5, 5]

        # Apply mask and flatten backbone pairs
        masked_distances = distances * distance_mask
        flat_distances = rearrange(masked_distances, "... r d -> ... (r d)")
        flat_mask = rearrange(distance_mask, "... r d -> ... (r d)")
        # [..., n_token, neighbors, 25]

        # Apply RBF encoding
        rbf_features = self.rbf(flat_distances) * flat_mask[..., None]
        # [..., n_token, neighbors, 25, num_rbf]

        return rearrange(rbf_features, "... pairs rbf -> ... (pairs rbf)")

    @typecheck
    def _compute_relpos_features(
        self,
        context: FullContext,
        edge_index: Int[Tensor, "batch n_token neighbours"],
    ) -> Float[Tensor, "batch n_token neighbours edge_dim"]:
        """Compute edge relpos features.

        Args:
            context: Full context containing token features.
            edge_index: Neighbor indices for edge computation.

        Returns:
            Edge relative position features.
        """
        edge_relpos_feat = self.positional_encoder(
            asym_id=context.token.asym_id,
            residue_index=context.token.residue_index,
            entity_id=context.token.entity_id,
            sym_id=context.token.sym_id,
            token_index=context.token.token_index,
            edge_index=edge_index,
        )
        return edge_relpos_feat

    @typecheck
    def forward(
        self,
        context: FullContext,
    ) -> tuple[
        Float[Tensor, "batch n_token node_dim"],
        Float[Tensor, "batch n_token neighbours edge_dim"],
        Int[Tensor, "batch n_token neighbours"],
    ]:
        """Compute MPNN features and connectivity for general biomolecular graphs.

        Args:
            context: Full context containing:
                - coordinate: centre_coordinate, backbone_coordinate
                - mask: centre_atom_mask (at token level), backbone_mask
                - token: asym_id, residue_index, entity_id, sym_id, token_index, restype
                - entity: is_protein_token, is_ligand_token, is_dna_token, is_rna_token, is_ligand
                - reference: ref_pos, ref_mask, ref_element, ref_charge
                - index: atom_to_token_idx

        Returns:
            node_embedding: Node features with entity and ligand information
            edge_embedding: Edge features from backbone distances
            edge_index: Graph edge connectivity index
        """
        # Extract coordinates and masks from context
        centre_atom_coord = context.coordinate.centre_coordinate
        token_mask = context.mask.token_mask
        backbone_coords = context.coordinate.backbone_coordinate
        backbone_mask = context.mask.backbone_mask

        # TODO: centre-random augment the backbone coordinates
        # Add coordinate noise during training
        if self.training:
            backbone_coords = (
                backbone_coords + torch.randn_like(backbone_coords) * self.coord_noise
            )

        # Compute k-nearest neighbor graph
        _, edge_index = self._compute_neighbour_graph(
            centre_atom_coords=centre_atom_coord, centre_atom_mask=token_mask
        )

        # Compute backbone distance edge features
        edge_dist_feats = self._compute_backbone_distance_features(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            edge_index=edge_index,
        )
        # Relpos
        edge_relpos_feats = self._compute_relpos_features(
            context=context, edge_index=edge_index
        )
        edge_embedding = self.edge_norm(
            self.embed_edge(torch.cat([edge_dist_feats, edge_relpos_feats], dim=-1))
        )
        # Compute node features
        node_embedding = self._compute_node_features(context)

        # Apply masks (convert to float for multiplication)
        token_mask_float = token_mask.float()
        node_embedding = node_embedding * token_mask_float[..., None]
        edge_embedding = edge_embedding * mp.gather_nodes(
            token_mask_float[..., None], edge_index
        )

        return node_embedding, edge_embedding, edge_index
