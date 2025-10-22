""" """

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

import protenix.model.modules.message_passing as mp
from protenix.model.modules.embedders import RelativePositionEncoding
from protenix.model.modules.primitives import LayerNorm, LinearNoBias
from protenix.model.utils import average_atom_to_token, flatten_final_dims
from protenix.utils.torch_utils import cdist


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
    ):
        """
        Args:
            asym_id: asymmetric unit IDs
            residue_index: residue indices
            entity_id: entity IDs
            sym_id: symmetry IDs
            token_index: token indices

        Returns:
            Tensor: relative position encoding
                [batch, n_token, n_neighbours, c_z]
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
        gamma = 1.0 / (spacing**2)
        self.register_buffer("gamma", torch.tensor(gamma))

    def forward(
        self,
        distances: Float[Tensor, "... distance_pairs"],
    ) -> Float[Tensor, "... distance_pairs num_kernels"]:
        """Apply RBF encoding to distances.

        Args:
            distances: Pairwise distances [..., distance_pairs]

        Returns:
            RBF encoded distances [..., distance_pairs, num_kernels]
        """
        # Expand distances for broadcasting: [..., distance_pairs, 1]
        distances = distances.unsqueeze(-1)

        # Compute Gaussian: exp(-gamma * (d - center)^2)
        # centers shape: [num_kernels]
        rbf = torch.exp(-self.gamma * (distances - self.centers) ** 2)

        return rbf


class BaseEdgeFeaturizer(nn.Module, ABC):
    """Base class for all edge featurizers.

    Defines the interface that all featurizers must implement for converting
    raw input data into feature representations suitable for neural networks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, input_feature_dict: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """Transform input data into feature representation.

        Args:
            input_feature_dict: Input data tensor

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

    def _compute_neighbour_graph(
        self,
        centre_atom_coords: Float[Tensor, "batch n_token 3"],
        centre_atom_mask: Bool[Tensor, "batch n_token"],
    ) -> Tuple[Float[Tensor, "batch n_token k"], Int[Tensor, "batch n_token k"]]:
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
        mask_2d = (
            centre_atom_mask[..., :, None] * centre_atom_mask[..., None, :]
        ).float()

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

    def _compute_node_features(self, input_feature_dict: Dict[str, Tensor]):
        """Compute node features with ligand and entity features."""
        # Grab data about the input
        atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
        atom_mask = input_feature_dict["atom_mask"]
        n_token = input_feature_dict["residue_index"].shape[-1]
        n_atom = input_feature_dict["ref_pos"].shape[-2]
        batch_shape = input_feature_dict["ref_pos"].shape[:-2]
        dtype = input_feature_dict["ref_pos"].dtype

        # Compute node features
        is_ligand = input_feature_dict["is_ligand"].unsqueeze(-1)  # [..., n_atom, 1]
        ligand_feats = (
            self.ligand_condition_embed(
                torch.cat(
                    [
                        input_feature_dict[name]
                        .reshape(*batch_shape, n_atom, self.ligand_input_feature[name])
                        .to(dtype)
                        for name in self.ligand_input_feature
                    ],
                    dim=-1,
                )
            )
            * is_ligand
        )  # [..., n_atom, node_dim]

        ligand_feats = average_atom_to_token(
            x_atom=ligand_feats,
            atom_to_token_idx=atom_to_token_idx,
            atom_mask=atom_mask,
            n_token=n_token,
        )  # [..., n_token, node_dim]

        # Embed entity flags
        entity_feats = self.entity_type_embed(
            torch.stack(
                [input_feature_dict[name].to(dtype) for name in self.entity_features],
                dim=-1,
            )
        )  # [..., n_token, node_dim]
        node_feats = ligand_feats + entity_feats
        return self.node_norm(node_feats)

    def _compute_backbone_distance_features(
        self,
        backbone_coords: Float[Tensor, "... n_token 5 3"],
        backbone_mask: Float[Tensor, "... n_token 5"],
        edge_index: Int[Tensor, "... n_token neighbors"],
    ) -> Float[Tensor, "... n_token neighbors edge_dim"]:
        """Compute edge features from backbone atom distances.

        Args:
            backbone_coords: Backbone atom coordinates [..., n_token, 5, 3]
            backbone_mask: Mask for valid backbone atoms [..., n_token, 5]
            edge_index: Neighbor indices [..., n_token, neighbors]

        Returns:
            Edge features from backbone distances [..., n_token, neighbors, edge_dim]
        """
        # Gather neighbor backbone coordinates and masks
        # Flatten for gather [..., n_token, 15]
        backbone_coords_flat = flatten_final_dims(backbone_coords, num_dims=2)
        neighbor_backbone_flat = mp.gather_nodes(backbone_coords_flat, edge_index)
        neighbor_backbone_coords = neighbor_backbone_flat.unflatten(
            dim=-1, sizes=(5, 3)
        )

        # Gather neighbor masks [..., n_token, neighbors, 5]
        neighbor_backbone_mask = mp.gather_nodes(backbone_mask, edge_index)

        # Compute pairwise distances between all backbone atoms
        bb_self = backbone_coords[..., None, :, None, :]
        bb_neighbor = neighbor_backbone_coords[..., None, :, :]

        # Distances [..., n_token, neighbors, 5, 5]
        distances = torch.norm(bb_self - bb_neighbor, dim=-1)

        # Create distance mask
        # [..., n_token, 1, 5, 1] * [..., n_token, neighbors, 1, 5]
        distance_mask = (
            neighbor_backbone_mask[..., :, None, :] * backbone_mask[..., None, :, None]
        )

        # Apply mask and flatten pairs [..., n_token, neighbors, 25]
        masked_distances = distances * distance_mask
        flat_distances = flatten_final_dims(masked_distances, 2)
        flat_mask = flatten_final_dims(distance_mask, 2)

        # Apply RBF encoding [..., n_token, neighbors, 25, num_rbf]
        rbf_features = self.rbf(flat_distances) * flat_mask[..., None]
        rbf_flat = flatten_final_dims(rbf_features, 2)
        return rbf_flat

    def _compute_relpos_features(
        self,
        input_feature_dict: Dict[str, Tensor],
        edge_index: Int[Tensor, "batch n_token neighbours"],
    ) -> Float[Tensor, "batch n_token neighbours edge_dim"]:
        """Compute edge relpos features."""
        edge_relpos_feat = self.positional_encoder(
            asym_id=input_feature_dict["asym_id"],
            residue_index=input_feature_dict["residue_index"],
            entity_id=input_feature_dict["entity_id"],
            sym_id=input_feature_dict["sym_id"],
            token_index=input_feature_dict["token_index"],
            edge_index=edge_index,
        )
        return edge_relpos_feat

    def forward(
        self,
        input_feature_dict: Dict[str, Tensor],
        centre_atom_coord: Float[Tensor, "batch n_token 3"],
        centre_atom_mask: Float[Tensor, "batch n_token"],
        backbone_coords: Float[Tensor, "batch n_token 5 3"],
        backbone_mask: Float[Tensor, "batch n_token 5"],
    ) -> Tuple[
        Float[Tensor, "batch n_token node_dim"],
        Float[Tensor, "batch n_token neighbours edge_dim"],
        Int[Tensor, "batch n_token neighbours"],
    ]:
        """
        Compute MPNN features and connectivity for general biomolecular graphs.

        Args:
            input_feature_dict: Input feature dict with the following tensors:
                - asym_id, residue_index, entity_id, sym_id, etc.
            centre_atom_coord: Centre atom coordinates per token
            centre_atom_mask: Mask for valid centre atoms
            backbone_coords: Backbone atom coordinates per token [..., n_token, 5, 3]
            backbone_mask: Mask for valid backbone atoms [..., n_token, 5]

        Returns:
            node_embedding: Node features with entity and ligand information
            edge_embedding: Edge features from backbone distances
            edge_index: Graph edge connectivity index
        """
        # Add coordinate noise during training
        if self.training:
            backbone_coords = (
                backbone_coords + torch.randn_like(backbone_coords) * self.coord_noise
            )

        # Compute k-nearest neighbor graph
        _, edge_index = self._compute_neighbour_graph(
            centre_atom_coords=centre_atom_coord, centre_atom_mask=centre_atom_mask
        )

        # Compute backbone distance edge features
        edge_dist_feats = self._compute_backbone_distance_features(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            edge_index=edge_index,
        )
        # Relpos
        edge_relpos_feats = self._compute_relpos_features(
            input_feature_dict=input_feature_dict, edge_index=edge_index
        )
        edge_embedding = self.edge_norm(
            self.embed_edge(torch.cat([edge_dist_feats, edge_relpos_feats], dim=-1))
        )
        # Compute node features
        node_embedding = self._compute_node_features(input_feature_dict)

        # Apply masks
        node_embedding = node_embedding * centre_atom_mask[..., None]
        edge_embedding = edge_embedding * mp.gather_nodes(
            centre_atom_mask[..., None], edge_index
        )

        return node_embedding, edge_embedding, edge_index
