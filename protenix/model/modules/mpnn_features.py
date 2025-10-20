""" """

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from protenix.model.modules.primitives import LinearNoBias
from protenix.model.utils import average_atom_to_token
from protenix.utils.torch_utils import cdist


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
        max_relative_feats: int,
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
        self.max_relative_feats = max_relative_feats
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
            # "ref_atom_name_chars": 4 * 64,
        }
        self.ligand_condition_embed = LinearNoBias(
            sum(self.ligand_input_feature.values()), node_dim
        )

        self.positional_encoder = None

        # TODO: the input dimensions of these will change.
        self.node_embedder = LinearNoBias(node_dim, node_dim)
        self.edge_embedder = LinearNoBias(edge_dim, edge_dim)

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
        mask_2d = centre_atom_mask[..., :, None] * centre_atom_mask[..., None, :]

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
        """
        Compute node features with ligand and
        """
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
        return node_feats

    def forward(
        self,
        input_feature_dict: Dict[str, Tensor],
        coordinate: Float[Tensor, "batch n_atom 3"],
        centre_atom_coord: Float[
            Tensor, "batch n_token 3"
        ],  # NOTE: will sort out later
        centre_atom_mask: Float[Tensor, "batch n_token"],
    ):
        """
        Compute MPNN features and connectivity for general biomolecular graphs.
        Args:
            input_feature_dict: Input feature dict with the following tensors:
                - asym_id:
                - residue_index:
                - entity_id:
                - sym_id:

        Returns:
            Relative position encoding
        """
        # Optionally add coordinate noise
        coordinate = coordinate + torch.randn_like(coordinate) * self.coord_noise

        _, edge_index = self._compute_neighbour_graph(
            centre_atom_coords=centre_atom_coord, centre_atom_mask=centre_atom_mask
        )

        # Compute RBF features

        # Relpos encoding

        # Compute node features
        node_feats = self._compute_node_features(input_feature_dict)

        return node_feats, edge_index
