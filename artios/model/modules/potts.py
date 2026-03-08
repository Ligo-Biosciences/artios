"""Potts model for sequence design.

Parameterizes fields h and couplings J from MPNN decoder features.
"""

from dataclasses import dataclass

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from artios.model.modules.primitives import LinearNoBias
from artios.utils.typing import Bool, Float, Int, typecheck


@typecheck
@dataclass
class PottsOutput:
    h: Float[Tensor, "batch n_token num_residue_types"]
    J: Float[Tensor, "batch n_token neighbours num_residue_types num_residue_types"]
    edge_index: Int[Tensor, "batch n_token neighbours"]
    edge_mask: Bool[Tensor, "batch n_token neighbours"]
    is_masked: Bool[Tensor, "batch n_token"]
    timestep: Float[Tensor, "batch"]


def symmetrize_J(
    J: Float[Tensor, "batch n_token neighbours num_residue_types num_residue_types"],
    edge_index: Int[Tensor, "batch n_token neighbours"],
    edge_mask: Bool[Tensor, "batch n_token neighbours"],
) -> Float[Tensor, "batch n_token neighbours num_residue_types num_residue_types"]:
    """Symmetrize Potts couplings: J_ij(a,b) = (J_ij(a,b) + J_ji(b,a)) / 2.

    For each edge (i,j), finds the reverse edge (j,i) in j's neighbor list,
    gathers J_ji, transposes its state dimensions, and averages with J_ij.

    Args:
        J: Potts couplings [B, N, K, R, R].
        edge_index: Neighbor indices [B, N, K].
        edge_mask: Edge mask [B, N, K].

    Returns:
        Symmetrized couplings [B, N, K, R, R].
    """
    B, N, K, R, _ = J.shape

    # 1. For each edge (i,j), find which slot k' in j's neighbor list points back to i
    edge_index_flat = repeat(
        rearrange(edge_index, "b n k -> b (n k)"),
        "b nk -> b nk k2",
        k2=K,
    )
    neighbors_of_neighbors = torch.gather(edge_index, 1, edge_index_flat)
    neighbors_of_neighbors = rearrange(
        neighbors_of_neighbors,
        "b (n k1) k2 -> b n k1 k2",
        n=N,
    )

    # Which slot k' at j maps back to i?
    residue_i = rearrange(torch.arange(N, device=edge_index.device), "n -> 1 n 1 1")
    match = (neighbors_of_neighbors == residue_i).float()
    has_reverse, reverse_k = torch.max(match, dim=-1)  # both [B, N, K]

    # 2. Build flat index into J viewed as [B, N*K, ...] to gather J_ji
    ji_flat_idx = rearrange(edge_index * K + reverse_k, "b n k -> b (n k)")

    J_flat = rearrange(J, "b n k r1 r2 -> b (n k) (r1 r2)")
    J_ji_flat = torch.gather(
        J_flat,
        1,
        repeat(ji_flat_idx, "b nk -> b nk rr", rr=R * R),
    )
    J_ji = rearrange(J_ji_flat, "b (n k) (r1 r2) -> b n k r1 r2", n=N, r1=R)

    # Transpose state dimensions: J_ji(b,a) -> J_ji(a,b)
    J_ji_T = J_ji.transpose(-2, -1)

    # 3. Average, masking edges where reverse doesn't exist
    mask_ji = edge_mask.float() * has_reverse  # [B, N, K]
    J_symm = rearrange(0.5 * mask_ji, "b n k -> b n k 1 1") * (J + J_ji_T)

    return J_symm


class GraphPotts(nn.Module):
    """Linear Potts parameterization from MPNN decoder features.

    Maps decoder node features to fields h and decoder edge features
    to couplings J.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_residue_types: int,
        symmetric_J: bool = True,
    ):
        super().__init__()
        self.num_residue_types = num_residue_types
        self.symmetric_J = symmetric_J
        self.W_h = LinearNoBias(node_dim, num_residue_types)
        self.W_J = LinearNoBias(edge_dim, num_residue_types**2)

    @typecheck
    def forward(
        self,
        node_feats: Float[Tensor, "batch n_token node_dim"],
        edge_feats: Float[Tensor, "batch n_token neighbours edge_dim"],
        edge_index: Int[Tensor, "batch n_token neighbours"],
        edge_mask: Bool[Tensor, "batch n_token neighbours"],
        token_mask: Bool[Tensor, "batch n_token"],
    ) -> tuple[
        Float[Tensor, "batch n_token num_residue_types"],
        Float[Tensor, "batch n_token neighbours num_residue_types num_residue_types"],
    ]:
        """Compute Potts fields h and couplings J.

        Args:
            node_feats: Decoder node features.
            edge_feats: Decoder edge features.
            edge_index: Neighbor indices.
            edge_mask: Valid edge mask from encoder.
            token_mask: Valid token mask.

        Returns:
            Tuple of (h, J) Potts parameters.
        """
        h = self.W_h(node_feats) * token_mask[..., None]
        J_flat = self.W_J(edge_feats) * edge_mask[..., None]
        J = rearrange(
            J_flat,
            "b n k (r1 r2) -> b n k r1 r2",
            r1=self.num_residue_types,
        )

        if self.symmetric_J:
            J = symmetrize_J(J, edge_index, edge_mask)

        return h, J
