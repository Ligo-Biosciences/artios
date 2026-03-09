"""Potts model for sequence design.

Parameterizes fields h and couplings J from MPNN decoder features.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from artios.model.modules.primitives import Linear
from artios.utils.typing import Bool, Float, Int, typecheck


class GraphPottsParameterisation(Enum):
    """Parameterisation modes for Potts model couplings.

    LINEAR: Direct linear projection of edge features to J.
    FACTOR: Factored J = W_left @ W_right with zero-sum gauge.
    SCORE: Low-rank factored J (via num_factors) with background fields/couplings and zero-sum gauge.
    """

    LINEAR = "linear"
    FACTOR = "factor"
    SCORE = "score"


@typecheck
@dataclass
class PottsOutput:
    h: Float[Tensor, "batch n_token num_residue_types"]
    J: Float[Tensor, "batch n_token neighbours num_residue_types num_residue_types"]
    edge_index: Int[Tensor, "batch n_token neighbours"]
    edge_mask: Bool[Tensor, "batch n_token neighbours"]
    is_masked: Bool[Tensor, "batch n_token"]
    timestep: Float[Tensor, "batch"]


def _zero_sum_gauge(
    h: Float[Tensor, "batch n_token num_residue_types"],
    J: Float[Tensor, "batch n_token neighbours num_residue_types num_residue_types"],
) -> tuple[
    Float[Tensor, "batch n_token num_residue_types"],
    Float[Tensor, "batch n_token neighbours num_residue_types num_residue_types"],
]:
    """Apply zero-sum gauge to h and J."""
    h = h - h.mean(-1, keepdim=True)
    J = (
        J
        - J.mean(-1, keepdim=True)
        - J.mean(-2, keepdim=True)
        + J.mean(dim=[-1, -2], keepdim=True)
    )
    return h, J


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
    """Potts parameterisation from MPNN decoder features.

    Maps decoder node features to fields h and decoder edge features
    to couplings J, using one of several parameterisation strategies.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_residue_types: int,
        parameterisation: GraphPottsParameterisation,
        symmetric_J: bool = True,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.num_residue_types = num_residue_types
        self.symmetric_J = symmetric_J
        self.parameterisation = parameterisation

        R = num_residue_types
        self.log_scale = nn.Parameter(np.log(init_scale) * torch.ones(1))

        if parameterisation == GraphPottsParameterisation.LINEAR:
            self.W_h = Linear(node_dim, R)
            self.W_J = Linear(edge_dim, R**2)

        elif parameterisation == GraphPottsParameterisation.FACTOR:
            self.W_h = Linear(node_dim, R)
            self.W_J_left = Linear(edge_dim, R**2)
            self.W_J_right = Linear(edge_dim, R**2)

        elif parameterisation == GraphPottsParameterisation.SCORE:
            num_factors = edge_dim  # TODO: make configurable
            self.num_factors = num_factors
            self.W_h = Linear(node_dim, R)
            self.W_h_bg = Linear(node_dim, 1)
            self.W_J_bg = Linear(edge_dim, 1)
            self.W_J_left = Linear(edge_dim, R * num_factors)
            self.W_J_right = Linear(edge_dim, R * num_factors)

        else:
            raise ValueError(f"Unknown parameterisation: {parameterisation}")

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
        scale = torch.exp(self.log_scale)
        mask_h = scale * token_mask[..., None]
        mask_J = scale * edge_mask[..., None]
        R = self.num_residue_types
        param = self.parameterisation

        if param == GraphPottsParameterisation.LINEAR:
            h = mask_h * self.W_h(node_feats)
            J_flat = mask_J * self.W_J(edge_feats)
            J = rearrange(J_flat, "b n k (r1 r2) -> b n k r1 r2", r1=R)

        elif param == GraphPottsParameterisation.FACTOR:
            h = mask_h * self.W_h(node_feats)
            J_left = rearrange(
                mask_J * self.W_J_left(edge_feats),
                "b n k (r1 r2) -> b n k r1 r2",
                r1=R,
            )
            J_right = rearrange(
                mask_J * self.W_J_right(edge_feats),
                "b n k (r1 r2) -> b n k r1 r2",
                r1=R,
            )
            J = torch.matmul(J_left, J_right)
            h, J = _zero_sum_gauge(h, J)

        elif param == GraphPottsParameterisation.SCORE:
            F = self.num_factors
            h = mask_h * self.W_h(node_feats)
            J_left = rearrange(
                mask_J * self.W_J_left(edge_feats),
                "b n k (r f) -> b n k r f",
                r=R,
                f=F,
            )
            J_right = rearrange(
                mask_J * self.W_J_right(edge_feats),
                "b n k (f r) -> b n k f r",
                f=F,
                r=R,
            )
            J = torch.matmul(J_left, J_right)

            # Zero-sum gauge
            h, J = _zero_sum_gauge(h, J)

            # Background components
            h = h + mask_h * self.W_h_bg(node_feats)
            J = J + (mask_J * self.W_J_bg(edge_feats)).unsqueeze(-1)

        if self.symmetric_J:
            J = symmetrize_J(J, edge_index, edge_mask)

        return h, J
