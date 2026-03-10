"""Potts model for sequence design.

Parameterizes fields h and couplings J from MPNN decoder features.
"""

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


@typecheck
def compute_potts_conditionals(
    S_idx: Int[Tensor, "batch n_token"],
    h: Float[Tensor, "batch n_token num_residue_types"],
    J: Float[Tensor, "batch n_token neighbours num_residue_types num_residue_types"],
    edge_index: Int[Tensor, "batch n_token neighbours"],
    edge_mask: Bool[Tensor, "batch n_token neighbours"],
) -> Float[Tensor, "batch n_token num_residue_types"]:
    """Compute conditional logits for the Potts model.

    For each node i and candidate state a, computes:
        logit_i(a) = -(h_i(a) + sum_j J_ij(a, s_j))
    where s_j is the current state of neighbor j.

    Args:
        S_idx: Current sequence indices [B, N].
        h: Potts fields [B, N, R].
        J: Potts couplings [B, N, K, R, R].
        edge_index: Neighbor indices [B, N, K].
        edge_mask: Valid edge mask [B, N, K].

    Returns:
        Conditional logits [B, N, R].
    """
    from artios.model.modules.message_passing import gather_nodes

    R = h.shape[-1]

    # Mask out self-edges
    ii = torch.arange(edge_index.shape[1], device=edge_index.device)
    not_self = edge_index != ii[None, :, None]  # [B, N, K]

    # Get neighbour states [B, N, K, 1]
    S_neighbours = gather_nodes(S_idx.unsqueeze(-1).float(), edge_index).long()

    # Index into J with neighbour states: J[..., s_j] -> [B, N, K, R]
    S_neighbours = repeat(S_neighbours, "... n k 1 -> ... n k r 1", r=R)
    J_at_S = torch.gather(J, -1, S_neighbours).squeeze(-1)  # [B, N, K, R]

    # Zero out self-edge and invalid-edge contributions
    J_at_S = J_at_S * (not_self & edge_mask).unsqueeze(-1)

    # Sum coupling contributions over neighbours
    J_node = J_at_S.sum(dim=-2)  # [B, N, R]

    return -(h + J_node)


@typecheck
def compute_potts_energy(
    S: Int[Tensor, "batch n_token"],
    h: Float[Tensor, "batch n_token num_residue_types"],
    J: Float[Tensor, "batch n_token neighbours num_residue_types num_residue_types"],
    edge_index: Int[Tensor, "batch n_token neighbours"],
    edge_mask: Bool[Tensor, "batch n_token neighbours"],
) -> Float[Tensor, "batch"]:
    """Compute per-sample Potts energy.

    E = sum_i h_i(s_i) + 0.5 * sum_{i,j} J_ij(s_i, s_j)

    The 0.5 factor accounts for each edge being counted twice in the
    sparse neighbour representation.

    Args:
        S: Sequence indices [B, N].
        h: Potts fields [B, N, R].
        J: Potts couplings [B, N, K, R, R].
        edge_index: Neighbor indices [B, N, K].
        edge_mask: Valid edge mask [B, N, K].

    Returns:
        Per-sample energy [B].
    """
    from artios.model.modules.message_passing import gather_nodes

    B, N, K, R, _ = J.shape

    # Field energy: sum_i h_i(s_i)
    h_s = torch.gather(h, -1, S.unsqueeze(-1)).squeeze(-1)  # [B, N]
    field_energy = h_s.sum(-1)  # [B]

    # Coupling energy: sum_{i,j} J_ij(s_i, s_j)
    S_j = (
        gather_nodes(S.unsqueeze(-1).float(), edge_index).long().squeeze(-1)
    )  # [B, N, K]

    # Index J with s_i on dim -2
    S_i_idx = S[:, :, None, None, None].expand(B, N, K, 1, R)
    J_si = torch.gather(J, -2, S_i_idx).squeeze(-2)  # [B, N, K, R]

    # Index with s_j on last dim
    J_si_sj = torch.gather(J_si, -1, S_j.unsqueeze(-1)).squeeze(-1)  # [B, N, K]

    # Mask self-edges and invalid edges
    ii = torch.arange(N, device=edge_index.device)
    not_self = edge_index != ii[None, :, None]
    J_si_sj = J_si_sj * (not_self & edge_mask)

    coupling_energy = 0.5 * J_si_sj.sum(dim=(-1, -2))  # [B]

    return field_energy + coupling_energy


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
            self.num_factors = num_residue_types
            self.W_h = Linear(node_dim, R)
            self.W_h_bg = Linear(node_dim, 1)
            self.W_J_bg = Linear(edge_dim, 1)
            self.W_J_left = Linear(edge_dim, R * self.num_factors)
            self.W_J_right = Linear(edge_dim, R * self.num_factors)

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
