"""Potts model for sequence design.

Parameterizes fields h and couplings J from MPNN decoder features.
"""

from enum import Enum

import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from typing_extensions import assert_never

from artios.model.modules.message_passing import gather_nodes
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


def build_mutable_mask(
    token_mask: Bool[Tensor, "batch n_token"],
    stitched_mask: Bool[Tensor, "batch n_token"] | None,
) -> Bool[Tensor, "batch n_token"]:
    """Build mask of positions that are mutable.
    In practice, this is just the complement of the stitched mask.

    Args:
        token_mask: Valid token mask [B, N].
        stitched_mask: Stitched (fixed) position mask [B, N], or None.

    Returns:
        Boolean mask of mutable positions [B, N].
    """
    if stitched_mask is not None:
        return token_mask & ~stitched_mask
    return token_mask


def build_alphabet_mask(
    mutable_mask: Bool[Tensor, "batch n_token"],
    num_residue_types: int,
    exclude_cys: bool = False,
    exclude_met: bool = False,
) -> Bool[Tensor, "batch n_token num_residue_types"]:
    """Build mask of allowed residue types per position.

    For mutable positions, allows the 20 standard amino acids (indices 0-19),
    optionally excluding cysteine (index 4) and/or methionine (index 12).
    Non-mutable positions are left as all-False (they are protected by
    token_mask in the sampler).

    Args:
        mutable_mask: Mutable position mask [B, N].
        num_residue_types: Total number of residue types R.
        exclude_cys: If True, disallow cysteine (index 4).
        exclude_met: If True, disallow methionine (index 12).

    Returns:
        Boolean mask of allowed residue types [B, N, R].
    """
    B, N = mutable_mask.shape
    device = mutable_mask.device
    R = num_residue_types

    mask = torch.zeros(B, N, R, dtype=torch.bool, device=device)
    # Allow first 20 amino acids for mutable positions
    mask[:, :, :20] = mutable_mask.unsqueeze(-1)

    if exclude_cys:
        mask[:, :, 4] = False
    if exclude_met:
        mask[:, :, 12] = False

    return mask


def build_s_init(
    h: Float[Tensor, "batch n_token num_residue_types"],
    temperature: float,
    stitched_mask: Bool[Tensor, "batch n_token"] | None,
    stitched_restype: Float[Tensor, "batch n_token num_residue_types"] | None,
    generator: torch.Generator | None,
) -> Float[Tensor, "batch n_token num_residue_types"] | None:
    """Build initial sequence for Potts sampling.

    If stitched_mask is None, returns None (sampler initializes from h internally).
    Otherwise, samples mutable positions from softmax(-h / temperature) and
    sets stitched positions to stitched_restype.

    Args:
        h: Potts fields (already alphabet-biased) [B, N, R].
        temperature: Initial sampling temperature.
        stitched_mask: Stitched position mask [B, N], or None.
        stitched_restype: One-hot stitched residue types [B, N, R], or None.
        generator: Optional torch Generator for reproducibility.

    Returns:
        Initial one-hot sequence [B, N, R], or None.
    """
    if stitched_mask is None:
        return None

    B, N, R = h.shape
    logits = torch.nn.functional.log_softmax(-h / temperature, dim=-1)
    indices = torch.multinomial(
        logits.exp().reshape(B * N, R), 1, generator=generator
    ).reshape(B, N)
    S_init = torch.nn.functional.one_hot(indices, R)
    S_init = torch.where(stitched_mask.unsqueeze(-1), stitched_restype, S_init).float()
    return S_init


@typecheck
def compute_potts_energy(
    S: Int[Tensor, "batch n_token"],
    h: Float[Tensor, "batch n_token num_residue_types"],
    J: Float[Tensor, "batch n_token neighbours num_residue_types num_residue_types"],
    edge_index: Int[Tensor, "batch n_token neighbours"],
    edge_mask: Bool[Tensor, "batch n_token neighbours"],
) -> Float[Tensor, "batch"]:
    """Compute total Potts energy for every sequence in a batch.

    E = sum_i h_i(s_i) + 0.5 * sum_{i,j} J_ij(s_i, s_j)

    The 0.5 factor accounts for edge double counting..

    Args:
        S: Sequence indices [B, N].
        h: Potts fields [B, N, R].
        J: Potts couplings [B, N, K, R, R].
        edge_index: Neighbor indices [B, N, K].
        edge_mask: Valid edge mask [B, N, K].

    Returns:
        Per-sample energy [B].
    """
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


def _symmetrize_J(
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


@typecheck
def compute_potts_single_logits(
    seq: Int[Tensor, "batch n_token"],
    h: Float[Tensor, "batch n_token num_residue_types"],
    J: Float[Tensor, "batch n_token neighbours num_residue_types num_residue_types"],
    edge_index: Int[Tensor, "batch n_token neighbours"],
    edge_mask: Bool[Tensor, "batch n_token neighbours"],
) -> Float[Tensor, "batch n_token num_residue_types"]:
    """Compute conditional logits for the Potts model.
    This is the logits for the conditional probability:
    p(state of i | state of neighbours of i ; h, J).

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
    R = h.shape[-1]

    # Mask out self-edges
    ii = torch.arange(edge_index.shape[1], device=edge_index.device)
    not_self = edge_index != ii[None, :, None]  # [B, N, K]

    # Get neighbour states [B, N, K, 1]
    S_neighbours = gather_nodes(seq.unsqueeze(-1).float(), edge_index).long()

    # Index into J with neighbour states: J[..., s_j] -> [B, N, K, R]
    S_neighbours = repeat(S_neighbours, "... n k 1 -> ... n k r 1", r=R)
    J_at_S = torch.gather(J, -1, S_neighbours).squeeze(-1)  # [B, N, K, R]

    # Apply mask and remove self-edges
    J_at_S = J_at_S * (not_self & edge_mask).unsqueeze(-1)

    # Sum coupling contributions over neighbours
    J_node = J_at_S.sum(dim=-2)  # [B, N, R]

    return -(h + J_node)


@typecheck
def compute_potts_joint_logits(
    seq: Int[Tensor, "batch n_token"],
    h: Float[Tensor, "batch n_token num_residue_types"],
    J: Float[Tensor, "batch n_token neighbours num_residue_types num_residue_types"],
    edge_index: Int[Tensor, "batch n_token neighbours"],
    edge_mask: Bool[Tensor, "batch n_token neighbours"],
) -> tuple[
    Float[Tensor, "batch n_token neighbours num_residue_types num_residue_types"],
    Bool[Tensor, "batch n_token neighbours"],
]:
    """Compute joint Potts logits for composite likelihood.

    The logits are the negative energy of the joint state pairs.
    For each edge (i, j), computes the joint logit over all state pairs
    (a, b), accounting for background contributions from all other nodes.

    Args:
        seq: Current sequence indices [B, N].
        h: Potts fields [B, N, R].
        J: Potts couplings [B, N, K, R, R].
        edge_index: Neighbor indices [B, N, K].
        edge_mask: Valid edge mask [B, N, K].

    Returns:
        Tuple of (joint_logits, combined_edge_mask):
            joint_logits: [B, N, K, R, R]
            combined_edge_mask: [B, N, K] (self-edges excluded)
    """
    _, N, K = edge_index.shape
    R = h.shape[-1]

    # Remove self edges (prevents leakage)
    ii = torch.arange(N, device=edge_index.device)
    not_self = edge_index != ii[None, :, None]
    combined_mask = not_self & edge_mask

    ### Get interactions between node states and GT neighbour states
    S_j = gather_nodes(seq.unsqueeze(-1).float(), edge_index).long()
    # Index into J with GT neighbour states J[..., s_j] -> [B, N, K, R]
    S_j = repeat(S_j, "... n k 1 -> ... n k r 1", r=R)
    J_j = torch.gather(J, -1, S_j).squeeze(-1)  # [B, N, K, R]

    ### Get interactions between GT node state and neighbour states
    # Index into J with GT node states J[..., s_i, :] -> [B, N, K, 1, R]
    S_i = repeat(seq.long(), "b n -> b n k 1 r", k=K, r=R)
    J_i = torch.gather(J, -2, S_i)  # [B, N, K, 1, R]

    # Get node energies from GT neighbour states
    J_j_self_edge_masked = J_j * not_self.unsqueeze(-1)  # zero out self-edges
    E_i = h + J_j_self_edge_masked.sum(dim=2)  # [B, N, R]

    # Get neighbour energies by gathering node energies
    E_j = gather_nodes(E_i, edge_index)  # [B, N, K, R]

    # Remove J_ij contribution from E_i (prevents leakage)
    E_i_minus_ij = repeat(E_i, "b n r -> b n k r 1", k=K) - J_j.unsqueeze(
        -1
    )  # [B, N, K, R, 1]

    # Remove J_ji contribution from E_j (prevents leakage)
    E_j_minus_ji = repeat(E_j, "b n k r -> b n k 1 r") - J_i  # [B, N, K, 1, R]

    # Joint logits (negative energy)
    joint_logits = -(E_i_minus_ij + E_j_minus_ji + J)  # [B, N, K, R, R]

    return joint_logits, combined_mask


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
        zero_sum_gauge: bool = True,
        init_scale: float = 0.1,
        num_factors: int = 32,
    ):
        super().__init__()
        self.num_residue_types = num_residue_types
        self.symmetric_J = symmetric_J
        self.zero_sum_gauge = zero_sum_gauge
        self.parameterisation = parameterisation
        self.num_factors = num_factors

        R = num_residue_types
        F = num_factors
        self.log_scale = nn.Parameter(np.log(init_scale) * torch.ones(1))

        match parameterisation:
            case GraphPottsParameterisation.LINEAR:
                self.W_h = Linear(node_dim, R)
                self.W_J = Linear(edge_dim, R**2)

            case GraphPottsParameterisation.FACTOR:
                self.W_h = Linear(node_dim, R)
                self.W_J_left = Linear(edge_dim, R * F)
                self.W_J_right = Linear(edge_dim, F * R)

            case GraphPottsParameterisation.SCORE:
                self.W_h = Linear(node_dim, R)
                self.W_h_bg = Linear(node_dim, 1)
                self.W_J_bg = Linear(edge_dim, 1)
                self.W_J_left = Linear(edge_dim, R * F)
                self.W_J_right = Linear(edge_dim, F * R)

            case _:
                assert_never(parameterisation)

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
        F = self.num_factors
        param = self.parameterisation

        match param:
            case GraphPottsParameterisation.LINEAR:
                h = mask_h * self.W_h(node_feats)
                J_flat = mask_J * self.W_J(edge_feats)
                J = rearrange(J_flat, "b n k (r1 r2) -> b n k r1 r2", r1=R)

            case GraphPottsParameterisation.FACTOR:
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

            case GraphPottsParameterisation.SCORE:
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

                h = h + mask_h * self.W_h_bg(node_feats)
                J = J + (mask_J * self.W_J_bg(edge_feats)).unsqueeze(-1)

        if self.zero_sum_gauge:
            h, J = _zero_sum_gauge(h, J)

        if self.symmetric_J:
            J = _symmetrize_J(J, edge_index, edge_mask)

        return h, J
