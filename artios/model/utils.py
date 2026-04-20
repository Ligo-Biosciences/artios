# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from typing import Optional

import torch
from torch import Tensor

from artios.utils.scatter_utils import scatter
from artios.utils.typing import Bool, Float, Int, typecheck


@typecheck
def one_hot(
    x: Float[Tensor, "*batch"],
    lower_bins: Float[Tensor, "bins"],
    upper_bins: Float[Tensor, "bins"],
) -> Float[Tensor, "*batch bins"]:
    """Get one hot embedding of x from lower_bins and upper_bins
    Args:
        x: the input x
        lower_bins: the lower bounds of bins
        upper_bins: the upper bounds of bins
    Returns:
        the one hot embedding of x from v_bins
    """
    dgram = (x[..., None] > lower_bins) * (x[..., None] < upper_bins).float()
    return dgram


# TODO: typecheck this
def batched_gather(
    data: Tensor, inds: Tensor, dim: int = 0, no_batch_dims: int = 0
) -> Tensor:
    """Gather data according to indices specify by inds

    Args:
        data (Tensor): the input data
            [..., K, ...]
        inds (Tensor): the indices for gathering data
            [..., N]
        dim (int, optional): along which dimension to gather data by inds (the dim of "K" "N"). Defaults to 0.
        no_batch_dims (int, optional): length of dimensions before the "dim" dimension. Defaults to 0.

    Returns:
        Tensor: gathered data
            [..., N, ...]
    """

    # for the naive case
    if len(inds.shape) == 1 and no_batch_dims == 0 and dim == 0:
        return data[inds]

    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        pre_index_dims = (1,) * i
        post_index_dims = (1,) * (len(inds.shape) - i - 1)
        r = r.view(*pre_index_dims, -1, *post_index_dims)
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[tuple(ranges)]


@typecheck
def broadcast_token_to_atom(
    x_token: Float[Tensor, "*batch n_token d"],
    atom_to_token_idx: Int[Tensor, "*batch n_atom"],
) -> Float[Tensor, "*batch n_atom d"]:
    """Broadcast token-level embeddings to atom-level embeddings

    Args:
        x_token: token embedding
        atom_to_token_idx: map atom idx to token idx

    Returns:
        atom embedding
    """

    if len(atom_to_token_idx.shape) == 1:
        # shape = [N_atom], easy index
        return x_token[..., atom_to_token_idx, :]

    return batched_gather(
        data=x_token,
        inds=atom_to_token_idx,
        dim=-2,
        no_batch_dims=len(x_token.shape[:-2]),
    )


@typecheck
def average_atom_to_token(
    x_atom: Float[Tensor, "... n_atom d"],
    atom_mask: Bool[Tensor, "... n_atom"],
    atom_to_token_idx: Int[Tensor, "... n_atom"],
    n_token: int,
) -> Float[Tensor, "*batch n_token d"]:
    """Aggregate atom embedding to obtain token embedding

    Args:
        x_atom: atom-level embedding
        atom_mask: atom mask indicating non-padding atoms
        atom_to_token_idx: map atom to token idx
        n_token (int): number of tokens in total.

    Returns:
        token-level embedding
    """
    assert (
        len(x_atom.shape[:-2]) == len(atom_to_token_idx.shape[:-1])
    ), f"got x_atom.shape[:-2] = {x_atom.shape[:-2]}, atom_to_token_idx.shape[:-1] = {atom_to_token_idx.shape[:-1]}"
    assert (
        len(x_atom.shape[:-2]) == len(atom_mask.shape[:-1])
    ), f"got x_atom.shape[:-2] = {x_atom.shape[:-2]}, atom_mask.shape[:-1] = {atom_mask.shape[:-1]}"

    # Mask out the padding atoms
    x_atom = x_atom * atom_mask[..., None]

    # Sum the atoms
    out_sum = scatter(
        src=x_atom, index=atom_to_token_idx, dim=-2, dim_size=n_token, reduce="sum"
    )
    out_count = scatter(
        src=atom_mask[..., None],
        index=atom_to_token_idx,
        dim=-2,
        dim_size=n_token,
        reduce="sum",
    )
    out_mean = out_sum / (out_count + 1e-4)

    return out_mean


def beta_function(
    t: Tensor,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """
    Evaluate the Beta probability density function at specified timestep values,
    with the distribution parameterized by either mean/std or alpha/beta.

    Args:
        t (Tensor): Timestep values between 0 and 1
        mean (float, optional): Mean of the Beta distribution (between 0 and 1)
        std (float, optional): Standard deviation of the Beta distribution
        alpha (float, optional): Alpha parameter of the Beta distribution (> 0)
        beta (float, optional): Beta parameter of the Beta distribution (> 0)
        device (torch.device, optional): Device for the computation
        dtype (torch.dtype, optional): Data type for the computation

    Returns:
        Tensor: Normalised Beta function values evaluated at the specified timesteps
    """
    if device is not None and dtype is not None:
        t = t.to(device=device, dtype=dtype)

    # Check parameter combinations
    mean_std_provided = mean is not None and std is not None
    alpha_beta_provided = alpha is not None and beta is not None

    if mean_std_provided and alpha_beta_provided:
        raise ValueError(
            "Cannot specify both (mean, std) and (alpha, beta). Choose one parameterization."
        )
    elif not mean_std_provided and not alpha_beta_provided:
        raise ValueError("Must specify either (mean, std) or (alpha, beta) parameters.")

    # Compute alpha and beta values (as floats first for validation)
    if mean_std_provided:
        assert mean is not None and std is not None
        var = std**2

        # Ensure the parameters are valid
        if mean * (1 - mean) <= var:
            raise ValueError(
                f"Invalid parameters: mean={mean}, std={std}. Requires mean*(1-mean) > std^2"
            )

        common_term = (mean * (1 - mean) / var) - 1
        alpha_val = mean * common_term
        beta_val = (1 - mean) * common_term
    else:
        assert alpha is not None and beta is not None
        if alpha <= 0 or beta <= 0:
            raise ValueError(
                f"Alpha and beta must be positive. Got alpha={alpha}, beta={beta}"
            )
        alpha_val = alpha
        beta_val = beta

    # Convert to tensors for computation
    alpha_t = torch.tensor(alpha_val, device=t.device, dtype=t.dtype)
    beta_t = torch.tensor(beta_val, device=t.device, dtype=t.dtype)

    # Calculate Beta function normalization constant using log-gamma
    # B(alpha, beta) = Γ(alpha)Γ(beta)/Γ(alpha+beta)
    log_norm = (
        torch.lgamma(alpha_t) + torch.lgamma(beta_t) - torch.lgamma(alpha_t + beta_t)
    )
    normalization = torch.exp(log_norm)

    return t ** (alpha_t - 1) * (1 - t) ** (beta_t - 1) / normalization


def masked_mean(
    input: Tensor,
    mask: Tensor,
    axes: int | list[int],
    keepdim: bool = False,
    eps: float = 1e-6,
) -> Tensor:
    """
    Calculate the masked mean of the input tensor. Assumes the input and the mask
    have the same number of dimensions but not necessarily the same shape.

    Args:
        input (Tensor): The input tensor.
        mask (Tensor): The mask tensor.
        axes (int | list[int]): The axes to reduce.
        keepdim (bool): Whether to keep the dimensions of the input tensor.
        eps (float): The epsilon to add to the denominator to avoid division by zero.

    Returns:
        Tensor: The masked mean of the input tensor.
    """
    assert len(input.shape) == len(
        mask.shape
    ), "Input and mask must have the same number of dimensions"

    if isinstance(axes, int):
        axes = [axes]

    broadcast_factor = 1.0
    for ax in axes:
        input_size = input.shape[ax]
        mask_size = mask.shape[ax]
        if mask_size == 1:
            broadcast_factor = broadcast_factor * input_size
        elif input_size == mask_size:
            continue
        else:
            raise ValueError(
                f"Input and mask must have the same size or mask must be 1 along the reduced axes. Got input size {input_size} and mask size {mask_size} along axis {ax}"
            )

    mean = (input * mask).sum(dim=axes, keepdim=keepdim) / (
        mask.sum(dim=axes, keepdim=keepdim) * broadcast_factor + eps
    )

    return mean


def build_backbone_ligand_proximity_mask(
    backbone_coordinate: Tensor,
    centre_coordinate: Tensor,
    is_ligand_token: Tensor,
    token_mask: Tensor,
    backbone_mask: Tensor,
    distance_cutoff: float,
) -> Tensor:
    """Identify non-ligand tokens with a backbone atom near any ligand token.

    For each non-ligand token, checks whether any of its valid backbone atoms
    is within ``distance_cutoff`` of any ligand token's centre coordinate.

    Args:
        backbone_coordinate: Backbone atom coordinates [B, N, 5, 3].
        centre_coordinate: Per-token centre atom coordinates [B, N, 3].
        is_ligand_token: Ligand token indicator [B, N].
        token_mask: Valid token mask [B, N].
        backbone_mask: Valid backbone atom mask [B, N, 5].
        distance_cutoff: Distance threshold in angstroms.

    Returns:
        Boolean mask [B, N] — True for non-ligand tokens that have at least
        one backbone atom within *distance_cutoff* of a ligand token.
    """
    B, N = is_ligand_token.shape
    n_bb = backbone_coordinate.shape[2]
    device = backbone_coordinate.device
    near_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

    for b in range(B):
        lig_mask = is_ligand_token[b] & token_mask[b]
        if not lig_mask.any():
            continue

        lig_centres = centre_coordinate[b][lig_mask]  # [L, 3]

        # Valid backbone atoms for non-ligand tokens only
        non_lig_bb_mask = backbone_mask[b] & (
            ~is_ligand_token[b] & token_mask[b]
        ).unsqueeze(-1)  # [N, 5]

        bb_flat = backbone_coordinate[b].reshape(N * n_bb, 3)  # [N*5, 3]
        bb_flat_mask = non_lig_bb_mask.reshape(N * n_bb)  # [N*5]

        valid_idx = bb_flat_mask.nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            continue

        valid_bb = bb_flat[valid_idx]  # [V, 3]
        dists = torch.cdist(valid_bb.unsqueeze(0), lig_centres.unsqueeze(0)).squeeze(
            0
        )  # [V, L]
        is_near = dists.min(dim=-1).values <= distance_cutoff  # [V]

        # Map back to token level: any backbone atom near → token is near
        near_per_flat = torch.zeros(N * n_bb, dtype=torch.bool, device=device)
        near_per_flat[valid_idx] = is_near
        near_mask[b] = near_per_flat.reshape(N, n_bb).any(dim=-1)

    return near_mask


class Mode(str, Enum):
    """Supported modes of operation for model forward passes."""

    TRAIN = "train"
    INFERENCE = "inference"
    EVAL = "eval"
