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

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from torch import Tensor

from protenix.utils.scatter_utils import scatter


def add(a, b, inplace_safe: bool = False):
    """
    Add two objects with optional in-place modification.

    Args:
        a: First object to add
        b: Second object to add
        in_place (bool): If True, modifies first argument in place. Defaults to False.

    Returns:
        The sum of the two objects
    """
    if inplace_safe:
        a += b
        return a
    else:
        return a + b


def mul(a, b, inplace_safe: bool = False):
    if inplace_safe:
        a *= b
        return a
    else:
        return a * b


def centre_random_augmentation(
    x_input_coords: Tensor,
    N_sample: int = 1,
    s_trans: float = 1.0,
    centre_only: bool = False,
    mask: Tensor = None,
    eps: float = 1e-12,
) -> Tensor:
    """Implements Algorithm 19 in AF3

    Args:
        x_input_coords (Tensor): input coords
            [..., N_atom, 3]
        N_sample (int, optional): the total number of augmentation. Defaults to 1.
        s_trans (float, optional): scale factor of trans. Defaults to 1.0.
        centre_only (bool, optional): if set true, will only perform centering without applying random translation and rotation.
        mask (Tensor, optional): masking for the coords
            [..., N_atom]
        eps (float, optional): small number used for masked mean
    Returns:
        Tensor:  the Augmentation version of input coords
            [..., N_sample, N_atom, 3]
    """

    N_atom = x_input_coords.size(-2)
    device = x_input_coords.device

    # Move to origin [..., N_atom, 3]
    if mask is None:
        x_input_coords = x_input_coords - torch.mean(
            input=x_input_coords, dim=-2, keepdim=True
        )
    else:
        center = (x_input_coords * mask.unsqueeze(dim=-1)).sum(dim=-2) / (
            mask.sum(dim=-1) + eps
        )
        x_input_coords = x_input_coords - center.unsqueeze(dim=-2)

    # Expand to [..., N_sample, N_atom, 3]
    x_input_coords = expand_at_dim(x_input_coords, dim=-3, n=N_sample)

    if centre_only:
        if mask is not None:
            x_input_coords = x_input_coords * mask[..., None, :, None]
        return x_input_coords

    # N_augment = batch_size * N_sample
    N_augment = torch.numel(x_input_coords[..., 0, 0])

    # Generate N_augment (rot, trans) pairs
    batch_size_shape = x_input_coords.shape[:-3]
    rot_matrix_random = (
        uniform_random_rotation(N_sample=N_augment)
        .to(device)
        .reshape(*batch_size_shape, N_sample, 3, 3)
    ).detach()  # [..., N_sample, 3, 3]
    trans_random = s_trans * torch.randn(size=(*batch_size_shape, N_sample, 3)).to(
        device
    )  # [..., N_sample, 3]
    x_augment_coords = (
        rot_vec_mul(
            r=expand_at_dim(rot_matrix_random, dim=-3, n=N_atom), t=x_input_coords
        )
        + trans_random[..., None, :]
    )  # [..., N_sample, N_atom, 3]

    # Apply the mask
    if mask is not None:
        x_augment_coords = x_augment_coords * mask[..., None, :, None]
    return x_augment_coords


# Comment: Rotation.random is not supported by torch.compile()
def uniform_random_rotation(N_sample: int = 1) -> Tensor:
    """Generate random rotation matrices with scipy.spatial.transform.Rotation

    Args:
        N_sample (int, optional): the total number of augmentation. Defaults to 1.

    Returns:
        Tensor: N_sample rot matrics
            [N_sample, 3, 3]
    """
    rotation = Rotation.random(num=N_sample)
    rot_matrix = torch.from_numpy(rotation.as_matrix()).float()  # [N_sample, 3, 3]
    return rot_matrix


# this is from openfold.utils.rigid_utils import rot_vec_mul
def rot_vec_mul(r: Tensor, t: Tensor) -> Tensor:
    """Apply rot matrix to vector
    Applies a rotation to a vector. Written out by hand to avoid transfer
    to avoid AMP downcasting.

    Args:
        r (Tensor): the rotation matrices
            [..., 3, 3]
        t (Tensor): the coordinate tensors
            [..., 3]

    Returns:
        Tensor: the rotated coordinates
    """
    x, y, z = torch.unbind(input=t, dim=-1)
    return torch.stack(
        tensors=[
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


# from openfold.utils.tensor_utils.permute_final_dims
# from openfold.utils.tensor_utils.flatten_final_dims
def permute_final_dims(tensor: Tensor, inds: list[int]) -> Tensor:
    """Permute final dims of tensor

    Args:
        tensor (Tensor): the input tensor
            [...]
        inds (List[int]): the dim to permute

    Returns:
        Tensor: the permuted tensor
    """
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: Tensor, num_dims: int) -> Tensor:
    """Flatten final dims of tensor

    Args:
        t (Tensor): the input tensor
            [...]
        num_dims (int): the number of final dims to flatten

    Returns:
        Tensor: the flattened tensor
    """
    return t.reshape(shape=t.shape[:-num_dims] + (-1,))


def one_hot(x: Tensor, lower_bins: Tensor, upper_bins: Tensor) -> Tensor:
    """Get one hot embedding of x from lower_bins and upper_bins
    Args:
        x (Tensor): the input x
            [...]
        lower_bins (Tensor): the lower bounds of bins
            [bins]
        upper_bins (Tensor): the upper bounds of bins
            [bins]
    Returns:
        Tensor: the one hot embedding of x from v_bins
            [..., bins]
    """
    dgram = (x[..., None] > lower_bins) * (x[..., None] < upper_bins).float()
    return dgram


# this is mostly from openfold.utils.torch_utils import batched_gather
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
    return data[ranges]


def broadcast_token_to_atom(x_token: Tensor, atom_to_token_idx: Tensor) -> Tensor:
    """Broadcast token-level embeddings to atom-level embeddings

    Args:
        x_token (Tensor): token embedding
            [..., N_token, d]
        atom_to_token_idx (Tensor): map atom idx to token idx
            [..., N_atom] or [N_atom]

    Returns:
        Tensor: atom embedding
            [..., N_atom, d]
    """

    if len(atom_to_token_idx.shape) == 1:
        # shape = [N_atom], easy index
        return x_token[..., atom_to_token_idx, :]
    else:
        assert atom_to_token_idx.shape[:-1] == x_token.shape[:-2]

    return batched_gather(
        data=x_token,
        inds=atom_to_token_idx,
        dim=-2,
        no_batch_dims=len(x_token.shape[:-2]),
    )


def average_atom_to_token(
    x_atom: Tensor,
    atom_mask: Tensor,
    atom_to_token_idx: Tensor,
    n_token: int,
) -> Tensor:
    """Aggregate atom embedding to obtain token embedding

    Args:
        x_atom (Tensor): atom-level embedding
            [..., N_atom, d]
        atom_mask (Tensor): atom mask indicating non-padding atoms
            [..., N_atom]
        atom_to_token_idx (Tensor): map atom to token idx
            [..., N_atom] or [N_atom]
        n_token (int): number of tokens in total.

    Returns:
        Tensor: token-level embedding
            [..., N_token, d]
    """

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


def sample_indices(
    n: int,
    device: torch.device = torch.device("cpu"),
    lower_bound=1,
    strategy: str = "random",
) -> Tensor:
    """Sample msa indices k from uniform[1,n]

    Args:
        n (int): the msa num
        strategy (str): the strategy to sample msa index, random or topk

    Returns:
        Tensor: the sampled indices k
    """
    assert strategy in ["random", "topk"]
    sample_size = torch.randint(low=min(lower_bound, n), high=n + 1, size=(1,)).item()
    if strategy == "random":
        indices = torch.randperm(n=n, device=device)[:sample_size]
    if strategy == "topk":
        indices = torch.arange(sample_size, device=device)
    return indices


def sample_msa_feature_dict_random_without_replacement(
    feat_dict: dict[str, Tensor],
    dim_dict: dict[str, int],
    cutoff: int = 512,
    lower_bound: int = 1,
    strategy: str = "random",
) -> dict[str, Tensor]:
    """Sample a dict of MSA features randomly without replacement.

    Args:
        feat_dict (dict[str, Tensor]): A dict containing the MSA features.
        dim_dict (dict[str, int]): A dict containing the dimensions of the MSA features.
        cutoff (int): The maximum number of features to sample.
        lower_bound (int): The minimum number of features to sample.
        strategy (str): The sampling strategy to use. Can be either "random" or "sequential".

    Returns:
        dict[str, Tensor]: A dict containing the sampled MSA features.
    """
    msa_len = feat_dict["msa"].size(dim=dim_dict["msa"])
    indices = sample_indices(
        n=msa_len,
        device=feat_dict["msa"].device,
        lower_bound=lower_bound,
        strategy=strategy,
    )
    if cutoff > 0:
        indices = indices[:cutoff]

    msa_feat_dict = {
        feat_name: torch.index_select(
            input=feat_dict[feat_name], dim=dim, index=indices
        )
        for feat_name, dim in dim_dict.items()
    }
    return msa_feat_dict


def expand_at_dim(x: Tensor, dim: int, n: int) -> Tensor:
    """expand a tensor at specific dim by n times

    Args:
        x (Tensor): input
        dim (int): dimension to expand
        n (int): expand size

    Returns:
        Tensor: expanded tensor of shape [..., n, ...]
    """
    x = x.unsqueeze(dim=dim)
    if dim < 0:
        dim = x.dim() + dim
    before_shape = x.shape[:dim]
    after_shape = x.shape[dim + 1 :]
    return x.expand(*before_shape, n, *after_shape)


def pad_at_dim(
    x: Tensor,
    dim: int,
    pad_length: Union[tuple[int], list[int]],
    value: float = 0,
) -> Tensor:
    """pad to input x at dimension dim with length pad_length[0] to the left and and pad_length[1] to the right.

    Args:
        x (Tensor): input
        dim (int): padding dimension
        pad_length (Union[Tuple[int], List[int]]): length to pad to the beginning and end.

    Returns:
        Tensor: padded tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    pad = (pad_length[0], pad_length[1])
    if pad == (0, 0):
        return x
    k = n_dim - (dim + 1)
    if k > 0:
        pad_skip = (0, 0) * k
        pad = (*pad_skip, *pad)
    return nn.functional.pad(x, pad=pad, value=value)


def reshape_at_dim(
    x: Tensor, dim: int, target_shape: Union[tuple[int], list[int]]
) -> Tensor:
    """reshape dimension dim of x to target_shape

    Args:
        x (Tensor): input
        dim (int): dimension to reshape
        target_shape (Union[Tuple[int], List[int]]): target_shape of dim

    Returns:
        Tensor: reshaped tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    target_shape = tuple(target_shape)
    target_shape = (*x.shape[:dim], *target_shape)
    if dim + 1 < n_dim:
        target_shape = (*target_shape, *x.shape[dim + 1 :])
    return x.reshape(target_shape)


def index_at_dim(x: Tensor, dim: int, idx: Tensor) -> Tensor:
    """Return x indexed by idx along `dim`, preserving all other dims.
    If idx is [T, K], the result inserts those two dims at `dim`."""
    if dim < 0:
        dim += x.ndim
    index = [slice(None)] * x.ndim
    index[dim] = idx
    return x[tuple(index)]


def move_final_dim_to_dim(x: Tensor, dim: int) -> Tensor:
    """
    Move the final dimension of a tensor to a specified dimension.

    Args:
        x (Tensor): Input tensor.
        dim (int): Target dimension to move the final dimension to.

    Returns:
        Tensor: Tensor with the final dimension moved to the specified dimension.
    """
    # permute_final_dims
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim
    if dim >= n_dim - 1:
        return x

    new_order = (n_dim - 1,)
    if dim > 0:
        new_order = tuple(range(dim)) + new_order
    if dim < n_dim - 1:
        new_order = new_order + tuple(range(dim, n_dim - 1))

    return x.permute(new_order)


def simple_merge_dict_list(dict_list: list[dict]) -> dict:
    """
    Merge a list of dictionaries into a single dictionary.

    Args:
        dict_list (list[dict]): List of dictionaries to merge.

    Returns:
        dict: Merged dictionary where values are concatenated arrays.
    """
    merged_dict = {}

    def add(key, value):
        merged_dict.setdefault(key, [])
        if isinstance(value, (float, int)):
            value = np.array([value])
        elif isinstance(value, Tensor):
            if value.dim() == 0:
                value = np.array([value.item()])
            else:
                value = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type for metric data: {type(value)}")
        merged_dict[key].append(value)

    for x in dict_list:
        for k, v in x.items():
            add(k, v)
    for k, v in merged_dict.items():
        merged_dict[k] = np.concatenate(v)
    return merged_dict


def sample_logit_normal(
    size, device, dtype=torch.float32, loc=0, scale=1, generator=None
):
    """Sample from a logit-normal distribution. Used to generate timesteps for flow matching."""
    normal_samples = torch.normal(
        loc, scale, size, device=device, dtype=dtype, generator=generator
    )
    return torch.sigmoid(normal_samples)


def noise_w_flow_matching(data, noise, timestep, sigma_min=1e-5):
    """Flow-matching interpolation between the data and noise.
    Requires data, noise and timestep to be broadcastable.
    """
    return timestep * data + (1 - (1 - sigma_min) * timestep) * noise


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

    if mean_std_provided:
        # Convert mean and std to alpha and beta parameters
        if device is not None and dtype is not None:
            mean = Tensor(mean, device=device, dtype=dtype)
            std = Tensor(std, device=device, dtype=dtype)

        var = std**2

        # Ensure the parameters are valid
        if mean * (1 - mean) <= var:
            raise ValueError(
                f"Invalid parameters: mean={mean}, std={std}. Requires mean*(1-mean) > std^2"
            )

        common_term = (mean * (1 - mean) / var) - 1
        alpha = mean * common_term
        beta = (1 - mean) * common_term
    else:
        # Use alpha and beta directly
        if alpha <= 0 or beta <= 0:
            raise ValueError(
                f"Alpha and beta must be positive. Got alpha={alpha}, beta={beta}"
            )

        if device is not None and dtype is not None:
            alpha = Tensor(alpha, device=device, dtype=dtype)
            beta = Tensor(beta, device=device, dtype=dtype)

    # Calculate Beta function normalization constant using log-gamma
    # B(alpha, beta) = Γ(alpha)Γ(beta)/Γ(alpha+beta)
    log_norm = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    normalization = torch.exp(log_norm)

    return t ** (alpha - 1) * (1 - t) ** (beta - 1) / normalization


def softmax_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    """Softmax cross entropy

    Args:
        logits (Tensor): classification logits
            [..., num_class]
        labels (Tensor): classification labels (value = probability)
            [..., num_class]

    Returns:
        Tensor: softmax cross entropy
            [...]
    """
    loss = -1 * torch.sum(
        labels * F.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


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
