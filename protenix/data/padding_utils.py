"""
Simplified padding utilities using get_data_shape_dict for shape inference.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from protenix.data.utils import get_data_shape_dict
from protenix.model.utils import pad_at_dim

# Default padding values for specific features
DEFAULT_PADDING_VALUES: Dict[str, Union[float, int]] = {
    "coordinate": 0,  # Pad coordinates with 0
    "ref_pos": 0,
    "atom_to_token_idx": 0,  # Will be adjusted after padding
    "atom_to_tokatom_idx": 0,
    "frame_atom_index": -1,
    "ref_space_uid": 0,
    "template_restype": 31,  # Gap token for templates
}


def get_padding_length(num: int, bin_ends: list[int]) -> int:
    """
    Given an input number of atoms or tokes as well as a list of bin ends, returns the bin
    the input lies in.

    Args
        num (int): Input number of atoms or tokens.
        bin_ends (list[int]): List of bin ends.

    Returns:
        int: Bin end for the given input number.
    """
    for end in bin_ends:
        if num <= end:
            return end
    return bin_ends[-1]


def pad_features_dict(
    features_dict: Dict[str, torch.Tensor],
    current_n_atom: int,
    max_n_atom: int,
    current_n_token: int,
    max_n_token: int,
    current_n_msa: int,
    max_n_msa: int,
    current_n_template: int,
    max_n_template: int,
    current_n_pocket: int,
    max_n_pocket: int,
    padding_values: Optional[Dict[str, Union[float, int]]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Pad features dictionary using shape inference from get_data_shape_dict.

    Algorithm:
    1. Get expected shapes from get_data_shape_dict for both current and max sizes
    2. For each feature, compare current shape with expected max shape
    3. Pad dimensions that need padding

    Args:
        features_dict (Dict[str, torch.Tensor]): Dictionary of features to pad
        current_n_atom (int): Current number of atoms
        max_n_atom (int): Maximum number of atoms to pad to
        current_n_token (int): Current number of tokens
        max_n_token (int): Maximum number of tokens to pad to
        current_n_msa (int): Current number of MSA sequences
        max_n_msa (int): Maximum number of MSA sequences
        current_n_template (int): Current number of templates
        max_n_template (int): Maximum number of templates
        current_n_pocket (int): Current number of pockets
        max_n_pocket (int): Maximum number of pockets
        padding_values (Optional[Dict[str, Union[float, int]]]): Custom padding values for specific features

    Returns:
        Dict[str, torch.Tensor]: Dictionary with padded features
    """

    # Get expected shapes for current and target dimensions
    current_shapes, current_label_shapes = get_data_shape_dict(
        num_token=current_n_token,
        num_atom=current_n_atom,
        num_msa=current_n_msa,
        num_templ=current_n_template,
        num_pocket=current_n_pocket,
    )

    padded_shapes, padded_label_shapes = get_data_shape_dict(
        num_token=max_n_token,
        num_atom=max_n_atom,
        num_msa=max_n_msa,
        num_templ=max_n_template,
        num_pocket=max_n_pocket,
    )

    # Combine all shape dicts (features and labels may overlap)
    all_current_shapes = {**current_shapes, **current_label_shapes}
    all_padded_shapes = {**padded_shapes, **padded_label_shapes}

    # Merge default and custom padding values
    pad_values = DEFAULT_PADDING_VALUES.copy()
    if padding_values:
        pad_values.update(padding_values)

    padded_dict = {}

    for key, value in features_dict.items():
        if not isinstance(value, torch.Tensor) or value.numel() == 1:
            # Keep non-tensor values as-is
            padded_dict[key] = value
            continue

        # Check if this feature is in our shape dictionary
        assert (
            key in all_padded_shapes
        ), f"Unknown feature '{key}' not in shape dictionary"

        # Get expected current and target shapes
        expected_current_shape = all_current_shapes[key]
        expected_target_shape = all_padded_shapes[key]

        # Verify current shape matches expected
        if value.shape != expected_current_shape:
            raise ValueError(
                f"Feature '{key}' has shape {value.shape} but expected {expected_current_shape} "
                f"based on n_atom={current_n_atom}, n_token={current_n_token}"
            )

        # Pad each dimension that needs it
        padded_tensor = value
        for dim_idx, (current_size, target_size) in enumerate(
            zip(expected_current_shape, expected_target_shape)
        ):
            if current_size < target_size:
                pad_amount = target_size - current_size
                pad_value = pad_values.get(key, 0)
                padded_tensor = pad_at_dim(
                    padded_tensor,
                    dim=dim_idx,
                    pad_length=[0, pad_amount],
                    value=pad_value,
                )

            elif current_size > target_size:
                raise ValueError(
                    f"Cannot pad feature '{key}' dimension {dim_idx} from {current_size} to smaller {target_size}"
                )

        padded_dict[key] = padded_tensor

    return padded_dict


def pad_tensor(
    tensor: torch.Tensor,
    pad_sizes: tuple[int, ...],
    num_batch_dims: int = 0,
) -> torch.Tensor:
    """Pad tensor along non-batch dimensions, with batch dimensions handled.

    Padding is applied to the dimensions immediately following the batch dimensions.
    Each entry in `pad_sizes` corresponds to a non-batch dimension (starting at dim=num_batch_dims).

    Args:
        tensor (torch.Tensor): Input tensor to pad. Shape: [*batch, D0, D1, ...].
        pad_sizes (tuple[int, ...]): Amounts to pad each non-batch dimension (in order, dim=num_batch_dims, num_batch_dims+1, ...).
            E.g., pad_sizes=(5,) with num_batch_dims=0 pads dim 0 by 5; pad_sizes=(5,2) with num_batch_dims=1 pads dims 1 and 2.
        num_batch_dims (int, optional): Number of leading batch dimensions (not padded). Defaults to 0.

    Returns:
        torch.Tensor: Tensor padded on specified non-batch dims, shape [*batch, padded_D...].

    Raises:
        AssertionError: if batch dimensions are not present or pad_sizes do not match tensor dims after batch.
    """
    ndim = tensor.ndim
    non_batch_dims = ndim - num_batch_dims
    assert non_batch_dims >= len(pad_sizes), (
        f"pad_sizes {pad_sizes} exceed available ({non_batch_dims}) non-batch dims "
        f"for tensor shape {tensor.shape} and num_batch_dims={num_batch_dims}"
    )

    # torch.nn.functional.pad expects: (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
    # Only pad the right side of each non-batch dim, starting from the dim after the batch dims
    padding = [0] * (2 * ndim)

    for i, pad_size in enumerate(pad_sizes):
        # For the i-th non-batch dim, its index in tensor is num_batch_dims + i
        dim = num_batch_dims + i
        # Position in padding tuple: -2 * (ndim - dim)
        pos = 2 * (ndim - 1 - dim)
        padding[pos + 1] = pad_size  # pad to the right only

    return F.pad(tensor, padding)


def pad_data(
    data: Dict[str, Any],
    max_n_atom: int,
    max_n_token: int,
    max_n_msa: int,
    max_n_template: int,
    max_n_pocket: int,
) -> Dict[str, Any]:
    """
    Pad a single data dictionary from the Dataset to the maximum dimensions. Each data dictionary
    has keys (input_feature_dict, label_dict, label_full_dict, and basic) and optional keys
    (cropped_atom_array, cropped_token_array).

    Args:
        data: Unpadded data dictionary
        max_n_atom: Target dimensions for padding
        max_n_token: Target dimensions for padding
        max_n_msa: Target dimensions for padding
        max_n_template: Target dimensions for padding
        max_n_pocket: Target dimensions for padding

    Returns:
        Padded data dictionary
    """
    # Get current dimensions from the sample
    feat = data["input_feature_dict"]
    pad_args = {
        "current_n_atom": feat["atom_to_token_idx"].shape[0],
        "max_n_atom": max_n_atom,
        "current_n_token": feat["token_index"].shape[0],
        "max_n_token": max_n_token,
        "current_n_msa": feat["msa"].shape[0],
        "max_n_msa": max_n_msa,
        "current_n_template": feat.get("template_restype", np.array([1])).shape[0],
        "max_n_template": max_n_template,
        "current_n_pocket": 1,
        "max_n_pocket": max_n_pocket,
    }

    # Pad input feature dict and label dict
    return {
        **data,
        "input_feature_dict": pad_features_dict(data["input_feature_dict"], **pad_args),
        "label_dict": pad_features_dict(data["label_dict"], **pad_args),
    }
