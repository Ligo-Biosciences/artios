"""
Simplified padding utilities using get_data_shape_dict for shape inference.
"""

from typing import Dict, Optional, Union

import torch

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
    assert max_n_atom > current_n_atom, "Need at least 1 padding token."
    assert max_n_token > current_n_token, "Need at least 1 padding token."

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
