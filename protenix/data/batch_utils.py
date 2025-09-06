"""Dedicated file for everything involving batching. This will include:
    - Collating the data coming out of the dataset

The data that comes out of the dataset is a dictionary with the following keys:
    - input_feature_dict
    - label_dict
    - label_full_dict
    - basic
    - (optional) cropped_atom_array
    - (optional) cropped_token_array

We need to collate this data into a batch. For now this includes:
    - Iterating over the different dictionaries (input_feature_dict, label_dict, label_full_dict and basic)
    - Iterating over the different keys in the dictionaries
    - Stacking the tensors along the first dimension (using torch.stack on dim=0)
    - Spotting exceptions to these rules and accomodating for this

Collating data into a batch is going to involve creating a DataCollator() class. this will have a
__call__ method that will be called on a batch ( which will come in as a list of dictionaries) and return
a tuple of dictionaries with the following keys:
    - input_feature_dict
    - label_dict
    - label_full_dict
    - basic
which now contain features with a batch dimension.

This will be used in the dataloader.py file.
"""

from typing import Any, Dict, List, Union

import numpy as np
import torch

from protenix.data.padding_utils import pad_features_dict
from protenix.data.utils import get_atom_padding_length
from protenix.utils.logger import get_logger

logger = get_logger(__name__)


class BinnedDataCollator:
    """
    Collates batches of data from the dataset into batched tensors.

    This collator handles padding of features to consistent shapes within
    each batch before stacking tensors along the batch dimension.
    Special handling is provided for non-tensor features that should
    remain as lists.
    """

    def __init__(
        self,
        crop_size: int = 384,
        bin_ends: List[int] = None,
        max_n_msa: int = 512,
        max_n_template: int = 4,
        max_n_pocket: int = 30,
        return_atom_token_arrays: bool = False,
    ):
        """
        Initialize the BinnedDataCollator.

        Args:
            crop_size: Maximum number of tokens (used to calculate max atoms)
            bin_ends: List of bin endpoints for atom padding
            max_n_msa: Maximum number of MSA sequences
            max_n_template: Maximum number of templates
            max_n_pocket: Maximum number of pockets
            return_atom_token_arrays: Whether to include cropped_atom_array and
                cropped_token_array in the output batch.
        """
        self.crop_size = crop_size
        self.bin_ends = bin_ends if bin_ends is not None else []
        self.max_n_msa = max_n_msa
        self.max_n_template = max_n_template
        self.max_n_pocket = max_n_pocket
        self.return_atom_token_arrays = return_atom_token_arrays

        # Features that should be kept as lists rather than stacked
        # These are typically strings or identifiers
        self.list_features = {
            "pdb_id",
            "chain_id",
            "bioassembly_dict_fpath",
        }

        # Features that may have variable first dimension (number of eval items)
        # but are already padded consistently within each sample
        self.eval_features = {"eval_type", "cluster_id", "chain_1_mask", "chain_2_mask"}

    def __call__(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[str, Union[Dict[str, torch.Tensor], List[Any]]]:
        """
        Collate a batch of samples from the dataset.

        This method first determines the maximum dimensions needed for the batch,
        pads all samples to these dimensions, then stacks them along the batch dimension.

        Args:
            batch: A list of dictionaries, where each dictionary is the output
                from a single sample from the dataset __getitem__ method.

        Returns:
            A dictionary containing the batched data with keys:
                - input_feature_dict: Batched input features
                - label_dict: Batched labels
                - label_full_dict: Batched full labels
                - basic: Batched basic information
                - (optional) cropped_atom_arrays: List of AtomArrays
                - (optional) cropped_token_arrays: List of TokenArrays
        """
        batch_size = len(batch)

        # Determine maximum dimensions across the batch
        # TODO: double check logic here.
        max_atom_in_batch = max(
            sample["input_feature_dict"]["atom_to_token_idx"].shape[0]
            for sample in batch
        )
        max_n_atom = get_atom_padding_length(
            max_atom_in_batch, self.bin_ends, self.crop_size
        )

        # Pad all samples to the maximum dimensions
        padded_batch = []
        for sample in batch:
            padded_sample = self._pad_sample(
                sample,
                max_n_atom,
                self.crop_size,
                self.max_n_msa,
                self.max_n_template,
                self.max_n_pocket,
            )
            padded_batch.append(padded_sample)

        # Initialise output dictionaries
        collated = {
            "input_feature_dict": {},
            "label_dict": {},
            "label_full_dict": {},
            "basic": {},
        }

        # Handle optional atom and token arrays if requested
        if self.return_atom_token_arrays:
            if "cropped_atom_array" in padded_batch[0]:
                collated["cropped_atom_arrays"] = [
                    sample["cropped_atom_array"] for sample in padded_batch
                ]
            if "cropped_token_array" in padded_batch[0]:
                collated["cropped_token_arrays"] = [
                    sample["cropped_token_array"] for sample in padded_batch
                ]

        # Process each dictionary type and stack padded tensors
        for dict_key in [
            "input_feature_dict",
            "label_dict",
            "label_full_dict",
            "basic",
        ]:
            if dict_key not in padded_batch[0]:
                continue

            collated_dict = collated[dict_key]

            # Get all keys from the first sample
            sample_dict = padded_batch[0][dict_key]

            for feature_key in sample_dict.keys():
                feature_values = [
                    sample[dict_key][feature_key] for sample in padded_batch
                ]

                # Check if this feature should be kept as a list
                if feature_key in self.list_features:
                    collated_dict[feature_key] = feature_values

                # Handle evaluation features that may not be present in all samples
                elif feature_key in self.eval_features:
                    # These features are for evaluation and may not be present in training
                    # Keep as list if shapes don't match, otherwise stack
                    if all(isinstance(v, torch.Tensor) for v in feature_values):
                        try:
                            collated_dict[feature_key] = torch.stack(
                                feature_values, dim=0
                            )
                        except RuntimeError:
                            # Different shapes (e.g., different number of eval items)
                            collated_dict[feature_key] = feature_values
                    else:
                        collated_dict[feature_key] = feature_values

                elif isinstance(feature_values[0], torch.Tensor):
                    # Stack tensors along batch dimension
                    # All tensors should now have the same shape due to padding
                    try:
                        collated_dict[feature_key] = torch.stack(feature_values, dim=0)
                    except RuntimeError as e:
                        logger.warning(
                            f"Could not stack {feature_key} after padding: {e}. "
                            f"Shapes: {[v.shape for v in feature_values]}. "
                            f"Keeping as list."
                        )
                        collated_dict[feature_key] = feature_values

                else:
                    # Keep other types (strings, objects) as lists
                    collated_dict[feature_key] = feature_values

        # Add batch size to basic info if not already present
        if "batch_size" not in collated["basic"]:
            collated["basic"]["batch_size"] = torch.tensor([batch_size])

        return collated

    def _pad_sample(
        self,
        sample: Dict[str, Any],
        max_n_atom: int,
        max_n_token: int,
        max_n_msa: int,
        max_n_template: int,
        max_n_pocket: int,
    ) -> Dict[str, Any]:
        """
        Pad a single sample to the maximum dimensions.

        Args:
            sample: Unpadded sample dictionary
            max_n_atom: Target dimensions for padding
            max_n_token: Target dimensions for padding
            max_n_msa: Target dimensions for padding
            max_n_template: Target dimensions for padding
            max_n_pocket: Target dimensions for padding

        Returns:
            Padded sample dictionary
        """
        padded_sample = {}

        # Get current dimensions from the sample
        feat = sample["input_feature_dict"]
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

        padded_sample["input_feature_dict"] = pad_features_dict(
            sample["input_feature_dict"],
            **pad_args,
        )

        padded_sample["label_dict"] = pad_features_dict(
            sample["label_dict"],
            **pad_args,
        )

        padded_sample["label_full_dict"] = sample["label_full_dict"]

        padded_sample["basic"] = sample["basic"]

        # Copy optional arrays if present
        if "cropped_atom_array" in sample:
            padded_sample["cropped_atom_array"] = sample["cropped_atom_array"]
        if "cropped_token_array" in sample:
            padded_sample["cropped_token_array"] = sample["cropped_token_array"]

        return padded_sample
