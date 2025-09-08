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

import torch

from protenix.data.padding_utils import get_atom_padding_length, pad_data


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
        max_n_msa: int = 512,
        max_n_template: int = 4,
        max_n_pocket: int = 30,
        bin_ends: List[int] = None,
        crop_size: int = 384,
        return_atom_token_arrays: bool = False,
    ):
        """
        Initialise the BinnedDataCollator.

        Args:
            max_n_msa: Maximum number of MSA sequences
            max_n_template: Maximum number of templates
            max_n_pocket: Maximum number of pockets
            bin_ends: List of bin endpoints for atom padding
            crop_size: Maximum number of tokens (used to calculate max atoms)
            return_atom_token_arrays: Whether to include cropped_atom_array and
                cropped_token_array in the output batch.
        """
        self.max_n_msa = max_n_msa
        self.max_n_template = max_n_template
        self.max_n_pocket = max_n_pocket
        self.bin_ends = bin_ends
        self.crop_size = crop_size
        self.return_atom_token_arrays = return_atom_token_arrays

        # Features that should be kept as lists rather than stacked
        # These are typically strings or identifiers
        self.list_features = {
            "pdb_id",
            "chain_id",
            "bioassembly_dict_fpath",
        }

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
        # batch_size = len(batch)
        # TODO: add a _validate_batch() method

        # Get max atom padding length for the batch
        atom_padding_length = self._get_atom_padding_length(batch)

        # Pad all samples to the maximum dimensions
        padded_batch = self._pad_batch(batch, atom_padding_length)

        # Initialise output dictionaries
        collated = {
            "input_feature_dict": {},
            "label_dict": {},
            # "label_full_dict": {},
            "basic": {},
        }

        # Process each dictionary type and stack padded tensors
        for dict_key in ["input_feature_dict", "label_dict", "basic"]:
            collated[dict_key] = self._collate_dict(dict_key, padded_batch)

        # Handle optional atom and token arrays if requested
        if self.return_atom_token_arrays:
            self._add_optional_arrays(collated, padded_batch)

        return collated

    def _get_atom_padding_length(self, batch: List[Dict[str, Any]]) -> int:
        """
        Get the maximum atom padding length for a batch.
        """
        max_atom_in_batch = max(
            data["input_feature_dict"]["atom_to_token_idx"].shape[0] for data in batch
        )
        return get_atom_padding_length(max_atom_in_batch, self.bin_ends, self.crop_size)

    def _pad_batch(
        self,
        batch: List[Dict[str, Any]],
        atom_padding_length: int,
    ) -> List[Dict[str, Any]]:
        """
        Pad a batch of samples to the maximum dimensions.
        """
        padded_batch = []
        for data in batch:
            padded_data = pad_data(
                data,
                max_n_atom=atom_padding_length,
                max_n_token=self.crop_size,
                max_n_msa=self.max_n_msa,
                max_n_template=self.max_n_template,
                max_n_pocket=self.max_n_pocket,
            )
            padded_batch.append(padded_data)
        return padded_batch

    def _add_optional_arrays(
        self, collated: Dict[str, Any], padded_batch: List[Dict]
    ) -> None:
        """Add optional atom/token arrays if requested."""
        if "cropped_atom_array" in padded_batch[0]:
            collated["cropped_atom_arrays"] = [
                sample["cropped_atom_array"] for sample in padded_batch
            ]
        if "cropped_token_array" in padded_batch[0]:
            collated["cropped_token_arrays"] = [
                sample["cropped_token_array"] for sample in padded_batch
            ]

    def _collate_dict(
        self, dict_key: str, padded_batch: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Collate a single dictionary type across the batch.

        Args:
            dict_key: The dictionary key to collate ("input_feature_dict", etc.)
            padded_batch: The list of datapoints, each item in the list is a dictionary with
                keys (input_feature_dict, label_dict, label_full_dict, and basic) and optional keys
                (cropped_atom_array, cropped_token_array).

        Returns:
            Collated dictionary for this key
        """
        collated_dict = {}
        sample_dict = padded_batch[0][dict_key]

        for feature_key in sample_dict.keys():
            # List of values for this feature across the batch
            feature_values = [sample[dict_key][feature_key] for sample in padded_batch]

            # Check if this feature should be kept as a list
            if feature_key in self.list_features:
                collated_dict[feature_key] = feature_values

            elif isinstance(feature_values[0], torch.Tensor):
                # Stack tensors along batch dimension
                collated_dict[feature_key] = torch.stack(feature_values, dim=0)

            else:
                # Raise error for unexpected feature types
                raise ValueError(f"Unexpected feature type: {type(feature_values[0])}")

        return collated_dict
