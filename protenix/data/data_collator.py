from typing import Any, Dict, List, Union

import torch

from protenix.data.padding_utils import get_atom_padding_length, pad_data


class DataCollatorWithPadding:
    """
    Collates batches of data from the dataset.

    This collator first pads all datapoints in the batch to a common shape that
    depends on the maximum number of atoms in the batch.
    Features are then collated across the batch.
        - For tensor features, this involves stacking along the batch dimension.
        - For list features, this involves storing in a list of lists.
        - For other types, an error is raised.
    """

    def __init__(
        self,
        max_n_msa: int = 512,
        max_n_template: int = 4,
        max_n_pocket: int = 30,
        bin_ends: List[int] = None,
        crop_size: int = 384,
        return_atom_token_arrays: bool = True,
        list_features: List[str] = [
            "pdb_id",
            "chain_id",
            "bioassembly_dict_fpath",
        ],
    ):
        """
        Initialise the DataCollatorWithPadding.

        Args:
            max_n_msa: Maximum number of MSA sequences
            max_n_template: Maximum number of templates
            max_n_pocket: Maximum number of pockets
            bin_ends: List of bin endpoints for atom padding
            crop_size: Maximum number of tokens (used to calculate max atoms)
            return_atom_token_arrays: Whether to include cropped_atom_array and
                cropped_token_array in the output batch.
            list_features: List of features which will be collated into a list.
        """
        self.max_n_msa = max_n_msa
        self.max_n_template = max_n_template
        self.max_n_pocket = max_n_pocket
        self.bin_ends = bin_ends
        self.crop_size = crop_size
        self.return_atom_token_arrays = return_atom_token_arrays

        self.list_features = list_features

    def __call__(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[str, Union[Dict[str, torch.Tensor], List[Any]]]:
        """
        Collate a batch of samples from the dataset.

        This method first determines the maximum dimensions needed for the batch,
        pads all samples to these dimensions, then stacks them along the batch dimension.

        Args:
            batch (list[dict]): A list of dictionaries, where each dictionary is the output
                from a single sample from the dataset __getitem__ method. Each sample will have keys
                (input_feature_dict, label_dict, label_full_dict, and basic) and optional keys
                (cropped_atom_array, cropped_token_array).

        Returns:
            A dictionary containing the batched data with keys:
                - input_feature_dict: Batched input features
                - label_dict: Batched labels
                - label_full_dict: Batched full labels
                - basic: Batched basic information
                - (optional) cropped_atom_arrays: List of AtomArrays
                - (optional) cropped_token_arrays: List of TokenArrays
        """
        # Get max atom padding length for the batch
        atom_padding_length = self._get_atom_padding_length(batch)

        # Pad all samples to the maximum dimensions
        padded_batch = self._pad_batch(batch, atom_padding_length)

        # Collate dicts
        collated = {}
        for dict_key in ["input_feature_dict", "label_dict", "basic"]:
            collated[dict_key] = self._collate_dicts(dict_key, padded_batch)

        # Add optional atom and token arrays if requested
        if self.return_atom_token_arrays:
            self._add_optional_arrays(collated, padded_batch)

        # NOTE: temp solution : Add an empty label full dict
        collated["label_full_dict"] = {}

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
                max_n_token=self.crop_size
                + 150,  # TODO: need to pad to max number of tokens. this should be crop size + max tokens you can get from the motif
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

    def _collate_dicts(
        self, dict_key: str, padded_batch: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Collate a single dictionary type across the batch.
        Features in the list_features are stored in a list.
            This is for features such as str's, like pdb_id.
        Features of type torch.Tensor are stacked along the batch dimension.
        Any other type will raise an error.

        Args:
            dict_key: The dictionary key to collate ("input_feature_dict", etc.)
            padded_batch: The list of datapoints, each item in the list is a dictionary with
                keys (input_feature_dict, label_dict, label_full_dict, and basic) and optional keys
                (cropped_atom_array, cropped_token_array).

        Returns:
            Collated dictionary for this dictionary key
        """
        collated_dict = {}
        sample_dict = padded_batch[0][dict_key]

        for feature_key in sample_dict.keys():
            feature_values = [data[dict_key][feature_key] for data in padded_batch]
            # Assert all values for feature have same type
            assert all(
                isinstance(value, type(feature_values[0])) for value in feature_values
            )

            # List features get stored in a list
            if feature_key in self.list_features:
                collated_dict[feature_key] = feature_values
            # torch tensors get stacked along batch dimension
            elif isinstance(feature_values[0], torch.Tensor):
                collated_dict[feature_key] = torch.stack(feature_values, dim=0)
            # Raise error for other types
            else:
                raise ValueError(
                    f"Unexpected feature type for key {feature_key}: {type(feature_values[0])}"
                )

        return collated_dict
