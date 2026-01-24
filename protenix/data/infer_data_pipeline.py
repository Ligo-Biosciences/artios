# Copyright 2026 Ligo Biosciences.
# All rights reserved.
# Proprietary and confidential.

import json
import logging
import traceback
import warnings
from typing import Any

from torch.utils.data import DataLoader, Dataset, DistributedSampler

from protenix.data.array_builder import process_input_to_atom_array
from protenix.data.data_collator import CollateContexts
from protenix.data.features import FullContext
from protenix.data.featurizer import ContextGenerator
from protenix.data.json_parser import parse_input_dict
from protenix.data.tokenizer import AtomArrayTokenizer
from protenix.utils.distributed import DIST_WRAPPER

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="biotite")


def get_inference_dataloader(
    configs: Any,
    use_distributed_sampler: bool = True,
) -> DataLoader:
    """
    Creates and returns a DataLoader for inference using the InferenceDataset.

    Args:
        configs: A configuration object containing the necessary parameters for the DataLoader.
        use_distributed_sampler: Whether to use DistributedSampler to split data across ranks.
            Set to False for training-time inference where only one rank runs inference
            but should process all samples.

    Returns:
        A DataLoader object configured for inference.
    """
    inference_dataset = InferenceDataset(
        input_json_path=configs.input_json_path,
    )

    if use_distributed_sampler:
        sampler = DistributedSampler(
            dataset=inference_dataset,
            num_replicas=DIST_WRAPPER.world_size,
            rank=DIST_WRAPPER.rank,
            shuffle=False,
        )
    else:
        sampler = None

    dataloader = DataLoader(
        # TODO: batch size and crop size, max motif residues etc. need to come from configs or determined
        # on the fly, not hardcoded.
        dataset=inference_dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=False,
        collate_fn=CollateContexts(
            atom_bin_ends=configs.data.padding.atom_bin_ends,
            token_bin_ends=configs.data.padding.token_bin_ends,
            crop_size=384,
            motif_training=False,
            max_motif_residues=10,
        ),
        num_workers=configs.num_workers,
    )
    return dataloader


class InferenceDataset(Dataset):
    def __init__(
        self,
        input_json_path: str,
    ) -> None:
        self.input_json_path = input_json_path
        with open(self.input_json_path, "r") as f:
            self.inputs = json.load(f)

    def __len__(
        self,
    ) -> int:  # TODO: length will be the m dimension of the AtomArrayStack or 1 in the case of motif scaffolding.
        return len(self.inputs)

    def __getitem__(self, index: int) -> FullContext:
        try:
            input_dict = self.inputs[index]

            sample_input = parse_input_dict(input_dict)

            sample_name = sample_input.name
            logger.info(f"Featurizing {sample_name}...")

            atom_array, original_motif_mapping = process_input_to_atom_array(
                sample_input
            )
            token_array = AtomArrayTokenizer(atom_array).get_token_array()
            context = ContextGenerator(
                token_array,
                atom_array,
                pdb_id=sample_input.name,
                bioassembly_dict_fpath=sample_input.name,
                original_motif_mapping=original_motif_mapping,
            ).get_full_context()

        except Exception as e:
            error_message = f"{e}:\n{traceback.format_exc()}"
            raise Exception(error_message)

        return context


class MotifDataset(Dataset):
    def __init__(
        self,
        input_json_path: str,
    ) -> None:
        self.input_json_path = input_json_path
        with open(self.input_json_path, "r") as f:
            self.inputs = json.load(f)

    def __len__(
        self,
    ) -> int:  # TODO: length will be the m dimension of the AtomArrayStack or 1 in the case of motif scaffolding.
        return len(self.inputs)

    def __getitem__(self, index: int) -> FullContext:
        try:
            input_dict = self.inputs[index]

            sample_input = parse_input_dict(input_dict)  # TODO: this might end up

            sample_name = sample_input.name
            logger.info(f"Featurizing {sample_name}...")

            atom_array, original_motif_mapping = process_input_to_atom_array(
                sample_input
            )  # TODO: rename this to be more specific to motifs
            token_array = AtomArrayTokenizer(atom_array).get_token_array()
            context = ContextGenerator(
                token_array,
                atom_array,
                pdb_id=sample_input.name,
                bioassembly_dict_fpath=sample_input.name,
                original_motif_mapping=original_motif_mapping,
            ).get_full_context()

        except Exception as e:
            error_message = f"{e}:\n{traceback.format_exc()}"
            raise Exception(error_message)

        return context


class MPNNDataset(Dataset):
    def __init__(
        self,
        input_json_path: str,
    ) -> None:
        self.input_json_path = input_json_path
        with open(self.input_json_path, "r") as f:
            self.inputs = json.load(f)

        # Read the input json
        # Read the NPZ file to create the atom array stack
        # checks for input properties

    def __len__(self) -> int:
        # TODO: length will be the m dimension of the AtomArrayStack
        return len(self.inputs)

    def __getitem__(self, index: int) -> FullContext:
        try:
            # TODO: for each index, create a separate atom array from the atom array stack
            input_dict = self.inputs[index]

            # TODO: make the sequence design input object here
            sample_input = parse_input_dict(input_dict)

            sample_name = sample_input.name
            logger.info(f"Featurizing {sample_name}...")

            atom_array, original_motif_mapping = process_input_to_atom_array(
                sample_input
            )  # TODO: we will not need this at all
            token_array = AtomArrayTokenizer(atom_array).get_token_array()
            context = ContextGenerator(
                token_array,
                atom_array,
                pdb_id=sample_input.name,
                bioassembly_dict_fpath=sample_input.name,
                original_motif_mapping=original_motif_mapping,
            ).get_full_context()

        except Exception as e:
            error_message = f"{e}:\n{traceback.format_exc()}"
            raise Exception(error_message)

        return context


class PackingDataset(Dataset):
    def __init__(
        self,
        input_json_path: str,
    ) -> None:
        self.input_json_path = input_json_path
        with open(self.input_json_path, "r") as f:
            self.inputs = json.load(f)

        # Read the input json
        # Read the NPZ file to create the atom array stack
        # checks for input properties

    def __len__(
        self,
    ) -> int:  # TODO: length will be the m dimension of the AtomArrayStack
        return len(self.inputs)

    def __getitem__(self, index: int) -> FullContext:
        try:
            input_dict = self.inputs[index]

            sample_input = parse_input_dict(input_dict)

            sample_name = sample_input.name
            logger.info(f"Featurizing {sample_name}...")

            atom_array, original_motif_mapping = process_input_to_atom_array(
                sample_input
            )  # TODO: packing will need a bit of atom array processing,
            # we need a new atom array that includes all the sidechain atoms
            # and prepares the array for centre atom extraction properly.
            token_array = AtomArrayTokenizer(atom_array).get_token_array()
            context = ContextGenerator(
                token_array,
                atom_array,
                pdb_id=sample_input.name,
                bioassembly_dict_fpath=sample_input.name,
                original_motif_mapping=original_motif_mapping,
            ).get_full_context()

        except Exception as e:
            error_message = f"{e}:\n{traceback.format_exc()}"
            raise Exception(error_message)

        return context
