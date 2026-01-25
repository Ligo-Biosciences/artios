# Copyright 2026 Ligo Biosciences.
# All rights reserved.
# Proprietary and confidential.

import json
import logging
import traceback
import warnings
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Dataset, DistributedSampler

from protenix.data.array_builder import (
    backbone_input_to_atom_array,
    process_input_to_atom_array,
)
from protenix.data.data_collator import CollateContexts
from protenix.data.features import FullContext
from protenix.data.featurizer import ContextGenerator
from protenix.data.json_parser import (
    BackboneInput,
    SequenceDesignInput,
    parse_input_dict,
    parse_motif_json,
    parse_sequence_design_json,
)
from protenix.data.tokenizer import AtomArrayTokenizer
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.file_io import load_atomarray_stack

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
        input_json_path: str | Path,
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
    """Dataset for motif scaffolding inference.

    Loads a single motif JSON file and produces a FullContext for inference.
    The JSON should contain a single BackboneInput specification (not a list).
    """

    def __init__(
        self,
        input_json_path: str | Path,
    ) -> None:
        self.input_json_path = input_json_path
        self.backbone_input: BackboneInput = parse_motif_json(input_json_path)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> FullContext:
        try:
            sample_name = self.backbone_input.name
            logger.info(f"Featurizing {sample_name}...")

            atom_array, original_motif_mapping = backbone_input_to_atom_array(
                self.backbone_input
            )
            token_array = AtomArrayTokenizer(atom_array).get_token_array()
            context = ContextGenerator(
                token_array,
                atom_array,
                pdb_id=sample_name,
                original_motif_mapping=original_motif_mapping,
            ).get_full_context()

        except Exception as e:
            error_message = f"{e}:\n{traceback.format_exc()}"
            raise Exception(error_message)

        return context


class MPNNDataset(Dataset):
    """Dataset for MPNN sequence design inference.

    Loads a SequenceDesignInput JSON file pointing to an NPZ file containing an
    AtomArrayStack (multiple backbone models). Each item in the dataset corresponds
    to one model from the stack.
    """

    def __init__(
        self,
        input_json_path: str | Path,
    ) -> None:
        self.input_json_path = input_json_path
        self.sequence_input: SequenceDesignInput = parse_sequence_design_json(
            input_json_path
        )
        self.atom_stack = load_atomarray_stack(self.sequence_input.npz_file.path)

    def __len__(self) -> int:
        # Use coord shape directly as stack_depth() can be unreliable
        return self.atom_stack.coord.shape[0]

    def __getitem__(self, index: int) -> FullContext:
        try:
            sample_name = f"{self.sequence_input.name}_model_{index}"
            logger.info(f"Featurizing {sample_name}...")

            # Index into the stack to get a single AtomArray
            atom_array = self.atom_stack[index]

            token_array = AtomArrayTokenizer(atom_array).get_token_array()
            context = ContextGenerator(
                token_array,
                atom_array,
                pdb_id=sample_name,
            ).get_full_context()

        except Exception as e:
            error_message = f"{e}:\n{traceback.format_exc()}"
            raise Exception(error_message)

        return context


class PackingDataset(Dataset):
    def __init__(
        self,
        input_json_path: str | Path,
    ) -> None:
        self.input_json_path = input_json_path
        with open(self.input_json_path, "r") as f:
            self.inputs = json.load(f)

        # TODO: Read the input json
        # TODO: Read the NPZ file to create the atom array stack
        # TODO: Read the Fasta file to create some list of sequences
        # TODO: checks for input properties
        # - check that m of atom array stack match the number of sequences in the fasta file.

    def __len__(
        self,
    ) -> int:  # TODO: length will be the m dimension of the AtomArrayStack
        return len(self.inputs)

    def __getitem__(self, index: int) -> FullContext:
        try:
            # TODO: index into AtomArrayStack to get AtomArray
            input_dict = self.inputs[index]

            # TODO: index into the list of sequences

            # TODO: will no longer be needed
            sample_input = parse_input_dict(input_dict)

            sample_name = sample_input.name
            logger.info(f"Featurizing {sample_name}...")

            # TODO: change to a packing specific function that can takes in the
            # design atom array as well as the sequence and builds the appropiate
            # array.
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
