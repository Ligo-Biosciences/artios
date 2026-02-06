# Copyright 2026 Ligo Biosciences.
# All rights reserved.
# Proprietary and confidential.

import json
import logging
import warnings
from typing import Any

import biotite.sequence.io as seqio
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from typing_extensions import assert_never

from protenix.data.array_builder import (
    backbone_input_to_atom_array,
    packing_input_to_atom_array_v2,
)
from protenix.data.data_collator import CollateContexts
from protenix.data.features import FullContext
from protenix.data.featurizer import ContextGenerator
from protenix.data.json_parser import (
    BackboneInput,
    PackingInput,
    SequenceDesignInput,
    parse_input_dict,
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
    Creates and returns a DataLoader for inference.

    Parses the input JSON (single task) and creates the appropriate dataset.

    Args:
        configs: Configuration object with input_json_path and data settings.
        use_distributed_sampler: Whether to use DistributedSampler to split data across ranks.

    Returns:
        A DataLoader configured for the specified inference task.
    """
    with open(configs.input_json_path) as f:
        input_dict = json.load(f)

    sample_input = parse_input_dict(input_dict)

    match sample_input:
        case BackboneInput():
            dataset = MotifDataset(sample_input)
            motif_training = True
        case SequenceDesignInput():
            dataset = MPNNDataset(sample_input)
            motif_training = False
        case PackingInput():
            dataset = PackingDataset(sample_input)
            motif_training = False
        case _:
            assert_never(sample_input)

    if use_distributed_sampler:
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=DIST_WRAPPER.world_size,
            rank=DIST_WRAPPER.rank,
            shuffle=False,
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=len(dataset),
        sampler=sampler,
        shuffle=False,
        collate_fn=CollateContexts(
            atom_bin_ends=configs.data.padding.atom_bin_ends,
            token_bin_ends=configs.data.padding.token_bin_ends,
            crop_size=384,
            motif_training=motif_training,
            max_motif_residues=10,
        ),
        num_workers=configs.num_workers,
    )
    return dataloader


class MotifDataset(Dataset):
    """Dataset for motif scaffolding inference.

    Produces a single FullContext from a BackboneInput specification.
    """

    def __init__(self, backbone_input: BackboneInput) -> None:
        self.backbone_input = backbone_input

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> FullContext:
        sample_name = self.backbone_input.name
        logger.info(f"Featurizing {sample_name}...")

        atom_array, original_motif_mapping = backbone_input_to_atom_array(
            self.backbone_input
        )
        # TODO: tidy me !!
        if np.any(atom_array.atom_name == "OXT"):
            logger.warning(
                "OXT atom present after running backbone_input_to_atom_array."
            )

        token_array = AtomArrayTokenizer(atom_array).get_token_array()
        return ContextGenerator(
            token_array,
            atom_array,
            pdb_id=sample_name,
            original_motif_mapping=original_motif_mapping,
        ).get_full_context()


class MPNNDataset(Dataset):  # TODO: need a different name
    """Dataset for MPNN sequence design inference.

    Each item corresponds to one model from the AtomArrayStack.
    """

    def __init__(self, sequence_input: SequenceDesignInput) -> None:
        self.sequence_input = sequence_input
        self.atom_stack = load_atomarray_stack(self.sequence_input.npz_file.path)
        assert len(self.atom_stack.coord.shape) == 3  # (m, n_atoms, 3)

    def __len__(self) -> int:
        # Use coord shape directly as stack_depth() can be unreliable
        return self.atom_stack.coord.shape[0]

    def __getitem__(self, index: int) -> FullContext:
        sample_name = f"{self.sequence_input.name}_model_{index}"
        logger.info(f"Featurizing {sample_name}...")

        atom_array = self.atom_stack[index]
        token_array = AtomArrayTokenizer(atom_array).get_token_array()
        return ContextGenerator(
            token_array,
            atom_array,
            pdb_id=sample_name,
        ).get_full_context()


class PackingDataset(Dataset):
    """Dataset for sidechain packing inference.

    Each item corresponds to one model from the AtomArrayStack paired
    with its corresponding designed sequence from the FASTA file.
    """

    def __init__(self, packing_input: PackingInput) -> None:
        self.packing_input = packing_input

        # Load the AtomArrayStack from NPZ
        self.atom_stack = load_atomarray_stack(self.packing_input.npz_file.path)
        assert len(self.atom_stack.coord.shape) == 3  # (m, n_atoms, 3)

        # Load sequences from FASTA file
        fasta_dict = seqio.load_sequences(self.packing_input.fasta_file.path)
        self.sequences = list(fasta_dict.values())

        # Validate that stack depth matches number of sequences
        stack_depth = self.atom_stack.coord.shape[0]
        if stack_depth != len(self.sequences):
            raise ValueError(
                f"AtomArrayStack has {stack_depth} models but FASTA has "
                f"{len(self.sequences)} sequences. They must match."
            )

    def __len__(self) -> int:
        return self.atom_stack.coord.shape[0]

    def __getitem__(self, index: int) -> FullContext:
        sample_name = f"{self.packing_input.name}_model_{index}"
        logger.info(f"Featurizing {sample_name}...")

        input_array = self.atom_stack[index]
        # # TODO: temporary fix while working on distillation backbones
        # input_array.res_id = input_array.res_id - input_array.res_id.min() + 1
        sequence = self.sequences[index]
        atom_array = packing_input_to_atom_array_v2(input_array, sequence)

        token_array = AtomArrayTokenizer(atom_array).get_token_array()
        return ContextGenerator(
            token_array,
            atom_array,
            pdb_id=sample_name,
        ).get_full_context()
