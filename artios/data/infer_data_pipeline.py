# Copyright 2024 ByteDance and/or its affiliates.
# Copyright 2026 Ligo Biosciences.
# Modified by Ligo Biosciences from the original Protenix implementation.
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

import json
import logging
import warnings
from typing import Any

import biotite.sequence.io as seqio
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from typing_extensions import assert_never

from artios.data.array_builder import (
    backbone_input_to_structure,
    packing_input_to_structure,
    sequence_design_input_to_structure,
)
from artios.data.collate.collate import CollateContexts
from artios.data.features import FullContext
from artios.data.features.featurizer import ContextGenerator
from artios.data.parsing.json import (
    BackboneInput,
    PackingInput,
    SequenceDesignInput,
    parse_input_dict,
)
from artios.utils.distributed import DIST_WRAPPER
from artios.utils.file_io import load_atomarray_stack

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

    try:
        motif_depth_norm = float(configs.motif_depth.inference_value)
    except (AttributeError, KeyError, TypeError):
        motif_depth_norm = -1.0

    try:
        rasa_value = float(configs.rasa.inference_value)
    except (AttributeError, KeyError, TypeError):
        rasa_value = -1.0

    match sample_input:
        case BackboneInput():
            dataset = MotifDataset(
                sample_input,
                motif_depth_norm=motif_depth_norm,
                rasa_value=rasa_value,
            )
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

    def __init__(
        self,
        backbone_input: BackboneInput,
        motif_depth_norm: float = -1.0,
        rasa_value: float = -1.0,
    ) -> None:
        self.backbone_input = backbone_input
        self.motif_depth_norm = motif_depth_norm
        self.rasa_value = rasa_value

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> FullContext:
        sample_name = self.backbone_input.name
        logger.info(f"Featurizing {sample_name}...")

        structure, original_motif_mapping = backbone_input_to_structure(
            self.backbone_input
        )
        return ContextGenerator(
            structure,
            pdb_id=sample_name,
            original_motif_mapping=original_motif_mapping,
            motif_depth_norm=self.motif_depth_norm,
            rasa_value=self.rasa_value,
        ).get_full_context()


class MPNNDataset(Dataset):
    """Dataset for MPNN sequence design inference.

    Each item corresponds to one sequence to generate. By default, one sequence
    per model in the AtomArrayStack (gen0). For redesign generations,
    ``backbone_index_map`` specifies which NPZ stack position each output
    sequence should be generated from. Multiple entries can point to the same
    stack position, and group sizes need not be uniform (supporting gen2+ where
    filter survival rates vary per backbone).

    The ordering contract: output position ``i`` corresponds to the UID whose
    ``batch_sample_idx == i`` in uid_df. The task builder in
    ``get_artios_potts_modal_tasks`` builds the map by reading
    ``backbone_sample_idx`` from uid_df sorted by ``batch_sample_idx``, so
    ``map[i]`` = the backbone stack position for the UID at FASTA position ``i``.
    """

    def __init__(self, sequence_input: SequenceDesignInput) -> None:
        self.sequence_input = sequence_input
        self.atom_stack = load_atomarray_stack(self.sequence_input.npz_file.path)
        assert len(self.atom_stack.coord.shape) == 3  # (m, n_atoms, 3)
        stack_len = self.atom_stack.coord.shape[0]
        self.backbone_index_map: list[int] = (
            sequence_input.backbone_index_map
            if sequence_input.backbone_index_map is not None
            else list(range(stack_len))
        )

    def __len__(self) -> int:
        return len(self.backbone_index_map)

    def __getitem__(self, index: int) -> FullContext:
        backbone_sample_idx = self.backbone_index_map[index]
        sample_name = f"{self.sequence_input.name}_model_{index}"
        logger.info(
            f"Featurizing {sample_name} (backbone_sample_idx {backbone_sample_idx})..."
        )

        structure, _ = sequence_design_input_to_structure(
            self.atom_stack[backbone_sample_idx]
        )

        # fixed_residues keyed by backbone stack position (backbone-dependent)
        fixed = self.sequence_input.fixed_residues.get(backbone_sample_idx)

        return ContextGenerator(
            structure,
            pdb_id=sample_name,
        ).get_full_context(fixed_residues=fixed)


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
        structure, _ = packing_input_to_structure(input_array, sequence)
        return ContextGenerator(
            structure,
            pdb_id=sample_name,
        ).get_full_context()
