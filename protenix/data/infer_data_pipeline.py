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

    def __len__(self) -> int:
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
