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

import torch
from biotite.structure import AtomArray
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from protenix.data.data_collator import CollateContexts
from protenix.data.json_parser import parse_input_dict
from protenix.data.json_to_context import InputToContext
from protenix.utils.distributed import DIST_WRAPPER

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="biotite")


def get_inference_dataloader(configs: Any) -> DataLoader:
    """
    Creates and returns a DataLoader for inference using the InferenceDataset.

    Args:
        configs: A configuration object containing the necessary parameters for the DataLoader.

    Returns:
        A DataLoader object configured for inference.
    """
    inference_dataset = InferenceDataset(
        input_json_path=configs.input_json_path,
        dump_dir=configs.dump_dir,
        use_msa=configs.use_msa,
    )
    sampler = DistributedSampler(
        dataset=inference_dataset,
        num_replicas=DIST_WRAPPER.world_size,
        rank=DIST_WRAPPER.rank,
        shuffle=False,
    )
    dataloader = DataLoader(
        dataset=inference_dataset,
        batch_size=1,
        sampler=sampler,
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
        dump_dir: str,
        use_msa: bool = True,
    ) -> None:
        self.input_json_path = input_json_path
        self.dump_dir = dump_dir
        self.use_msa = use_msa
        with open(self.input_json_path, "r") as f:
            self.inputs = json.load(f)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], AtomArray, str]:
        try:
            input_dict = self.inputs[index]

            sample_input = parse_input_dict(input_dict)

            sample_name = sample_input.name
            logger.info(f"Featurizing {sample_name}...")

            context = InputToContext(
                input=sample_input,
            ).get_context()

            error_message = ""
        except Exception as e:
            context = None
            error_message = f"{e}:\n{traceback.format_exc()}"

        return context, error_message
