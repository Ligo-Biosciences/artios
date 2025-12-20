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
import os
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from os.path import join as opjoin
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from protenix.config import parse_configs, parse_sys_args
from protenix.data.features import FullContext
from protenix.data.infer_data_pipeline import get_inference_dataloader
from protenix.model.artios_backbone import ArtiosBackbone
from protenix.model.artios_mpnn import ArtiosMPNN
from protenix.model.artios_packing import ArtiosPacking
from protenix.model.output import ModelOutput
from protenix.model.utils import Mode
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.inference import ModelType
from protenix.utils.seed import seed_everything
from protenix.utils.structure_io import (
    save_model_output_to_cif,
    save_model_output_to_fasta,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a specific model type."""

    model_class: type[nn.Module]
    checkpoint_path_key: str
    n_sample_key: str = ""


MODEL_REGISTRY: dict[ModelType, ModelConfig] = {
    ModelType.BACKBONE: ModelConfig(
        model_class=ArtiosBackbone,
        checkpoint_path_key="load_backbone_checkpoint_path",
        n_sample_key="sample_diffusion.N_sample",
    ),
    ModelType.SEQUENCE_DESIGN: ModelConfig(
        model_class=ArtiosMPNN,
        checkpoint_path_key="load_mpnn_checkpoint_path",
    ),
    ModelType.PACKING: ModelConfig(
        model_class=ArtiosPacking,
        checkpoint_path_key="load_packing_checkpoint_path",
        n_sample_key="artios_packing.sample_diffusion.N_sample",
    ),
}


class InferenceRunner(object):
    def __init__(self, configs: Any, model_type: ModelType) -> None:
        self.configs = configs
        self.model_type = model_type
        self.model_config = MODEL_REGISTRY[model_type]

        self.init_basics()
        self.init_env()
        self.init_model()
        self.load_checkpoint()

    def init_env(self) -> None:
        self.print(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            + f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device("cuda:{}".format(DIST_WRAPPER.local_rank))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        if DIST_WRAPPER.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # Set high precision for tensor core matmul using new API (PyTorch 2.9+)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logging.info("Finished init ENV.")

    def init_basics(self) -> None:
        """Initialize dump directories and error directory.

        Sets up a list of dump directories for each sample, and ensures the error directory exists.

        Creates N_sample dump subdirectories inside dump_dir: dump_dir/0, dump_dir/1, ... etc if not
        given by a dump_dir_list.

        Raises:
            ValueError: If configs.N_sample is not a positive integer
        """
        self.dump_dir = self.configs.dump_dir
        os.makedirs(self.dump_dir, exist_ok=True)

        n_sample = self.configs.get(self.model_config.n_sample_key, None)

        if self.configs.dump_dir_list:
            self.dump_dir_list = self.configs.dump_dir_list
        else:
            self.dump_dir_list = [
                os.path.join(self.dump_dir, str(i))
                for i in range(n_sample if n_sample is not None else 1)
            ]
        for dump_path in self.dump_dir_list:
            os.makedirs(dump_path, exist_ok=True)

        self.error_dir = os.path.join(self.dump_dir, "ERR")
        os.makedirs(self.error_dir, exist_ok=True)

    def init_model(self) -> None:
        model_class = self.model_config.model_class
        self.model = model_class(self.configs).to(self.device)

    def load_checkpoint(self) -> None:
        def _load_checkpoint(
            checkpoint_path: str,
            model: nn.Module,
        ):
            if not os.path.exists(checkpoint_path):
                raise Exception(f"Given checkpoint path not exist [{checkpoint_path}]")
            self.print(
                f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
            )
            checkpoint = torch.load(checkpoint_path, self.device)

            sample_key = [k for k in checkpoint["model"].keys()][0]
            self.print(f"Sampled key: {sample_key}")
            if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
                checkpoint["model"] = {
                    k[len("module.") :]: v for k, v in checkpoint["model"].items()
                }

            model.load_state_dict(
                state_dict=checkpoint["model"],
                strict=self.configs.load_strict,
            )
            model.eval()

            return model

        checkpoint_path_key = self.model_config.checkpoint_path_key
        checkpoint_path = getattr(self.configs, checkpoint_path_key)
        self.model = _load_checkpoint(checkpoint_path, self.model)
        self.print(f"Loaded {self.model_type.value} model")
        self.print("Finished loading checkpoint.")

    @torch.no_grad()
    def predict(self, context: FullContext) -> ModelOutput:
        enable_amp = (
            torch.autocast(device_type="cuda", dtype=torch.float32)
            if torch.cuda.is_available()
            else nullcontext()
        )
        context = context.to(self.device)

        with enable_amp:
            output = self.model(
                context=context,
                mode=Mode.INFERENCE,
            )

        return output

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            logger.info(msg)


def infer_predict(runner: InferenceRunner, configs: Any, model_type: ModelType) -> None:
    logger.info(f"Loading data from\n{configs.input_json_path}")
    try:
        dataloader = get_inference_dataloader(configs=configs)
    except Exception as e:
        error_message = f"{e}:\n{traceback.format_exc()}"
        logger.info(error_message)
        with open(opjoin(runner.error_dir, "error.txt"), "a") as f:
            f.write(error_message)
        return

    for seed in configs.seeds:
        seed_everything(seed=seed, deterministic=configs.deterministic)
        for context in dataloader:
            try:
                sample_name = context.basic.pdb_id[0]
                n_atom = context.basic.n_atom[0]
                n_token = context.basic.n_token[0]

                output = runner.predict(context)

                match model_type:
                    case ModelType.BACKBONE:
                        _ = save_model_output_to_cif(
                            output_dir_list=runner.dump_dir_list,
                            output=output,
                            model_type=model_type,
                            atom_array=context.basic.atom_array,
                            n_atom=n_atom,
                            pdb_id=sample_name,
                            field_in_b_factor_col="is_motif",
                        )
                        parent_dir = os.path.dirname(runner.dump_dir)
                        mapping_path = opjoin(
                            parent_dir, f"{sample_name}_motif_mapping.json"
                        )
                        if not os.path.exists(mapping_path):
                            with open(mapping_path, "w") as f:
                                json.dump(context.basic.original_motif_mapping, f)

                    case ModelType.PACKING:
                        _ = save_model_output_to_cif(
                            output_dir_list=runner.dump_dir_list,
                            output=output,
                            model_type=model_type,
                            atom_array=context.basic.atom_array,
                            n_atom=n_atom,
                            pdb_id=sample_name,
                            field_in_b_factor_col="is_motif",
                        )

                    case ModelType.SEQUENCE_DESIGN:
                        _ = save_model_output_to_fasta(
                            output_dir_list=runner.dump_dir_list,
                            output=output,
                            n_token=n_token,
                            pdb_id=sample_name,
                        )

                logger.info(
                    f"[Rank {DIST_WRAPPER.rank}] {sample_name} succeeded.\n"
                    f"Results saved to {configs.dump_dir}"
                )
                torch.cuda.empty_cache()
            except Exception as e:
                error_message = f"[Rank {DIST_WRAPPER.rank}]{sample_name} {e}:\n{traceback.format_exc()}"
                logger.info(error_message)
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()


def main(configs: Any, model_type: ModelType = ModelType.BACKBONE) -> None:
    runner = InferenceRunner(configs, model_type)
    infer_predict(runner, configs, model_type)


def run() -> None:
    LOG_FORMAT = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        arg_str=parse_sys_args(),
        fill_required_with_null=True,
    )
    main(configs, model_type=ModelType(configs.model_type))


if __name__ == "__main__":
    run()
