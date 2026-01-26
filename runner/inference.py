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

# from contextlib import nullcontext
from dataclasses import dataclass
from os.path import join as opjoin
from pathlib import Path
from typing import Any, Optional

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
from protenix.model.output import BackboneOutput, ModelOutput
from protenix.model.utils import Mode
from protenix.utils.casting import disable_autocast
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.inference import ModelType
from protenix.utils.seed import seed_everything
from protenix.utils.structure_io import SaveContext, save_output

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
    def __init__(
        self, configs: Any, model_type: ModelType, model: Optional[nn.Module] = None
    ) -> None:
        self.configs = configs
        self.model_type = model_type
        self.model_config = MODEL_REGISTRY[model_type]
        self._model_provided = model is not None

        self.init_basics()
        self.init_env()
        self.init_model(model)
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

        # TODO: I'm not sure if we need n_sample different directories anymore
        # If the model outputs are going to be NPZ files, we can shove all of the
        # n_sample coordinates into the same file and save it as an atom array stack or so.

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

    def init_model(self, model: Optional[nn.Module] = None) -> None:
        model_class = self.model_config.model_class
        if model is None:
            self.model = model_class(self.configs).to(self.device)
        else:
            self.model = model.to(self.device)

    def load_checkpoint(self) -> None:
        if self._model_provided:
            self.model.eval()
            return

        def _load_checkpoint(
            checkpoint_path: str,
            model: nn.Module,
        ):
            if not os.path.exists(checkpoint_path):
                raise Exception(f"Given checkpoint path not exist [{checkpoint_path}]")
            self.print(
                f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
            )
            # TODO: this exact code is repeated and looks quite unsafe
            # We should make this into a method that's safer and call it both during training.
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
    @disable_autocast
    def predict(self, context: FullContext) -> ModelOutput:
        context = context.to(self.device)
        output = self.model(
            context=context,
            mode=Mode.INFERENCE,
        )
        return output

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            logger.info(msg)


def infer_predict(
    runner: InferenceRunner,
    configs: Any,
    use_distributed_sampler: bool = False,
) -> Path | None:
    """Run inference on all samples in the dataloader.

    The output type (BackboneOutput, PackingOutput, SequenceOutput) determines
    how results are saved - no explicit model_type dispatch needed.

    Args:
        runner: InferenceRunner with loaded model.
        configs: Configuration object.
        use_distributed_sampler: Whether to use distributed sampling.

    Returns:
        Path to the saved output file, or None if failed.
    """
    logger.info(f"Loading data from\n{configs.input_json_path}")
    try:
        dataloader = get_inference_dataloader(
            configs=configs,
            use_distributed_sampler=use_distributed_sampler,
        )
        logger.info(f"Loaded {len(dataloader)} samples to do inference on.")
    except Exception as e:
        error_message = f"{e}:\n{traceback.format_exc()}"
        logger.info(error_message)
        with open(opjoin(runner.error_dir, "error.txt"), "a") as f:
            f.write(error_message)
        return None

    saved_path = None
    for seed in configs.seeds:
        seed_everything(seed=seed, deterministic=configs.deterministic)
        for context in dataloader:
            try:
                sample_name = context.basic.pdb_id[0]
                n_atom = context.basic.n_atom[0]
                n_token = context.basic.n_token[0]

                logger.info(f"Running inference for sample {sample_name}...")
                output = runner.predict(context)

                ctx = SaveContext(
                    output_dir=runner.dump_dir,
                    sample_dirs=runner.dump_dir_list,
                    sample_name=sample_name,
                    atom_array=context.basic.atom_array[0],
                    n_atom=n_atom,
                    n_token=n_token,
                    output_format=configs.get("output_format", "npz"),
                    trajectory_step_stride=configs.get(
                        "sample_diffusion.trajectory_step_stride", 1
                    ),
                )

                saved_path = save_output(output, ctx)

                # Save motif mapping for backbone outputs (from context, not output)
                if isinstance(output, BackboneOutput):
                    parent_dir = os.path.dirname(runner.dump_dir)
                    mapping_path = opjoin(
                        parent_dir, f"{sample_name}_motif_mapping.json"
                    )
                    if not os.path.exists(mapping_path):
                        with open(mapping_path, "w") as f:
                            json.dump(context.basic.original_motif_mapping, f)

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

    return saved_path


def main(configs: Any, model_type: ModelType = ModelType.BACKBONE) -> None:
    runner = InferenceRunner(configs, model_type)
    infer_predict(runner, configs)


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
