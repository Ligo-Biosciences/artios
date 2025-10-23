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

import datetime
import hashlib
import logging
import os
import time
from contextlib import nullcontext
from typing import Optional

import numpy as np
import rootutils
import torch
import torch.distributed as dist
from neptune_scale import Run
from neptune_scale.api.run import SourceTrackingConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from protenix.config import parse_configs, parse_sys_args
from protenix.config.config import save_config
from protenix.data.dataloader import get_dataloaders

# from protenix.metrics.lddt_metrics import LDDTMetrics
from protenix.metrics.sequence_metrics import SequenceMetrics
from protenix.model.allatom_mpnn import AllatomMPNN
from protenix.model.loss import MPNNLoss
from protenix.model.modules.atom_attention import (
    AtomAttentionEncoder,
)
from protenix.model.modules.transformer import (
    LocalAttnTransformerBlock,
    Transformer,
)
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.lr_scheduler import get_lr_scheduler
from protenix.utils.metrics import SimpleMetricAggregator
from protenix.utils.monitoring import grad_norm
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import autocasting_disable_decorator, to_device
from protenix.utils.torchwatcher import TorchWatcher
from protenix.utils.training import flatten_nested, get_optimizer, is_loss_nan_check
from runner.dumper import DataDumper
from runner.ema import EMAWrapper

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Disable WANDB's console output capture to reduce unnecessary logging
os.environ["WANDB_CONSOLE"] = "off"


class ArtiosAllAtomTrainer(object):
    def __init__(self, configs):
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_model()
        self.init_log()
        self.init_loss()
        self.init_data()
        self.try_load_checkpoint()

    def init_basics(self):
        # Step means effective step considering accumulation
        self.step = 0
        # Global_step equals to self.step * self.iters_to_accumulate
        self.global_step = 0
        self.start_step = 0
        # Add for grad accumulation, it can increase real batch size
        self.iters_to_accumulate = self.configs.iters_to_accumulate

        self.run_name = self.configs.run_name + "_" + time.strftime("%Y%m%d_%H%M%S")
        run_names = DIST_WRAPPER.all_gather_object(
            self.run_name if DIST_WRAPPER.rank == 0 else None
        )
        self.run_name = [name for name in run_names if name is not None][0]
        self.run_dir = f"{self.configs.base_dir}/{self.run_name}"
        self.checkpoint_dir = f"{self.run_dir}/checkpoints"
        self.ema_checkpoint_dir = f"{self.run_dir}/ema_checkpoints"
        self.prediction_dir = f"{self.run_dir}/predictions"
        self.structure_dir = f"{self.run_dir}/structures"
        self.dump_dir = f"{self.run_dir}/dumps"
        self.error_dir = f"{self.run_dir}/errors"
        self.logs_dir = f"{self.run_dir}/logs"

        # Add dump dir for training time inference
        self.configs.dump_dir = self.prediction_dir
        self.dumper = DataDumper(self.configs.dump_dir)

        if DIST_WRAPPER.rank == 0:
            os.makedirs(self.run_dir)
            os.makedirs(self.checkpoint_dir)
            os.makedirs(self.ema_checkpoint_dir)
            os.makedirs(self.prediction_dir)
            os.makedirs(self.structure_dir)
            os.makedirs(self.dump_dir)
            os.makedirs(self.error_dir)
            os.makedirs(self.logs_dir)
            save_config(
                self.configs,
                os.path.join(self.configs.base_dir, self.run_name, "config.yaml"),
            )

        self.print(
            f"Using run name: {self.run_name}, run dir: {self.run_dir}, checkpoint_dir: "
            + f"{self.checkpoint_dir}, ema_checkpoint_dir: {self.ema_checkpoint_dir}, prediction_dir: {self.prediction_dir}, structure_dir: "
            + f"{self.structure_dir}, error_dir: {self.error_dir}"
        )

    def init_log(self):
        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
            source_config = SourceTrackingConfig(  # source-code tracking
                namespace="source_code",
                upload_entry_point=True,
                upload_diff_head=True,
                upload_diff_upstream=True,
                upload_run_command=True,
            )
            self.neptune_run = Run(  # neptune
                project=self.configs.project,
                experiment_name=self.run_name,
                source_tracking_config=source_config,
            )
            self.torchwatcher = TorchWatcher(
                model=self.model.module if DIST_WRAPPER.world_size > 1 else self.model,
                run=self.neptune_run,
                track_layers=[
                    LocalAttnTransformerBlock,
                    Transformer,
                    AtomAttentionEncoder,
                ],  # optionally specify
                track_layers_name_includes=["linear_q", "linear_k"],
                tensor_stats=["channel_rms", "abs_mean"],
                base_namespace="model_metrics",
            )  # torchwatcher for model internals
            self.neptune_run.log_configs(flatten_nested(self.configs.to_dict()))
        self.train_metric_wrapper = SimpleMetricAggregator(["nan_avg"])

    def init_env(self):
        """Init pytorch/cuda envs."""
        logging.info(
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
        if DIST_WRAPPER.world_size > 1:
            timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
            dist.init_process_group(
                backend="nccl", timeout=datetime.timedelta(seconds=timeout_seconds)
            )

        if not self.configs.deterministic_seed:
            # use rank-specific seed
            hash_string = f"({self.configs.seed},{DIST_WRAPPER.rank},init_seed)"
            rank_seed = int(hashlib.sha256(hash_string.encode("utf8")).hexdigest(), 16)
            rank_seed = rank_seed % (2**32)
        else:
            rank_seed = self.configs.seed

        seed_everything(
            seed=rank_seed,
            deterministic=self.configs.deterministic,
        )  # different ddp processes have diff seeds

        if self.configs.use_deepspeed_evo_attention:
            env = os.getenv("CUTLASS_PATH", None)
            print(f"env: {env}")
            assert (
                env is not None
            ), "if use ds4sci, set env as https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"

        # Set high precision for tensor core matmul
        torch.set_float32_matmul_precision("high")
        logging.info("Finished init ENV.")

        # Set torch compile cache limit
        torch._dynamo.config.cache_size_limit = 8

    def init_loss(self):
        self.loss = MPNNLoss(self.configs)
        self.symmetric_permutation = SymmetricPermutation(
            self.configs, error_dir=self.error_dir
        )
        # self.lddt_metrics = LDDTMetrics(self.configs)
        self.sequence_metrics = SequenceMetrics()
        self.curr_loss_dict = {}

    def init_model(self):
        self.raw_model = AllatomMPNN(self.configs).to(self.device)  # TODO: AllatomMPNN
        self.use_ddp = False
        if DIST_WRAPPER.world_size > 1:
            self.print("Using DDP")
            self.use_ddp = True
            # Fix DDP/checkpoint https://discuss.pytorch.org/t/ddp-and-gradient-checkpointing/132244
            self.model = DDP(
                self.raw_model,
                find_unused_parameters=self.configs.find_unused_parameters,
                device_ids=[DIST_WRAPPER.local_rank],
                output_device=DIST_WRAPPER.local_rank,
                static_graph=True,
            )
        else:
            self.model = self.raw_model

        def count_parameters(model):
            total_params = sum(p.numel() for p in model.parameters())
            return total_params / 1000.0 / 1000.0

        self.print(f"Model Parameters: {count_parameters(self.model)}")
        if self.configs.get("ema_vae_decay", -1) > 0:
            assert self.configs.ema_vae_decay < 1
            self.ema_wrapper = EMAWrapper(
                self.model,
                self.configs.ema_vae_decay,
                self.configs.ema_mutable_param_keywords,
            )
            self.ema_wrapper.register()

        torch.cuda.empty_cache()
        self.optimizer = get_optimizer(self.configs, self.model)
        self.init_scheduler()

    def init_scheduler(self, **kwargs):
        self.lr_scheduler = get_lr_scheduler(self.configs, self.optimizer, **kwargs)

    def init_data(self):
        print("beginning init_data...")
        self.train_dl, self.test_dls = get_dataloaders(
            self.configs,
            DIST_WRAPPER.world_size,
            seed=self.configs.seed,
            error_dir=self.error_dir,
        )
        print("finished init_data")

    def save_checkpoint(self, ema_suffix="", save_model_only=False):
        if DIST_WRAPPER.rank == 0:
            if ema_suffix:
                path = f"{self.ema_checkpoint_dir}/{self.step}_{ema_suffix}.pt"
            else:
                path = f"{self.checkpoint_dir}/{self.step}.pt"

            if not save_model_only:
                # Save model, optimizer, scheduler, and step
                checkpoint = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": (
                        self.lr_scheduler.state_dict()
                        if self.lr_scheduler is not None
                        else None
                    ),
                    "step": self.step,
                }
            else:
                # Save model and step (used for EMA wrappers)
                checkpoint = {"model": self.model.state_dict(), "step": self.step}

            torch.save(checkpoint, path)
            self.print(f"Saved checkpoint to {path}")

    def try_load_checkpoint(self):
        def _load_checkpoint(
            checkpoint_path: str,
            load_params_only: bool,
            skip_load_optimizer: bool = False,
            skip_load_step: bool = False,
            skip_load_scheduler: bool = False,
        ):
            if not os.path.exists(checkpoint_path):
                raise Exception(f"Given checkpoint path not exist [{checkpoint_path}]")
            self.print(
                f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
            )
            checkpoint = torch.load(checkpoint_path, self.device)
            sample_key = [k for k in checkpoint["model"].keys()][0]
            self.print(f"Sampled key: {sample_key}")
            if sample_key.startswith("module.") and not self.use_ddp:
                # DDP checkpoint has module. prefix
                checkpoint["model"] = {
                    k[len("module.") :]: v for k, v in checkpoint["model"].items()
                }

            self.model.load_state_dict(
                state_dict=checkpoint["model"],
                strict=self.configs.load_strict,
            )
            if not load_params_only:
                if not skip_load_optimizer:
                    self.print("Loading optimizer state")
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                if not skip_load_step:
                    self.print("Loading checkpoint step")
                    self.step = checkpoint["step"] + 1
                    self.start_step = self.step
                    self.global_step = self.step * self.iters_to_accumulate
                if not skip_load_scheduler:
                    self.print("Loading scheduler state")
                    self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
                else:
                    # reinitialize LR scheduler using the updated optimizer and step
                    self.init_scheduler(last_epoch=self.step - 1)
            self.print(f"Finish loading checkpoint, current step: {self.step}")

        # Load EMA model parameters
        if (
            self.configs.load_ema_mpnn_checkpoint_path
        ):  # TODO: different checkpoint paths
            _load_checkpoint(
                self.configs.load_mpnn_vae_checkpoint_path,
                load_params_only=True,
            )
            self.ema_wrapper.register()

        # Load model parameters (With no EMA) from checkpoint
        if self.configs.load_mpnn_checkpoint_path:
            _load_checkpoint(
                self.configs.load_mpnn_checkpoint_path,
                self.configs.load_params_only,
                skip_load_optimizer=self.configs.skip_load_optimizer,
                skip_load_scheduler=self.configs.skip_load_scheduler,
                skip_load_step=self.configs.skip_load_step,
            )

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            logging.info(msg)

    def model_forward(
        self, batch: dict, mode: str = "train", seed: Optional[int] = None
    ) -> tuple[dict, dict]:
        """Forward pass of the model.

        Args:
            batch (dict): Batch of data.
            mode (str): Mode of the forward pass.
            seed (Optional[int]): Seed for random number generation. Defaults to None.

        Returns:
            tuple[dict, dict]: Predictions and log dict.
        """
        assert mode in ["train", "eval"]
        try:
            batch["pred_dict"], batch["label_dict"], log_dict = self.model(
                input_feature_dict=batch["input_feature_dict"],
                label_dict=batch["label_dict"],
                label_full_dict=batch["label_full_dict"],
                mode=mode,
                current_step=self.step if mode == "train" else None,
                # symmetric_permutation=self.symmetric_permutation,
                # seed=seed,
            )
        except Exception as e:
            print(f"Error in model forward: {e}")
            print(f"pdb_id: {batch['basic']['pdb_id']}, step: {self.step}")
            raise e
        return batch, log_dict

    def get_loss(self, batch: dict, mode: str = "train") -> tuple[torch.Tensor, dict]:
        assert mode in ["train", "eval"]
        self.loss.to(batch["label_dict"]["coordinate"].device)
        loss, loss_dict = autocasting_disable_decorator(self.configs.skip_amp.loss)(
            self.loss
        )(
            feat_dict=batch["input_feature_dict"],
            pred_dict=batch["pred_dict"],
            label_dict=batch["label_dict"],
        )
        return loss, loss_dict

    @torch.no_grad()
    def get_metrics(self, batch: dict) -> dict:
        metrics_dict = {}

        # LDDT metrics
        # metrics_dict["lddt"] = self.lddt_metrics.compute_lddt(
        #    batch["pred_dict"], batch["label_dict"]
        # )

        # Sequence metrics
        metrics_dict["sequence"] = self.sequence_metrics.compute_metrics(
            batch["pred_dict"], batch["input_feature_dict"]
        )

        return metrics_dict

    @torch.no_grad()
    def aggregate_metrics(self, metrics_dict: dict) -> dict:
        simple_metrics = {}

        # lddt_dict, _ = self.lddt_metrics.aggregate_lddt(metrics_dict["lddt"])
        # simple_metrics.update(lddt_dict)

        sequence_dict, _ = self.sequence_metrics.aggregate_sequence_metrics(
            metrics_dict["sequence"]
        )
        simple_metrics.update(sequence_dict)

        return simple_metrics

    @torch.no_grad()
    def evaluate(self, mode: str = "eval"):
        if not self.configs.eval_ema_only:
            self._evaluate()
        if hasattr(self, "ema_wrapper"):
            self.ema_wrapper.apply_shadow()
            self._evaluate(ema_suffix=f"ema{self.ema_wrapper.decay}_", mode=mode)
            self.ema_wrapper.restore()

    @torch.no_grad()
    def _evaluate(self, ema_suffix: str = "", mode: str = "eval"):
        # NOTE: currently, the test datasets are not being batched.
        # Init Metric Aggregator
        simple_metric_wrapper = SimpleMetricAggregator(["nonzero_avg"])
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )
        self.model.eval()

        # Initialize master monitor dict to collect all metrics
        # master_monitor_dict = defaultdict(lambda: defaultdict(list))

        self.latents = (
            {"mean": [], "restype": []}
            if self.configs.monitor.latents.track_during_vae_training
            else None
        )

        for test_name, test_dl in self.test_dls.items():
            self.print(f"Testing on {test_name}")
            evaluated_pids = []
            total_batch_num = len(test_dl)

            for index, batch in enumerate(tqdm(test_dl)):
                batch = to_device(batch, self.device)
                pid = batch["basic"]["pdb_id"]

                if index + 1 == total_batch_num and DIST_WRAPPER.world_size > 1:
                    # Gather all pids across ranks for avoiding duplicated evaluations when drop_last = False
                    all_data_ids = DIST_WRAPPER.all_gather_object(evaluated_pids)
                    dedup_ids = set(sum(all_data_ids, []))
                    if pid in dedup_ids:
                        print(
                            f"Rank {DIST_WRAPPER.rank}: Drop data_id {pid} as it is already evaluated."
                        )
                        break
                evaluated_pids.append(pid)

                simple_metrics = {}
                with enable_amp:
                    # Model forward - adds the pred_dict and label_dict to batch
                    batch, _ = self.model_forward(batch, mode=mode, seed=index)
                    # Loss forward - forms loss and loss_dict, additionally returns the batch
                    loss, loss_dict = self.get_loss(
                        batch, mode="train"
                    )  # eval, loss evaluated in the same way as train

                    simple_metrics.update(loss_dict)

                    # Additional metrics
                    metrics_dict = self.get_metrics(batch)
                    simple_metrics.update(self.aggregate_metrics(metrics_dict))

                # TODO: this might not work with batching.
                if (index + 1) % 10 == 0 and DIST_WRAPPER.rank == 0:  # one in 10
                    coordinate = batch["pred_dict"]["coordinate"]
                    n_atoms = len(batch["cropped_atom_array"])
                    batch["pred_dict"]["coordinate"] = coordinate[..., :n_atoms, :]
                    saved_files = self.dumper.dump(
                        dataset_name=test_name,
                        pdb_id=pid,
                        seed=0,
                        pred_dict=batch["pred_dict"],
                        atom_array=batch["cropped_atom_array"],
                        entity_poly_type=None,  # batch["cropped_atom_array"].entity_poly_type,
                    )

                    # Log structure files to Neptune Scale
                    if self.configs.use_wandb and saved_files:
                        files_dict = {}
                        for file_path in saved_files:
                            # Create a relative key for Neptune using test_name/pdb_id/filename
                            filename = os.path.basename(file_path)
                            neptune_key = f"structures/{test_name}/{pid}/{filename}"
                            files_dict[neptune_key] = file_path

                        self.neptune_run.log_files(
                            files=files_dict,
                            step=self.step,
                        )

                # Metrics
                for key, value in simple_metrics.items():
                    simple_metric_wrapper.add(key, value, namespace="val")

                del batch, simple_metrics
                if index % 5 == 0:
                    # Release some memory periodically
                    torch.cuda.empty_cache()

            metrics = simple_metric_wrapper.calc()
            self.print(f"Step {self.step}, eval {test_name}: {metrics}")

    def update(self):
        # Clip the gradient
        if self.configs.grad_clip_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.configs.grad_clip_norm
            )

    def train_step(self, batch: dict):
        self.model.train()
        # FP16 training has not been verified yet
        train_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        enable_amp = (
            torch.autocast(
                device_type="cuda", dtype=train_precision, cache_enabled=False
            )
            if torch.cuda.is_available()
            else nullcontext()
        )

        scaler = torch.GradScaler(
            device="cuda" if torch.cuda.is_available() else "cpu",
            enabled=(self.configs.dtype == "float16"),
        )

        with enable_amp:
            batch, _ = self.model_forward(batch, mode="train")
            loss, loss_dict = self.get_loss(batch, mode="train")

        if self.configs.dtype in ["bf16", "fp32"]:
            if is_loss_nan_check(loss):
                self.print(f"Skip iteration with NaN loss: {self.step} steps")
                loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        scaler.scale(loss / self.iters_to_accumulate).backward()

        # For simplicity, the global training step is used
        if (self.global_step + 1) % self.iters_to_accumulate == 0:
            self.print(
                f"self.step {self.step}, self.iters_to_accumulate: {self.iters_to_accumulate}"
            )
            # Unscales the gradients of optimizer's assigned parameters in-place
            scaler.unscale_(self.optimizer)
            self.on_before_optimizer_step()
            # Do grad clip only
            self.update()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
        for key, value in loss_dict.items():
            if "loss" not in key:
                continue
            self.train_metric_wrapper.add(key, value, namespace="train")

    def progress_bar(self, desc: str = ""):
        if DIST_WRAPPER.rank != 0:
            return
        if self.global_step % (
            self.configs.eval_interval * self.iters_to_accumulate
        ) == 0 or (not hasattr(self, "_ipbar")):
            # Start a new progress bar
            self._pbar = tqdm(
                range(
                    self.global_step
                    % (self.iters_to_accumulate * self.configs.eval_interval),
                    self.iters_to_accumulate * self.configs.eval_interval,
                )
            )
            self._ipbar = iter(self._pbar)

        step = next(self._ipbar)
        self._pbar.set_description(
            f"[step {self.step}: {step}/{self.iters_to_accumulate * self.configs.eval_interval}] {desc}"
        )
        return

    def on_validation_epoch_end(self):
        """Hook that is called at the end of each validation epoch."""
        if (
            not self.configs.use_wandb
            or DIST_WRAPPER.rank != 0
            or not self.configs.monitor.weight_norm
        ):
            return

        # Calculate and log weight norms for each parameter
        weight_norms = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:  # Skip non-trainable parameters
                continue

            if len(param.shape) == 2:  # Linear layer weights (2D tensors)
                # Compute L2 norm along fan_in dimension (dim=1) and average across output dimension
                # [out_features, in_features] -> [out_features] -> scalar
                fan_in_norms = torch.norm(
                    param, p=2, dim=1
                )  # L2 norm along input dimension
                norm = torch.mean(
                    fan_in_norms
                ).item()  # Average across output dimension
            else:  # Other parameters (biases, layernorm weights, etc.)
                # Standard L2 norm of the entire tensor divided by sqrt(num_elements)
                # although this is not the norm anymore, makes variable-length tensors comparable
                norm = param.norm().item() / np.sqrt(param.numel())
            weight_norms[f"weight_norms/{name}"] = norm

            if DIST_WRAPPER.rank == 0:
                # Log to wandb
                self.neptune_run.log_metrics(
                    flatten_nested(weight_norms), step=self.step
                )

    def on_before_optimizer_step(self):
        """Hook that is called before the optimizer step."""
        if self.configs.monitor.grad_norm:
            _, total_norm = grad_norm(self.model, norm_type=2)

            # Average across all ranks
            # def numpy_mean_aggregation(values):
            #    return float(np.mean(values))  # Convert to float in case of numpy types
            # total_norm = gather_and_merge(total_norm, numpy_mean_aggregation)

            # Log to wandb
            if DIST_WRAPPER.rank == 0:
                self.neptune_run.log_metrics({"grad_norm": total_norm}, step=self.step)

    def run(self):
        """
        Main entry for the ArtiosAllAtomTrainer.

        This function handles the training process, evaluation, logging, and checkpoint saving.
        """
        if self.configs.eval_only or self.configs.eval_first:
            self.evaluate()
            if self.configs.eval_only:
                return
        use_ema = hasattr(self, "ema_wrapper")
        self.print(f"Using EMA: {use_ema}")

        def trace_handler(p):
            output = p.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=20
            )
            print(output)
            # Only save trace from rank 0 to avoid file conflicts
            if DIST_WRAPPER.rank == 0:
                tracename = f"trace_{p.step_num}_rank{DIST_WRAPPER.rank}.json"
                filepath = f"{self.logs_dir}/{tracename}"
                # Ensure directory exists before writing
                os.makedirs(self.logs_dir, exist_ok=True)
                p.export_chrome_trace(filepath)
                # self.neptune_log_files({f"profiling/{tracename}": filepath})

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=10,
                warmup=10,
                active=50,
                skip_first=50,
                skip_first_wait=1,
                repeat=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            while True:
                for batch in self.train_dl:
                    # Print pdb id and step
                    is_update_step = (
                        self.global_step + 1
                    ) % self.iters_to_accumulate == 0
                    is_last_step = (self.step + 1) == self.configs.max_steps
                    step_need_log = (self.step + 1) % self.configs.log_interval == 0

                    step_need_eval = (
                        self.configs.eval_interval > 0
                        and (self.step + 1) % self.configs.eval_interval == 0
                    )
                    step_need_save = (
                        self.configs.checkpoint_interval > 0
                        and (self.step + 1) % self.configs.checkpoint_interval == 0
                    )

                    is_last_step &= is_update_step
                    step_need_log &= is_update_step
                    step_need_eval &= is_update_step
                    step_need_save &= is_update_step

                    batch = to_device(batch, self.device)
                    self.progress_bar()
                    self.train_step(batch)
                    if use_ema and is_update_step:
                        self.ema_wrapper.update()
                    if step_need_log or is_last_step:
                        metrics = self.train_metric_wrapper.calc()
                        self.print(f"Step {self.step} train: {metrics}")
                        last_lr = self.lr_scheduler.get_last_lr()[0]
                        if DIST_WRAPPER.rank == 0:
                            if self.configs.use_wandb:
                                self.neptune_run.log_metrics(
                                    {"train/lr": last_lr},
                                    step=self.step,
                                )
                                self.torchwatcher.watch(
                                    step=self.step,
                                    track_gradients=False,
                                    track_activations=True,
                                    track_parameters=False,
                                    prefix="train",
                                )
                            self.print(f"Step {self.step}, lr: {last_lr}")
                        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                            self.neptune_run.log_metrics(metrics, step=self.step)

                    if step_need_save or is_last_step:
                        self.print(f"beginning Saving checkpoint at step {self.step}")
                        self.save_checkpoint()  # Saves non-ema checkpoint
                        if use_ema:
                            self.ema_wrapper.apply_shadow()
                            self.save_checkpoint(
                                ema_suffix=f"_ema_{self.ema_wrapper.decay}"
                            )
                            self.ema_wrapper.restore()
                            # wait_for_all_processes()

                    if step_need_eval or is_last_step:
                        self.evaluate()
                        self.on_validation_epoch_end()

                    self.global_step += 1
                    if self.global_step % self.iters_to_accumulate == 0:
                        self.step += 1
                    if self.step >= self.configs.max_steps:
                        self.print(f"Finish training after {self.step} steps")
                        break

                    # Profiler step
                    p.step()

                if self.step >= self.configs.max_steps:
                    break


def main():
    LOG_FORMAT = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        filename="output.log",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="a",
    )
    configs_base["use_deepspeed_evo_attention"] = (
        os.environ.get("USE_DEEPSPEED_EVO_ATTTENTION", False) == "true"
    )
    configs = {**configs_base, **{"data": data_configs}}
    configs = parse_configs(
        configs,
        parse_sys_args(),
    )

    print(configs.run_name)
    print(configs)
    trainer = ArtiosAllAtomTrainer(configs)
    trainer.run()


if __name__ == "__main__":
    main()
