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
import logging
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from glob import glob
from pathlib import Path
from typing import Optional

from biotite.structure import AtomArray
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
import torch.distributed as dist
import wandb
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from protenix.common.atom_types import DENSE_ATOM
from protenix.common.atom_utils import dense_atoms_to_flat_atoms, create_atom_array_from_predictions
from protenix.common.constants import STD_RESIDUES_AND_GAP
from protenix.config import parse_configs, parse_sys_args
from protenix.config.config import save_config
from protenix.data.dataloader import get_dataloaders
from protenix.data.utils import save_structure_cif
from protenix.metrics.lddt_metrics import LDDTMetrics
from protenix.model.artios_allatom import ArtiosAllatom
from protenix.model.artios_vae import ArtiosAllAtomVAE
from protenix.model.utils import expand_at_dim
from protenix.utils.distributed import DIST_WRAPPER, wait_for_all_processes
from protenix.utils.ema_utils import calculate_post_hoc_ema, log_ema_validation_curves
from protenix.utils.lr_scheduler import get_lr_scheduler
from protenix.utils.metrics import SimpleMetricAggregator
from protenix.utils.monitoring import grad_norm
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.seed import seed_everything
from protenix.utils.structure_io import cif_to_pdb
from protenix.utils.torch_utils import autocasting_disable_decorator, to_device
from protenix.utils.training import get_optimizer, is_loss_nan_check
from runner.ema import PowerLawEMAWrapper
from runner.dumper import DataDumper
from runner.evaluate import Evaluator
from runner.inference import InferenceRunner, infer_predict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Disable WANDB's console output capture to reduce unnecessary logging
os.environ["WANDB_CONSOLE"] = "off"


class ArtiosAllAtomFlowTrainer(object):
    def __init__(self, configs):
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_log()
        self.init_model()
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
        self.post_hoc_ema_checkpoint_dir = f"{self.run_dir}/post_hoc_ema_checkpoints"
        self.prediction_dir = f"{self.run_dir}/predictions"
        self.structure_dir = f"{self.run_dir}/structures"
        self.dump_dir = f"{self.run_dir}/dumps"
        self.error_dir = f"{self.run_dir}/errors"

        # Add dump dir for training time inference
        self.configs.dump_dir = self.prediction_dir
        self.dumper = DataDumper(self.configs.dump_dir)

        if DIST_WRAPPER.rank == 0:
            os.makedirs(self.run_dir)
            os.makedirs(self.checkpoint_dir)
            os.makedirs(self.ema_checkpoint_dir)
            os.makedirs(self.post_hoc_ema_checkpoint_dir)
            os.makedirs(self.prediction_dir)
            os.makedirs(self.structure_dir)
            os.makedirs(self.dump_dir)
            os.makedirs(self.error_dir)
            save_config(
                self.configs,
                os.path.join(self.configs.base_dir, self.run_name, "config.yaml"),
            )

        self.print(
            f"Using run name: {self.run_name}, run dir: {self.run_dir}, checkpoint_dir: "
            + f"{self.checkpoint_dir}, ema_checkpoint_dir: {self.ema_checkpoint_dir}, post_hoc_ema_checkpoint_dir: {self.post_hoc_ema_checkpoint_dir}, prediction_dir: {self.prediction_dir}, structure_dir: "
            + f"{self.structure_dir}, error_dir: {self.error_dir}"
        )

    def init_log(self):
        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
            wandb.init(
                project=self.configs.project,
                name=self.run_name,
                config=vars(self.configs),
                id=self.configs.wandb_id or None,
            )
        self.train_metric_wrapper = SimpleMetricAggregator(["avg"])

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
        # All ddp process got the same seed
        seed_everything(
            seed=self.configs.seed,
            deterministic=self.configs.deterministic,
        )

        if self.configs.use_deepspeed_evo_attention:
            env = os.getenv("CUTLASS_PATH", None)
            print(f"env: {env}")
            assert (
                env is not None
            ), "if use ds4sci, set env as https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"

        # Set high precision for tensor core matmul
        torch.set_float32_matmul_precision("high")
        logging.info("Finished init ENV.")

    def init_loss(self):
        # self.loss = ProtenixLoss(self.configs) #TODO: mse losses for latents and centre coords
        self.symmetric_permutation = (
            SymmetricPermutation(  # TODO: still needed for ligands
                self.configs, error_dir=self.error_dir
            )
        )
        self.lddt_metrics = LDDTMetrics(self.configs)

    def init_model(self):
        #self.raw_model = ArtiosLigand(self.configs).to(
        #    self.device
        #) 
        self.raw_flow_model = ArtiosAllatom(self.configs).to(self.device)
        self.vae = ArtiosAllAtomVAE(self.configs).to(self.device)
        # DDP not required for VAE loading since model is frozen - no need for gradient aggregation
        self.use_ddp = False
        if DIST_WRAPPER.world_size > 1:
            self.print("Using DDP")
            self.use_ddp = True
            # Fix DDP/checkpoint https://discuss.pytorch.org/t/ddp-and-gradient-checkpointing/132244
            self.flow_model = DDP(
                self.raw_flow_model,
                find_unused_parameters=self.configs.find_unused_parameters,
                device_ids=[DIST_WRAPPER.local_rank],
                output_device=DIST_WRAPPER.local_rank,
                static_graph=True,
            )
        else:
            self.flow_model = self.raw_flow_model

        if self.configs.get("power_law_ema_sigma", []):  # Check if list is not empty
            self.power_law_ema_wrappers = []  # Store wrappers in a list
            for sigma in self.configs.power_law_ema_sigma:
                wrapper = PowerLawEMAWrapper(
                    model=self.flow_model,
                    sigma=sigma,
                    mutable_param_keywords=self.configs.ema_mutable_param_keywords,
                )
                wrapper.register()
                self.power_law_ema_wrappers.append(wrapper)
            self.print(
                f"Initialized {len(self.power_law_ema_wrappers)} power law EMA wrappers with standard deviations: {self.configs.power_law_ema_sigma}"
            )

        torch.cuda.empty_cache()
        self.optimizer = get_optimizer(self.configs, self.flow_model)
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
                    "model": self.flow_model.state_dict(),
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
                checkpoint = {"model": self.flow_model.state_dict(), "step": self.step}

            torch.save(checkpoint, path)
            self.print(f"Saved checkpoint to {path}")

    # TODO: new loading logic required here for loading in different models (the encoder, decoder or
    # flow model).
    def try_load_checkpoint(self):
        def _load_checkpoint(
            checkpoint_path: str,
            load_params_only: bool,
            model: nn.Module,
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

            model.load_state_dict(
                    checkpoint["model"],
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

            return model

        if self.configs.load_ema_flow_checkpoint_paths_directory:
            checkpoint_dir = self.configs.load_ema_flow_checkpoint_paths_directory
            if not os.path.exists(checkpoint_dir):
                raise Exception(
                    f"EMA checkpoint directory does not exist: {checkpoint_dir}"
                )

            # Get all checkpoint files from directory
            checkpoint_files = [
                f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")
            ]
            self.print(
                f"Found {len(checkpoint_files)} EMA checkpoints in {checkpoint_dir}"
            )

            # Create mapping of sigma values to their corresponding wrappers
            sigma_to_wrapper = {
                wrapper.sigma: wrapper for wrapper in self.power_law_ema_wrappers
            }

            # Load each checkpoint and match with corresponding wrapper
            for checkpoint_file in checkpoint_files:
                # Extract sigma value from filename (assuming format like "*_ema_6.94.pt")
                try:
                    sigma_str = checkpoint_file.split("_ema_")[-1].replace(".pt", "")
                    sigma = float(sigma_str)
                except (IndexError, ValueError):
                    self.print(
                        f"Warning: Could not extract a sigma value from checkpoint filename: {checkpoint_file}"
                    )
                    continue

                if sigma not in sigma_to_wrapper:
                    self.print(
                        f"Warning: No matching wrapper found for sigma={sigma} in checkpoint: {checkpoint_file}"
                    )
                    continue

                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
                _load_checkpoint(
                    checkpoint_path,
                    load_params_only=True,
                    model="flow",
                )

                # Register only the wrapper corresponding to this sigma
                matching_wrapper = sigma_to_wrapper[sigma]
                matching_wrapper.register()
                self.print(
                    f"Loaded and registered EMA checkpoint for sigma={sigma}: {checkpoint_file}"
                )

        # Load model parameters (With no EMA) from checkpoint
        if self.configs.load_flow_checkpoint_path:
            self.flow_model = _load_checkpoint(
                self.configs.load_flow_checkpoint_path,
                self.configs.load_params_only,
                model=self.flow_model,
                skip_load_optimizer=self.configs.skip_load_optimizer,
                skip_load_scheduler=self.configs.skip_load_scheduler,
                skip_load_step=self.configs.skip_load_step,
            )

        if self.configs.load_vae_checkpoint_path:
            self.vae = _load_checkpoint(
                self.configs.load_vae_checkpoint_path,
                load_params_only=True,
                model=self.vae,
            )

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            logging.info(msg)

    def model_forward(
        self, 
        batch: dict, 
        mode: str = "train", 
        encoder: Optional[nn.Module] = None, 
        decoder: Optional[nn.Module] = None,
        seed: Optional[int] = None
    ) -> tuple[dict, dict]:
        """Forward pass of the model.

        Args:
            batch (dict): Batch of data.
            mode (str): Mode of the forward pass.
            encoder (Optional[nn.Module]): Encoder model. Defaults to None.
            decoder (Optional[nn.Module]): Decoder model. Defaults to None.
            seed (Optional[int]): Seed for random number generation. Defaults to None.

        Returns:
            tuple[dict, dict]: Predictions and log dict.
        """
        assert mode in ["train", "eval"]
        batch["pred_dict"], batch["label_dict"], log_dict = self.flow_model(
            input_feature_dict=batch["input_feature_dict"],
            label_dict=batch["label_dict"],
            label_full_dict=batch["label_full_dict"],
            mode=mode,
            current_step=self.step if mode == "train" else None,
            symmetric_permutation=self.symmetric_permutation,
            encoder=encoder,
            decoder=decoder,
            seed=seed,
        )
        return batch, log_dict

    def get_loss(
        self, batch: dict, mode: str = "train"
    ) -> tuple[torch.Tensor, dict, dict]:
        assert mode in ["train", "eval"]
        self.loss.to(batch["label_dict"]["coordinate"].device)  # Move loss to device

        loss, loss_dict, monitor_data = autocasting_disable_decorator(
            self.configs.skip_amp.loss
        )(self.loss)(
            feat_dict=batch["input_feature_dict"],
            pred_dict=batch["pred_dict"],
            label_dict=batch["label_dict"],
            mode=mode,
        )
        return loss, loss_dict, monitor_data

    @torch.no_grad()
    def get_metrics(self, batch: dict) -> dict:
        lddt_dict = self.lddt_metrics.compute_lddt(
            batch["pred_dict"], batch["label_dict"]
        )
        # TODO: add the get_latents metric logic. It will be valuable to see the same
        # visualisations of the latents coming out of the flow model to compare to what
        # we see training the VAE.

        # TODO: building on the above todo, comparing the latent distribution to the VAE's
        # latent distribution is going to be key for analysing training performance. Ideas
        # for what we want to see:
        # - Wasserstein dist aggregated over seeds, per sample
        # - histograms, tsne, qq aggregated over samples
        # - histograms, tsne, qq aggregated over seeds, per sample
        # - it is likely that specific molecular properties get mapped to specific subsets of
        # the latent distribution. Is this retained by the flow model?

        return lddt_dict

    @torch.no_grad()
    def aggregate_metrics(self, lddt_dict: dict, batch: dict) -> dict:
        simple_metrics, _ = self.lddt_metrics.aggregate_lddt(lddt_dict)

        return simple_metrics

    @torch.no_grad()
    def get_latents(self, batch: dict) -> dict:
        """Get the latents from the batch."""
        # Get num_sample latents and flatten
        latents = batch["pred_dict"]["latents"]  # [..., N_sample, N_token, c_latent]

        batch_dims = latents.shape[:-3]

        # Collapse N_sample and N_token dimensions together
        latents = latents.reshape(batch_dims, -1, latents.shape[-1])  # [..., N_sample*N_token, c_latent]

        return latents.detach().cpu().float().numpy()

    @torch.no_grad()
    def evaluate(self, mode: str = "eval"):
        all_eval_metrics = {}
        if not self.configs.eval_ema_only:
            # Evaluate None-EMA
            metrics = self._evaluate()
            all_eval_metrics["None-EMA"] = metrics

        # Evaluate post-hoc EMA checkpoints
        if self.configs.calc_post_hoc_ema_during_training:
            # Perform post hoc ema calculation
            post_hoc_checkpoints = self.calc_post_hoc_ema()

            # Store current model state
            current_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            for checkpoint_path in post_hoc_checkpoints:
                # Extract sigma value from filename
                sigma = float(checkpoint_path.stem.split("_post_hoc_ema_")[-1])

                # Load and evaluate the post-hoc EMA checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=self.device)

                # Load post-hoc EMA weights
                self.model.load_state_dict(checkpoint["model"])

                # Evaluate
                print(f"Beginning evaluation for post-hoc ema with sigma:{sigma}")
                metrics = self._evaluate(ema_suffix=f"_post_hoc_ema_{sigma}", mode=mode)
                all_eval_metrics[f"post_hoc_ema_{sigma}"] = metrics

                # Delete post-hoc checkpoint from RAM and disk
                del checkpoint
                wait_for_all_processes()
                if DIST_WRAPPER.rank == 0:
                    os.remove(checkpoint_path)

            # Restore original model state
            self.model.load_state_dict(current_state)

            # Plot all the metrics on wandb
            if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                log_ema_validation_curves(all_eval_metrics, self.step)

    @torch.no_grad()
    def _evaluate(self, ema_suffix: str = "", mode: str = "eval"):
        # Init Metric Aggregator
        simple_metric_wrapper = SimpleMetricAggregator(["avg"])
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

        self.flow_model.eval()

        self.latents = (
            {"latents": [], "restype": []}
            if self.configs.monitor.latents.track_latents
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
                    #batch, _ = self.model_forward(batch, mode=mode, seed=index)

                    batch, _ = self.model_forward(batch, encoder=self.vae.encoder, mode=mode, seed=index)

                    # Loss forward - forms loss and loss_dict, additionally returns the batch
                    loss, loss_dict, monitor_data = self.get_loss(
                        batch, mode="train"
                    )  # eval, loss evaluated in the same way as train
                    simple_metrics.update(loss_dict)

                # Metrics
                for key, value in simple_metrics.items():
                    simple_metric_wrapper.add(key, value, namespace="val")

                # Inference
                if self.configs.run_training_time_inference and index < 20 and DIST_WRAPPER.rank == 0:
                    pred_dict, _, _ = self.flow_model(
                        input_feature_dict=batch["input_feature_dict"], 
                        label_full_dict=batch["label_full_dict"],
                        label_dict=batch["label_dict"],
                        decoder=self.vae.decoder, 
                        mode="inference"
                    )

                    saved_files = self.save_outputs(
                        pred_dict=pred_dict,
                        batch=batch,
                    )

                    if self.latents is not None:
                        self.latents["latents"].append(self.get_latents(batch))
                        #self.latents["restype"].append(something_or_other)

                    # Log structure files to Neptune Scale
                    if self.configs.use_wandb:
                        if saved_files:
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
                        if self.latents is not None:
                            pass

                del batch, simple_metrics, saved_files
                if index % 5 == 0:
                    # Release some memory periodically
                    torch.cuda.empty_cache()

            metrics = simple_metric_wrapper.calc()
            self.print(f"Step {self.step}, eval {test_name}: {metrics}")

        return metrics

    def update(self):
        # Clip the gradient
        if self.configs.grad_clip_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.configs.grad_clip_norm
            )

    def train_step(self, batch: dict):
        self.flow_model.train()
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
            #batch, _ = self.model_forward(batch, mode="train")

            # TODO: pass in encoder to forward pass
            batch, _ = self.model_forward(batch, encoder=self.vae.encoder, mode="train")
            loss, loss_dict, _ = self.get_loss(batch, mode="train")

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
        torch.cuda.empty_cache()

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

    def calc_post_hoc_ema(self):
        """
        Calculate post-hoc EMA checkpoints using the utility function from ema_utils.
        This combines checkpoints with different sigma values to create new EMA profiles
        with target sigma values specified in configs.post_hoc_ema_decay.
        """
        # Get all EMA checkpoint paths
        ema_checkpoint_paths = [
            Path(os.path.join(self.ema_checkpoint_dir, f))
            for f in os.listdir(self.ema_checkpoint_dir)
            if f.endswith(".pt")
        ]

        if len(ema_checkpoint_paths) == 0:
            self.print(f"No EMA checkpoints found in {self.ema_checkpoint_dir}")
            return

        self.print(
            f"Found {len(ema_checkpoint_paths)} EMA checkpoints in {self.ema_checkpoint_dir}"
        )

        # Calculate post-hoc EMA checkpoints
        save_dir = Path(self.post_hoc_ema_checkpoint_dir)
        target_sigma_values = self.configs.post_hoc_ema_sigma

        # Use the utility function to calculate post-hoc EMA checkpoints
        output_paths = calculate_post_hoc_ema(
            ema_checkpoint_paths=ema_checkpoint_paths,
            target_sigma_values=target_sigma_values,
            save_dir=save_dir,
        )
        self.print(f"Created {len(output_paths)} post-hoc EMA checkpoints.")
        return output_paths

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
        for name, param in self.flow_model.named_parameters():
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
                wandb.log(weight_norms, step=self.step)

    def on_before_optimizer_step(self):
        """Hook that is called before the optimizer step."""
        if self.configs.monitor.grad_norm:
            _, total_norm = grad_norm(self.flow_model, norm_type=2)

            # Average across all ranks
            # def numpy_mean_aggregation(values):
            #    return float(np.mean(values))  # Convert to float in case of numpy types
            # total_norm = gather_and_merge(total_norm, numpy_mean_aggregation)

            # Log to wandb
            if DIST_WRAPPER.rank == 0:
                wandb.log({"grad_norm": total_norm}, step=self.step)

    def save_outputs(self, pred_dict: dict, batch: dict):
        """
        Saves outputs to CIFs. Will be using pred_dict["coordinate"] which is 
        shape [..., N_sample, N_token, NUM_DENSE_ATOMS, 3]

        Tokenwise chain information can be gotten from the batch["cropped_atom_array"] and then
        filtering the cropped atom array for the centre atoms and extracting their chain_ids, then
        these can be passed to the flattened coordinates
        """
        idx_to_resname = {idx: res for res, idx in STD_RESIDUES_AND_GAP.items()}

        # Get data
        cropped_atom_array = batch["cropped_atom_array"]
        pdb_id = batch["basic"]["pdb_id"]
        is_ligand_token = batch["input_feature_dict"]["is_ligand_token"].cpu().numpy()
        
        # Get predictions
        coordinate = pred_dict["coordinate"].detach().cpu().numpy()
        seq_logits = pred_dict["seq_logits"].detach().cpu().numpy()

        restype_onehot = np.eye(seq_logits.shape[-1])[
            np.argmax(seq_logits, axis=-1)
        ].astype(np.float32)  # [..., N_sample, N_token, NUM_RESIDUES]

        # Get chain, res_id's
        centre_atom_indices = batch["cropped_token_array"].get_annotation("centre_atom_index")
        centre_atoms = cropped_atom_array[centre_atom_indices]

        chain_ids = centre_atoms.chain_id  # [N_token]
        res_ids = centre_atoms.res_id  # [N_token]
        
        # Save each sample
        saved_files = []
        for idx in range(coordinate.shape[-4]):
            sample_dense_coords = coordinate[idx]
            sample_restype_onehot = restype_onehot[idx]

            # Calculate flattened coordinates
            flattened_coords, atoms_per_token = dense_atoms_to_flat_atoms(
                restype_onehot=sample_restype_onehot,
                dense_atom_coords=sample_dense_coords,
                is_ligand_token=is_ligand_token,
            ) # [..., N_atom_flat, 3], [..., N_token]

            predicted_atom_array = create_atom_array_from_predictions(
                flattened_coords=flattened_coords,
                atoms_per_token=atoms_per_token,
                restype_onehot=sample_restype_onehot,
                chain_ids=chain_ids,
                res_ids=res_ids,
                is_ligand_token=is_ligand_token,
            )
            
            # Determine output file path
            output_path = os.path.join(self.structure_dir, f"{pdb_id}_sample_{idx}.cif")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Call save_to_cif function
            try:
                save_structure_cif(
                    atom_array=predicted_atom_array,
                    pred_coordinate=predicted_atom_array.coord,
                    output_fpath=output_path,
                    entity_poly_type=None,
                    pdb_id=pdb_id,
                )
                saved_files.append(output_path)
                self.print(f"Saved sample {idx} to {output_path}")
                
            except Exception as e:
                self.print(f"Error saving sample {idx}: {e}")
                continue
        
        return saved_files

    def log_structure_files_to_wandb(self, cif_files: list[Path], ema_suffix: str = ""):
        if not self.configs.use_wandb or DIST_WRAPPER.rank != 0:
            return
        for cif_path in cif_files:
            try:
                # Create temporary PDB file path with full path
                cif_path = os.path.abspath(cif_path)
                temp_pdb_path = os.path.abspath(str(Path(cif_path).with_suffix(".pdb")))

                # Convert CIF to PDB
                cif_to_pdb(cif_path, temp_pdb_path)

                # Verify the PDB file was created
                if not os.path.exists(temp_pdb_path):
                    self.print(f"Warning: Failed to create PDB file at {temp_pdb_path}")
                    continue

                # Get relative path for the wandb key
                rel_path = os.path.relpath(cif_path, self.prediction_dir)
                structure_name = os.path.splitext(rel_path)[0].replace(os.sep, "_")

                # Log PDB file to wandb
                wandb.log(
                    {
                        f"structures/{ema_suffix}_{structure_name}_step_{self.step}": wandb.Molecule(
                            temp_pdb_path
                        )
                    },
                    step=self.step,
                )

                # Clean up both files
                if os.path.exists(temp_pdb_path):
                    os.remove(temp_pdb_path)

            except Exception as e:
                self.print(f"Error processing CIF file {cif_path}: {e}")
                continue

            # Explicitly clean up memory
            torch.cuda.empty_cache()

    def run_evaluation_tests(self, cif_files: list[Path], ema_suffix: str = ""):
        if not self.configs.use_wandb or DIST_WRAPPER.rank != 0:
            return

        print(f"Evaluating tests: {self.configs.monitor.eval_metrics.eval_tests}...")
        results = {}

        # Valid evaluation tests
        valid_eval_tests = {
            "ss_distribution": Evaluator.ss_distribution,
            "posebusters": Evaluator.evaluate_posebusters,
        }

        for eval_test in self.configs.monitor.eval_metrics.eval_tests:
            assert (
                eval_test in valid_eval_tests
            ), f"Invalid evaluation type: {eval_test}. Valid types are: {list(valid_eval_tests.keys())}"

            print(f"Running {eval_test} evaluation...")
            try:
                results[eval_test] = valid_eval_tests[eval_test](cif_files=cif_files)
                print(f"{eval_test} evaluation complete.")
            except Exception as e:
                self.print(f"Error running {eval_test} evaluation: {e}")
                results.pop(eval_test, None)
                continue

            # Clean up after each test
            torch.cuda.empty_cache()

        # Log metrics to WandB
        wandb_metrics = {}
        for test_name, df in results.items():
            for col in df.columns:
                if col == "Ligand CCD":
                    continue
                wandb_metrics[f"{test_name}/{ema_suffix}_{col}"] = (
                    df[col].astype(float).mean()
                )

        wandb.log(wandb_metrics, step=self.step)

        # Clean up memory again
        del results, wandb_metrics
        torch.cuda.empty_cache()

    def run(self):
        """
        Main entry for the ArtiosAllAtomFlowTrainer.

        This function handles the training process, evaluation, logging, and checkpoint saving.
        """
        if self.configs.eval_only or self.configs.eval_first:
            self.evaluate()
            if self.configs.eval_only:
                return
        use_ema = hasattr(self, "power_law_ema_wrappers")
        self.print(f"Using power law ema's: {use_ema}")

        while True:
            for batch in self.train_dl:
                is_update_step = (self.global_step + 1) % self.iters_to_accumulate == 0
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
                if use_ema:
                    for wrapper in self.power_law_ema_wrappers:
                        wrapper.update(t=self.step + 1)
                if step_need_log or is_last_step:
                    metrics = self.train_metric_wrapper.calc()
                    self.print(f"Step {self.step} train: {metrics}")
                    last_lr = self.lr_scheduler.get_last_lr()[0]
                    if DIST_WRAPPER.rank == 0:
                        if self.configs.use_wandb:
                            wandb.log(
                                {"train/lr": last_lr},
                                step=self.step,
                            )
                        self.print(f"Step {self.step}, lr: {last_lr}")
                    if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                        wandb.log(metrics, step=self.step)

                if step_need_save or is_last_step:
                    self.print(f"beginning Saving checkpoint at step {self.step}")
                    self.save_checkpoint()  # Saves non-ema checkpoint
                    if use_ema:
                        for wrapper in self.power_law_ema_wrappers:
                            wrapper.apply_shadow()
                            self.save_checkpoint(
                                ema_suffix=f"ema_{wrapper.sigma}", save_model_only=True
                            )
                            wrapper.restore()
                        wait_for_all_processes()

                if step_need_eval or is_last_step:
                    self.evaluate()
                    self.on_validation_epoch_end()

                self.global_step += 1
                if self.global_step % self.iters_to_accumulate == 0:
                    self.step += 1
                if self.step >= self.configs.max_steps:
                    self.print(f"Finish training after {self.step} steps")
                    break

            if self.step >= self.configs.max_steps:
                break


def main():
    LOG_FORMAT = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
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
    trainer = ArtiosAllAtomFlowTrainer(configs)
    trainer.run()


if __name__ == "__main__":
    main()
