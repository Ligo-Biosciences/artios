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
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
import torch.distributed as dist
import torch.nn as nn
from neptune_scale import Run
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from protenix.common.atom_utils import (
    create_atom_array_from_predictions,
    dense_atoms_to_flat_atoms,
)
from protenix.config import parse_configs, parse_sys_args
from protenix.config.config import save_config
from protenix.data.dataloader import get_dataloaders
from protenix.data.utils import save_structure_cif
from protenix.metrics.lddt_metrics import LDDTMetrics
from protenix.model.artios_allatom import ArtiosAllatom
from protenix.model.artios_vae import ArtiosAllAtomVAE
from protenix.model.loss import FlowLoss
from protenix.model.utils import expand_at_dim
from protenix.utils.distributed import DIST_WRAPPER, wait_for_all_processes
from protenix.utils.ema_utils import calculate_post_hoc_ema, log_ema_validation_curves
from protenix.utils.lr_scheduler import get_lr_scheduler
from protenix.utils.metrics import SimpleMetricAggregator
from protenix.utils.monitoring import (
    analyse_evaluation_latents,
    analyse_inference_latents,
    grad_norm,
)
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.profiler import profile_function
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import autocasting_disable_decorator, to_device
from protenix.utils.training import flatten_nested, get_optimizer, is_loss_nan_check
from runner.ema import PowerLawEMAWrapper

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Neptune logging configuration handled in init_log


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
        self.logs_dir = f"{self.run_dir}/logs"

        # Add dump dir for training time inference
        self.configs.dump_dir = self.prediction_dir

        if DIST_WRAPPER.rank == 0:
            os.makedirs(self.run_dir)
            os.makedirs(self.checkpoint_dir)
            os.makedirs(self.ema_checkpoint_dir)
            os.makedirs(self.post_hoc_ema_checkpoint_dir)
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
            + f"{self.checkpoint_dir}, ema_checkpoint_dir: {self.ema_checkpoint_dir}, post_hoc_ema_checkpoint_dir: {self.post_hoc_ema_checkpoint_dir}, prediction_dir: {self.prediction_dir}, structure_dir: "
            + f"{self.structure_dir}, error_dir: {self.error_dir}"
        )

    def init_log(self):
        self.neptune_run = None
        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
            self.neptune_run = Run(
                project=self.configs.project,
                experiment_name=self.run_name,
            )
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
        self.loss = FlowLoss(self.configs)
        self.symmetric_permutation = (
            SymmetricPermutation(  # TODO: still needed for ligands
                self.configs, error_dir=self.error_dir
            )
        )
        self.lddt_metrics = LDDTMetrics(self.configs)

    def init_model(self):
        # self.raw_model = ArtiosLigand(self.configs).to(
        #    self.device
        # )
        self.raw_flow_model = ArtiosAllatom(self.configs).to(self.device)
        self.vae_shell = ArtiosAllAtomVAE(self.configs).to(self.device)
        self.vae_shell.eval()
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
            use_ddp = (
                self.use_ddp
            )  # workaround for loading VAE checkpoint trained with DDP
            self.use_ddp = False
            self.vae_shell = _load_checkpoint(
                self.configs.load_vae_checkpoint_path,
                load_params_only=True,
                model=self.vae_shell,
            )
            self.use_ddp = use_ddp

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            logging.info(msg)

    def model_forward(
        self,
        batch: dict,
        mode: str = "train",
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        seed: Optional[int] = None,
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

    @profile_function("get_loss")
    def get_loss(
        self, batch: dict, mode: str = "train"
    ) -> tuple[torch.Tensor, dict, dict]:
        assert mode in ["train", "eval"]
        self.loss.to(batch["label_dict"]["coordinate"].device)  # Move loss to device

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
    def get_latents_data(self, batch: dict) -> dict:
        """Get the latents, seq_logits, timesteps, restype, and is_X_token from a batch.

        Returns:
            data (dict): A dictionary containing the data.
                Keys: "latents", "seq_logits", "latent_timestep", "restype", "is_X_token"
                where X is [protein, rna, dna, ligand]
                Values: np.ndarray, each of shape [..., N_sample*N_token, *]
        """
        data = {}

        # Latents
        latents = batch["pred_dict"]["latents"]  # [..., N_sample, N_token, c_latent]
        data["latents"] = latents

        # Seq logits
        seq_logits = batch["pred_dict"][
            "seq_logits"
        ]  # [..., N_sample, N_token, NUM_RESIDUES]
        data["seq_logits"] = seq_logits

        # Latent timestep
        latent_timestep = expand_at_dim(
            batch["pred_dict"]["latent_timestep"], -1, latents.shape[-2]
        ).unsqueeze(-1)  # [..., N_sample, N_token, 1]
        data["latent_timestep"] = latent_timestep

        # Restype
        restype = expand_at_dim(
            batch["input_feature_dict"]["restype"], -3, latents.shape[-3]
        )  # [..., N_sample, N_token, NUM_RESIDUES]
        data["restype"] = restype

        # is_X_token
        for mol_type in ["protein", "rna", "dna", "ligand"]:
            is_x_token = batch["input_feature_dict"][f"is_{mol_type}_token"]
            is_x_token = expand_at_dim(
                is_x_token, -3, latents.shape[-3]
            )  # [..., N_sample, N_token, 1]
            is_x_token = is_x_token.unsqueeze(-1)  # [..., N_sample, N_token, 1]
            data[f"is_{mol_type}_token"] = is_x_token

        # Reshape, move to numpy
        batch_dims = latents.shape[:-3]
        for name, tensor in data.items():
            data[name] = (
                tensor.reshape(*batch_dims, -1, tensor.shape[-1])
                .detach()
                .cpu()
                .float()
                .numpy()
            )

        return data

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
            current_state = {
                k: v.clone() for k, v in self.flow_model.state_dict().items()
            }

            for checkpoint_path in post_hoc_checkpoints:
                # Extract sigma value from filename
                sigma = float(checkpoint_path.stem.split("_post_hoc_ema_")[-1])

                # Load and evaluate the post-hoc EMA checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=self.device)

                # Load post-hoc EMA weights
                self.flow_model.load_state_dict(checkpoint["model"])

                # Evaluate
                self.print(f"Beginning evaluation for post-hoc ema with sigma:{sigma}")
                metrics = self._evaluate(ema_suffix=f"_post_hoc_ema_{sigma}", mode=mode)
                all_eval_metrics[f"post_hoc_ema_{sigma}"] = metrics

                # Delete post-hoc checkpoint from RAM and disk
                del checkpoint
                wait_for_all_processes()
                if DIST_WRAPPER.rank == 0:
                    os.remove(checkpoint_path)

            # Restore original model state
            self.flow_model.load_state_dict(current_state)

            # Plot all the metrics on Neptune
            if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                log_ema_validation_curves(all_eval_metrics, self.step, self.neptune_run)

    @torch.no_grad()
    def _evaluate(self, ema_suffix: str = "", mode: str = "eval"):
        # Init Metric Aggregator
        simple_metric_wrapper = SimpleMetricAggregator(["nonzero_avg"])
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        enable_amp = (
            torch.autocast(
                device_type="cuda",
                dtype=eval_precision,
            )
            if torch.cuda.is_available()
            else nullcontext()
        )

        self.flow_model.eval()

        self.latents_evaluation = (
            {"latents": [], "seq_logits": [], "latent_timestep": [], "restype": []}
            if self.configs.monitor.latents.track_evaluation_latents
            else None
        )

        self.latents_inference = (
            {"latents": [], "seq_logits": [], "latent_timestep": [], "restype": []}
            if self.configs.monitor.latents.track_inference_latents
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
                    batch, _ = self.model_forward(
                        batch, encoder=self.vae_shell.vae.encoder, mode=mode, seed=index
                    )

                    # Loss forward - forms loss and loss_dict, additionally returns the batch
                    loss, loss_dict = self.get_loss(
                        batch, mode="train"
                    )  # eval, loss evaluated in the same way as train
                    simple_metrics.update(loss_dict)

                # Track latents
                if self.latents_evaluation is not None and index < 20:
                    for name, tensor in self.get_latents_data(batch).items():
                        self.latents_evaluation[name].append(tensor)

                # Metrics
                for key, value in simple_metrics.items():
                    simple_metric_wrapper.add(key, value, namespace="val")

                # Inference
                if (
                    self.configs.run_training_time_inference
                    and index < 20
                    and DIST_WRAPPER.rank == 0
                ):
                    pred_dict, _, _ = self.flow_model(
                        input_feature_dict=batch["input_feature_dict"],
                        label_full_dict=batch["label_full_dict"],
                        label_dict=batch["label_dict"],
                        decoder=self.vae_shell.vae.decoder,
                        mode="inference",
                    )

                    saved_files = self.save_outputs(
                        pred_dict=pred_dict, batch=batch, ema_suffix=ema_suffix
                    )

                    if self.latents_inference is not None:
                        for name, tensor in self.get_latents_data(batch).items():
                            self.latents_inference[name].append(tensor)

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
                    del saved_files
                del batch, simple_metrics
                # if index % 5 == 0:
                # Release some memory periodically
                torch.cuda.empty_cache()

            metrics = simple_metric_wrapper.calc()
            self.print(f"Step {self.step}, eval {test_name}: {metrics}")

            ### Plots
            if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                ### Evaluation latent plots
                if self.latents_evaluation is not None:
                    for name, fig in analyse_evaluation_latents(
                        self.latents_evaluation,
                    ).items():
                        # Save figure to file and log to neptune
                        fig_path = f"{self.run_dir}/figures/{name}.png"
                        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
                        fig.savefig(fig_path, dpi=300, bbox_inches="tight")

                        self.neptune_run.log_files(
                            files={f"figures/{name}": fig_path},
                            step=self.step,
                        )
                        plt.close(fig)
                    # os.remove(fig_path)

                ### Inference latent plots
                if self.latents_inference is not None:
                    for name, fig in analyse_inference_latents(
                        self.latents_inference,
                    ).items():
                        # Save figure to file and log to neptune
                        fig_path = f"{self.run_dir}/figures/{name}.png"
                        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
                        fig.savefig(fig_path, dpi=300, bbox_inches="tight")

                        self.neptune_run.log_files(
                            files={f"figures/{name}": fig_path},
                            step=self.step,
                        )
                        plt.close(fig)
                        # os.remove(fig_path)
        return metrics

    def update(self):
        # Clip the gradient
        if self.configs.grad_clip_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.flow_model.parameters(), self.configs.grad_clip_norm
            )

    @profile_function("backward_pass")
    def backward_pass(self, loss, scaler):
        scaler.scale(loss / self.iters_to_accumulate).backward()
        # scaler.scale(loss).backward()

    @profile_function("optimizer_step")
    def optimizer_step(self, scaler):
        scaler.step(self.optimizer)
        scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

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
            # batch, _ = self.model_forward(batch, mode="train")

            batch, _ = self.model_forward(
                batch, encoder=self.vae_shell.vae.encoder, mode="train"
            )
            loss, loss_dict = self.get_loss(batch, mode="train")

        if self.configs.dtype in ["bf16", "fp32"]:
            if is_loss_nan_check(loss):
                self.print(f"Skip iteration with NaN loss: {self.step} steps")
                loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        # with profile_function("backward_pass"):
        # scaler.scale(loss / self.iters_to_accumulate).backward()
        self.backward_pass(loss, scaler)

        # For simplicity, the global training step is used
        if (self.global_step + 1) % self.iters_to_accumulate == 0:
            self.print(
                f"self.step {self.step}, self.iters_to_accumulate: {self.iters_to_accumulate}"
            )
            # Unscales the gradients of optimizer's assigned parameters in-place
            scaler.unscale_(self.optimizer)
            self.on_before_optimizer_step()
            # Do grad clip only
            # with profile_function("optimizer_step"):
            self.update()
            self.optimizer_step(scaler)
            # scaler.step(self.optimizer)
            # scaler.update()
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
                # Log to neptune
                self.neptune_run.log_metrics(
                    flatten_nested(weight_norms), step=self.step
                )

    @profile_function("on_before_optimizer_step")
    def on_before_optimizer_step(self):
        """Hook that is called before the optimizer step."""
        if self.configs.monitor.grad_norm:
            _, total_norm = grad_norm(self.flow_model, norm_type=2)

            # Average across all ranks
            # def numpy_mean_aggregation(values):
            #    return float(np.mean(values))  # Convert to float in case of numpy types
            # total_norm = gather_and_merge(total_norm, numpy_mean_aggregation)

            # Log to neptune
            if DIST_WRAPPER.rank == 0:
                self.neptune_run.log_metrics({"grad_norm": total_norm}, step=self.step)

    def save_outputs(self, pred_dict: dict, batch: dict, ema_suffix: str):
        """
        Saves outputs to CIFs.
        """
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
        centre_atom_indices = batch["cropped_token_array"].get_annotation(
            "centre_atom_index"
        )
        centre_atoms = cropped_atom_array[centre_atom_indices]

        chain_ids = centre_atoms.chain_id  # [N_token]
        res_ids = centre_atoms.res_id  # [N_token]
        label_entity_ids = centre_atoms.label_entity_id

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
            )  # [..., N_atom_flat, 3], [..., N_token]

            predicted_atom_array = create_atom_array_from_predictions(
                flattened_coords=flattened_coords,
                atoms_per_token=atoms_per_token,
                restype_onehot=sample_restype_onehot,
                chain_ids=chain_ids,
                res_ids=res_ids,
                is_ligand_token=is_ligand_token,
                label_entity_ids=label_entity_ids,
            )

            # Determine output file path
            output_path = os.path.join(
                self.structure_dir, f"{pdb_id}_sample_{idx}_{ema_suffix}.cif"
            )

            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Call save_to_cif function
            try:
                save_structure_cif(
                    atom_array=predicted_atom_array,
                    pred_coordinate=torch.Tensor(predicted_atom_array.coord),
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

        files_dict = {}
        for cif_path in cif_files:
            try:
                # Get relative path for the neptune key
                rel_path = os.path.relpath(cif_path, self.prediction_dir)
                structure_name = os.path.splitext(rel_path)[0].replace(os.sep, "_")

                # Create Neptune key and add to files dict
                neptune_key = (
                    f"structures/{ema_suffix}_{structure_name}_step_{self.step}"
                )
                files_dict[neptune_key] = str(cif_path)

            except Exception as e:
                self.print(f"Error processing CIF file {cif_path}: {e}")
                continue

        # Log all files to Neptune in one call
        if files_dict:
            self.neptune_run.log_files(
                files=files_dict,
                step=self.step,
            )

        # Explicitly clean up memory
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
                active=10,
                skip_first=50,
                skip_first_wait=1,
                repeat=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            while True:
                for batch in self.train_dl:
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
                        for wrapper in self.power_law_ema_wrappers:
                            wrapper.update(t=self.step + 1)
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
                            self.print(f"Step {self.step}, lr: {last_lr}")
                        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                            self.neptune_run.log_metrics(metrics, step=self.step)

                    if step_need_save or is_last_step:
                        self.print(f"beginning Saving checkpoint at step {self.step}")
                        self.save_checkpoint()  # Saves non-ema checkpoint
                        if use_ema:
                            for wrapper in self.power_law_ema_wrappers:
                                wrapper.apply_shadow()
                                self.save_checkpoint(
                                    ema_suffix=f"ema_{wrapper.sigma}",
                                    save_model_only=True,
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

                    # Profiler step
                    p.step()

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
