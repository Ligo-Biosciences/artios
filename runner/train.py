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

import torch
import torch.distributed as dist
import numpy as np
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from protenix.config import parse_configs, parse_sys_args
from protenix.config.config import save_config
from protenix.data.dataloader import get_dataloaders
from protenix.metrics.lddt_metrics import LDDTMetrics
from protenix.model.loss import ProtenixLoss
from protenix.model.artios_ligand import ArtiosLigand
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.lr_scheduler import get_lr_scheduler
from protenix.utils.metrics import SimpleMetricAggregator
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import autocasting_disable_decorator, to_device
from protenix.utils.training import get_optimizer, is_loss_nan_check, extract_gamma_step
from runner.ema import EMAWrapper, PowerLawEMAWrapper, solve_weights
from protenix.data.utils import to_pdb
from io import StringIO

# Disable WANDB's console output capture to reduce unnecessary logging
os.environ["WANDB_CONSOLE"] = "off"
PDB_TO_VISUALISE = [
    "7xzr", "7uaw", "7zhd", "7x0b", 
    "7vkq", "7pk0", "7ugk", "7vkr", 
    "7uo8", "7uq1", "7wcf", "7oas", 
    "8dik", "7te2", "8ajp", "7r0t", 
    "8dl3", "7u1o", "7qa0", "7pd7",
    "8dp7", "7z05", "7t7h", "7ulc"
]

class AF3Trainer(object):
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
        logging.info("Finished init ENV.")

    def init_loss(self):
        self.loss = ProtenixLoss(self.configs)
        self.symmetric_permutation = SymmetricPermutation(
            self.configs, error_dir=self.error_dir
        )
        self.lddt_metrics = LDDTMetrics(self.configs)

    def init_model(self):
        self.raw_model = ArtiosLigand(self.configs).to(self.device)
        self.use_ddp = False
        if DIST_WRAPPER.world_size > 1:
            self.print(f"Using DDP")
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

        if self.configs.get("power_law_ema_decay", []):  # Check if list is not empty
            self.power_law_ema_wrappers = []  # Store wrappers in a list
            for gamma in self.configs.power_law_ema_decay:
                wrapper = PowerLawEMAWrapper(
                    model=self.model,
                    gamma=gamma,
                    mutable_param_keywords=self.configs.ema_mutable_param_keywords,
                )
                wrapper.register()
                self.power_law_ema_wrappers.append(wrapper)
            self.print(f"Initialized {len(self.power_law_ema_wrappers)} power law EMA wrappers with decays: {self.configs.power_law_ema_decay}")

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

    def init_sampler(self): 
        #TODO: here we should perhaps initialise some sampler object with some sampler configs
        # To establish how we should formulate the sampler object let's basicalyl copy the evaluate 
        # method below which is exactly we should be doing a forward pass
        # They do this via a "batch". #TODO: determine how they get batch and what is stored inside it.
        pass

    def save_checkpoint(self, ema_suffix=""):
        if DIST_WRAPPER.rank == 0:
            if ema_suffix:
                path = f"{self.ema_checkpoint_dir}/{self.step}{ema_suffix}.pt"
            else:
                path = f"{self.checkpoint_dir}/{self.step}.pt"
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
                    self.print(f"Loading optimizer state")
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                if not skip_load_step:
                    self.print(f"Loading checkpoint step")
                    self.step = checkpoint["step"] + 1
                    self.start_step = self.step
                    self.global_step = self.step * self.iters_to_accumulate
                if not skip_load_scheduler:
                    self.print(f"Loading scheduler state")
                    self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
                else:
                    # reinitialize LR scheduler using the updated optimizer and step
                    self.init_scheduler(last_epoch=self.step - 1)
            self.print(f"Finish loading checkpoint, current step: {self.step}")

        if self.configs.load_ema_checkpoint_paths_directory:
            checkpoint_dir = self.configs.load_ema_checkpoint_paths_directory
            if not os.path.exists(checkpoint_dir):
                raise Exception(f"EMA checkpoint directory does not exist: {checkpoint_dir}")
            
            # Get all checkpoint files from directory
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            self.print(f"Found {len(checkpoint_files)} EMA checkpoints in {checkpoint_dir}")
            
            # Create mapping of gamma values to their corresponding wrappers
            gamma_to_wrapper = {wrapper.gamma: wrapper for wrapper in self.power_law_ema_wrappers}

            # Load each checkpoint and match with corresponding wrapper
            for checkpoint_file in checkpoint_files:
                # Extract gamma value from filename (assuming format like "*_ema_6.94.pt")
                try:
                    gamma_str = checkpoint_file.split('_ema_')[-1].replace('.pt', '')
                    gamma = float(gamma_str)
                except (IndexError, ValueError):
                    self.print(f"Warning: Could not extract a gamma value from checkpoint filename: {checkpoint_file}")
                    continue
                
                if gamma not in gamma_to_wrapper:
                    self.print(f"Warning: No matching wrapper found for gamma={gamma} in checkpoint: {checkpoint_file}")
                    continue
                
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
                _load_checkpoint(
                    checkpoint_path,
                    load_params_only=True,
                )
                
                # Register only the wrapper corresponding to this gamma
                matching_wrapper = gamma_to_wrapper[gamma]
                matching_wrapper.register()
                self.print(f"Loaded and registered EMA checkpoint for gamma={gamma}: {checkpoint_file}")

        # Load model parameters (With no EMA) from checkpoint 
        if self.configs.load_checkpoint_path:
            _load_checkpoint(
                self.configs.load_checkpoint_path,
                self.configs.load_params_only,
                skip_load_optimizer=self.configs.skip_load_optimizer,
                skip_load_scheduler=self.configs.skip_load_scheduler,
                skip_load_step=self.configs.skip_load_step,
            )

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            logging.info(msg)

    def model_forward(self, batch: dict, mode: str = "train") -> tuple[dict, dict]:
        assert mode in ["train", "eval"]
        batch["pred_dict"], batch["label_dict"], log_dict = self.model(
            input_feature_dict=batch["input_feature_dict"],
            label_dict=batch["label_dict"],
            label_full_dict=batch["label_full_dict"],
            mode=mode,
            current_step=self.step if mode == "train" else None,
            symmetric_permutation=self.symmetric_permutation,
        )
        return batch, log_dict

    def get_loss(
        self, batch: dict, mode: str = "train"
    ) -> tuple[torch.Tensor, dict, dict]:
        assert mode in ["train", "eval"]
        self.loss.to(batch["pred_dict"]["coordinate"].device)  # Move loss to device

        loss, loss_dict = autocasting_disable_decorator(self.configs.skip_amp.loss)(
            self.loss
        )(
            feat_dict=batch["input_feature_dict"],
            pred_dict=batch["pred_dict"],
            label_dict=batch["label_dict"],
            mode=mode,
        )
        return loss, loss_dict, batch

    @torch.no_grad()
    def get_metrics(self, batch: dict) -> dict:

        lddt_dict = self.lddt_metrics.compute_lddt(
            batch["pred_dict"], batch["label_dict"]
        )

        return lddt_dict

    @torch.no_grad()
    def aggregate_metrics(self, lddt_dict: dict, batch: dict) -> dict:

        simple_metrics, _ = self.lddt_metrics.aggregate_lddt(
            lddt_dict
        )

        return simple_metrics

    @torch.no_grad()
    def evaluate(self, mode: str = "eval"):
        all_eval_metrics = {}
        if not self.configs.eval_ema_only:
            # Evaluate None-EMA
            metrics = self._evaluate()
            all_eval_metrics["None-EMA"] = metrics
        if hasattr(self, "power_law_ema_wrappers"):
            # Evaluate regular EMA checkpoints
            for wrapper in self.power_law_ema_wrappers:
                wrapper.apply_shadow() 
                metrics = self._evaluate(ema_suffix=f"power_law_ema{wrapper.gamma}_", mode=mode) 
                all_eval_metrics[f"train_ema{wrapper.gamma}_"] = metrics
                wrapper.restore() 
            
        # Evaluate post-hoc EMA checkpoints
        if self.configs.calc_post_hoc_ema_during_training:
            # Perform post hoc ema calculation
            self.print(f"Calculating post-hoc EMA checkpoints")
            self.calc_post_hoc_ema()
            self.print(f"Finished calculating post-hoc EMA checkpoints")

            # Get post hoc ema checkpoint paths
            post_hoc_checkpoints = [f for f in os.listdir(self.post_hoc_ema_checkpoint_dir) 
                                    if f"post_hoc_ema" in f and f.endswith('.pt')]
            self.print(f"There are {len(post_hoc_checkpoints)} post-hoc EMA checkpoints to evaluate")

            # Store current model state
            current_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            
            for checkpoint_file in post_hoc_checkpoints:
                # Extract gamma value from filename
                gamma = float(checkpoint_file.split('post_hoc_ema')[-1].replace('.pt', ''))
                
                # Load and evaluate the post-hoc EMA checkpoint
                checkpoint_path = os.path.join(self.post_hoc_ema_checkpoint_dir, checkpoint_file)
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Load post-hoc EMA weights
                self.model.load_state_dict(checkpoint['model'])
                
                # Evaluate
                print(f"Beginning evaluation for post-hoc ema {gamma}")
                metrics = self._evaluate(ema_suffix=f"post_hoc_ema{gamma}_", mode=mode)
                all_eval_metrics[f"post_hoc_ema{gamma}_"] = metrics
                
                del checkpoint
                torch.cuda.empty_cache()
            
            # Restore original model state
            self.model.load_state_dict(current_state)

            # Plot all the metrics on wandb
            if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                # Create separate plots for each metric
                print("all_eval_metrics.keys(): ", all_eval_metrics.keys())
                print("The metrics are: ", next(iter(all_eval_metrics.values())).keys())

                for metric_name in next(iter(all_eval_metrics.values())).keys():
                    print("metric_name: ", metric_name)
                    metric_data = {
                        model_name: metrics[metric_name] 
                        for model_name, metrics in all_eval_metrics.items()
                    }
                    wandb.log({
                        f"comparison/{metric_name}": wandb.plot.line_series(
                            xs=list(range(len(metric_data))),
                            ys=list(metric_data.values()),
                            keys=list(metric_data.keys()),
                            title=f"Comparison of {metric_name}",
                            xname="Step"
                        )
                    })

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
        self.model.eval()

        for test_name, test_dl in self.test_dls.items():
            self.print(f"Testing on {test_name}")
            evaluated_pids = []
            total_batch_num = len(test_dl)
            for index, batch in enumerate(tqdm(test_dl)):
                ###
                # TEMPORARY RESTRICTION ON VALIDATION SIZE
                if index >= 40:
                    break
                ###

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
                    batch, _ = self.model_forward(batch, mode=mode)
                    # Loss forward - forms loss and loss_dict, additionally returns the batch
                    loss, loss_dict, batch = self.get_loss(batch, mode="train")  # eval, loss evaluated in the same way as train
                    # lDDT metrics TODO: now with our modified lddt calculations this will need fixing
                    #lddt_dict = self.get_metrics(batch)
                    #lddt_metrics = self.aggregate_metrics(lddt_dict, batch)
                    #simple_metrics.update(
                    #    {k: v for k, v in lddt_metrics.items() if "diff" not in k}
                    #)
                    simple_metrics.update(loss_dict)

                # Metrics
                for key, value in simple_metrics.items():
                    simple_metric_wrapper.add(
                        key, value, namespace=test_name
                    )
                
                # Visualise selected structures in wandb
                # FIXME: just temporary sampling logic for intial training runs
                if pid in PDB_TO_VISUALISE:
                    pdb_str = to_pdb(batch["pred_dict"]["coordinate_mini"], batch["input_feature_dict"])  # output of the mini-rollout
                    if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                        wandb.log({
                           f"{ema_suffix}structure_{pid}": wandb.Molecule(StringIO(pdb_str), file_type="pdb")
                        }, step=self.step)

                del batch, simple_metrics
                if index % 5 == 0:
                    # Release some memory periodically
                    torch.cuda.empty_cache()

            metrics = simple_metric_wrapper.calc()
            self.print(f"Step {self.step}, eval {test_name}: {metrics}")
            #if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
            #    wandb.log(metrics, step=self.step)

            return metrics

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
        Notes:
        This function could live outside of the trainer class.
        Having it elsewhere would allow for us to use it after training.
        In the case it lives outside, the inputs must include:
            - an ema_checkpoint_dir
            - the gamma's used during training
            - the gamma's wanted for post hoc ema
            - post hoc ema checkpoint dir
            - access to the current model's optimizer and lr_scheduler
        """
        # Create list of the ema checkpoint paths from the ema_checkpoint_dir
        ema_checkpoint_paths = [os.path.join(self.ema_checkpoint_dir, f) for f in os.listdir(self.ema_checkpoint_dir) if f.endswith('.pt')]
        if len(ema_checkpoint_paths) == 0:
            self.print(f"No ema checkpoints found in {self.ema_checkpoint_dir}")
            return
        self.print(f"Found {len(ema_checkpoint_paths)} ema checkpoints in {self.ema_checkpoint_dir}")

        # load the gamma and step for all checkpoints
        gamma_step_pairs = [extract_gamma_step(path) for path in ema_checkpoint_paths]

        # Create the timesteps, gamma values and post hoc gamma values to be given to solve_weights
        t_i = sorted(list(set(step for _, step in gamma_step_pairs)))  # The timesteps we have checkpoints at
        gamma_i = list(self.configs.power_law_ema_decay) # The gamma values we have checkpoints at
        gamma_r = list(self.configs.post_hoc_ema_decay) # The post hoc gamma values

        # Calculate the weights vector x_i to be ascrined to the checkpoints for each post hoc gamma value
        weights = solve_weights(t_i, gamma_i, gamma_r) # [len(t_i)*len(gamma_i), len(gamma_r)]

        # Calculate the post hoc ema network parameters for each gamma value
        for i, ph_gamma in enumerate(gamma_r):
            # Find each path's weight in the weights vector by ordering the weights vector to match the gamma_step_pairs
            ordered_weights = []
            for gamma, step in gamma_step_pairs:
                # Get gamma index and step index using list.index()
                gamma_index = gamma_i.index(gamma)
                step_index = t_i.index(step)
                # Get weight value
                weight = weights[gamma_index * len(t_i) + step_index, i]
                # Append weight to ordered_weights
                ordered_weights.append(weight)

            # Initialise post hoc ema parameters 
            final_weights = None

            # Process each checkpoint in turn to avoid memory issues
            for path, weight in zip(ema_checkpoint_paths, ordered_weights):
                checkpoint = torch.load(path)
                state_dict = checkpoint["model"]

                if final_weights is None:
                    # Initialize final_checkpoint with weighted values
                    final_weights = {k: v * weight for k, v in state_dict.items()}
                else:
                    # Accumulate weighted sum
                    for k in final_weights.keys():
                        final_weights[k] += state_dict[k] * weight

                del checkpoint  # Free memory
                torch.cuda.empty_cache()  # Clear GPU memory if needed

            # Save the final post hoc ema checkpoints
            path = f"{self.post_hoc_ema_checkpoint_dir}/{self.step}post_hoc_ema{ph_gamma}.pt"
            checkpoint = {
                "model": final_weights,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (
                    self.lr_scheduler.state_dict()
                    if self.lr_scheduler is not None
                    else None
                ),
                "step": self.step,
            }
            torch.save(checkpoint, path)
            self.print(f"Saved post hoc ema checkpoint to {path}")

    def run(self):
        """
        Main entry for the AF3Trainer.

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
                        wrapper.update(t=self.step+1)
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
                    self.save_checkpoint() # Saves non-ema checkpoint
                    if use_ema:
                        for wrapper in self.power_law_ema_wrappers:
                            wrapper.apply_shadow()
                            self.save_checkpoint(
                                ema_suffix=f"ema{wrapper.gamma}"
                            )
                            wrapper.restore()

                if step_need_eval or is_last_step:
                    self.evaluate()
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
    trainer = AF3Trainer(configs)
    trainer.run()


if __name__ == "__main__":
    main()
