import datetime
import hashlib
import logging
import os
import time
import warnings
from contextlib import nullcontext
from typing import Optional

import numpy as np
import rootutils
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from artios.config import parse_configs, parse_sys_args
from artios.config.config import save_config
from artios.data.dataloader import get_dataloaders
from artios.data.features import FullContext
from artios.metrics.lddt_metrics import LDDTMetrics
from artios.model.artios_backbone import ArtiosBackbone
from artios.model.output import ModelOutput
from artios.model.utils import Mode
from artios.utils.distributed import DIST_WRAPPER
from artios.utils.inference import ModelType
from artios.utils.lr_scheduler import get_lr_scheduler
from artios.utils.metrics import SimpleMetricAggregator
from artios.utils.monitoring import grad_norm
from artios.utils.profiler import profile_function
from artios.utils.seed import seed_everything
from artios.utils.training import flatten_nested, get_optimizer
from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from runner.ema import EMAWrapper
from runner.inference import InferenceRunner, infer_predict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Disable WANDB's console output capture to reduce unnecessary logging
os.environ["WANDB_CONSOLE"] = "off"


class ArtiosBackboneTrainer(object):
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
        self.training_samples_dir = f"{self.run_dir}/training_samples"
        self.dump_dir = f"{self.run_dir}/dumps"
        self.error_dir = f"{self.run_dir}/errors"
        self.logs_dir = f"{self.run_dir}/logs"

        # TODO: Re-add DataDumper when inference output saving is needed
        self.configs.dump_dir = self.prediction_dir

        if DIST_WRAPPER.rank == 0:
            os.makedirs(self.run_dir)
            os.makedirs(self.checkpoint_dir)
            os.makedirs(self.ema_checkpoint_dir)
            os.makedirs(self.prediction_dir)
            os.makedirs(self.training_samples_dir)
            os.makedirs(self.dump_dir)
            os.makedirs(self.error_dir)
            os.makedirs(self.logs_dir)
            save_config(
                self.configs,
                os.path.join(self.configs.base_dir, self.run_name, "config.yaml"),
            )

        self.print(
            f"Using run name: {self.run_name}, run dir: {self.run_dir}, checkpoint_dir: "
            + f"{self.checkpoint_dir}, ema_checkpoint_dir: {self.ema_checkpoint_dir}, prediction_dir: {self.prediction_dir}, "
            + f"{self.error_dir} training_samples_dir: {self.training_samples_dir}"
        )

    def init_log(self):
        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
            wandb.init(
                project=self.configs.project,
                name=self.run_name,
                config=self.configs.to_dict(),
                id=self.configs.wandb_id if self.configs.wandb_id else None,
                resume="allow" if self.configs.wandb_id else None,
            )
        self.train_metric_wrapper = SimpleMetricAggregator(["nonzero_avg"])

    def init_env(self):
        """Init pytorch/cuda envs."""
        logging.info(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            + f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        mp.set_sharing_strategy("file_system")
        # mp.set_start_method("spawn", force=True)

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
        # Use rank-specific seed for different augmentations across DDP processes
        if not self.configs.deterministic_seed:
            hash_string = f"({self.configs.seed},{DIST_WRAPPER.rank},init_seed)"
            rank_seed = int(hashlib.sha256(hash_string.encode("utf8")).hexdigest(), 16)
            rank_seed = rank_seed % (2**32)
        else:
            rank_seed = self.configs.seed
        seed_everything(
            seed=rank_seed,
            deterministic=self.configs.deterministic,
        )

        # Set high precision for tensor core matmul using new API (PyTorch 2.9+)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logging.info("Finished init ENV.")

        # Suppress annoying atomworks warnings
        warnings.filterwarnings("ignore", module="atomworks")

    def init_loss(self):
        self.lddt_metrics = LDDTMetrics(self.configs)

    def init_model(self):
        self.raw_backbone_model = ArtiosBackbone(self.configs).to(self.device)
        # TODO: we might need to initialize the packing model here.
        # DDP not required for VAE loading since model is frozen - no need for gradient aggregation
        self.use_ddp = False
        if DIST_WRAPPER.world_size > 1:
            self.print("Using DDP")
            self.use_ddp = True
            # Fix DDP/checkpoint https://discuss.pytorch.org/t/ddp-and-gradient-checkpointing/132244
            self.backbone_model = DDP(
                self.raw_backbone_model,
                find_unused_parameters=self.configs.find_unused_parameters,
                device_ids=[DIST_WRAPPER.local_rank],
                output_device=DIST_WRAPPER.local_rank,
                static_graph=True,
            )
        else:
            self.backbone_model = self.raw_backbone_model

        if self.configs.get("ema_decay", -1) > 0:
            assert self.configs.ema_decay < 1
            self.ema_wrapper = EMAWrapper(
                self.backbone_model,
                self.configs.ema_decay,
                self.configs.ema_mutable_param_keywords,
            )
            self.ema_wrapper.register()

        torch.cuda.empty_cache()
        self.optimizer = get_optimizer(self.configs, self.backbone_model)
        self.init_scheduler()

        # Initialize GradScaler once for use across all training steps
        self.scaler = torch.GradScaler(
            device="cuda" if self.use_cuda else "cpu",
            enabled=(self.configs.dtype == "fp16"),
        )

    def init_scheduler(self, **kwargs):
        self.lr_scheduler = get_lr_scheduler(self.configs, self.optimizer, **kwargs)

    def init_data(self):
        self.print("beginning init_data...")
        train_dl, self.test_dls = get_dataloaders(
            self.configs,
            DIST_WRAPPER.world_size,
            seed=self.configs.seed,
            error_dir=self.error_dir,
        )

        self.train_dl = train_dl
        self.print("finished init_data")

    def save_checkpoint(self, ema_suffix="", save_model_only=False):
        if DIST_WRAPPER.rank == 0:
            if ema_suffix:
                path = f"{self.ema_checkpoint_dir}/{self.step}_{ema_suffix}.pt"
            else:
                path = f"{self.checkpoint_dir}/{self.step}.pt"

            if not save_model_only:
                # Save model, optimizer, scheduler, and step
                checkpoint = {
                    "model": self.backbone_model.state_dict(),
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
                checkpoint = {
                    "model": self.backbone_model.state_dict(),
                    "step": self.step,
                }

            torch.save(checkpoint, path)
            self.print(f"Saved checkpoint to {path}")

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

        # Load model parameters from checkpoint
        if self.configs.load_backbone_checkpoint_path:
            self.backbone_model = _load_checkpoint(
                self.configs.load_backbone_checkpoint_path,
                self.configs.load_params_only,
                model=self.backbone_model,
                skip_load_optimizer=self.configs.skip_load_optimizer,
                skip_load_scheduler=self.configs.skip_load_scheduler,
                skip_load_step=self.configs.skip_load_step,
            )

        # Load EMA shadow: from a dedicated checkpoint if provided,
        # otherwise re-register from the loaded backbone weights
        if hasattr(self, "ema_wrapper"):
            if self.configs.load_ema_checkpoint_path:
                _load_checkpoint(
                    self.configs.load_ema_checkpoint_path,
                    load_params_only=True,
                    model=self.backbone_model,
                )
            self.ema_wrapper.register()

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            logging.info(msg)

    @profile_function("model_forward")
    def model_forward(
        self,
        context: FullContext,
        mode: Mode = Mode.TRAIN,
        seed: Optional[int] = None,
    ) -> ModelOutput:
        """Forward pass of the model.

        Args:
            context: Full context containing all features and labels.
            mode: Mode of the forward pass (TRAIN or EVAL).
            seed: Seed for random number generation. Defaults to None.

        Returns:
            ModelOutput containing the prediction.
        """
        output = self.backbone_model(context=context, mode=mode, seed=seed)
        return output

    @profile_function("get_loss")
    def get_loss(
        self,
        output: ModelOutput,
        context: FullContext,
    ) -> tuple[torch.Tensor, dict]:
        """Compute loss from model output and context.

        Args:
            output: Model output containing predictions.
            context: Full context containing labels and masks.

        Returns:
            Tuple of (loss tensor, loss metrics dict).
        """
        return self.raw_backbone_model.compute_loss(output, context)

    @torch.no_grad()
    def evaluate(self):
        """Run evaluation on all test datasets."""
        if not self.configs.eval_ema_only:
            self._evaluate(ema_suffix="")
        if hasattr(self, "ema_wrapper"):
            self.ema_wrapper.apply_shadow()
            self._evaluate(ema_suffix=f"ema_{self.ema_wrapper.decay}")
            self.ema_wrapper.restore()

    @torch.no_grad()
    def _evaluate(self, ema_suffix: str = "") -> dict:
        """Run evaluation loop on all test datasets.

        Args:
            ema_suffix: Suffix for logging (e.g., "post_hoc_ema_0.28" or "").

        Returns:
            Dictionary of evaluation metrics.
        """
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

        self.backbone_model.eval()

        for test_name, test_dl in self.test_dls.items():
            self.print(f"Testing on {test_name}")

            for index, context in enumerate(tqdm(test_dl)):
                context = context.to(self.device)

                with enable_amp:
                    output = self.model_forward(context, mode=Mode.EVAL, seed=index)
                    _, loss_dict = self.get_loss(output, context)

                for key, value in loss_dict.items():
                    simple_metric_wrapper.add(key, value, namespace="val")

                del context, output
                if index % 5 == 0:
                    torch.cuda.empty_cache()

            metrics = simple_metric_wrapper.calc()
            self.print(f"Step {self.step}, eval {test_name}: {metrics}")

            # Log validation metrics to WandB
            if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                prefix = f"{ema_suffix}/" if ema_suffix else ""
                prefixed_metrics = {
                    f"{prefix}{test_name}/{k}": v for k, v in metrics.items()
                }
                wandb.log(prefixed_metrics, step=self.step)

        return metrics

    def update(self):
        # Clip the gradient
        if self.configs.grad_clip_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.backbone_model.parameters(), self.configs.grad_clip_norm
            )

    @profile_function("backward_pass")
    def backward_pass(self, loss):
        self.scaler.scale(loss / self.iters_to_accumulate).backward()

    @profile_function("optimizer_step")
    def optimizer_step(self):
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    @profile_function("train_step")
    def train_step(self, context: FullContext):
        """Execute a single training step.

        Args:
            context: Full context containing features and labels.
        """
        self.backbone_model.train()

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

        with enable_amp:
            output = self.model_forward(context, mode=Mode.TRAIN)
            loss, loss_dict = self.get_loss(output, context)

        # if self.configs.dtype in ["bf16", "fp32"]:
        #     if is_loss_nan_check(loss):
        #         self.print(f"Skip iteration with NaN loss: {self.step} steps")
        #         loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        self.backward_pass(loss)

        # For simplicity, the global training step is used
        if (self.global_step + 1) % self.iters_to_accumulate == 0:
            self.print(
                f"self.step {self.step}, self.iters_to_accumulate: {self.iters_to_accumulate}"
            )
            # Unscales the gradients of optimizer's assigned parameters in-place
            self.scaler.unscale_(self.optimizer)
            self.on_before_optimizer_step()
            # Do grad clip only
            self.update()
            self.optimizer_step()
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

    def _run_inference_and_log(self):
        """
        Runs inference during evaluation.
        """
        self.configs.dump_dir = os.path.join(self.prediction_dir, f"step_{self.step}")

        inference_runner = InferenceRunner(
            self.configs,
            ModelType.BACKBONE,
            model=self.backbone_model,
        )

        self.print(f"Running inference for step {self.step}...")
        infer_predict(
            inference_runner,
            self.configs,
            use_distributed_sampler=False,
        )

    def on_validation_epoch_end(self):
        """Hook that is called at the end of each validation epoch."""
        if (
            not self.configs.use_wandb
            or DIST_WRAPPER.rank != 0
            or not self.configs.monitor.weight_norm
        ):
            return

        # Perform inference during evaluation
        if self.configs.run_training_time_inference and DIST_WRAPPER.rank == 0:
            self._run_inference_and_log()

        # Calculate and log weight norms for each parameter
        weight_norms = {}
        for name, param in self.backbone_model.named_parameters():
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
                wandb.log(flatten_nested(weight_norms), step=self.step)

    @profile_function("on_before_optimizer_step")
    def on_before_optimizer_step(self):
        """Hook that is called before the optimizer step."""
        if self.configs.monitor.grad_norm:
            _, total_norm = grad_norm(self.backbone_model, norm_type=2)

            if DIST_WRAPPER.rank == 0:
                wandb.log({"grad_norm": total_norm}, step=self.step)

    def save_training_sample(
        self,
        context: FullContext,
        motif_as_b_factor: bool = False,
    ):
        """Saves a training sample to the training samples directory.

        Args:
            context: Full context containing atom arrays and pdb ids.
            motif_as_b_factor: If True, set b_factor to is_motif annotation.

        Note: This method is a no-op if atom_array is None (disabled for performance).
        """
        atom_arrays = context.basic.atom_array
        if atom_arrays is None:
            return

        from biotite.structure.io import save_structure

        pdb_ids = context.basic.pdb_id

        for idx, atom_array in enumerate(atom_arrays):
            if atom_array is None:
                continue
            pdb_id = pdb_ids[idx]
            if motif_as_b_factor:
                atom_array.b_factor = atom_array.is_motif
            save_structure(
                os.path.join(self.training_samples_dir, f"{pdb_id}.cif"), atom_array
            )

    def run(self):
        """
        Main entry for the ArtiosBackboneTrainer.

        This function handles the training process, evaluation, logging, and checkpoint saving.
        """
        if self.configs.eval_only or self.configs.eval_first:
            self.evaluate()
            if self.configs.eval_only:
                return
        use_ema = hasattr(self, "ema_wrapper")
        self.print(f"Using EMA: {use_ema}")

        enable_profiling = self.configs.get("enable_profiling", False)

        def trace_handler(p):
            output = p.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=20
            )
            self.print(output)
            if DIST_WRAPPER.rank == 0:
                tracename = f"trace_{p.step_num}_rank{DIST_WRAPPER.rank}.json"
                filepath = f"{self.logs_dir}/{tracename}"
                os.makedirs(self.logs_dir, exist_ok=True)
                p.export_chrome_trace(filepath)

        profiler_context = (
            torch.profiler.profile(
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
            )
            if enable_profiling
            else nullcontext()
        )

        with profiler_context as p:
            while True:
                for context in self.train_dl:
                    context = context.to(self.device, non_blocking=True)

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

                    is_save_training_sample_step = self.global_step < 20

                    is_last_step &= is_update_step
                    step_need_log &= is_update_step
                    step_need_eval &= is_update_step
                    step_need_save &= is_update_step

                    # Note: context moved to device above

                    if is_save_training_sample_step:
                        self.save_training_sample(context, motif_as_b_factor=True)

                    self.progress_bar()
                    self.train_step(context)

                    if use_ema and is_update_step:
                        self.ema_wrapper.update()

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
                        self.print(f"Saving checkpoint at step {self.step}")
                        self.save_checkpoint()
                        if use_ema:
                            self.ema_wrapper.apply_shadow()
                            self.save_checkpoint(
                                ema_suffix=f"ema_{self.ema_wrapper.decay}",
                            )
                            self.ema_wrapper.restore()

                    if step_need_eval or is_last_step:
                        self.evaluate()
                        self.on_validation_epoch_end()

                    self.global_step += 1
                    if self.global_step % self.iters_to_accumulate == 0:
                        self.step += 1

                    if self.step >= self.configs.max_steps:
                        self.print(f"Finish training after {self.step} steps")
                        break

                    if enable_profiling:
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
    trainer = ArtiosBackboneTrainer(configs)
    trainer.run()


if __name__ == "__main__":
    main()
