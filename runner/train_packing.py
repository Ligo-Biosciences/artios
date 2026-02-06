import datetime
import logging
import os
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import biotite.structure as bs
import numpy as np
import rootutils
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from biotite.structure import AtomArray
from biotite.structure.io import load_structure
from neptune_scale import Run
from neptune_scale.api.run import SourceTrackingConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from protenix.common.constants import BACKBONE_DEFS
from protenix.config import parse_configs, parse_sys_args
from protenix.config.config import save_config
from protenix.data.dataloader import get_dataloaders
from protenix.data.features import FullContext
from protenix.metrics.lddt_metrics import LDDTMetrics

# from protenix.metrics.rmsd import compute_per_atom_rmsf
from protenix.model.artios_packing import ArtiosPacking
from protenix.model.modules.sidechain_packer import SidechainPacker
from protenix.model.modules.transformer import DiffusionTransformer, DiTBlock
from protenix.model.output import ModelOutput
from protenix.model.utils import Mode
from protenix.utils.distributed import DIST_WRAPPER, wait_for_all_processes
from protenix.utils.lr_scheduler import get_lr_scheduler
from protenix.utils.metrics import SimpleMetricAggregator
from protenix.utils.monitoring import grad_norm
from protenix.utils.profiler import profile_function
from protenix.utils.seed import seed_everything
from protenix.utils.torchwatcher import TorchWatcher
from protenix.utils.training import flatten_nested, get_optimizer, is_loss_nan_check
from runner.ema import EMAWrapper

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class ArtiosPackingTrainer(object):
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
        self.neptune_run = None
        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
            source_config = SourceTrackingConfig(
                namespace="source_code",
                upload_entry_point=True,
                upload_diff_head=True,
                upload_diff_upstream=True,
                upload_run_command=True,
            )
            self.neptune_run = Run(
                project=self.configs.project,
                experiment_name=self.run_name,
                source_tracking_config=source_config,
                run_id=self.configs.neptune_run_id
                if self.configs.neptune_run_id
                else None,
                fork_run_id=self.configs.neptune_fork_run_id
                if self.configs.neptune_fork_run_id
                else None,
                fork_step=int(self.configs.neptune_fork_step)
                if self.configs.neptune_fork_step
                else None,
            )
            self.torchwatcher = TorchWatcher(
                model=self.packing_model.module
                if DIST_WRAPPER.world_size > 1
                else self.packing_model,
                run=self.neptune_run,
                track_layers=[
                    DiffusionTransformer,
                    DiTBlock,
                    SidechainPacker,
                ],
                track_layers_name_includes=["linear_q", "linear_k"],
                tensor_stats=["channel_rms"],
                base_namespace="model_metrics",
            )
            self.neptune_run.log_configs(flatten_nested(self.configs.to_dict()))
        self.train_metric_wrapper = SimpleMetricAggregator(["nonzero_avg"])

    def init_env(self):
        """Init pytorch/cuda envs."""
        logging.info(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            + f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        mp.set_sharing_strategy("file_system")

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

        # Set high precision for tensor core matmul using new API (PyTorch 2.9+)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logging.info("Finished init ENV.")

        # Suppress annoying atomworks warnings
        warnings.filterwarnings("ignore", module="atomworks")

    def init_loss(self):
        self.lddt_metrics = LDDTMetrics(self.configs)

    def init_model(self):
        self.raw_packing_model = ArtiosPacking(self.configs).to(self.device)
        self.use_ddp = False
        if DIST_WRAPPER.world_size > 1:
            self.print("Using DDP")
            self.use_ddp = True
            self.packing_model = DDP(
                self.raw_packing_model,
                find_unused_parameters=self.configs.find_unused_parameters,
                device_ids=[DIST_WRAPPER.local_rank],
                output_device=DIST_WRAPPER.local_rank,
                static_graph=True,
            )
        else:
            self.packing_model = self.raw_packing_model

        if self.configs.get("ema_decay", None):
            self.ema_wrapper = EMAWrapper(
                model=self.packing_model,
                decay=self.configs.ema_decay,
                mutable_param_keywords=self.configs.ema_mutable_param_keywords,
            )
            self.ema_wrapper.register()
            self.print(f"Initialized EMA wrapper with decay: {self.configs.ema_decay}")

        torch.cuda.empty_cache()
        self.optimizer = get_optimizer(self.configs, self.packing_model)
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
        self.train_dl, self.test_dls = get_dataloaders(
            self.configs,
            DIST_WRAPPER.world_size,
            seed=self.configs.seed,
            error_dir=self.error_dir,
        )
        self.print("finished init_data")

    def save_checkpoint(self, ema_suffix="", save_model_only=False):
        if DIST_WRAPPER.rank == 0:
            if ema_suffix:
                path = f"{self.ema_checkpoint_dir}/{self.step}_{ema_suffix}.pt"
            else:
                path = f"{self.checkpoint_dir}/{self.step}.pt"

            if not save_model_only:
                checkpoint = {
                    "model": self.packing_model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": (
                        self.lr_scheduler.state_dict()
                        if self.lr_scheduler is not None
                        else None
                    ),
                    "step": self.step,
                }
            else:
                checkpoint = {
                    "model": self.packing_model.state_dict(),
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
                    self.init_scheduler(last_epoch=self.step - 1)
            self.print(f"Finish loading checkpoint, current step: {self.step}")

            return model

        # Load EMA checkpoint if specified
        if self.configs.get("load_ema_checkpoint_path"):
            ema_checkpoint_path = self.configs.load_ema_checkpoint_path
            if not os.path.exists(ema_checkpoint_path):
                raise Exception(f"EMA checkpoint does not exist: {ema_checkpoint_path}")
            _load_checkpoint(
                ema_checkpoint_path, load_params_only=True, model=self.packing_model
            )
            if hasattr(self, "ema_wrapper"):
                self.ema_wrapper.register()
            self.print(f"Loaded EMA checkpoint from {ema_checkpoint_path}")

        # Load model parameters from checkpoint
        if self.configs.load_packing_checkpoint_path:
            self.packing_model = _load_checkpoint(
                self.configs.load_packing_checkpoint_path,
                self.configs.load_params_only,
                model=self.packing_model,
                skip_load_optimizer=self.configs.skip_load_optimizer,
                skip_load_scheduler=self.configs.skip_load_scheduler,
                skip_load_step=self.configs.skip_load_step,
            )

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
        output = self.packing_model(context=context, mode=mode, seed=seed)
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
        return self.raw_packing_model.compute_loss(output, context)

    @torch.no_grad()
    def evaluate(self):
        """Run evaluation on all test datasets, including EMA if configured."""
        if not self.configs.get("eval_ema_only", False):
            self._evaluate()

        # Run EMA evaluation if EMA wrapper is configured
        use_ema = hasattr(self, "ema_wrapper")
        if use_ema:
            self.ema_wrapper.apply_shadow()
            self.print("Beginning evaluation with EMA weights")
            self._evaluate(ema_suffix=f"ema_{self.configs.ema_decay}")
            self.ema_wrapper.restore()

    @torch.no_grad()
    def _evaluate(self, ema_suffix: str = "") -> dict:
        """Run evaluation loop on all test datasets.

        Args:
            ema_suffix: Suffix for logging (e.g., "ema_0.999" or "none-EMA").

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

        self.packing_model.eval()

        # Get inference config
        run_inference = self.configs.get("run_training_time_inference", False)
        inference_log_interval = self.configs.get("inference_log_interval", 10)

        for test_name, test_dl in self.test_dls.items():
            self.print(f"Testing on {test_name}")

            for index, context in enumerate(tqdm(test_dl)):
                context = context.to(self.device)

                with enable_amp:
                    output = self.model_forward(context, mode=Mode.EVAL, seed=index)
                    _, loss_dict = self.get_loss(output, context)

                for key, value in loss_dict.items():
                    simple_metric_wrapper.add(key, value, namespace="val")

                # Training-time inference for a subset of samples
                is_logging_element = (index + 1) % inference_log_interval == 0
                if run_inference and is_logging_element:
                    self._run_eval_set_inference_and_log(
                        context=context,
                        test_name=test_name,
                        ema_suffix=ema_suffix,
                        enable_amp=enable_amp,
                    )

                del context, output
                if index % 5 == 0:
                    torch.cuda.empty_cache()

            metrics = simple_metric_wrapper.calc()
            self.print(f"Step {self.step}, eval {test_name}: {metrics}")

            # Log validation metrics to Neptune
            if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                prefix = f"{ema_suffix}/" if ema_suffix else ""
                prefixed_metrics = {
                    f"{prefix}{test_name}/{k}": v for k, v in metrics.items()
                }
                self.neptune_run.log_metrics(prefixed_metrics, step=self.step)

        return metrics

    # def _compute_rmsf_metrics(
    #     self,
    #     structure_paths: list[Path],
    # ) -> dict[str, float]:
    #     """Compute overall and backbone RMSF metrics from a list of packed structures.

    #     Args:
    #         structure_paths: List of paths to CIF files (packed samples).

    #     Returns:
    #         Dict with 'rmsf_overall' and 'rmsf_backbone' keys.
    #     """
    #     structures = [load_structure(str(p)) for p in structure_paths]

    #     if len(structures) < 2:
    #         return {"rmsf_overall": 0.0, "rmsf_backbone": 0.0}

    #     reference = structures[0]
    #     aligned_structures = [reference]
    #     for structure in structures[1:]:
    #         superimposed, _ = bs.superimpose(reference, structure)
    #         aligned_structures.append(superimposed)

    #     per_atom_rmsf = compute_per_atom_rmsf(aligned_structures)

    #     rmsf_overall = float(np.mean(per_atom_rmsf))

    #     backbone_mask = np.isin(reference.atom_name, BACKBONE_DEFS["protein"])
    #     backbone_rmsf = per_atom_rmsf[backbone_mask]
    #     rmsf_backbone = float(np.mean(backbone_rmsf)) if len(backbone_rmsf) > 0 else 0.0

    #     return {"rmsf_overall": rmsf_overall, "rmsf_backbone": rmsf_backbone}

    def _compute_rmsd_to_ground_truth(
        self,
        structure_paths: list[Path],
        ground_truth: AtomArray,
    ) -> dict[str, float]:
        """Compute RMSD between predicted structures and ground truth, averaged over samples.

        Args:
            structure_paths: List of paths to predicted CIF files.
            ground_truth: Ground truth AtomArray from the context.

        Returns:
            Dict with 'rmsd_overall' and 'rmsd_backbone' keys (averaged over samples).
        """
        if len(structure_paths) == 0:
            return {"rmsd_overall": 0.0, "rmsd_backbone": 0.0}

        structures = [load_structure(str(p)) for p in structure_paths]

        backbone_atoms = BACKBONE_DEFS["protein"]
        gt_backbone_mask = np.isin(ground_truth.atom_name, backbone_atoms)

        rmsd_overall_list = []
        rmsd_backbone_list = []

        for pred_structure in structures:
            # Align predicted to ground truth
            superimposed, _ = bs.superimpose(ground_truth, pred_structure)

            # Compute overall RMSD
            diff = superimposed.coord - ground_truth.coord
            per_atom_sq_dist = np.sum(diff**2, axis=-1)
            rmsd_overall = float(np.sqrt(np.mean(per_atom_sq_dist)))
            rmsd_overall_list.append(rmsd_overall)

            # Compute backbone RMSD
            if np.sum(gt_backbone_mask) > 0:
                backbone_sq_dist = per_atom_sq_dist[gt_backbone_mask]
                rmsd_backbone = float(np.sqrt(np.mean(backbone_sq_dist)))
                rmsd_backbone_list.append(rmsd_backbone)

        avg_rmsd_overall = (
            float(np.mean(rmsd_overall_list)) if rmsd_overall_list else 0.0
        )
        avg_rmsd_backbone = (
            float(np.mean(rmsd_backbone_list)) if rmsd_backbone_list else 0.0
        )

        return {"rmsd_overall": avg_rmsd_overall, "rmsd_backbone": avg_rmsd_backbone}

    # @torch.no_grad()
    # def _run_full_inference_and_log(self, log_outputs_to_neptune: bool = True) -> None:
    #     """Run inference on a batch and log structures to Neptune.

    #     Args:
    #         log_outputs_to_neptune: Whether to log outputs to Neptune.
    #     """
    #     step_dir = os.path.join(self.prediction_dir, f"step_{self.step}")
    #     self.configs.dump_dir = step_dir

    #     ### Run inference
    #     inference_runner = InferenceRunner(
    #         self.configs,
    #         ModelType.PACKING,
    #         model=self.packing_model,
    #     )

    #     self.print(f"Running inference for step {self.step}...")
    #     infer_predict(
    #         inference_runner,
    #         self.configs,
    #         use_distributed_sampler=False,
    #     )

    #     if not (log_outputs_to_neptune and self.configs.use_wandb):
    #         return

    #     filepaths = glob(os.path.join(step_dir, "**/*.cif"), recursive=True)
    #     self.print(f"Found {len(filepaths)} files to log to Neptune")
    #     files_dict = {f"structures/{Path(fp).name}": fp for fp in filepaths}
    #     self.neptune_run.log_files(files=files_dict, step=self.step)

    #     ### Compute RMSF metrics
    #     with open(self.configs.input_json_path, "r") as f:
    #         inference_samples = json.load(f)
    #     sample_names = [sample["name"] for sample in inference_samples]

    #     n_sample = self.configs.artios_packing.sample_diffusion.N_sample

    #     all_rmsf_metrics = {}
    #     for sample_name in sample_names:
    #         structure_paths = []
    #         for sample_idx in range(n_sample):
    #             sample_dir = os.path.join(step_dir, str(sample_idx))
    #             cif_pattern = os.path.join(sample_dir, f"{sample_name}*.cif")
    #             matching_files = glob(cif_pattern)
    #             structure_paths.extend([Path(f) for f in matching_files])

    #         if len(structure_paths) < 2:
    #             self.print(
    #                 f"Skipping RMSF for {sample_name}: found {len(structure_paths)} structures"
    #             )
    #             continue

    #         try:
    #             rmsf_metrics = self._compute_rmsf_metrics(structure_paths)
    #             all_rmsf_metrics[f"inference/{sample_name}/rmsf_overall"] = (
    #                 rmsf_metrics["rmsf_overall"]
    #             )
    #             all_rmsf_metrics[f"inference/{sample_name}/rmsf_backbone"] = (
    #                 rmsf_metrics["rmsf_backbone"]
    #             )
    #             self.print(
    #                 f"  {sample_name}: overall={rmsf_metrics['rmsf_overall']:.4f}, "
    #                 f"backbone={rmsf_metrics['rmsf_backbone']:.4f}"
    #             )
    #         except Exception as e:
    #             self.print(f"Error computing RMSF for {sample_name}: {e}")

    #     if all_rmsf_metrics:
    #         self.neptune_run.log_metrics(all_rmsf_metrics, step=self.step)
    #         self.print(f"Logged RMSF metrics for {len(sample_names)} samples")

    @torch.no_grad()
    def _run_eval_set_inference_and_log(
        self,
        context: FullContext,
        test_name: str,
        ema_suffix: str,
        enable_amp,
    ) -> None:
        """Run inference on a batch and log structures to Neptune.

        Args:
            context: Full context for inference (may be batched).
            test_name: Name of the test dataset.
            ema_suffix: EMA suffix for logging.
            enable_amp: Autocast context for mixed precision.
        """
        # Get pdb_ids and atom_arrays - ensure they're lists
        pdb_ids = context.basic.pdb_id
        if not isinstance(pdb_ids, list):
            pdb_ids = [pdb_ids]

        atom_arrays = context.basic.atom_array
        if not isinstance(atom_arrays, list):
            atom_arrays = [atom_arrays]

        # Check if we have any valid atom arrays
        valid_count = sum(1 for aa in atom_arrays if aa is not None)
        if valid_count == 0:
            self.print("Skipping inference: all atom_arrays are None")
            return

        try:
            with enable_amp:
                # Run inference (sampling) to get full structures
                inference_output = self.model_forward(
                    context, mode=Mode.INFERENCE, seed=42
                )

            if DIST_WRAPPER.rank == 0:
                # Create output directory
                prefix = f"{ema_suffix}_" if ema_suffix else ""
                output_dir = os.path.join(
                    self.prediction_dir,
                    test_name,
                    f"step_{self.step}",
                )
                if prefix:
                    output_dir = os.path.join(output_dir, prefix.rstrip("_"))

                # Save structures to CIF files (handles full batch)
                all_saved_files = inference_output.save_samples_to_cif(
                    atom_arrays=atom_arrays,
                    output_dir=output_dir,
                    pdb_ids=pdb_ids,
                )

                # Log structure files to Neptune
                if self.configs.use_wandb and all_saved_files:
                    # Create mapping from pdb_id to ground truth atom_array
                    pdb_id_to_atom_array = dict(zip(pdb_ids, atom_arrays))

                    all_metrics = {}
                    files_dict = {}
                    total_files = 0
                    for pdb_id, saved_files in all_saved_files.items():
                        for file_path in saved_files:
                            filename = os.path.basename(file_path)
                            neptune_key = (
                                f"structures/{prefix}{test_name}/{pdb_id}/{filename}"
                            )
                            files_dict[neptune_key] = file_path
                            total_files += 1

                        # Calc and log RMSF metrics (variance across samples)
                        rmsf_metrics = self._compute_rmsf_metrics(saved_files)
                        all_metrics[
                            f"inference/{prefix}{test_name}/{pdb_id}/rmsf_overall"
                        ] = rmsf_metrics["rmsf_overall"]
                        all_metrics[
                            f"inference/{prefix}{test_name}/{pdb_id}/rmsf_backbone"
                        ] = rmsf_metrics["rmsf_backbone"]

                        # Calc and log RMSD to ground truth (averaged over samples)
                        ground_truth = pdb_id_to_atom_array.get(pdb_id)
                        if ground_truth is not None:
                            rmsd_metrics = self._compute_rmsd_to_ground_truth(
                                saved_files, ground_truth
                            )
                            all_metrics[
                                f"inference/{prefix}{test_name}/{pdb_id}/rmsd_overall"
                            ] = rmsd_metrics["rmsd_overall"]
                            all_metrics[
                                f"inference/{prefix}{test_name}/{pdb_id}/rmsd_backbone"
                            ] = rmsd_metrics["rmsd_backbone"]

                            self.print(
                                f"  {pdb_id}: rmsf_overall={rmsf_metrics['rmsf_overall']:.4f}, "
                                f"rmsf_backbone={rmsf_metrics['rmsf_backbone']:.4f}, "
                                f"rmsd_overall={rmsd_metrics['rmsd_overall']:.4f}, "
                                f"rmsd_backbone={rmsd_metrics['rmsd_backbone']:.4f}"
                            )
                        else:
                            self.print(
                                f"  {pdb_id}: rmsf_overall={rmsf_metrics['rmsf_overall']:.4f}, "
                                f"rmsf_backbone={rmsf_metrics['rmsf_backbone']:.4f} "
                                f"(no ground truth for RMSD)"
                            )

                        self.neptune_run.log_metrics(all_metrics, step=self.step)
                        self.print(f"Logged metrics for {pdb_id}")

                    if files_dict:
                        self.neptune_run.log_files(files=files_dict, step=self.step)
                        self.print(
                            f"Logged {total_files} structure files for "
                            f"{len(all_saved_files)} structures"
                        )

        except Exception as e:
            self.print(f"Failed inference/logging for batch: {e}")

    def update(self):
        # Clip the gradient
        if self.configs.grad_clip_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.packing_model.parameters(), self.configs.grad_clip_norm
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
        self.packing_model.train()

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

        if self.configs.dtype in ["bf16", "fp32"]:
            if is_loss_nan_check(loss):
                self.print(f"Skip iteration with NaN loss: {self.step} steps")
                loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        self.backward_pass(loss)

        if (self.global_step + 1) % self.iters_to_accumulate == 0:
            self.print(
                f"self.step {self.step}, self.iters_to_accumulate: {self.iters_to_accumulate}"
            )
            self.scaler.unscale_(self.optimizer)
            self.on_before_optimizer_step()
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

        if self.configs.run_training_time_inference and DIST_WRAPPER.rank == 0:
            # self._run_full_inference_and_log(log_outputs_to_neptune=True)
            pass

        weight_norms = {}
        for name, param in self.packing_model.named_parameters():
            if not param.requires_grad:
                continue

            if len(param.shape) == 2:
                fan_in_norms = torch.norm(param, p=2, dim=1)
                norm = torch.mean(fan_in_norms).item()
            else:
                norm = param.norm().item() / np.sqrt(param.numel())
            weight_norms[f"weight_norms/{name}"] = norm

            if DIST_WRAPPER.rank == 0:
                self.neptune_run.log_metrics(
                    flatten_nested(weight_norms), step=self.step
                )

    @profile_function("on_before_optimizer_step")
    def on_before_optimizer_step(self):
        """Hook that is called before the optimizer step."""
        if self.configs.monitor.grad_norm:
            _, total_norm = grad_norm(self.packing_model, norm_type=2)

            if DIST_WRAPPER.rank == 0:
                self.neptune_run.log_metrics({"grad_norm": total_norm}, step=self.step)

    def save_training_sample(
        self,
        context: FullContext,
        motif_as_b_factor: bool = False,
    ):
        """Saves a training sample to the training samples directory.

        Args:
            context: Full context containing atom arrays and pdb ids.
            motif_as_b_factor: If True, set b_factor to is_motif annotation.
        """
        from biotite.structure.io import save_structure

        atom_arrays = context.basic.atom_array
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
        Main entry for the ArtiosPackingTrainer.

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

                    context = context.to(self.device, non_blocking=True)

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
                                self.neptune_run.log_metrics(
                                    {"train/lr": last_lr},
                                    step=self.step,
                                )
                                # Log torchwatcher 10x less frequently
                                if (self.step + 1) % (
                                    self.configs.log_interval * 10
                                ) == 0:
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
                        self.print(f"Saving checkpoint at step {self.step}")
                        self.save_checkpoint()
                        if use_ema:
                            self.ema_wrapper.apply_shadow()
                            self.save_checkpoint(
                                ema_suffix=f"ema_{self.configs.ema_decay}",
                                save_model_only=True,
                            )
                            self.ema_wrapper.restore()
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
    trainer = ArtiosPackingTrainer(configs)
    trainer.run()


if __name__ == "__main__":
    main()
