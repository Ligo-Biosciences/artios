import time
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from protenix.model import sample_confidence
from protenix.model.generator import (
    InferenceTimestepScheduler,
    VariableLinQuadScheduler,
    LaProteinaScheduler,
    TrainingTimestepSamplerV2,
    sample_diffusion,
    sample_diffusion_training,
)
from protenix.model.loss import ProtenixLoss
from protenix.model.modules.diffusion import DiffusionModule
from protenix.model.modules.embedders import InputFeatureEmbedder, RelativePositionEncoding
from protenix.model.modules.primitives import LinearNoBias
from protenix.model.utils import add, simple_merge_dict_list
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.torch_utils import autocasting_disable_decorator


class ArtiosLigand(nn.Module):
    def __init__(self, configs):
        super(ArtiosLigand, self).__init__()
        self.configs = configs

        # Some constants
        self.N_model_seed = self.configs.model.N_model_seed

        # Fixed eval generator for reproducibility
        # self.eval_generator = torch.Generator(device=self.device).manual_seed(42)

        # Diffusion
        self.train_timestep_sampler = TrainingTimestepSamplerV2(**configs.train_timestep_sampler)
        self.eval_timestep_sampler = TrainingTimestepSamplerV2(
            **configs.train_timestep_sampler,
        )
        #self.inference_timestep_scheduler = VariableLinQuadScheduler(
        #    **configs.inference_timestep_scheduler.variable_linear_quadratic
        #)
        #self.inference_timestep_scheduler = LaProteinaScheduler(
        #    **configs.inference_timestep_scheduler.la_proteina
        #)
        self.inference_timestep_scheduler = InferenceTimestepScheduler(
            **configs.inference_timestep_scheduler.ligo
        )

        self.diffusion_batch_size = self.configs.diffusion_batch_size

        # Model
        self.c_s, self.c_z = (
            configs.c_s,
            configs.c_z,
        )

        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)

        # Initialize loss as part of model for checkpointing and uncertainty learning
        self.loss = ProtenixLoss(configs)

        if configs.compile_model:
            self.diffusion_module.diffusion_transformer = torch.compile(
                self.diffusion_module.diffusion_transformer
            )
            # self.input_feature_embedder = torch.compile(self.input_feature_embedder)

        self.sigma_data = configs.sigma_data

    def sample_diffusion(self, **kwargs) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
        torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "stoch_type",
                "alpha",
                "gamma",
                "sigma_data",
                "guidance_weight",
                "beta_mean",
                "beta_std",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size if not self.training else None
                ),
                "diffusion_chunk_size": (
                    self.configs.infer_setting.sample_diffusion_chunk_size
                    if not self.training
                    else None
                ),
                "second_order": (
                    self.configs.sample_diffusion.second_order_train
                    if self.training
                    else self.configs.sample_diffusion.second_order_inference
                ),
            }
        )
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)

    def main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        N_model_seed: int = 1,
        symmetric_permutation: SymmetricPermutation = None,
        guiding_net: Optional[nn.Module] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (multiple model seeds) for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_dict (dict[str, Any]): Label dictionary.
            mode (str): Mode of operation (e.g., 'inference').
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to True.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to 4.
            N_model_seed (int): Number of model seeds. Defaults to 1.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.
            guiding_net (Optional[Callable]): Guiding net. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        pred_dicts = []
        log_dicts = []
        time_trackers = []
        for _ in range(N_model_seed):
            pred_dict, log_dict, time_tracker = self._main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                symmetric_permutation=symmetric_permutation,
                guiding_net=guiding_net,
            )
            pred_dicts.append(pred_dict)
            log_dicts.append(log_dict)
            time_trackers.append(time_tracker)

        # Combine outputs of multiple models
        def _cat(dict_list, key):
            return torch.cat([x[key] for x in dict_list], dim=0)

        def _list_join(dict_list, key):
            return sum([x[key] for x in dict_list], [])

        # all_pred_dict = {
        #    "coordinate": _cat(pred_dicts, "coordinate"),
        #    "summary_confidence": _list_join(pred_dicts, "summary_confidence"),
        #    "full_data": _list_join(pred_dicts, "full_data"),
        #    "plddt": _cat(pred_dicts, "plddt"),
        #    "pae": _cat(pred_dicts, "pae"),
        #    "pde": _cat(pred_dicts, "pde"),
        #    "resolved": _cat(pred_dicts, "resolved"),
        # }
        all_pred_dict = {
            "coordinate": _cat(pred_dicts, "coordinate"),
            "timestep": _cat(pred_dicts, "timestep"),
        }

        all_log_dict = simple_merge_dict_list(log_dicts)
        all_time_dict = simple_merge_dict_list(time_trackers)
        return all_pred_dict, all_log_dict, all_time_dict

    def _main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        mode: str,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = 4,
        symmetric_permutation: SymmetricPermutation = None,
        guiding_net: Optional[nn.Module] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (single model seed) for the Alphafold3 model.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        step_st = time.time()
        N_token = input_feature_dict["residue_index"].shape[-1]
        device = input_feature_dict["residue_index"].device
        dtype = input_feature_dict["residue_index"].dtype

        log_dict = {}
        pred_dict = {}
        time_tracker = {}

        if mode == "inference":
            keys_to_delete = []
            for key in input_feature_dict.keys():
                if "template_" in key or key in [
                    "msa",
                    "has_deletion",
                    "deletion_value",
                    "profile",
                    "deletion_mean",
                    # "token_bonds",
                ]:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del input_feature_dict[key]
            torch.cuda.empty_cache()
        step_trunk = time.time()
        time_tracker.update({"input_embed": step_trunk - step_st})

        ### Diffusion
        # Initialise guiding diffusion module
        guiding_diffusion_module = None
        if guiding_net is not None:
            guiding_diffusion_module = guiding_net.diffusion_module

        # Number of samples to generate from the same conditioning
        N_sample = self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]
        # Generate timesteps for solving the FM SDE
        timesteps = self.inference_timestep_scheduler(N_step=N_step, device=device, dtype=dtype)
        # N_sample different ouputs created via diffusion
        pred_dict["coordinate"] = self.sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            N_sample=N_sample,
            timesteps=timesteps,
            inplace_safe=inplace_safe,
            guiding_denoise_net=guiding_diffusion_module,  # TODO: flow net
        )

        timestep = timesteps[-1]  # Get final timestep
        pred_dict["timestep"] = timestep.expand(
            *pred_dict["coordinate"].shape[:-2]
        )  # Expand to [..., N_sample]

        step_diffusion = time.time()

        time_tracker.update({"diffusion": step_diffusion - step_trunk})
        if mode == "inference" and N_token > 2000:
            torch.cuda.empty_cache()
        time_tracker.update({"model_forward": time.time() - step_st})

        ### Permutation
        # When label is given, permute coordinates and other heads
        step_permutation = time.time()
        if label_dict is not None and symmetric_permutation is not None:
            pred_dict, log_dict = symmetric_permutation.permute_inference_pred_dict(
                input_feature_dict=input_feature_dict,
                pred_dict=pred_dict,
                label_dict=label_dict,
                permute_by_pocket=("pocket_mask" in label_dict)
                and ("interested_ligand_mask" in label_dict),
            )
            last_step_seconds = step_permutation
            time_tracker.update({"permutation": time.time() - last_step_seconds})

        # Summary Confidence & Full Data
        # Computed after coordinates and logits are permuted #TODO: can we get rid of this?
        if label_dict is None:
            interested_atom_mask = None
        else:
            interested_atom_mask = label_dict.get("interested_ligand_mask", None)

        return pred_dict, log_dict, time_tracker

    def main_train_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict,
        symmetric_permutation: SymmetricPermutation,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main training (and evaluation) loop for the ArtiosLigand model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict): Label dictionary (cropped).
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object.
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.
            seed (Optional[int]): Seed for random number generation. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """

        log_dict = {}
        pred_dict = {}

        device = input_feature_dict["residue_index"].device

        ### Mini-rollout for evaluation
        """
        if not self.training:
            # Rollout for visualizing the structures during evaluation
            with torch.no_grad():
                # [..., 1, N_atom, 3]
                N_sample_mini_rollout = self.configs.sample_diffusion[
                    "N_sample_mini_rollout"
                ]  # =1
                N_step = self.configs.sample_diffusion["N_step"]

                coordinate_mini = self.sample_diffusion(
                    denoise_net=self.diffusion_module,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs.detach(),
                    timesteps=self.inference_timestep_scheduler(
                        N_step=N_step,
                        device=s_inputs.device,
                        dtype=s_inputs.dtype,
                    ),
                    N_sample=N_sample_mini_rollout
                )
                coordinate_mini.detach_()
                pred_dict["coordinate_mini"] = coordinate_mini

                # Permute ground truth to match mini-rollout prediction
                label_dict, perm_log_dict = (
                    symmetric_permutation.permute_label_to_match_mini_rollout(
                        coordinate_mini,
                        input_feature_dict,
                        label_dict,
                        label_full_dict,
                    )
                )
                log_dict.update(perm_log_dict)
        """

        # Set seed for reproducibility
        if seed:
            self.eval_timestep_sampler.seed = seed

        ### Diffusion training
        N_sample = self.diffusion_batch_size
        x_target_augment, coords_pred, timestep = autocasting_disable_decorator(
            self.configs.skip_amp.sample_diffusion_training
        )(sample_diffusion_training)(
            timestep_sampler=(
                self.train_timestep_sampler if self.training else self.eval_timestep_sampler
            ),
            denoise_net=self.diffusion_module,
            label_dict=label_dict,
            input_feature_dict=input_feature_dict,
            sigma_data=self.sigma_data,
            N_sample=N_sample,
            diffusion_chunk_size=self.configs.diffusion_chunk_size,
            generator=(torch.Generator(device).manual_seed(seed) if not self.training else None),
        )

        # Update dicts
        pred_dict.update(
            {
                # [..., N_sample=48, N_atom, 3]: diffusion loss
                "coordinate": coords_pred,
                "timestep": timestep,
            }
        )

        # Permute symmetric atom/chain in each sample to match true structure
        # Note: currently chains cannot be permuted since label is cropped
        (
            pred_dict,
            perm_log_dict,
            _,
            _,
        ) = symmetric_permutation.permute_diffusion_sample_to_match_label(
            input_feature_dict, pred_dict, label_dict, stage="train"
        )
        log_dict.update(perm_log_dict)

        return pred_dict, label_dict, log_dict

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        mode: str = "inference",
        current_step: Optional[int] = None,
        symmetric_permutation: SymmetricPermutation = None,
        guiding_net: Optional[nn.Module] = None,
        seed: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Forward pass of the Artios model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict[str, Any]): Label dictionary (cropped).
            mode (str): Mode of operation ('train', 'inference', 'eval'). Defaults to 'inference'.
            current_step (Optional[int]): Current training step. Defaults to None.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.
            seed (Optional[int]): Seed for random number generation. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """

        assert mode in ["train", "inference", "eval"]
        inplace_safe = not (self.training or torch.is_grad_enabled())
        chunk_size = self.configs.infer_setting.chunk_size if inplace_safe else None

        if mode == "train" or mode == "eval":
            nc_rng = np.random.RandomState(current_step)
            assert label_dict is not None
            assert symmetric_permutation is not None

            pred_dict, label_dict, log_dict = self.main_train_loop(
                input_feature_dict=input_feature_dict,
                label_full_dict=label_full_dict,
                label_dict=label_dict,
                symmetric_permutation=symmetric_permutation,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                seed=seed,
            )
        elif mode == "inference":
            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=None,
                mode=mode,
                inplace_safe=False,  # inplace_safe,  TODO: temporary workaround
                chunk_size=chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=None,
                guiding_net=guiding_net,
            )
            log_dict.update({"time": time_tracker})

        """
        elif mode == "eval":
            # NOTE: this is not accessed for now, eval is the same as training
            if label_dict is not None:
                assert (
                    label_dict["coordinate"].size()
                    == label_full_dict["coordinate"].size()
                )
                label_dict.update(label_full_dict)

            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=symmetric_permutation,
            )
            log_dict.update({"time": time_tracker})
        """

        return pred_dict, label_dict, log_dict
