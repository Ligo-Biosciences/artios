from typing import Any, Optional

import torch
import torch.nn as nn

from protenix.model.generator_backbone import (
    LaProteinaCoordScheduler,
    TrainingTimestepSampler,
    sample_diffusion,
    sample_noising,
)
from protenix.model.modules.diffusion_backbone import DiffusionModule
from protenix.utils.torch_utils import autocasting_disable_decorator


class ArtiosBackbone(nn.Module):
    def __init__(self, configs):
        super(ArtiosBackbone, self).__init__()
        self.configs = configs

        self.N_model_seed = self.configs.model.N_model_seed

        # Fixed eval generator for reproducibility
        # self.eval_generator = torch.Generator(device=self.device).manual_seed(42)

        self.timestep_sampler = TrainingTimestepSampler(
            **configs.train_timestep_sampler.coord
        )

        self.inference_timestep_scheduler = LaProteinaCoordScheduler(
            **configs.inference_timestep_scheduler.la_proteina.coord
        )

        self.diffusion_batch_size = self.configs.diffusion_batch_size
        self.cfg_drop_probability = self.configs.cfg_drop_probability

        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        self.sigma_coord = configs.sigma_coord

    def sample_diffusion(self, **kwargs) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
        torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "sigma_coord",
                "stoch_type",
                "alpha",
                "gamma",
                "second_order",
                "guidance_weight",
                "beta_mean",
                "beta_std",
                "lower_guidance_limit",
                "upper_guidance_limit",
            ]
        }
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)

    def main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        mode: str,
        inplace_safe: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop for ArtiosBackbone.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        device = input_feature_dict["residue_index"].device
        dtype = input_feature_dict["residue_index"].dtype

        N_sample = self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]

        log_dict = {}
        pred_dict = {}
        time_tracker = {}  # TODO: remove this

        # Generate timesteps for solving the FM SDE
        timesteps = self.inference_timestep_scheduler(
            N_step=N_step, device=device, dtype=dtype
        )

        pred_dict["coordinate"] = self.sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            timesteps=timesteps,
            N_sample=N_sample,
            inplace_safe=inplace_safe,
        )

        return pred_dict, log_dict, time_tracker

    def main_train_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict,
        inplace_safe: bool = False,
        seed: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main training (and evaluation) loop for the Artios All-Atom model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_dict (dict): Label dictionary (cropped).
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
            seed (Optional[int]): Seed for random number generation. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """
        log_dict = {}
        pred_dict = {}

        device = label_dict["coordinate"].device

        N_sample = self.diffusion_batch_size
        generator = (
            torch.Generator(device).manual_seed(seed) if not self.training else None
        )

        # Noising
        noisy_coords, timestep, _ = sample_noising(
            input=label_dict["coordinate"],
            mask=label_dict["coordinate_mask"],
            timestep_sampler=self.timestep_sampler,
            num_sample=N_sample,
            scale=self.sigma_coord,
            generator=generator,
        )

        # With CFG drop probability, train the unconditional model
        if torch.rand(1).item() < self.cfg_drop_probability:
            input_feature_dict["is_ligand"] = torch.zeros_like(
                input_feature_dict["is_ligand"]
            )
            input_feature_dict["is_ligand_token"] = torch.zeros_like(
                input_feature_dict["is_ligand_token"]
            )
            input_feature_dict["is_target_token"] = torch.zeros_like(
                input_feature_dict["is_target_token"]
            )
            input_feature_dict["is_hotspot"] = torch.zeros_like(
                input_feature_dict["is_hotspot"]
            )
            # TODO: drop motif related features

        # Denoising
        pred = self.diffusion_module(
            r_noisy=noisy_coords,
            timestep=timestep,
            input_feature_dict=input_feature_dict,
            inplace_safe=inplace_safe,
        )

        pred_dict.update(
            {
                "timestep": timestep,
                "coordinate": pred,
            }
        )

        return pred_dict, label_dict, log_dict

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        mode: str = "inference",
        current_step: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Forward pass of the Artios model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_dict (dict[str, Any]): Label dictionary (cropped).
            mode (str): Mode of operation ('train', 'inference', 'eval'). Defaults to 'inference'.
            current_step (Optional[int]): Current training step. Defaults to None.
            seed (Optional[int]): Seed for random number generation. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """

        assert mode in ["train", "inference", "eval"]
        inplace_safe = not (self.training or torch.is_grad_enabled())
        # chunk_size = self.configs.infer_setting.chunk_size if inplace_safe else None

        if mode == "train" or mode == "eval":
            # nc_rng = np.random.RandomState(current_step)
            assert label_dict is not None

            pred_dict, label_dict, log_dict = self.main_train_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                inplace_safe=inplace_safe,
                seed=seed,
            )
        elif mode == "inference":
            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                mode=mode,
                inplace_safe=False,
            )
            log_dict.update({"time": time_tracker})

        return pred_dict, label_dict, log_dict
