import time
from typing import Any, Optional

import torch
import torch.nn as nn

from protenix.model.generator import (
    LaProteinaCoordScheduler,
    LaProteinaLatentScheduler,
    TrainingTimestepSampler,
    sample_latent_diffusion,
    sample_noising,
)
from protenix.model.modules.diffusion_allatom import DiffusionModule
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.torch_utils import autocasting_disable_decorator


class ArtiosAllatom(nn.Module):
    def __init__(self, configs):
        super(ArtiosAllatom, self).__init__()
        self.configs = configs

        # Some constants
        self.N_model_seed = self.configs.model.N_model_seed

        # Fixed eval generator for reproducibility
        # self.eval_generator = torch.Generator(device=self.device).manual_seed(42)

        # Diffusion
        self.coord_timestep_sampler = TrainingTimestepSampler(
            **configs.train_timestep_sampler.coord
        )
        self.latent_timestep_sampler = TrainingTimestepSampler(
            **configs.train_timestep_sampler.latent
        )
        self.coord_inference_timestep_scheduler = LaProteinaCoordScheduler(
            **configs.inference_timestep_scheduler.la_proteina.coord
        )
        self.latent_inference_timestep_scheduler = LaProteinaLatentScheduler()

        self.diffusion_batch_size = self.configs.diffusion_batch_size
        self.cfg_drop_probability = self.configs.cfg_drop_probability

        # Model
        self.c_s, self.c_z = (
            configs.c_s,
            configs.c_z,
        )

        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)

        self.sigma_coord = configs.sigma_coord
        self.sigma_latent = configs.sigma_latent

    def sample_latent_diffusion(self, **kwargs) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
        torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "sigma_coord",
                "sigma_latent",
                "stoch_type_coord",
                "stoch_type_latent",
                "alpha_coord",
                "alpha_latent",
                "gamma_coord",
                "gamma_latent",
                "second_order",
                "guidance_weight",
                "beta_mean",
                "beta_std",
                "lower_guidance_limit",
                "upper_guidance_limit",
            ]
        }
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_latent_diffusion
        )(**_configs, **kwargs)

    def main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        mode: str,
        inplace_safe: bool = False,
        decoder: Optional[nn.Module] = None,
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

        N_sample = self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]

        log_dict = {}
        pred_dict = {}
        time_tracker = {}

        # if mode == "inference":
        #    keys_to_delete = []
        #    for key in input_feature_dict.keys():
        #        if "template_" in key or key in [
        #            "msa",
        #            "has_deletion",
        #            "deletion_value",
        #            "profile",
        #            "deletion_mean",
        #            "token_bonds",
        #        ]:
        #            keys_to_delete.append(key)

        #    for key in keys_to_delete:
        #        del input_feature_dict[key]
        #    torch.cuda.empty_cache()

        # Generate timesteps for solving the FM SDE
        coord_timesteps = self.coord_inference_timestep_scheduler(
            N_step=N_step, device=device, dtype=dtype
        )
        latent_timesteps = self.latent_inference_timestep_scheduler(
            N_step=N_step, device=device, dtype=dtype
        )

        # N_sample different ouputs created via diffusion
        # NOTE: the coords are sent to the decoder, treated like they are working at sigma_coord
        # NOTE: the latents are immediately layernormed and then sent through a linear.
        # TODO: coming out of sample_latent_diffusion, are our coords actually working at the right sigma_coord?
        # TODO: is our decoder used to seeing coords at this scale ??
        pred_dict["centre_coordinate"], pred_dict["latents"] = (
            self.sample_latent_diffusion(
                denoise_net=self.diffusion_module,
                input_feature_dict=input_feature_dict,
                coord_timesteps=coord_timesteps,
                latent_timesteps=latent_timesteps,
                c_latent=self.configs.c_latent,
                N_sample=N_sample,
                inplace_safe=inplace_safe,
            )
        )

        # timestep = timesteps[-1]  # Get final timestep
        # pred_dict["timestep"] = timestep.expand(
        #    *pred_dict["coordinate"].shape[:-2]
        # )  # Expand to [..., N_sample]

        step_diffusion = time.time()

        time_tracker.update({"diffusion": step_diffusion - step_st})
        if mode == "inference" and N_token > 2000:
            torch.cuda.empty_cache()
        time_tracker.update({"model_forward": time.time() - step_st})

        if decoder is not None:
            pred_dict["coordinate"], pred_dict["seq_logits"] = decoder(
                latents=pred_dict["latents"],
                centre_atom_augment_coords=pred_dict["centre_coordinate"],
                input_feature_dict=input_feature_dict,
                inplace_safe=inplace_safe,
            )  # [..., N_sample, N_token, NUM_DENSE_ATOMS, 3], [..., N_sample, N_token, NUM_RESIDUES]

        return pred_dict, log_dict, time_tracker

    def main_train_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict,
        encoder: nn.Module,
        symmetric_permutation: SymmetricPermutation,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main training (and evaluation) loop for the Artios All-Atom model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict): Label dictionary (cropped).
            encoder (nn.Module): Encoder model.
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

        device = label_dict["coordinate"].device

        N_sample = self.diffusion_batch_size
        generator = (
            torch.Generator(device).manual_seed(seed) if not self.training else None
        )

        # Noising
        noisy_latent, latent_timestep, gt_latents = sample_noising(
            input=label_dict["coordinate"],
            mask=label_dict["coordinate_mask"],
            timestep_sampler=self.latent_timestep_sampler,
            num_sample=N_sample,
            scale=self.sigma_latent,  # TODO: sigma_latent
            generator=generator,
            input_feature_dict=input_feature_dict,
            encoder=encoder,
        )

        noisy_centre_coords, coord_timestep, _ = sample_noising(
            input=label_dict["centre_coordinate"],
            mask=label_dict["has_centre_atom"],
            timestep_sampler=self.coord_timestep_sampler,
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

        # Denoising
        coord_pred, latent_pred = self.diffusion_module(
            noisy_coord=noisy_centre_coords,
            noisy_latent=noisy_latent,
            coord_timestep=coord_timestep,
            latent_timestep=latent_timestep,
            input_feature_dict=input_feature_dict,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        pred_dict.update(
            {
                "latent_timestep": latent_timestep,
                "coord_timestep": coord_timestep,
                "latents": latent_pred,
                "centre_coordinate": coord_pred,
            }
        )

        label_dict.update(
            {
                "latents": gt_latents,  # [..., N_sample, N_token, c_latent]
            }
        )

        # Permute symmetric atom/chain in each sample to match true structure
        # (
        #    pred_dict,
        #    perm_log_dict,
        #    _,
        #    _,
        # ) = symmetric_permutation.permute_diffusion_sample_to_match_label(
        #    input_feature_dict, pred_dict, label_dict, stage="train"
        # )
        # log_dict.update(perm_log_dict)

        return pred_dict, label_dict, log_dict

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        mode: str = "inference",
        current_step: Optional[int] = None,
        symmetric_permutation: SymmetricPermutation = None,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
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
            encoder (Optional[nn.Module]): Encoder model. Defaults to None.
            decoder (Optional[nn.Module]): Decoder model. Defaults to None.
            seed (Optional[int]): Seed for random number generation. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """

        assert mode in ["train", "inference", "eval"]
        inplace_safe = not (self.training or torch.is_grad_enabled())
        chunk_size = self.configs.infer_setting.chunk_size if inplace_safe else None

        if mode == "train" or mode == "eval":
            # nc_rng = np.random.RandomState(current_step)
            assert label_dict is not None
            assert symmetric_permutation is not None

            pred_dict, label_dict, log_dict = self.main_train_loop(
                input_feature_dict=input_feature_dict,
                label_full_dict=label_full_dict,
                label_dict=label_dict,
                encoder=encoder,
                symmetric_permutation=symmetric_permutation,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                seed=seed,
            )
        elif mode == "inference":
            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                mode=mode,
                inplace_safe=False,  # inplace_safe,
                decoder=decoder,
            )
            log_dict.update({"time": time_tracker})

        return pred_dict, label_dict, log_dict
