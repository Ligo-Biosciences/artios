import time
from typing import Any, Optional

import torch
import torch.nn as nn

from protenix.model.generator import (
    LaProteinaScheduler,
    TrainingTimestepSampler,
    sample_diffusion,
    sample_noising,
)
from protenix.model.loss import FlowLoss
from protenix.model.modules.diffusion_allatom import DiffusionModule
from protenix.model.utils import (
    simple_merge_dict_list, 
)
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
        # TODO: separate timestep schedulers for the sidechain and backbone coordinates
        self.coord_inference_timestep_scheduler = LaProteinaScheduler(
            **configs.inference_timestep_scheduler.la_proteina.coord
        )
        self.latent_inference_timestep_scheduler = LaProteinaScheduler(
            **configs.inference_timestep_scheduler.la_proteina.latent
        )

        self.diffusion_batch_size = self.configs.diffusion_batch_size

        # Model
        self.c_s, self.c_z = (
            configs.c_s,
            configs.c_z,
        )

        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)

        self.loss = FlowLoss(configs)

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
        symmetric_permutation: SymmetricPermutation = None,
        guiding_net: Optional[nn.Module] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        TODO: this should be completely re-written for the all-atom diffusion model.

        Main inference loop for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_dict (dict[str, Any]): Label dictionary.
            mode (str): Mode of operation (e.g., 'inference').
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to True.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to 4.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.
            guiding_net (Optional[Callable]): Guiding net. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        pred_dict, log_dict, time_tracker = self._main_inference_loop(
            input_feature_dict=input_feature_dict,
            label_dict=label_dict,
            mode=mode,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            symmetric_permutation=symmetric_permutation,
            guiding_net=guiding_net,
        )

        return pred_dict, log_dict, time_tracker

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

        N_sample = self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]

        log_dict = {}
        pred_dict = {}
        time_tracker = {}

        #if mode == "inference":
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

        ### Diffusion
        # Initialise guiding diffusion module
        guiding_diffusion_module = None
        if guiding_net is not None:
            guiding_diffusion_module = guiding_net.diffusion_module

        # Generate timesteps for solving the FM SDE
        coord_timesteps = self.coord_inference_timestep_scheduler(
            N_step=N_step, device=device, dtype=dtype
        )
        latent_timesteps = self.latent_inference_timestep_scheduler(
            N_step=N_step, device=device, dtype=dtype
        )

        # N_sample different ouputs created via diffusion
        #TODO: sample_diffusion needs big rewrite
        pred_dict["coordinate"] = self.sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            N_sample=N_sample,
            timesteps=timesteps,
            inplace_safe=inplace_safe,
            guiding_denoise_net=guiding_diffusion_module,
        )

        timestep = timesteps[-1]  # Get final timestep
        pred_dict["timestep"] = timestep.expand(
            *pred_dict["coordinate"].shape[:-2]
        )  # Expand to [..., N_sample]

        step_diffusion = time.time()

        time_tracker.update({"diffusion": step_diffusion - step_st})
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
        # if label_dict is None:
        #    interested_atom_mask = None
        # else:
        #    interested_atom_mask = label_dict.get("interested_ligand_mask", None)

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
        generator = torch.Generator(device).manual_seed(seed) if not self.training else None

        # Noising
        noisy_latent, latent_timestep, gt_latents = sample_noising(
            input=label_dict["coordinate"],
            mask=label_dict["coordinate_mask"],
            timestep_sampler=self.latent_timestep_sampler,
            num_sample=N_sample,
            scale=1.0,
            generator=generator,
            input_feature_dict=input_feature_dict,
            encoder=encoder,
        )

        noisy_centre_coords, coord_timestep, _ = sample_noising(
            input=label_dict["centre_coordinate"],
            mask=label_dict["has_centre_atom"],
            timestep_sampler=self.coord_timestep_sampler,
            num_sample=N_sample,
            scale=self.sigma_data,
            generator=generator,
        )

        # Denoising
        denoise_dict = self.diffusion_module(
            noisy_coord=noisy_centre_coords,
            noisy_latent=noisy_latent,
            coord_timestep=coord_timestep,
            latent_timestep=latent_timestep,
            input_feature_dict=input_feature_dict,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        
        pred_dict.update(denoise_dict)

        pred_dict.update(
            {
                "latent_timestep": latent_timestep,
                "coord_timestep": coord_timestep,
            }
        )

        label_dict.update(
            {
                "latents": gt_latents, # [..., N_sample, N_token, c_latent]
            }
        )

        # Permute symmetric atom/chain in each sample to match true structure
        #(
        #    pred_dict,
        #    perm_log_dict,
        #    _,
        #    _,
        #) = symmetric_permutation.permute_diffusion_sample_to_match_label(
        #    input_feature_dict, pred_dict, label_dict, stage="train"
        #)
        #log_dict.update(perm_log_dict)

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
                label_dict=None,
                mode=mode,
                inplace_safe=False,  # inplace_safe,  TODO: temporary workaround
                chunk_size=chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=None,
                guiding_net=guiding_net,
            )
            log_dict.update({"time": time_tracker})

        return pred_dict, label_dict, log_dict
