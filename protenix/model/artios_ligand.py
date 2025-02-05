import torch
import torch.nn as nn
from typing import Any
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.model.modules.primitives import LinearNoBias
from .modules.diffusion import DiffusionModule
from .modules.embedders import InputFeatureEmbedder, RelativePositionEncoding
from typing import Optional
from protenix.model.generator import (
    sample_diffusion_training, 
    TrainingNoiseSampler,
    InferenceNoiseScheduler,
    sample_diffusion
)
from protenix.utils.torch_utils import autocasting_disable_decorator
from protenix.utils.utils import _add

class ArtiosLigand(nn.Module):
    def __init__(self, configs):
        super(ArtiosLigand, self).__init__()
        self.configs = configs

        # Diffusion 
        self.train_noise_sampler = TrainingNoiseSampler(**configs.train_noise_sampler)
        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )
        self.diffusion_batch_size = self.configs.diffusion_batch_size

        # Model 
        self.c_s, self.c_z, self.c_s_inputs = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
        )

        self.input_feature_embedder = InputFeatureEmbedder(
            **configs.model.input_feature_embedder
        )

        self.linear_no_bias_s_init = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_z_init1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_z_init2 = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_z
        )
        self.linear_no_bias_token_bonds = LinearNoBias(
            in_features=1, out_features=self.c_z
        )

        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )

        self.diffusion_module = DiffusionModule(
            **configs.model.diffusion_module
        )

    def sample_diffusion(self, **kwargs) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
        torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
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
        }
        )
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)

    def forward(
            self,
            input_feature_dict: dict[str, Any],
            label_full_dict: dict[str, Any],
            label_dict: dict[str, Any],
            symmetric_permutation: SymmetricPermutation,
            inplace_safe: bool = False,
            chunk_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Forward pass for the ArtiosLigand model.

        Args:
            input_feature_dict (dict[str, Any]): Input feature dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict[str, Any]): Label dictionary (cropped).
            symmetric_permutation (SymmetricPermutation, optional): Symmetric permutation object. Defaults to None.
            inplace_safe (bool, optional): Whether to use inplace operations safely. Defaults to False.
            chunk_size (Optional[int], optional): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """
        # This function is an attempt at unifying the code from main_train_loop(), main_inference_loop() and forward()
        # which all feature some fairly clunky boilerplate and common code.
        # TODO: Unfinished. Need to add main_inference_loop logic after the _main_inference_loop() section.
        # TODO: It will be neat if this function takes in noisy_coords and just performs denoising. This will require moving the 
        # seperate training and inference logic elsewhere which will be easier with lightning.
        # This will require a lot of refactoring, including a noise generator (we can keep generator.py?), the noising of the coords
        # (which should take place in the model_step of the lightning module), a seperate sampler.py file that performs the sampling 
        # in the many ways we will sampling to occur,

        N_token = input_feature_dict["residue_index"].shape[-1]
        if N_token <= 16:
            # Deepspeed_evo_attention do not support token <= 16
            deepspeed_evo_attention_condition_satisfy = False
        else:
            deepspeed_evo_attention_condition_satisfy = True

        # Initialize dictionaries
        log_dict = {} 
        pred_dict = {}

        # Creating single and pair embeddings used to condition the diffusion module
        s_inputs = self.input_feature_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )  # [..., N_token, 449] TODO: is 449 c_token? Why not set in configs?
        s_init = self.linear_no_bias_s_init(s_inputs)  #  [..., N_token, c_s]
        z_init = _add(
            self.linear_no_bias_z_init1(s_init)[..., None, :],
            self.linear_no_bias_z_init2(s_init)[..., None, :, :],
            inplace_safe=inplace_safe
        )  #  [..., N_token, N_token, c_z]

        # Add relative position encoding and token bonds to pair embeddings
        pos_enc = self.relative_position_encoding(input_feature_dict)
        z_init = _add(z_init, pos_enc, inplace_safe=inplace_safe)

        token_bonds_emb = self.linear_no_bias_token_bonds(input_feature_dict["token_bonds"].unsqueeze(dim=-1))
        z_init = _add(z_init, token_bonds_emb, inplace_safe=inplace_safe)

        # Initialize z and s
        z = z_init
        s = s_init

        # Training logic (mainly from main_train_loop)
        if self.training:
            N_sample = self.diffusion_batch_size
            _, x_denoised, x_noise_level = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion_training
            )(sample_diffusion_training)(
                noise_sampler=self.train_noise_sampler,
                denoise_net=self.diffusion_module,
                label_dict=label_dict,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s,
                z_trunk=z,
                N_sample=N_sample,
                diffusion_chunk_size=self.configs.diffusion_chunk_size,
            )

            pred_dict.update(
                {
                    "coordinate": x_denoised,
                    "noise_level": x_noise_level,
                }
            )

            # Mini-rollout: used for label permutation during training and confidence head (the latter we have ablated)

            with torch.no_grad():
                # [..., 1, N_atom, 3]
                N_sample_mini_rollout = self.configs.sample_diffusion[
                    "N_sample_mini_rollout"
                ]  # =1
                N_step_mini_rollout = self.configs.sample_diffusion["N_step_mini_rollout"]

                coordinate_mini = self.sample_diffusion(
                    denoise_net=self.diffusion_module,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs.detach(),
                    s_trunk=s.detach(),
                    z_trunk=z.detach(),
                    N_sample=N_sample_mini_rollout,
                    noise_schedule=self.inference_noise_scheduler(
                        N_step=N_step_mini_rollout,
                        device=s_inputs.device,
                        dtype=s_inputs.dtype,
                    ),
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

                # Permute symmetric atom/chain in each sample to match true structure
                # Note: currently chains cannot be permuted since label is cropped TODO: what does this mean?
                pred_dict, perm_log_dict, _, _ = (
                    symmetric_permutation.permute_diffusion_sample_to_match_label(
                        input_feature_dict, pred_dict, label_dict, stage="train"
                    )
                )
                log_dict.update(perm_log_dict)

        # Inference logic (mainly from _main_inference_loop)
        if not self.training:
            keys_to_delete = []
            for key in input_feature_dict.keys():
                if "template_" in key or key in [
                    "msa",
                    "has_deletion",
                    "deletion_value",
                    "profile",
                    "deletion_mean",
                    "token_bonds",
                ]:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del input_feature_dict[key]
            torch.cuda.empty_cache()

            # Sample diffusion 
            N_sample = self.configs.sample_diffusion["N_sample"]
            N_step = self.configs.sample_diffusion["N_step"]

            noise_schedule = self.inference_noise_scheduler(
                N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype
            )
            pred_dict["coordinate"] = self.sample_diffusion(
                denoise_net=self.diffusion_module,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s,
                z_trunk=z,
                N_sample=N_sample,
                noise_schedule=noise_schedule,
                inplace_safe=inplace_safe,
            )

            if N_token > 2000:
                torch.cuda.empty_cache()

            # Permutation: when label is given, permute coordinates and other heads
            if label_dict is not None and symmetric_permutation is not None:
                pred_dict, log_dict = symmetric_permutation.permute_inference_pred_dict(
                    input_feature_dict=input_feature_dict,
                    pred_dict=pred_dict,
                    label_dict=label_dict,
                    permute_by_pocket=("pocket_mask" in label_dict)
                    and ("interested_ligand_mask" in label_dict),
                )

            if label_dict is None:
                interested_atom_mask = None
            else:
                interested_atom_mask = label_dict.get("interested_ligand_mask", None)
            pred_dict["full_data"] = None  #TODO: modify sample_confidence.compute_full_data_and_summary to just return data values not confidence information 
            
            # TODO: add in the main_inference_loop logic 

        return pred_dict, label_dict, log_dict
