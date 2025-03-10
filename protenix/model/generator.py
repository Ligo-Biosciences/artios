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

from typing import Any, Callable, Optional

import torch
import numpy as np

from protenix.model.utils import centre_random_augmentation, sample_logit_normal, noise_w_flow_matching


class TrainingTimestepSampler:
    """
    Sample the timestep used for noising training samples via a logit-normal distribution. Used in the flow matching paradigm.
    """
    def __init__(
            self,
            loc: float = 0,
            scale: float = 1,
            generator: torch.Generator = None,
    ) -> None:
        self.loc = loc
        self.scale = scale
        self.generator = generator

    def __call__(
        self, 
        size: torch.Size, 
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sampling.

        Args:
            shape (torch.Size): the target shape
            device (torch.device, optional): target device. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: sampled timestep 
        """
        return sample_logit_normal(size, device, dtype, self.loc, self.scale, self.generator)
    

class InferenceTimestepScheduler:
    """
    Schedule the timesteps used for solving the FM SDE at inference time via the p-Ligo schedule.
    """
    def __init__(
            self,
            p: float = -0.7
    ) -> None:
        self.p = p

    def __call__(
        self, 
        N_step: int = 50,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        "Timesteps to be used at inference time."
        t = (torch.tensor(range(N_step+1), dtype=dtype, device=device) / N_step) ** np.exp(self.p)
        return t
    

def _calculate_velocity_and_score(
    x_noisy: torch.Tensor,
    t_cur: torch.Tensor,
    denoise_net: Callable,
    input_feature_dict: dict,
    s_inputs: torch.Tensor,
    t: torch.Tensor,
    attn_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    guiding_denoise_net: Optional[Callable] = None,
    guiding_s_inputs: Optional[torch.Tensor] = None,
    guidance_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate velocity and scaled score for the diffusion process for a single timestep.

    Args:
        x_noisy: Current noisy coordinates [..., N_sample, N_atom, 3]
        t_cur: Current timestep (float)
        denoise_net: Network that performs denoising
        input_feature_dict: Input feature dictionary
        s_inputs: Single embedding tensor
        t: Broadcasted timestep tensor 
            [..., N_sample]
        attn_chunk_size: Chunk size for attention operation
        inplace_safe: Whether to use inplace operations
        guiding_denoise_net: Optional guiding network
        guiding_s_inputs: Optional guiding single embedding tensor
        guidance_weight: Weight for guidance

    Returns:
        tuple: (velocity_pred, scaled_score)
    """
    # Predict denoised coords from our noised coords
    coords_pred = denoise_net(
        x_noisy=x_noisy,
        timestep=t,
        input_feature_dict=input_feature_dict,
        s_inputs=s_inputs,
        chunk_size=attn_chunk_size,
        inplace_safe=inplace_safe,
    )

    # Calculate velocity
    velocity_pred = (coords_pred - x_noisy) / (1 - t_cur)

    # Predict score (division by 1-t absorbed into the stochasticity schedule for numerical stability)
    scaled_score = (t_cur * velocity_pred - x_noisy)

    # Apply guidance if specified
    if guiding_denoise_net is not None:
        guidance_coords_pred = guiding_denoise_net(
            x_noisy=x_noisy,
            timestep=t,
            input_feature_dict=input_feature_dict,
            s_inputs=guiding_s_inputs,
            chunk_size=attn_chunk_size,
            inplace_safe=inplace_safe,
        )

        # Calculate velocity and scaled score for guidance
        guidance_velocity_pred = (guidance_coords_pred - x_noisy) / (1 - t_cur)
        scaled_guidance_score = (t_cur * guidance_velocity_pred - x_noisy)
        
        # Velocity and score interpolation for guidance
        velocity_pred = guidance_weight * velocity_pred + (1 - guidance_weight) * guidance_velocity_pred
        scaled_score = guidance_weight * scaled_score + (1 - guidance_weight) * scaled_guidance_score

    return velocity_pred, scaled_score


def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    timesteps: torch.Tensor,
    N_sample: int = 1,
    sigma_data: float = 16.0,
    stoch_type: str = "deterministic",
    alpha: float = 1.0,
    gamma: float = 1.0,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    second_order: bool = False,
    guiding_denoise_net : Optional[Callable] = None,
    guiding_s_inputs : Optional[torch.Tensor] = None,
    guidance_weight : float = 1.0,
) -> torch.Tensor:
    """Implements Algorithm 18 in AF3.
    It performs denoising steps from time 0 to time T with Euler's method.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        timesteps (torch.Tensor): Timestep discretisation for solving the SDE.
            [N_steps]
        N_sample (int): number of generated samples
        sigma_data (float): variance of data 
        stoch_type (str): Stochasticity schedule type (choice of beta(t) in FM SDE)
        alpha (float): Stochasticity parameter.
        gamma (float): Noise scaling parameter.
        p (float): Time step discretisation exponent used in the Ligo scheme.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
        attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.
        second_order (bool): Whether to use second order Euler-Maruyama. Defaults to False.
        guiding_denoise_net (Optional[Callable]): The guiding network to use for guidance. Defaults to None.
        guiding_s_inputs (Optional[torch.Tensor]): The guiding single embedding from InputFeatureEmbedder. Defaults to None.
        guidance_weight (float): The weight to use for guidance. Defaults to 1.0.
    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype

    # Stochasticity schedule g (scaled by 1/(1-t) for numerical stability)
    stoch_schedules = {
        "deterministic": lambda t: torch.zeros_like(t) / (1 - t),
        "constant": lambda t: torch.ones_like(t) * (0.5 * (alpha**2)) / (1 - t),
        "non_singular": lambda t: torch.ones_like(t) * 0.5 * (alpha**2),
        "zero_ends": lambda t: 0.5 * (alpha**2) * t,
        #TODO: check the divergence minimising choice used in ma et al.
        #"proteina": lambda t: alpha / (t + 0.01)
    }

    if stoch_type not in stoch_schedules:
        raise ValueError(f"Invalid stochasticity type: {stoch_type}")
        
    g = stoch_schedules[stoch_type](timesteps)

    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
        """ 
        Performs all sampling logic for a single chunk of samples 
        i.e generate noise, send noise through Euler steps, return denoised coordinates
        """
        # Generate noise
        x_0 = sigma_data * torch.randn(
            size=(*batch_shape, chunk_n_sample, N_atom, 3), 
            device=device, 
            dtype=dtype
        )  # [..., N_sample, N_atom, 3] # NOTE: set seed in distributed training

        # Initialise Euler/Euler-Maruyama
        x = x_0 
        noise = torch.zeros_like(x)

        # Sampling loop
        for i, (t_cur, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
   
            # Shape wrangling for current timestep
            t = t_cur.new_full((batch_shape + (chunk_n_sample,)), t_cur).to(dtype) # [..., N_sample]

            # Generate noise (for stochastic sampling)
            if stoch_type != "deterministic":
                noise = torch.randn_like(x)

            # Stochastic schedule at current time step
            g_t = g[i]

            # Centre and rotate coords [..., N_sample, N_atom, 3] 
            x = (
                centre_random_augmentation(x_input_coords=x, N_sample=1)
                .squeeze(dim=-3)
                .to(dtype)
            )

            # Calculate velocity and scaled score
            velocity_pred, scaled_score = _calculate_velocity_and_score(
                x_noisy=x,
                t_cur=t_cur,
                denoise_net=denoise_net,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                t=t,
                attn_chunk_size=attn_chunk_size,
                inplace_safe=inplace_safe,
                guiding_denoise_net=guiding_denoise_net,
                guiding_s_inputs=guiding_s_inputs,
                guidance_weight=guidance_weight,
            )

            # Calculate step length
            step = t_next - t_cur

            # Calculate update
            update = step * (velocity_pred.to(torch.float64) + g_t * scaled_score) 

            # Euler step
            if not second_order:
                x = x + update + torch.sqrt(2 * step * g_t * gamma) * noise
            else:
                # Calculate intermediate prediction
                x_middle = x + update

                # Calculate intermediate prediction
                t = t_next.new_full((batch_shape + (chunk_n_sample,)), t_next).to(dtype) # [..., N_sample]

                # Calculate update at intermediate point
                velocity_pred_2, scaled_score_2 = _calculate_velocity_and_score(
                    x_noisy=x_middle.to(dtype),
                    t_cur=t_cur,
                    denoise_net=denoise_net,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    t=t,
                    attn_chunk_size=attn_chunk_size,
                    inplace_safe=inplace_safe,
                    guiding_denoise_net=guiding_denoise_net,
                    guiding_s_inputs=guiding_s_inputs,
                    guidance_weight=guidance_weight,
                )

                # Calculate update at intermediate point
                update_2 = step * (velocity_pred_2.to(torch.float64) + g_t * scaled_score_2)

                # Perform second order Euler step
                x = x + 1/2 * (update + update_2) + torch.sqrt(2 * step * g_t * gamma) * noise

        return x

    if diffusion_chunk_size is None:
        coords_pred = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
    else:
        coords_pred = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            chunk_n_sample = (
                diffusion_chunk_size
                if i < no_chunks - 1
                else N_sample - i * diffusion_chunk_size
            )
            chunk_coords_pred = _chunk_sample_diffusion(
                chunk_n_sample, inplace_safe=inplace_safe
            )
            coords_pred.append(chunk_coords_pred)
        coords_pred = torch.cat(coords_pred, -3)  # [..., N_sample, N_atom, 3]

    return coords_pred


def sample_diffusion_training(
    timestep_sampler: TrainingTimestepSampler,
    denoise_net: Callable,
    label_dict: dict[str, Any],
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    sigma_data: float = 16.0,
    N_sample: int = 1,
    diffusion_chunk_size: Optional[int] = None,
    generator: torch.Generator = None,
) -> tuple[torch.Tensor, ...]:
    """Implements diffusion training. Creates N_sample's of randomly augmented ground truth coordinates, 
    noises the coords via FM style interpolation, sends them through the network to get denoised coords.

    Args:
        timestep_sampler (TrainingTimestepSampler): the timestep sampler
        denoise_net (Callable): the network that performs the denoising step.
        label_dict (dict, optional) : a dictionary containing the followings.
            "coordinate": the ground-truth coordinates
                [..., N_atom, 3]
            "coordinate_mask": whether true coordinates exist.
                [..., N_atom]
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        N_sample (int): number of training samples
        generator (torch.Generator, optional): the generator to use for sampling. Defaults to None during training.
    Returns:
        x_target_augment (torch.Tensor): the augmented ground truth coordinates 
            [..., N_sample, N_atom, 3]
        coords_pred (torch.Tensor): the predicted coordinates
            [..., N_sample, N_atom, 3]
        timestep (torch.Tensor): the timestep
            [..., N_sample]
    """
    batch_size_shape = label_dict["coordinate"].shape[:-2]
    device = label_dict["coordinate"].device
    dtype = label_dict["coordinate"].dtype

    # Create N_sample versions of the input structure by randomly rotating and translating
    x_target_augment = centre_random_augmentation(
        x_input_coords=label_dict["coordinate"],
        N_sample=N_sample,
        mask=label_dict["coordinate_mask"],
    ).to(
        dtype
    )  # [..., N_sample, N_atom, 3]

    # Get timestep for noising
    # timestep: independent timesteps for each sample [..., N_sample]
    timestep = timestep_sampler(size=(*batch_size_shape, N_sample), device=device).to(dtype) 
    
    # Generate noise
    noise = torch.randn_like(
        x_target_augment, 
        dtype=dtype, 
        generator=generator
    ) * sigma_data  # [..., N_sample, N_atom, 3]

    # Noise coords FM style ( x_t = (1-t)*x_0 + t*x_1 )
    x_noisy = noise_w_flow_matching(x_target_augment, noise, timestep[..., :, None, None])

    # Get denoised output predictions [..., N_sample, N_atom, 3]
    if diffusion_chunk_size is None:
        coords_pred = denoise_net( 
            x_noisy=x_noisy, 
            timestep=timestep, 
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs
        )
    else:
        coords_pred = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            x_noisy_i = x_noisy[ 
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
            ]
            timestep_i = timestep[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size
            ]
            coords_pred_i = denoise_net(
                x_noisy=x_noisy_i,
                timestep=timestep_i,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs
            )

            coords_pred.append(coords_pred_i)
        coords_pred = torch.cat(coords_pred, dim=-3)

    return x_target_augment, coords_pred, timestep 
