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

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from protenix.model.utils import (
    centre_random_augmentation,
    noise_w_flow_matching,
    beta_function,
)


class TrainingTimestepSampler:
    """Sample timesteps using a mixture of Beta(alpha,beta) and Uniform(0,1) distributions.

    Args:
        alpha (float): First shape parameter for Beta(alpha,beta) distribution. Defaults to 2.2.
        beta (float): Second shape parameter for Beta(alpha,beta) distribution. Defaults to 1.0.
        uniform_weight (float): Weight allocated to the Uniform distribution. Defaults to 0.02.
    """

    def __init__(
        self,
        alpha: float = 2.2,
        beta: float = 1.0,
        uniform_weight: float = 0.02,
    ):
        self.alpha = alpha
        self.beta = beta
        self.p_beta = 1.0 - uniform_weight

    def __call__(
        self, 
        size: torch.Size,
        device: torch.device = torch.device("cpu"),
        generator: Optional[torch.Generator] = None,
        eps: float = 1e-4
    ) -> torch.Tensor:
        """Sample timesteps from the mixture distribution. It is a mixture of Beta(alpha,beta) and Uniform(0,1) distributions.

        Args:
            size (torch.Size): Shape of output tensor
            device (torch.device): Device to place tensor on
            generator (torch.Generator, optional): Generator for reproducibility
            eps (float, optional): Epsilon value to clamp timesteps

        Returns:
            torch.Tensor: Sampled timesteps tensor of shape 'size'
        """
        # Component selector: 1 → take Beta, 0 → take Uniform
        take_beta = (torch.rand(size, generator=generator, device=device) < self.p_beta)

        # Beta(alpha, beta) distribution using torch.distributions
        beta_dist = torch.distributions.Beta(self.alpha, self.beta)
        
        # Sample from Beta(alpha, beta)
        beta_part = beta_dist.sample(size).to(device)
        
        # Sample from Uniform(0, 1)
        uniform_part = torch.rand(size, generator=generator, device=device)

        timesteps = torch.where(take_beta, beta_part, uniform_part)

        # Clamp timesteps to avoid numerical instability at t=0 or t=1
        return timesteps.clamp(min=eps, max=1.0-eps)
    

class InferenceTimestepScheduler:
    """
    Schedule the timesteps used for solving the FM SDE at inference time via the p-Ligo schedule.
    """

    def __init__(self, p: float = -0.7) -> None:
        self.p = p

    def __call__(
        self,
        N_step: int = 50,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Generate timesteps for inference using p-Ligo schedule.

        Args:
            N_step (int): Number of timesteps. Defaults to 50.
            device (torch.device): Device to place tensor on. Defaults to CPU.
            dtype (torch.dtype): Data type of tensor. Defaults to float32.

        Returns:
            torch.Tensor: Timesteps tensor of shape [N_step + 1]
        """
        steps = torch.arange(N_step + 1, dtype=dtype, device=device)
        t = (steps / N_step) ** np.exp(self.p)
        return t
    

class VariableLinQuadScheduler:
    """
    This scheduler will be lienar between the points (0,0) and (m,n)
    and then will be concave quadratic between (m,n) and (1,1)
    """
    def __init__(self,
                 m = 0.097,
                 n = 0.418) -> None:
        self.m = m
        self.n = n

    def __call__(
        self,
        N_step: int = 50,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:

        steps = torch.arange(N_step + 1, dtype=dtype, device=device) / N_step
        linear_steps = steps[steps < self.m]
        quadratic_steps = steps[steps > self.m]

        # Create the timesteps on the linear section
        linear_t = linear_steps * (self.n/self.m) 
        
        # Create the timeteps on the quadratic section
        a = (self.n - 1) / ((self.m - 1)**2)
        b = -2 * a
        c = a + 1
        quadratic_t = a * (quadratic_steps**2) + b * quadratic_steps + c

        # Concatenate all the timesteps 
        t = torch.cat([linear_t, quadratic_t])
        return t
    

class LaProteinaScheduler:
    """
    This shceduler is the same as the shceduler in the La-Proteina paper.
    """
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def __call__(
        self,
        N_step: int = 50,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        steps = torch.arange(N_step + 1, dtype=dtype, device=device) / N_step
        t = (1 - self.scale ** (-2 * steps)) / (1 - self.scale ** (-2))
        return t
    

def _calculate_velocity_and_score(
    x_noisy: torch.Tensor,
    t_cur: torch.Tensor,
    denoise_net: Callable,
    input_feature_dict: dict,
    t: torch.Tensor,
    attn_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    guiding_denoise_net: Optional[Callable] = None,
    guidance_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate velocity and scaled score for the diffusion process for a single timestep.

    Args:
        x_noisy: Current noisy coordinates [..., N_sample, N_atom, 3]
        t_cur: Current timestep (float)
        denoise_net: Network that performs denoising
        input_feature_dict: Input feature dictionary
        t: Broadcasted timestep tensor
            [..., N_sample]
        attn_chunk_size: Chunk size for attention operation
        inplace_safe: Whether to use inplace operations
        guiding_denoise_net: Optional guiding network
        guidance_weight: Weight for guidance

    Returns:
        tuple: (velocity_pred, scaled_score)
    """
    # Predict denoised coords from our noised coords
    coords_pred = denoise_net(
        x_noisy=x_noisy,
        timestep=t,
        input_feature_dict=input_feature_dict,
        chunk_size=attn_chunk_size,
        inplace_safe=inplace_safe,
    )

    # Calculate velocity
    velocity_pred = (coords_pred - x_noisy) / (1 - t_cur)

    # Predict score (division by 1-t absorbed into the stochasticity schedule for numerical stability)
    scaled_score = t_cur * velocity_pred - x_noisy

    # Apply guidance if specified
    if guiding_denoise_net is not None:
        guidance_coords_pred = guiding_denoise_net(
            x_noisy=x_noisy,
            timestep=t,
            input_feature_dict=input_feature_dict,
            chunk_size=attn_chunk_size,
            inplace_safe=inplace_safe,
        )

        # Calculate velocity and scaled score for guidance
        guidance_velocity_pred = (guidance_coords_pred - x_noisy) / (1 - t_cur)
        scaled_guidance_score = t_cur * guidance_velocity_pred - x_noisy

        # Velocity and score interpolation for guidance
        velocity_pred = (
            guidance_weight * velocity_pred + (1 - guidance_weight) * guidance_velocity_pred
        )
        scaled_score = (
            guidance_weight * scaled_score + (1 - guidance_weight) * scaled_guidance_score
        )

    return velocity_pred, scaled_score


def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    timesteps: torch.Tensor,
    N_sample: int = 1,
    sigma_data: float = 12.0,
    stoch_type: str = "deterministic",
    alpha: float = 1.0,
    gamma: float = 1.0,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    second_order: bool = False,
    guiding_denoise_net: Optional[Callable] = None,
    guidance_weight: float = 1.0,
    beta_mean: Optional[float] = 0.5,
    beta_std: Optional[float] = 0.224,
) -> torch.Tensor:
    """Implements Algorithm 18 in AF3.
    It performs denoising steps from time 0 to time T with Euler's method.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
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
        guidance_weight (float): The weight to use for guidance. Defaults to 1.0.
        beta_mean (float): The mean of the beta distribution. Defaults to 0.5.
        beta_std (float): The standard deviation of the beta distribution. Defaults to 0.1.
    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = input_feature_dict["residue_index"].shape[:-1]
    device = input_feature_dict["residue_index"].device
    dtype = torch.float32  # input_feature_dict["residue_index"].dtype

    # Stochasticity schedule g (scaled by 1/(1-t) for numerical stability)
    # We also normalise by the functions' integral over [0,1].
    # This allows for the parameter alpha to fully control the level of noise added.
    # Now, the area under the noise schedule is equal to alpha.
    stoch_schedules = {
        "deterministic": lambda t: torch.zeros_like(t) / (1 - t),
        "constant": lambda t: torch.ones_like(t) * (0.5 * (alpha**2)) / (1 - t),
        "non_singular": lambda t: torch.where(
            t > 0.95,  # do not noise close to 1 for ligand's sake
            torch.zeros_like(t),
            4 * torch.ones_like(t) * 0.5 * (alpha**2)
        ),  # int_{0}^{1}(non_sing) = 1/4
        "zero_ends": lambda t: 12 * 0.5 * (alpha**2) * t,  # int_{0}^{1}(zero_ends) = 1/12
        "proteina": lambda t: torch.where(
            t > 0.95,
            torch.zeros_like(t),
            0.5 * ((alpha**2) / (t + 0.01)) / (1 - t)
        ),
        "beta": lambda t: torch.where(
            t > 0.95,  
            torch.zeros_like(t),
            2 * 0.5 * (alpha**2) * beta_function(t, beta_mean, beta_std, device=device, dtype=dtype) / (1 - t)
        ),
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
            size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype
        )  # [..., N_sample, N_atom, 3]

        # Initialise Euler/Euler-Maruyama
        x = x_0
        noise = torch.zeros_like(x)

        # Sampling loop
        for i, (t_cur, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            # Shape wrangling for current timestep
            t = t_cur.new_full((batch_shape + (chunk_n_sample,)), t_cur).to(
                dtype
            )  # [..., N_sample]

            # Generate noise (for stochastic sampling)
            if stoch_type != "deterministic":
                noise = torch.randn_like(x)

            # Stochastic schedule at current time step. No stochasticity near end of sampling.
            #if t_cur < 0.95:
            #    g_t = g[i]
            #else:
            #    g_t = 0.0
            g_t = g[i]

            # Centre and rotate coords [..., N_sample, N_atom, 3]
            x = centre_random_augmentation(x_input_coords=x, N_sample=1).squeeze(dim=-3).to(dtype)

            # Calculate velocity and scaled score
            velocity_pred, scaled_score = _calculate_velocity_and_score(
                x_noisy=x,
                t_cur=t_cur,
                denoise_net=denoise_net,
                input_feature_dict=input_feature_dict,
                t=t,
                attn_chunk_size=attn_chunk_size,
                inplace_safe=inplace_safe,
                guiding_denoise_net=guiding_denoise_net,
                guidance_weight=guidance_weight,
            )

            # Calculate step length
            step = t_next - t_cur

            # Calculate update
            update = step * (velocity_pred.to(torch.float64) + g_t * scaled_score)

            # Euler step
            if not second_order:
                x = x + update + torch.sqrt(2 * step * g_t * gamma) * noise
            else:  # FIXME: my current issue with this is that we add nosie twice essentially: our initial update already uses noise.
                # Calculate intermediate prediction
                x_middle = x + update

                # Calculate intermediate prediction
                t = t_next.new_full((batch_shape + (chunk_n_sample,)), t_next).to(
                    dtype
                )  # [..., N_sample]

                # Calculate update at intermediate point
                velocity_pred_2, scaled_score_2 = _calculate_velocity_and_score(
                    x_noisy=x_middle.to(dtype),
                    t_cur=t_cur,
                    denoise_net=denoise_net,
                    input_feature_dict=input_feature_dict,
                    t=t,
                    attn_chunk_size=attn_chunk_size,
                    inplace_safe=inplace_safe,
                    guiding_denoise_net=guiding_denoise_net,
                    guidance_weight=guidance_weight,
                )

                # Calculate update at intermediate point
                update_2 = step * (velocity_pred_2.to(torch.float64) + g_t * scaled_score_2)

                # Perform second order Euler step
                x = x + 1 / 2 * (update + update_2) + torch.sqrt(2 * step * g_t * gamma) * noise

        return x

    if diffusion_chunk_size is None:
        coords_pred = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
    else:
        coords_pred = []
        no_chunks = N_sample // diffusion_chunk_size + (N_sample % diffusion_chunk_size != 0)
        for i in range(no_chunks):
            chunk_n_sample = (
                diffusion_chunk_size if i < no_chunks - 1 else N_sample - i * diffusion_chunk_size
            )
            chunk_coords_pred = _chunk_sample_diffusion(chunk_n_sample, inplace_safe=inplace_safe)
            coords_pred.append(chunk_coords_pred)
        coords_pred = torch.cat(coords_pred, -3)  # [..., N_sample, N_atom, 3]

    return coords_pred


def sample_noising(
    input: torch.Tensor, 
    mask: torch.Tensor,
    timestep_sampler: TrainingTimestepSampler, 
    num_sample: int, 
    scale: float,
    generator: torch.Generator,
    input_feature_dict: Optional[dict[str, Any]] = None,
    encoder: Optional[nn.Module] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The core noising function used for sample training FM models. 
    1. Augment input
    2. (Optional) Encode input
    3. Generate timesteps
    4. Generate noise
    5. Noise input FM style ( x_t = (1-t)*x_0 + t*x_1 )

    Args:
        input (torch.Tensor): The input to noise.
            [..., num_tokens, num_features]
        mask (torch.Tensor): The mask of the input.
            [..., num_tokens]
        timestep_sampler (TrainingTimestepSampler): The timestep sampler to use.
        num_sample (int): The number of samples to generate.
        scale (float): The scale of the noise.
        generator (torch.Generator): The generator to use for sampling.
        input_feature_dict (Optional[dict[str, Any]]): Input feature dictionary, used for encoding.
        encoder (Optional[nn.Module]): The encoder to use for encoding the input

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The noised input, associated noising timesteps, and augmented input.
            [..., num_sample, num_tokens, num_features], [..., num_sample], [..., num_sample, num_tokens, num_features]
    """
    batch_size_shape = input.shape[:-2]
    device = input.device
    dtype = input.dtype

    # Augment
    x_target_augment = centre_random_augmentation(
        x_input_coords=input,
        N_sample=num_sample,
        mask=mask,
    ).to(dtype) 

    # (Optional) Encode
    if encoder is not None:
        x_target_augment, _ = encoder(input_feature_dict, x_target_augment)

    # Generate timestep
    timestep = timestep_sampler(
        size=(*batch_size_shape, num_sample), 
        device=device, 
        generator=generator
    ).to(dtype) 

    # Generate noise
    noise = torch.randn(
        x_target_augment.shape,
        dtype=dtype,
        device=x_target_augment.device,
        generator=generator,
    ) * scale

    # Noise input
    x_noisy = noise_w_flow_matching(
        x_target_augment, noise, timestep[..., :, None, None])

    return x_noisy, timestep, x_target_augment


def sample_diffusion_training(
    timestep_sampler: TrainingTimestepSampler,
    denoise_net: Callable,
    label_dict: dict[str, Any],
    input_feature_dict: dict[str, Any],
    sigma_data: float = 12.0,
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
        N_sample (int): number of training samples
        generator (torch.Generator): the generator to use for sampling
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
    timestep = timestep_sampler(
        size=(*batch_size_shape, N_sample), 
        device=device, 
        generator=generator
    ).to(dtype) 

    # Generate noise
    noise = (
        torch.randn(
            x_target_augment.shape,
            dtype=dtype,
            device=x_target_augment.device,
            generator=generator,
        )
        * sigma_data
    )  # [..., N_sample, N_atom, 3]

    # Noise coords FM style ( x_t = (1-t)*x_0 + t*x_1 )
    x_noisy = noise_w_flow_matching(x_target_augment, noise, timestep[..., :, None, None])

    # Get denoised output predictions [..., N_sample, N_atom, 3]
    if diffusion_chunk_size is None:
        coords_pred = denoise_net(
            x_noisy=x_noisy,
            timestep=timestep,
            input_feature_dict=input_feature_dict,
        )
    else:
        coords_pred = []
        no_chunks = N_sample // diffusion_chunk_size + (N_sample % diffusion_chunk_size != 0)
        for i in range(no_chunks):
            x_noisy_i = x_noisy[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
            ]
            timestep_i = timestep[..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size]
            coords_pred_i = denoise_net(
                x_noisy=x_noisy_i,
                timestep=timestep_i,
                input_feature_dict=input_feature_dict,
            )

            coords_pred.append(coords_pred_i)
        coords_pred = torch.cat(coords_pred, dim=-3)

    return x_target_augment, coords_pred, timestep
