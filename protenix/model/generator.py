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


def _denoise_and_compute_velocity_score(
    denoise_net: Callable,
    noisy_coord: torch.Tensor,
    noisy_latent: torch.Tensor,
    t_coord: torch.Tensor,
    t_latent: torch.Tensor,
    t_coord_cur: torch.Tensor,
    t_latent_cur: torch.Tensor,
    input_feature_dict: dict,
    inplace_safe: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run denoising network and compute velocity and score for both coord and latent."""
    denoise_dict = denoise_net(
        noisy_coord=noisy_coord,
        noisy_latent=noisy_latent,
        coord_timestep=t_coord,
        latent_timestep=t_latent,
        input_feature_dict=input_feature_dict,
        inplace_safe=inplace_safe,
    )
    pred_coord = denoise_dict["centre_coordinate"]
    pred_latent = denoise_dict["latents"]

    velocity_coord = (pred_coord - noisy_coord) / (1 - t_coord_cur)
    velocity_latent = (pred_latent - noisy_latent) / (1 - t_latent_cur)

    score_coord = (t_coord_cur * velocity_coord - noisy_coord) / (1 - t_coord_cur)
    score_latent = (t_latent_cur * velocity_latent - noisy_latent) / (1 - t_latent_cur)

    return velocity_coord, velocity_latent, score_coord, score_latent


def _calculate_velocity_and_score(
    denoise_net: Callable,
    noisy_coord: torch.Tensor,
    noisy_latent: torch.Tensor,
    t_coord_cur: torch.Tensor,
    t_latent_cur: torch.Tensor,
    input_feature_dict: dict,
    inplace_safe: bool = False,
    guiding_denoise_net: Optional[Callable] = None,
    guidance_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    Calculate velocity and scaled score for the diffusion process for a single timestep.
    """
    batch_shape = input_feature_dict["residue_index"].shape[:-1]
    N_sample = noisy_coord.shape[-3]
    dtype = noisy_coord.dtype

    t_coord = t_coord_cur.new_full((batch_shape + (N_sample,)), t_coord_cur).to(dtype)
    t_latent = t_latent_cur.new_full((batch_shape + (N_sample,)), t_latent_cur).to(dtype)

    # Denoising
    v_c, v_l, s_c, s_l = _denoise_and_compute_velocity_score(
        denoise_net, noisy_coord, noisy_latent,
        t_coord, t_latent,
        t_coord_cur, t_latent_cur,
        input_feature_dict, inplace_safe
    )

    # (Optional) Guidance denoising
    if guiding_denoise_net is not None:
        v_c_g, v_l_g, s_c_g, s_l_g = _denoise_and_compute_velocity_score(
            guiding_denoise_net, noisy_coord, noisy_latent,
            t_coord, t_latent,
            t_coord_cur, t_latent_cur,
            input_feature_dict, inplace_safe
        )

        v_c = guidance_weight * v_c + (1 - guidance_weight) * v_c_g
        v_l = guidance_weight * v_l + (1 - guidance_weight) * v_l_g
        s_c = guidance_weight * s_c + (1 - guidance_weight) * s_c_g
        s_l = guidance_weight * s_l + (1 - guidance_weight) * s_l_g

    return {
        "centre_velocity": v_c,
        "latent_velocity": v_l,
        "centre_score": s_c,
        "latent_score": s_l,
    }


def sample_latent_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    coord_timesteps: torch.Tensor,
    latent_timesteps: torch.Tensor,
    c_latent: int = 8,
    N_sample: int = 1,
    sigma_coord: float = 12.0,
    sigma_latent: float = 1.0,
    stoch_type_coord: str = "deterministic",
    stoch_type_latent: str = "deterministic",
    alpha: float = 1.0,
    gamma: float = 1.0,
    second_order: bool = False,
    inplace_safe: bool = False,
    guiding_denoise_net: Optional[Callable] = None,
    guidance_weight: float = 1.0,
    beta_mean: Optional[float] = 0.5,
    beta_std: Optional[float] = 0.224,
) -> torch.Tensor:
    """
    Diffuses latents and centre coords with the allatom Flow model by discretising the FM SDE.
    Stochasticity implemented with a beta distribution, scaled by alpha, and temperature controlled by gamma.
    Optional guidance net can be supplied.
    Second order functionality (basic Midpoint ODE solver).
    Centre coords and latents denoise at different speeds, as recommended by LaProteina paper. 

    Args:
        denoise_net (Callable): The denoising network.
        input_feature_dict (dict[str, Any]): The input feature dictionary.
        coord_timesteps (torch.Tensor): The denoising timesteps for the centre coords.
        latent_timesteps (torch.Tensor): The denoising timesteps for the latents.
        c_latent (int): The latent dimension.
        N_sample (int): The number of samples.
        sigma_coord (float): Scale of the centre coords.
        sigma_latent (float): Scale of the latents.
        stoch_type_coord (str): The stochasticity function for the centre coords.
        stoch_type_latent (str): The stochasticity function for the latents.
        alpha (float): Size of stochasticity.
        gamma (float): Temperature of the stochasticity.
        second_order (bool): Whether to use the second order method.
        inplace_safe (bool): Whether to use inplace operations.
        guiding_denoise_net (Optional[Callable]): The guiding denoising network.
        guidance_weight (float): The weight of the guidance.
        beta_mean (Optional[float]): The mean of the beta distribution.
        beta_std (Optional[float]): The std of the beta distribution.

    """
    batch_shape = input_feature_dict["residue_index"].shape[:-1]
    N_token = input_feature_dict["residue_index"].shape[-1]
    device = input_feature_dict["residue_index"].device
    dtype = torch.float32  

    # Stochasticity functions (normalised such that int_{0}^{1}(g(t)) = 1)
    #TODO: check these integrals. I think off by factor of 2.
    stoch_schedules = {
        "deterministic": lambda t: torch.zeros_like(t),
        "constant": lambda t: torch.ones_like(t) * (0.5 * (alpha**2)),
        "non_singular": lambda t: torch.where(
            t > 0.95,  # do not noise close to 1 for ligand's sake
            torch.zeros_like(t),
            4 * 0.5 * (alpha**2) * (1 - t)
        ),  # int_{0}^{1}(non_sing) = 1/4
        "zero_ends": lambda t: 12 * 0.5 * (alpha**2) * t * (1 - t),  # int_{0}^{1}(zero_ends) = 1/12
        "proteina": lambda t: torch.where(
            t > 0.95,
            torch.zeros_like(t),
            0.5 * ((alpha**2) / (t + 0.01)) 
        ),
        "beta": lambda t: torch.where(
            t > 0.95,  
            torch.zeros_like(t),
            2 * 0.5 * (alpha**2) * beta_function(t, beta_mean, beta_std, device=device, dtype=dtype)
        ),
    }
    if stoch_type_coord not in stoch_schedules:
        raise ValueError(f"Invalid stochasticity type: {stoch_type_coord}")
    if stoch_type_latent not in stoch_schedules:
        raise ValueError(f"Invalid stochasticity type: {stoch_type_latent}")

    # Stochasticity schedules for centre coords and latents
    g_coord = stoch_schedules[stoch_type_coord](coord_timesteps)
    g_latent = stoch_schedules[stoch_type_latent](latent_timesteps)

    ### Sampling logic
    # Generate noise
    r_0 = sigma_coord * torch.randn(
        size=(*batch_shape, N_sample, N_token, 3), device=device, dtype=dtype
    )
    l_0 = sigma_latent * torch.randn(
        size=(*batch_shape, N_sample, N_token, c_latent), device=device, dtype=dtype
    )

    # Initialise Euler/Euler-Maruyama
    r = r_0
    l = l_0
    noise_r = torch.zeros_like(r)
    noise_l = torch.zeros_like(l)

    # Denoising loop
    for i, (t_coord_cur, t_coord_next, t_latent_cur, t_latent_next) in enumerate(zip(coord_timesteps[:-1], coord_timesteps[1:], latent_timesteps[:-1], latent_timesteps[1:])):
        # Generate noise (for stochastic sampling)
        if stoch_type_coord != "deterministic":
            noise_r = torch.randn_like(r)
        if stoch_type_latent != "deterministic":
            noise_l = torch.randn_like(l)

        # Stochastic schedule at current t
        g_coord_t = g_coord[i]
        g_latent_t = g_latent[i]

        # Centre and rotate coords [..., N_sample, N_atom, 3]
        r = centre_random_augmentation(x_input_coords=r, N_sample=1).squeeze(dim=-3).to(dtype)

        # Calculate velocity and score
        pred = _calculate_velocity_and_score(
            denoise_net=denoise_net,
            noisy_coord=r,
            noisy_latent=l,
            t_coord_cur=t_coord_cur,
            t_latent_cur=t_latent_cur,
            input_feature_dict=input_feature_dict,
            inplace_safe=inplace_safe,
            guiding_denoise_net=guiding_denoise_net,
            guidance_weight=guidance_weight,
        )

        # Calculate updates
        step_coord = t_coord_next - t_coord_cur
        step_latent = t_latent_next - t_latent_cur

        update_coord = step_coord * (pred["centre_velocity"] + g_coord_t * pred["centre_score"])
        update_latent = step_latent * (pred["latent_velocity"] + g_latent_t * pred["latent_score"])

        # Update
        if not second_order:
            r = r + update_coord + torch.sqrt(2 * step_coord * g_coord_t * gamma) * noise_r
            l = l + update_latent + torch.sqrt(2 * step_latent * g_latent_t * gamma) * noise_l
        else:
            # Calculate intermediate prediction
            r_middle = r + update_coord
            l_middle = l + update_latent

            pred_2 = _calculate_velocity_and_score(
                denoise_net=denoise_net,
                noisy_coord=r_middle,
                noisy_latent=l_middle,
                t_coord_cur=t_coord_cur,
                t_latent_cur=t_latent_cur,
                input_feature_dict=input_feature_dict,
                inplace_safe=inplace_safe,
                guiding_denoise_net=guiding_denoise_net,
                guidance_weight=guidance_weight,
            )

            update_2_coord = step_coord * (pred_2["centre_velocity"] + g_coord_t * pred_2["centre_score"])
            update_2_latent = step_latent * (pred_2["latent_velocity"] + g_latent_t * pred_2["latent_score"])

            r = r + 1 / 2 * (update_coord + update_2_coord) + torch.sqrt(2 * step_coord * g_coord_t * gamma) * noise_r
            l = l + 1 / 2 * (update_latent + update_2_latent) + torch.sqrt(2 * step_latent * g_latent_t * gamma) * noise_l

    return r, l 
        

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
