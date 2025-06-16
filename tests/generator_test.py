import torch
import pytest
import numpy as np
from artios.model.generator import (
    TrainingTimestepSampler,
    InferenceTimestepScheduler,
    sample_diffusion,
    sample_diffusion_training
)
import os

class TestGenerator:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def sample_data(self):
        data_path = os.path.join(os.path.dirname(__file__), "testdata", "sample_data_dict.pt")
        if not os.path.exists(data_path):
            pytest.skip(f"Test data not found at {data_path}")
        return torch.load(data_path)

    @pytest.fixture
    def input_dict(self, device, sample_data):
        return {k: v.to(device) if torch.is_tensor(v) else v 
                for k, v in sample_data["input_feature_dict"].items()}

    @pytest.fixture
    def label_full_dict(self, device, sample_data):
        return {k: v.to(device) if torch.is_tensor(v) else v
                for k, v in sample_data["label_full_dict"].items()}

    @pytest.fixture
    def label_dict(self, device, sample_data):
        return {k: v.to(device) if torch.is_tensor(v) else v
                for k, v in sample_data["label_dict"].items()}

    def test_training_timestep_sampler(self, device):
        # Initialize sampler
        sampler = TrainingTimestepSampler(loc=0.0, scale=1.0)
        
        # Test sampling
        batch_size = (2, 3)
        timesteps = sampler(size=batch_size, device=device)
        
        # Check output shape and values
        assert timesteps.shape == batch_size
        assert ((timesteps >= 0) & (timesteps <= 1)).all()
        assert not torch.isnan(timesteps).any()

    def test_inference_timestep_scheduler(self, device):
        # Initialize scheduler
        scheduler = InferenceTimestepScheduler(p=-0.7)
        
        # Test scheduling
        n_steps = 50
        timesteps = scheduler(N_step=n_steps, device=device)
        
        # Check output shape and values
        assert timesteps.shape == (n_steps + 1,)
        assert timesteps[0] == 0
        assert torch.isclose(timesteps[-1], torch.tensor(1.0, device=device))
        assert torch.all(torch.diff(timesteps) > 0)  # Monotonically increasing
        assert not torch.isnan(timesteps).any()

    def test_sample_diffusion_basic(self, device, input_dict, sample_data):
        # Mock denoise network
        def mock_denoise_net(x_noisy, timestep, input_feature_dict, s_inputs, chunk_size=None, inplace_safe=False):
            return torch.zeros_like(x_noisy)
        
        # Get dimensions from sample data
        n_atoms = input_dict["atom_to_token_idx"].size(-1)
        batch_shape = input_dict["atom_to_token_idx"].shape[:-1]
        n_tokens = sample_data["input_feature_dict"]["atom_to_token_idx"].size(-1)
        c_s_inputs = 416
        n_steps = 5
        n_samples = 3
        
        s_inputs = torch.randn((*batch_shape, n_tokens, c_s_inputs), device=device)
        timesteps = torch.linspace(0, 1, n_steps, device=device)
        
        # Test basic functionality
        coords_pred = sample_diffusion(
            denoise_net=mock_denoise_net,
            input_feature_dict=input_dict,
            s_inputs=s_inputs,
            timesteps=timesteps,
            N_sample=n_samples,
            sigma_data=16.0,
            alpha=1.0,
            gamma=1.0
        )
        
        # Check output shape and values
        assert coords_pred.shape == (*batch_shape, n_samples, n_atoms, 3)
        assert not torch.isnan(coords_pred).any()

    def test_sample_diffusion_stochastic_types(self, device, input_dict, sample_data):
        def mock_denoise_net(x_noisy, timestep, input_feature_dict, s_inputs, chunk_size=None, inplace_safe=False):
            return torch.zeros_like(x_noisy)
        
        n_atoms = input_dict["atom_to_token_idx"].size(-1)
        batch_shape = input_dict["atom_to_token_idx"].shape[:-1]
        n_tokens = sample_data["input_feature_dict"]["atom_to_token_idx"].size(-1)
        c_s_inputs = 416
        n_steps = 5
        n_samples = 3
        
        s_inputs = torch.randn((*batch_shape, n_tokens, c_s_inputs), device=device)
        timesteps = torch.linspace(0, 1, n_steps, device=device)

        for stoch_type in ["deterministic", "constant", "non_singular", "zero_ends"]:
            coords_pred = sample_diffusion(
                denoise_net=mock_denoise_net,
                input_feature_dict=input_dict,
                s_inputs=s_inputs,
                timesteps=timesteps,
                N_sample=n_samples,
                stoch_type=stoch_type
            )
            assert coords_pred.shape == (*batch_shape, n_samples, n_atoms, 3)
            assert not torch.isnan(coords_pred).any()

    def test_sample_diffusion_with_guidance(self, device, input_dict, sample_data):
        def mock_denoise_net(x_noisy, timestep, input_feature_dict, s_inputs, chunk_size=None, inplace_safe=False):
            return torch.zeros_like(x_noisy)
            
        def mock_guiding_net(x_noisy, timestep, input_feature_dict, s_inputs, chunk_size=None, inplace_safe=False):
            return torch.ones_like(x_noisy)
            
        n_atoms = input_dict["atom_to_token_idx"].size(-1)
        batch_shape = input_dict["atom_to_token_idx"].shape[:-1]
        n_tokens = sample_data["input_feature_dict"]["atom_to_token_idx"].size(-1)
        c_s_inputs = 416
        n_steps = 5
        n_samples = 3
        
        s_inputs = torch.randn((*batch_shape, n_tokens, c_s_inputs), device=device)
        timesteps = torch.linspace(0, 1, n_steps, device=device)

        coords_pred = sample_diffusion(
            denoise_net=mock_denoise_net,
            input_feature_dict=input_dict,
            s_inputs=s_inputs,
            timesteps=timesteps,
            N_sample=n_samples,
            guiding_denoise_net=mock_guiding_net,
            guiding_s_inputs=s_inputs,
            guidance_weight=0.5
        )
        
        assert coords_pred.shape == (*batch_shape, n_samples, n_atoms, 3)
        assert not torch.isnan(coords_pred).any()

    def test_sample_diffusion_second_order(self, device, input_dict, sample_data):
        def mock_denoise_net(x_noisy, timestep, input_feature_dict, s_inputs, chunk_size=None, inplace_safe=False):
            return torch.zeros_like(x_noisy)
            
        n_atoms = input_dict["atom_to_token_idx"].size(-1)
        batch_shape = input_dict["atom_to_token_idx"].shape[:-1]
        n_tokens = sample_data["input_feature_dict"]["atom_to_token_idx"].size(-1)
        c_s_inputs = 416
        n_steps = 5
        n_samples = 3
        
        s_inputs = torch.randn((*batch_shape, n_tokens, c_s_inputs), device=device)
        timesteps = torch.linspace(0, 1, n_steps, device=device)

        # Test with second order solver
        coords_pred = sample_diffusion(
            denoise_net=mock_denoise_net,
            input_feature_dict=input_dict,
            s_inputs=s_inputs,
            timesteps=timesteps,
            N_sample=n_samples,
            second_order=True
        )
        
        assert coords_pred.shape == (*batch_shape, n_samples, n_atoms, 3)
        assert not torch.isnan(coords_pred).any()

    def test_sample_diffusion_training(self, device, input_dict, label_dict, sample_data):
        # Mock flow network
        def mock_flow_net(x_noisy, timestep, input_feature_dict, s_inputs):
            return torch.zeros_like(x_noisy)
            
        # Get dimensions from sample data
        n_atoms = input_dict["atom_to_token_idx"].size(-1)
        batch_shape = input_dict["atom_to_token_idx"].shape[:-1]
        n_tokens = sample_data["input_feature_dict"]["atom_to_token_idx"].size(-1)
        c_s_inputs = 416
        n_samples = 3
        
        s_inputs = torch.randn((*batch_shape, n_tokens, c_s_inputs), device=device)
        timestep_sampler = TrainingTimestepSampler()
        
        target_flow, velocity_pred, timestep = sample_diffusion_training(
            timestep_sampler=timestep_sampler,
            flow_net=mock_flow_net,
            label_dict=label_dict,
            input_feature_dict=input_dict,
            s_inputs=s_inputs,
            N_sample=n_samples
        )
            
        # Check output shapes and values
        assert target_flow.shape == (*batch_shape, n_samples, n_atoms, 3)
        assert velocity_pred.shape == (*batch_shape, n_samples, n_atoms, 3)
        assert timestep.shape == (*batch_shape, n_samples)
            
        assert not torch.isnan(target_flow).any()
        assert not torch.isnan(velocity_pred).any()
        assert not torch.isnan(timestep).any()
        assert ((timestep >= 0) & (timestep <= 1)).all()

if __name__ == "__main__":
    pytest.main([__file__])
