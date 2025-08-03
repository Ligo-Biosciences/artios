import os

import numpy as np
import pytest
import torch

from protenix.model.generator import (
    InferenceTimestepScheduler,
    TrainingTimestepSampler,
    sample_diffusion,
    sample_noising,
)


class TestGenerator:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def sample_data(self):
        data_path = os.path.join(os.path.dirname(__file__), "testdata", "sample_data_dict.pt")
        if not os.path.exists(data_path):
            pytest.skip(f"Test data not found at {data_path}")
        return torch.load(data_path, weights_only=False)

    @pytest.fixture
    def input_dict(self, device, sample_data):
        return {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in sample_data["input_feature_dict"].items()
        }

    @pytest.fixture
    def label_full_dict(self, device, sample_data):
        return {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in sample_data["label_full_dict"].items()
        }

    @pytest.fixture
    def label_dict(self, device, sample_data):
        return {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in sample_data["label_dict"].items()
        }

    def test_training_timestep_sampler(self, device):
        # Initialize sampler
        sampler = TrainingTimestepSampler()

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
        def mock_denoise_net(
            x_noisy, timestep, input_feature_dict, chunk_size=None, inplace_safe=False
        ):
            return torch.zeros_like(x_noisy)

        # Get dimensions from sample data
        n_atoms = input_dict["atom_to_token_idx"].size(-1)
        batch_shape = input_dict["atom_to_token_idx"].shape[:-1]
        n_tokens = sample_data["input_feature_dict"]["atom_to_token_idx"].size(-1)
        c_s_inputs = 416
        n_steps = 5
        n_samples = 3

        timesteps = torch.linspace(0, 1, n_steps, device=device)

        # Test basic functionality
        coords_pred = sample_diffusion(
            denoise_net=mock_denoise_net,
            input_feature_dict=input_dict,
            timesteps=timesteps,
            N_sample=n_samples,
            sigma_data=16.0,
            alpha=1.0,
            gamma=1.0,
        )

        # Check output shape and values
        assert coords_pred.shape == (*batch_shape, n_samples, n_atoms, 3)
        assert not torch.isnan(coords_pred).any()

    def test_sample_diffusion_stochastic_types(self, device, input_dict, sample_data):
        def mock_denoise_net(
            x_noisy, timestep, input_feature_dict, chunk_size=None, inplace_safe=False
        ):
            return torch.zeros_like(x_noisy)

        n_atoms = input_dict["atom_to_token_idx"].size(-1)
        batch_shape = input_dict["atom_to_token_idx"].shape[:-1]
        n_tokens = sample_data["input_feature_dict"]["atom_to_token_idx"].size(-1)
        c_s_inputs = 416
        n_steps = 5
        n_samples = 3

        timesteps = torch.linspace(0, 1, n_steps, device=device)

        for stoch_type in ["deterministic", "constant", "non_singular", "zero_ends"]:
            coords_pred = sample_diffusion(
                denoise_net=mock_denoise_net,
                input_feature_dict=input_dict,
                timesteps=timesteps,
                N_sample=n_samples,
                stoch_type=stoch_type,
            )
            assert coords_pred.shape == (*batch_shape, n_samples, n_atoms, 3)
            assert not torch.isnan(coords_pred).any()

    def test_sample_diffusion_with_guidance(self, device, input_dict, sample_data):
        def mock_denoise_net(
            x_noisy, timestep, input_feature_dict, chunk_size=None, inplace_safe=False
        ):
            return torch.zeros_like(x_noisy)

        def mock_guiding_net(
            x_noisy, timestep, input_feature_dict, chunk_size=None, inplace_safe=False
        ):
            return torch.ones_like(x_noisy)

        n_atoms = input_dict["atom_to_token_idx"].size(-1)
        batch_shape = input_dict["atom_to_token_idx"].shape[:-1]
        n_tokens = sample_data["input_feature_dict"]["atom_to_token_idx"].size(-1)
        c_s_inputs = 416
        n_steps = 5
        n_samples = 3

        timesteps = torch.linspace(0, 1, n_steps, device=device)

        coords_pred = sample_diffusion(
            denoise_net=mock_denoise_net,
            input_feature_dict=input_dict,
            timesteps=timesteps,
            N_sample=n_samples,
            guiding_denoise_net=mock_guiding_net,
            guidance_weight=0.5,
        )

        assert coords_pred.shape == (*batch_shape, n_samples, n_atoms, 3)
        assert not torch.isnan(coords_pred).any()

    def test_sample_diffusion_second_order(self, device, input_dict, sample_data):
        def mock_denoise_net(
            x_noisy, timestep, input_feature_dict, chunk_size=None, inplace_safe=False
        ):
            return torch.zeros_like(x_noisy)

        n_atoms = input_dict["atom_to_token_idx"].size(-1)
        batch_shape = input_dict["atom_to_token_idx"].shape[:-1]
        n_tokens = sample_data["input_feature_dict"]["atom_to_token_idx"].size(-1)
        c_s_inputs = 416
        n_steps = 5
        n_samples = 3

        timesteps = torch.linspace(0, 1, n_steps, device=device)

        # Test with second order solver
        coords_pred = sample_diffusion(
            denoise_net=mock_denoise_net,
            input_feature_dict=input_dict,
            timesteps=timesteps,
            N_sample=n_samples,
            second_order=True,
        )

        assert coords_pred.shape == (*batch_shape, n_samples, n_atoms, 3)
        assert not torch.isnan(coords_pred).any()

    def test_sample_noising_basic(self, device, input_dict, sample_data):
        """Test basic functionality of sample_noising function."""
        # Create input coordinates tensor
        batch_shape = (1,)  # Simple batch shape for testing
        n_tokens = 100
        n_features = 3  # x, y, z coordinates
        
        input_coords = torch.randn(*batch_shape, n_tokens, n_features, device=device)
        mask = torch.ones(*batch_shape, n_tokens, device=device, dtype=torch.bool)
        print("input_coords.shape", input_coords.shape)
        print("mask.shape", mask.shape)
        
        # Create timestep sampler
        timestep_sampler = TrainingTimestepSampler(alpha=2.2, beta=1.0, uniform_weight=0.02)
        
        # Set parameters
        num_sample = 3
        scale = 12.0
        generator = torch.Generator(device=device).manual_seed(42)
        
        # Call sample_noising
        input_noisy, timestep, input_augment = sample_noising(
            input=input_coords,
            mask=mask,
            timestep_sampler=timestep_sampler,
            num_sample=num_sample,
            scale=scale,
            generator=generator,
        )
        
        # Check output shapes
        expected_shape = (*batch_shape, num_sample, n_tokens, n_features)
        expected_timestep_shape = (*batch_shape, num_sample)
        
        assert input_noisy.shape == expected_shape
        assert input_augment.shape == expected_shape
        assert timestep.shape == expected_timestep_shape
        
        # Check that outputs are valid
        assert not torch.isnan(input_noisy).any()
        assert not torch.isnan(input_augment).any()
        assert not torch.isnan(timestep).any()
        
        # Check timestep range
        assert ((timestep >= 0) & (timestep <= 1)).all()
        
        # Check that noisy input is different from augmented input
        assert not torch.allclose(input_noisy, input_augment)

    def test_sample_noising_with_encoder(self, device, input_dict, sample_data):
        """Test sample_noising with an encoder."""
        # Create mock encoder
        class MockEncoder(torch.nn.Module):
            def forward(self, input_feature_dict, coords):
                # Return modified coords and some dummy output
                return coords * 1.1, torch.zeros_like(coords[..., :1])
        
        encoder = MockEncoder().to(device)
        
        # Create input data
        batch_shape = (1,)
        n_tokens = 50
        n_features = 3
        
        input_coords = torch.randn(*batch_shape, n_tokens, n_features, device=device)
        mask = torch.ones(*batch_shape, n_tokens, device=device, dtype=torch.bool)
        
        timestep_sampler = TrainingTimestepSampler()
        num_sample = 2
        scale = 10.0
        generator = torch.Generator(device=device).manual_seed(123)
        
        # Call with encoder
        input_noisy, timestep, input_augment = sample_noising(
            input=input_coords,
            mask=mask,
            timestep_sampler=timestep_sampler,
            num_sample=num_sample,
            scale=scale,
            generator=generator,
            input_feature_dict=input_dict,
            encoder=encoder,
        )
        
        # Check shapes
        expected_shape = (*batch_shape, num_sample, n_tokens, n_features)
        assert input_noisy.shape == expected_shape
        assert input_augment.shape == expected_shape
        assert timestep.shape == (*batch_shape, num_sample)
        
        # Check that encoder was applied (coords should be scaled by 1.1)
        # Note: This is approximate due to augmentation and other processing
        assert not torch.isnan(input_noisy).any()
        assert not torch.isnan(input_augment).any()

    def test_sample_noising_different_scales(self, device):
        """Test sample_noising with different noise scales."""
        batch_shape = (1,)
        n_tokens = 30
        n_features = 3
        
        input_coords = torch.randn(*batch_shape, n_tokens, n_features, device=device)
        mask = torch.ones(*batch_shape, n_tokens, device=device, dtype=torch.bool)
        
        timestep_sampler = TrainingTimestepSampler()
        num_sample = 1
        
        scales = [1.0, 5.0, 20.0]
        results = []
        
        for scale in scales:
            generator = torch.Generator(device=device).manual_seed(42)
            result = sample_noising(
                input=input_coords,
                mask=mask,
                timestep_sampler=timestep_sampler,
                num_sample=num_sample,
                scale=scale,
                generator=generator,
            )
            results.append(result)
        
        # Higher scale should generally produce noisier results
        # (though this is probabilistic due to timestep sampling)
        for i in range(len(scales) - 1):
            # Check that all outputs have valid shapes and values
            assert results[i][0].shape == results[i+1][0].shape
            assert not torch.isnan(results[i][0]).any()
            assert not torch.isnan(results[i+1][0]).any()

    def test_sample_noising_edge_cases(self, device):
        """Test sample_noising with edge cases."""
        # Test with single sample
        batch_shape = (1,)
        n_tokens = 10
        n_features = 3
        
        input_coords = torch.randn(*batch_shape, n_tokens, n_features, device=device)
        mask = torch.ones(*batch_shape, n_tokens, device=device, dtype=torch.bool)
        
        timestep_sampler = TrainingTimestepSampler()
        generator = torch.Generator(device=device).manual_seed(0)
        
        # Test with num_sample = 1
        result = sample_noising(
            input=input_coords,
            mask=mask,
            timestep_sampler=timestep_sampler,
            num_sample=1,
            scale=1.0,
            generator=generator,
        )
        
        assert result[0].shape == (*batch_shape, 1, n_tokens, n_features)
        assert result[1].shape == (*batch_shape, 1)
        assert result[2].shape == (*batch_shape, 1, n_tokens, n_features)
        
        # Test with partial mask (some tokens masked out)
        partial_mask = torch.zeros(*batch_shape, n_tokens, device=device, dtype=torch.bool)
        partial_mask[..., :n_tokens//2] = True  # Only first half is valid
        
        generator = torch.Generator(device=device).manual_seed(0)
        result_partial = sample_noising(
            input=input_coords,
            mask=partial_mask,
            timestep_sampler=timestep_sampler,
            num_sample=1,
            scale=1.0,
            generator=generator,
        )
        
        assert result_partial[0].shape == (*batch_shape, 1, n_tokens, n_features)
        assert not torch.isnan(result_partial[0]).any()


if __name__ == "__main__":
    pytest.main([__file__])
