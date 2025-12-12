"""Tests for the SidechainPacker module.

Tests cover the forward pass, f_forward, as_predictor, and sample methods.
Uses context_utils to create proper FullContext test data.
"""

import pytest
import torch

from protenix.model.generator import (
    Deterministic,
    LaProteinaCoordScheduler,
    NonSingular,
)
from protenix.model.modules.sidechain_packer import SidechainPacker
from tests.context_utils import create_test_context


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def context(device):
    """Create test FullContext."""
    return create_test_context(
        batch_size=1,
        n_tokens=32,
        n_atoms=64,
        device=device,
    )


@pytest.fixture
def model_config():
    """Small model configuration for fast testing."""
    return {
        "c_s": 64,
        "c_token": 48,
        "c_atom": 32,
        "c_atompair": 8,
        "c_z": 16,
        "atom_encoder": {"num_blocks": 1, "num_heads": 2},
        "transformer": {"num_blocks": 1, "num_heads": 2},
        "atom_decoder": {"num_blocks": 1, "num_heads": 2},
        "num_queries": 16,
        "num_keys": 32,
        "sigma_coord": 10.0,
        "local_sigma_coord": 1.0,
    }


class TestSidechainPacker:
    """Tests for SidechainPacker model."""

    @pytest.fixture
    def sidechain_packer(self, model_config, device):
        """Create SidechainPacker model instance."""
        model = SidechainPacker(**model_config)
        return model.to(device)

    def test_model_initialization(self, sidechain_packer):
        """Test that model initializes correctly with all components."""
        assert hasattr(sidechain_packer, "atom_attention_encoder")
        assert hasattr(sidechain_packer, "diffusion_transformer")
        assert hasattr(sidechain_packer, "atom_attention_decoder")
        assert hasattr(sidechain_packer, "embed_restype")
        assert hasattr(sidechain_packer, "fourier_embedding")
        assert hasattr(sidechain_packer, "embed_timestep")
        assert hasattr(sidechain_packer, "relpos")
        assert hasattr(sidechain_packer, "layernorm_a")

    def test_forward(self, sidechain_packer, context, device):
        """Test the forward pass produces valid output."""
        batch_size = 1
        n_sample = 2
        n_atoms = context.reference.ref_pos.shape[-2]

        # Create noisy coordinates and timesteps
        r_noisy = torch.randn(batch_size, n_sample, n_atoms, 3, device=device)
        timestep = torch.rand(batch_size, n_sample, device=device)

        # Forward pass
        output = sidechain_packer.forward(
            r_noisy=r_noisy,
            timestep=timestep,
            context=context,
        )

        # Check output shape
        expected_shape = (batch_size, n_sample, n_atoms, 3)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Check for finite values
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    def test_f_forward(self, sidechain_packer, context, device):
        """Test the f_forward (inner forward) pass."""
        batch_size = 1
        n_sample = 2
        n_atoms = context.reference.ref_pos.shape[-2]

        # Create noisy coordinates and timesteps
        r_noisy = torch.randn(batch_size, n_sample, n_atoms, 3, device=device)
        timestep = torch.rand(batch_size, n_sample, device=device)

        # f_forward pass
        output = sidechain_packer.f_forward(
            r_noisy=r_noisy,
            timestep=timestep,
            context=context,
        )

        # Check output shape
        expected_shape = (batch_size, n_sample, n_atoms, 3)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Check for finite values
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    def test_c_out(self, sidechain_packer, device):
        """Test the output scaling function c_out."""
        timestep = torch.tensor([0.0, 0.5, 1.0], device=device)
        c_out = sidechain_packer.c_out(timestep)

        expected = torch.tensor([1.0, 0.5, 0.0], device=device)
        assert torch.allclose(c_out, expected), f"Expected {expected}, got {c_out}"

    def test_as_predictor(self, sidechain_packer, context, device):
        """Test that as_predictor returns a valid VelocityScorePredictor."""
        batch_size = 1
        n_sample = 2
        n_atoms = context.reference.ref_pos.shape[-2]

        # Create predictor
        predictor = sidechain_packer.as_predictor(context)

        # Create test inputs
        x_noisy = torch.randn(batch_size, n_sample, n_atoms, 3, device=device)
        timestep = torch.rand(batch_size, n_sample, device=device)

        # Call predictor
        velocity, score = predictor(x_noisy, timestep)

        # Check output shapes
        expected_shape = (batch_size, n_sample, n_atoms, 3)
        assert (
            velocity.shape == expected_shape
        ), f"Expected velocity shape {expected_shape}, got {velocity.shape}"
        assert (
            score.shape == expected_shape
        ), f"Expected score shape {expected_shape}, got {score.shape}"

        # Check for finite values
        assert torch.isfinite(velocity).all(), "Velocity contains non-finite values"
        assert torch.isfinite(score).all(), "Score contains non-finite values"

    def test_sample_deterministic(self, sidechain_packer, context, device):
        """Test the sample method with deterministic stochasticity."""
        batch_size = 1
        n_sample = 2
        n_atoms = context.reference.ref_pos.shape[-2]
        # n_tokens = context.token.residue_index.shape[-1]

        # Create residual centre atoms
        residual_centre_atoms = torch.randn(
            batch_size, n_sample, n_atoms, 3, device=device
        )

        # Sample with deterministic stochasticity
        output = sidechain_packer.sample(
            context=context,
            timestep_scheduler=LaProteinaCoordScheduler(scale=10.0),
            n_steps=3,  # Small for fast testing
            residual_centre_atoms=residual_centre_atoms,
            stochasticity=Deterministic(),
            gamma=1.0,
            n_sample=n_sample,
        )

        # Check output shape
        expected_shape = (batch_size, n_sample, n_atoms, 3)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Check for finite values
        assert torch.isfinite(output).all(), "Sample output contains non-finite values"

    def test_sample_stochastic(self, sidechain_packer, context, device):
        """Test the sample method with non-singular stochasticity."""
        batch_size = 1
        n_sample = 2
        n_atoms = context.reference.ref_pos.shape[-2]

        # Create residual centre atoms
        residual_centre_atoms = torch.randn(
            batch_size, n_sample, n_atoms, 3, device=device
        )

        # Sample with non-singular stochasticity
        output = sidechain_packer.sample(
            context=context,
            timestep_scheduler=LaProteinaCoordScheduler(scale=10.0),
            n_steps=3,  # Small for fast testing
            residual_centre_atoms=residual_centre_atoms,
            stochasticity=NonSingular(alpha=0.5),
            gamma=1.0,
            n_sample=n_sample,
        )

        # Check output shape
        expected_shape = (batch_size, n_sample, n_atoms, 3)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        # Check for finite values
        assert torch.isfinite(output).all(), "Sample output contains non-finite values"

    def test_gradient_flow(self, sidechain_packer, context, device):
        """Test that gradients flow properly through the model."""
        sidechain_packer.train()

        batch_size = 1
        n_sample = 2
        n_atoms = context.reference.ref_pos.shape[-2]

        # Create inputs
        r_noisy = torch.randn(batch_size, n_sample, n_atoms, 3, device=device)
        timestep = torch.rand(batch_size, n_sample, device=device)

        # Forward pass
        output = sidechain_packer.forward(
            r_noisy=r_noisy,
            timestep=timestep,
            context=context,
        )

        # Compute simple loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Check that gradients exist and are finite
        has_grad = False
        for name, param in sidechain_packer.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                assert torch.isfinite(
                    param.grad
                ).all(), f"Non-finite gradients for parameter {name}"

        assert has_grad, "No gradients computed for any parameter"

    def test_output_device(self, sidechain_packer, context, device):
        """Test that outputs are on the correct device."""
        batch_size = 1
        n_sample = 2
        n_atoms = context.reference.ref_pos.shape[-2]

        r_noisy = torch.randn(batch_size, n_sample, n_atoms, 3, device=device)
        timestep = torch.rand(batch_size, n_sample, device=device)

        output = sidechain_packer.forward(
            r_noisy=r_noisy,
            timestep=timestep,
            context=context,
        )

        assert (
            output.device == device
        ), f"Output on wrong device: {output.device} vs {device}"

    def test_different_batch_sizes(self, model_config, context, device):
        """Test model with different batch sizes."""
        model = SidechainPacker(**model_config).to(device)
        n_atoms = context.reference.ref_pos.shape[-2]

        for batch_size in [1, 2]:
            # Create context with different batch size
            ctx = create_test_context(
                batch_size=batch_size,
                n_tokens=32,
                n_atoms=64,
                device=device,
            )

            n_sample = 2
            r_noisy = torch.randn(batch_size, n_sample, n_atoms, 3, device=device)
            timestep = torch.rand(batch_size, n_sample, device=device)

            output = model.forward(
                r_noisy=r_noisy,
                timestep=timestep,
                context=ctx,
            )

            expected_shape = (batch_size, n_sample, n_atoms, 3)
            assert (
                output.shape == expected_shape
            ), f"Expected shape {expected_shape}, got {output.shape}"

    def test_different_n_sample(self, sidechain_packer, context, device):
        """Test model with different number of samples."""
        batch_size = 1
        n_atoms = context.reference.ref_pos.shape[-2]

        for n_sample in [1, 2, 4]:
            r_noisy = torch.randn(batch_size, n_sample, n_atoms, 3, device=device)
            timestep = torch.rand(batch_size, n_sample, device=device)

            output = sidechain_packer.forward(
                r_noisy=r_noisy,
                timestep=timestep,
                context=context,
            )

            expected_shape = (batch_size, n_sample, n_atoms, 3)
            assert (
                output.shape == expected_shape
            ), f"Expected shape {expected_shape} for n_sample={n_sample}, got {output.shape}"

    def test_timestep_boundary_values(self, sidechain_packer, context, device):
        """Test model with boundary timestep values (near 0 and 1)."""
        batch_size = 1
        n_sample = 2
        n_atoms = context.reference.ref_pos.shape[-2]

        r_noisy = torch.randn(batch_size, n_sample, n_atoms, 3, device=device)

        # Test near 0
        timestep_low = torch.full((batch_size, n_sample), 0.01, device=device)
        output_low = sidechain_packer.forward(
            r_noisy=r_noisy,
            timestep=timestep_low,
            context=context,
        )
        assert torch.isfinite(
            output_low
        ).all(), "Output at t=0.01 contains non-finite values"

        # Test near 1
        timestep_high = torch.full((batch_size, n_sample), 0.99, device=device)
        output_high = sidechain_packer.forward(
            r_noisy=r_noisy,
            timestep=timestep_high,
            context=context,
        )
        assert torch.isfinite(
            output_high
        ).all(), "Output at t=0.99 contains non-finite values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
