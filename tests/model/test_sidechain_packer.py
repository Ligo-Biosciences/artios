"""Tests for the SidechainPacker module.

Tests cover the forward pass, f_forward, as_predictor, and sample methods.
Uses context_utils to create proper FullContext test data.
"""

import pytest
import torch

from artios.model.generator import (
    Deterministic,
    LaProteinaCoordScheduler,
    SamplingResult,
)
from artios.model.modules.sidechain_packer import SidechainPacker
from tests._helpers.context import create_test_context


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
        result = sidechain_packer.sample(
            context=context,
            timestep_scheduler=LaProteinaCoordScheduler(scale=10.0),
            n_steps=3,  # Small for fast testing
            residual_centre_atoms=residual_centre_atoms,
            stochasticity=Deterministic(),
            gamma=1.0,
            n_sample=n_sample,
        )

        # Check return type
        assert isinstance(
            result, SamplingResult
        ), f"Expected SamplingResult, got {type(result)}"

        # Check output shape
        expected_shape = (batch_size, n_sample, n_atoms, 3)
        assert (
            result.final.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {result.final.shape}"

        # Check for finite values
        assert torch.isfinite(
            result.final
        ).all(), "Sample output contains non-finite values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
