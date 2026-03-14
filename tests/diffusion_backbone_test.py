"""Tests for the refactored diffusion_backbone module using FullContext.

This test file uses the new typed FullContext dataclass instead of raw dicts.
Uses context_utils to create proper FullContext test data.
"""

import pytest
import torch

from artios.data.features import FullContext
from artios.model.generator import Deterministic, LaProteinaCoordScheduler
from artios.model.modules.diffusion_backbone import (
    CachedConditioning,
    DiffusionConditioning,
    DiffusionModule,
)
from tests.context_utils import create_test_context


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def context(device) -> FullContext:
    """Create test FullContext."""
    return create_test_context(
        batch_size=2,
        n_tokens=32,
        n_atoms=64,
        device=device,
    )


class TestDiffusionConditioning:
    """Tests for DiffusionConditioning module."""

    def test_forward(self, device, context):
        """Test DiffusionConditioning forward pass."""
        diff_cond = DiffusionConditioning(sigma_coord=12.0).to(device)

        n_tokens = context.token.residue_index.shape[-1]
        n_atoms = context.reference.ref_pos.shape[-2]
        batch_size = context.token.residue_index.shape[0]
        n_samples = 2

        # Create noisy coordinates and timesteps
        r_noisy = torch.randn(batch_size, n_samples, n_atoms, 3).to(device)
        timestep = torch.ones(batch_size, n_samples).to(device) * 0.5

        print("ref_pos shape: ", context.reference.ref_pos.shape)

        # Run forward pass
        s, z = diff_cond(
            r_noisy=r_noisy,
            timestep=timestep,
            context=context,
            inplace_safe=False,
        )

        # Check output shapes
        assert s.shape == (
            batch_size,
            n_samples,
            n_tokens,
            384,
        ), f"Expected s shape {(batch_size, n_samples, n_tokens, 384)}, got {s.shape}"
        assert z.shape == (
            batch_size,
            n_tokens,
            n_tokens,
            128,
        ), f"Expected z shape {(batch_size, n_tokens, n_tokens, 128)}, got {z.shape}"
        assert not torch.isnan(s).any(), "s contains NaN values"
        assert not torch.isnan(z).any(), "z contains NaN values"

    def test_get_atom_condition(self, device, context):
        """Test atom condition generation."""
        diff_cond = DiffusionConditioning(sigma_coord=12.0).to(device)

        n_tokens = context.token.residue_index.shape[-1]
        batch_size = context.token.residue_index.shape[0]

        # Test atom condition generation
        s_atom, z_atom = diff_cond.get_atom_condition(context=context)

        # Check shapes
        assert s_atom.shape == (batch_size, n_tokens, 384)
        assert z_atom.shape == (batch_size, n_tokens, n_tokens, 128)
        assert not torch.isnan(s_atom).any()
        assert not torch.isnan(z_atom).any()

    def test_get_motif_condition(self, device, context):
        """Test motif condition generation."""
        diff_cond = DiffusionConditioning(sigma_coord=12.0).to(device)

        n_tokens = context.token.residue_index.shape[-1]
        batch_size = context.token.residue_index.shape[0]

        # Test motif condition generation
        s_motif, z_motif = diff_cond.get_motif_condition(context=context)

        # Check shapes
        assert s_motif.shape == (batch_size, n_tokens, 384)
        assert z_motif.shape == (batch_size, n_tokens, n_tokens, 128)
        assert not torch.isnan(s_motif).any()
        assert not torch.isnan(z_motif).any()

    def test_motif_distance_binning(self, device, context):
        """Test motif distance binning parameters."""
        diff_cond = DiffusionConditioning(
            sigma_coord=12.0,
            motif_dist_min=1.0,
            motif_dist_max=15.0,
            motif_dist_bin_size=0.25,
        ).to(device)

        # Check binning parameters
        assert diff_cond.motif_dist_min == 1.0
        assert diff_cond.motif_dist_max == 15.0
        assert diff_cond.motif_dist_bin_size == 0.25
        expected_bins = int((15.0 - 1.0) / 0.25)
        assert diff_cond.n_motif_dist_bins == expected_bins

        # Test motif condition with specific binning
        s_motif, z_motif = diff_cond.get_motif_condition(context=context)

        assert s_motif.shape[-1] == 384
        assert z_motif.shape[-1] == 128

    def test_combined_conditions(self, device, context):
        """Test that atom and motif conditions combine correctly."""
        diff_cond = DiffusionConditioning(sigma_coord=12.0).to(device)

        n_tokens = context.token.residue_index.shape[-1]
        batch_size = context.token.residue_index.shape[0]

        # Get individual conditions
        s_atom, z_atom = diff_cond.get_atom_condition(context=context)
        s_motif, z_motif = diff_cond.get_motif_condition(context=context)

        # Combine conditions (as done in forward)
        s_combined = s_atom + s_motif
        z_combined = z_atom + z_motif

        # Check combined shapes
        assert s_combined.shape == (batch_size, n_tokens, 384)
        assert z_combined.shape == (batch_size, n_tokens, n_tokens, 128)
        assert not torch.isnan(s_combined).any()
        assert not torch.isnan(z_combined).any()

    def test_compute_cache(self, device, context):
        """Test that compute_cache returns correct shapes and dtypes."""
        diff_cond = DiffusionConditioning(sigma_coord=12.0).to(device)

        n_tokens = context.token.residue_index.shape[-1]
        batch_size = context.token.residue_index.shape[0]

        cache = diff_cond.compute_cache(context)

        assert isinstance(cache, CachedConditioning)
        assert cache.s_base.shape == (batch_size, n_tokens, 384)
        assert cache.z.shape == (batch_size, n_tokens, n_tokens, 128)
        assert cache.s_base.dtype == context.reference.ref_pos.dtype
        assert cache.z.dtype == context.reference.ref_pos.dtype
        assert not torch.isnan(cache.s_base).any()
        assert not torch.isnan(cache.z).any()

    def test_cache_equivalence(self, device, context):
        """Test that forward(cache=None) produces identical output to forward(cache=compute_cache(...))."""
        diff_cond = DiffusionConditioning(sigma_coord=12.0).to(device)
        diff_cond.eval()

        n_atoms = context.reference.ref_pos.shape[-2]
        batch_size = context.token.residue_index.shape[0]
        n_samples = 2

        r_noisy = torch.randn(batch_size, n_samples, n_atoms, 3).to(device)
        timestep = torch.ones(batch_size, n_samples).to(device) * 0.5

        # Disable random augmentation for deterministic comparison
        torch.manual_seed(42)
        s_no_cache, z_no_cache = diff_cond(
            r_noisy=r_noisy, timestep=timestep, context=context, cache=None
        )

        torch.manual_seed(42)
        cache = diff_cond.compute_cache(context)
        s_cached, z_cached = diff_cond(
            r_noisy=r_noisy, timestep=timestep, context=context, cache=cache
        )

        torch.testing.assert_close(s_cached, s_no_cache)
        torch.testing.assert_close(z_cached, z_no_cache)

    def test_cache_timestep_invariance(self, device, context):
        """Test that compute_cache output is identical regardless of what timestep is later used."""
        diff_cond = DiffusionConditioning(sigma_coord=12.0).to(device)
        diff_cond.eval()

        # compute_cache does not depend on timestep at all
        torch.manual_seed(42)
        cache1 = diff_cond.compute_cache(context)

        torch.manual_seed(42)
        cache2 = diff_cond.compute_cache(context)

        torch.testing.assert_close(cache1.s_base, cache2.s_base)
        torch.testing.assert_close(cache1.z, cache2.z)


class TestDiffusionModule:
    """Tests for DiffusionModule."""

    @pytest.fixture
    def diffusion_module(self, device):
        """Create a DiffusionModule with minimal configuration for testing."""
        config = {
            "sigma_coord": 12.0,
            "c_z": 128,
            "c_token": 64,  # Smaller for faster testing
            "c_atom": 128,
            "c_s": 384,
            "transformer": {"num_blocks": 2, "num_heads": 4},  # Minimal transformer
            "dropout": 0.0,
            "blocks_per_ckpt": None,
        }
        return DiffusionModule(**config).to(device)

    def test_readout(self, diffusion_module, device):
        """Test readout layer."""
        batch_size, n_samples, n_atoms, c_atom = 2, 1, 100, 128
        q_atom = torch.randn(batch_size, n_samples, n_atoms, c_atom).to(device)

        output = diffusion_module.readout(q_atom)

        # Check output shape - should be 3D coordinates
        assert output.shape == (batch_size, n_samples, n_atoms, 3)
        assert not torch.isnan(output).any()

    def test_scaling_functions(self, diffusion_module, device):
        """Test c_in, c_out, c_skip scaling functions."""
        timestep = torch.tensor([0.1, 0.5, 0.9]).to(device)

        c_in = diffusion_module.c_in(timestep)
        c_out = diffusion_module.c_out(timestep)
        c_skip = diffusion_module.c_skip(timestep)

        assert c_in.shape == timestep.shape
        assert c_out.shape == timestep.shape
        assert c_skip == 1.0  # c_skip returns a constant

        assert not torch.isnan(c_in).any()
        assert not torch.isnan(c_out).any()

    def test_f_forward(self, diffusion_module, context, device):
        """Test the f_forward method."""
        n_atoms = context.reference.ref_pos.shape[-2]
        batch_size = context.token.residue_index.shape[0]
        n_samples = 1

        # Create input tensors
        r_noisy = torch.randn(batch_size, n_samples, n_atoms, 3).to(device)
        timestep = torch.rand(batch_size, n_samples).to(device)

        # Run f_forward
        output = diffusion_module.f_forward(
            r_noisy=r_noisy,
            timestep=timestep,
            context=context,
            inplace_safe=False,
        )

        # Check output shape
        assert output.shape == (batch_size, n_samples, n_atoms, 3)
        assert not torch.isnan(output).any()

    def test_forward_pass(self, diffusion_module, context, device):
        """Test the complete forward pass."""
        n_atoms = context.reference.ref_pos.shape[-2]
        batch_size = context.token.residue_index.shape[0]
        n_samples = 1

        # Create input tensors
        r_noisy = torch.randn(batch_size, n_samples, n_atoms, 3).to(device)
        timestep = torch.rand(batch_size, n_samples).to(device)

        # Run forward pass
        r_pred = diffusion_module(
            r_noisy=r_noisy,
            timestep=timestep,
            context=context,
            inplace_safe=False,
        )

        assert r_pred.shape == (batch_size, n_samples, n_atoms, 3)
        assert not torch.isnan(r_pred).any()

    def test_different_timesteps(self, diffusion_module, context, device):
        """Test forward pass with different timesteps."""
        n_atoms = context.reference.ref_pos.shape[-2]
        batch_size = context.token.residue_index.shape[0]
        n_samples = 2

        # Create input tensors
        r_noisy = torch.randn(batch_size, n_samples, n_atoms, 3).to(device)

        # Different timesteps
        timestep = torch.tensor([[0.2, 0.8]] * batch_size).to(device)

        # Run forward pass
        output = diffusion_module(
            r_noisy=r_noisy,
            timestep=timestep,
            context=context,
        )

        assert output.shape == (batch_size, n_samples, n_atoms, 3)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, diffusion_module, device):
        """Test that gradients flow through the model correctly."""
        # Create a single-batch context for gradient test
        single_context = create_test_context(
            batch_size=1,
            n_tokens=32,
            n_atoms=64,
            device=device,
        )
        n_atoms = single_context.reference.ref_pos.shape[-2]
        n_samples = 1

        # Create input tensors that require gradients
        r_noisy = torch.randn(1, n_samples, n_atoms, 3, requires_grad=True).to(device)
        timestep = torch.rand(1, n_samples).to(device)

        # Forward pass
        output = diffusion_module(
            r_noisy=r_noisy,
            timestep=timestep,
            context=single_context,
        )

        # Compute loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        assert r_noisy.grad is not None
        assert not torch.isnan(r_noisy.grad).any()

    def test_with_motif_features(self, diffusion_module, context, device):
        """Test that motif features are correctly processed."""
        n_atoms = context.reference.ref_pos.shape[-2]
        batch_size = context.token.residue_index.shape[0]
        n_samples = 1

        # Ensure motif features are present
        assert context.motif.is_motif is not None
        assert context.motif.is_motif_related is not None
        assert context.motif.is_motif_token is not None
        assert context.motif.motif_restype is not None
        assert context.motif.motif_coordinate is not None

        # Create input tensors
        r_noisy = torch.randn(batch_size, n_samples, n_atoms, 3).to(device)
        timestep = torch.tensor([[0.5]] * batch_size).to(device)

        # Run forward pass
        output = diffusion_module(
            r_noisy=r_noisy,
            timestep=timestep,
            context=context,
        )

        assert output.shape == (batch_size, n_samples, n_atoms, 3)
        assert not torch.isnan(output).any()

    def test_as_predictor(self, diffusion_module, context, device):
        """Test that as_predictor returns a callable that produces velocity and score."""
        n_atoms = context.reference.ref_pos.shape[-2]
        batch_size = context.token.residue_index.shape[0]
        n_samples = 1

        # Create predictor
        predictor = diffusion_module.as_predictor(context=context, inplace_safe=False)

        # Create input tensors
        x_noisy = torch.randn(batch_size, n_samples, n_atoms, 3).to(device)
        timestep = torch.rand(batch_size, n_samples).to(device)

        # Call predictor
        velocity, score = predictor(x_noisy, timestep)

        # Check output shapes
        assert velocity.shape == (batch_size, n_samples, n_atoms, 3)
        assert score.shape == (batch_size, n_samples, n_atoms, 3)
        assert not torch.isnan(velocity).any()
        assert not torch.isnan(score).any()

    def test_as_predictor_with_different_timesteps(
        self, diffusion_module, context, device
    ):
        """Test as_predictor with varying timesteps."""
        n_atoms = context.reference.ref_pos.shape[-2]
        batch_size = context.token.residue_index.shape[0]
        n_samples = 2

        predictor = diffusion_module.as_predictor(context=context)

        x_noisy = torch.randn(batch_size, n_samples, n_atoms, 3).to(device)
        timestep = torch.tensor([[0.1, 0.9]] * batch_size).to(device)

        velocity, score = predictor(x_noisy, timestep)

        assert velocity.shape == (batch_size, n_samples, n_atoms, 3)
        assert score.shape == (batch_size, n_samples, n_atoms, 3)
        assert not torch.isnan(velocity).any()
        assert not torch.isnan(score).any()

    def test_sample(self, diffusion_module, device):
        """Test the sample method produces valid outputs."""
        # Create a small context for sampling test
        context = create_test_context(
            batch_size=1,
            n_tokens=16,
            n_atoms=32,
            device=device,
        )
        n_atoms = context.reference.ref_pos.shape[-2]
        n_sample = 2
        n_steps = 3  # Small for fast testing

        # Create timestep scheduler
        timestep_scheduler = LaProteinaCoordScheduler(scale=10.0)

        with torch.no_grad():
            output = diffusion_module.sample(
                context=context,
                timestep_scheduler=timestep_scheduler,
                n_steps=n_steps,
                stochasticity=Deterministic(),
                gamma=1.0,
                n_sample=n_sample,
                second_order=False,
                inplace_safe=False,
            )

        # Check output shape: [batch, n_sample, n_atom, 3]
        assert output.shape == (1, n_sample, n_atoms, 3)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()

    def test_sample_different_n_samples(self, diffusion_module, device):
        """Test sample method with different n_sample values."""
        context = create_test_context(
            batch_size=1,
            n_tokens=16,
            n_atoms=32,
            device=device,
        )
        n_atoms = context.reference.ref_pos.shape[-2]
        timestep_scheduler = LaProteinaCoordScheduler(scale=10.0)

        for n_sample in [1, 2, 4]:
            with torch.no_grad():
                output = diffusion_module.sample(
                    context=context,
                    timestep_scheduler=timestep_scheduler,
                    n_steps=2,
                    n_sample=n_sample,
                )

            assert output.shape == (1, n_sample, n_atoms, 3)
            assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
