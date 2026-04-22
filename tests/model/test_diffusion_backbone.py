"""Tests for the refactored diffusion_backbone module using FullContext.

This test file uses the new typed FullContext dataclass instead of raw dicts.
Uses context_utils to create proper FullContext test data.
"""

import pytest
import torch

from artios.data.features import FullContext
from artios.model.modules.diffusion_backbone import (
    DiffusionConditioning,
    DiffusionModule,
)
from tests._helpers.context import create_test_context


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

    def test_motif_distance_bin_count_formula(self, device):
        """n_motif_dist_bins must be ``int((max - min) / bin_size)``."""
        diff_cond = DiffusionConditioning(
            sigma_coord=12.0,
            motif_dist_min=1.0,
            motif_dist_max=15.0,
            motif_dist_bin_size=0.25,
        ).to(device)
        assert diff_cond.n_motif_dist_bins == int((15.0 - 1.0) / 0.25)

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
        }
        return DiffusionModule(**config).to(device)

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


class TestSelfConditioning:
    """Tests for self-conditioning in DiffusionModule."""

    @pytest.fixture
    def diffusion_module(self, device):
        config = {
            "sigma_coord": 12.0,
            "c_z": 128,
            "c_token": 64,
            "c_atom": 128,
            "c_s": 384,
            "transformer": {"num_blocks": 2, "num_heads": 4},
            "dropout": 0.0,
        }
        return DiffusionModule(**config).to(device)

    def test_prev_pred_changes_output(self, diffusion_module, context, device):
        """Providing prev_pred should change the model output."""
        n_atoms = context.reference.ref_pos.shape[-2]
        batch_size = context.token.residue_index.shape[0]

        # Many layers are zero-init (readout, prev_pred_embed, linear_s, etc.)
        # so the network squashes differences at init. Set all weights to
        # small non-zero values to simulate a partially-trained model.
        with torch.no_grad():
            for p in diffusion_module.parameters():
                if p.ndim >= 2:
                    p.normal_(std=0.02)

        r_noisy = torch.randn(batch_size, 1, n_atoms, 3, device=device)
        timestep = torch.rand(batch_size, 1, device=device)
        prev_pred = torch.randn(batch_size, 1, n_atoms, 3, device=device) * 10.0

        diffusion_module.eval()
        with torch.no_grad():
            out_without = diffusion_module(
                r_noisy=r_noisy, timestep=timestep, context=context
            )
            out_with = diffusion_module(
                r_noisy=r_noisy,
                timestep=timestep,
                context=context,
                prev_pred=prev_pred,
            )

        assert not torch.allclose(out_without, out_with, atol=1e-5)

    def test_predictor_self_conditions_across_steps(
        self, diffusion_module, context, device
    ):
        """The predictor should use self-conditioning from the previous step."""
        n_atoms = context.reference.ref_pos.shape[-2]
        batch_size = context.token.residue_index.shape[0]

        # Set zero-init weights to non-zero so the network produces distinct outputs
        with torch.no_grad():
            for p in diffusion_module.parameters():
                if p.ndim >= 2:
                    p.normal_(std=0.02)

        diffusion_module.eval()
        predictor = diffusion_module.as_predictor(context=context)

        x = torch.randn(batch_size, 1, n_atoms, 3, device=device)
        t = torch.full((batch_size, 1), 0.5, device=device)

        with torch.no_grad():
            # First call: no prev_pred available
            v1, s1 = predictor(x, t)

            # Second call: should use prev_pred from first call
            v2, s2 = predictor(x, t)

        # Same input but different output because second call has self-conditioning
        assert not torch.allclose(v1, v2, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
