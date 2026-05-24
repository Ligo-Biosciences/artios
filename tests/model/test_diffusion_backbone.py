"""Tests for the refactored diffusion_backbone module using FullContext.

This test file uses the new typed FullContext dataclass instead of raw dicts.
Uses context_utils to create proper FullContext test data.
"""

import pytest
import torch
import torch.nn as nn

from artios.data.features import FullContext
from artios.model.generator import SIGMA_DATA, EDMInferenceNoiseScheduler
from artios.model.modules.diffusion_backbone import (
    DiffusionConditioning,
    DiffusionConditioningConfig,
    DiffusionModule,
    DiffusionModuleConfig,
)
from artios.model.modules.transformer import DiffusionTransformerConfig
from tests._helpers.context import create_test_context


class SpyFourierEmbedding(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c = c
        self.seen_input = None

    def forward(self, t_hat_timestep):
        self.seen_input = t_hat_timestep.detach().clone()
        return torch.zeros(*t_hat_timestep.shape, self.c, device=t_hat_timestep.device)


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
            config=DiffusionConditioningConfig(
                motif_dist_min=1.0,
                motif_dist_max=15.0,
                motif_dist_bin_size=0.25,
            ),
        ).to(device)
        assert diff_cond.n_motif_dist_bins == int((15.0 - 1.0) / 0.25)

    def test_cache_equivalence(self, device, context):
        """forward(cache=None) matches forward(cache=compute_cache(...))."""
        diff_cond = DiffusionConditioning().to(device)
        diff_cond.eval()

        n_atoms = context.reference.ref_pos.shape[-2]
        batch_size = context.token.residue_index.shape[0]
        n_samples = 2

        r_noisy = torch.randn(batch_size, n_samples, n_atoms, 3).to(device)
        noise_level = torch.ones(batch_size, n_samples).to(device) * 16.0

        # Disable random augmentation for deterministic comparison
        torch.manual_seed(42)
        s_no_cache, z_no_cache = diff_cond(
            r_noisy=r_noisy, noise_level=noise_level, context=context, cache=None
        )

        torch.manual_seed(42)
        cache = diff_cond.compute_cache(context)
        s_cached, z_cached = diff_cond(
            r_noisy=r_noisy, noise_level=noise_level, context=context, cache=cache
        )

        torch.testing.assert_close(s_cached, s_no_cache)
        torch.testing.assert_close(z_cached, z_no_cache)

    def test_cache_noise_level_invariance(self, device, context):
        """Test that compute_cache output is identical regardless of later noise levels."""
        diff_cond = DiffusionConditioning().to(device)
        diff_cond.eval()

        # compute_cache does not depend on noise level at all
        torch.manual_seed(42)
        cache1 = diff_cond.compute_cache(context)

        torch.manual_seed(42)
        cache2 = diff_cond.compute_cache(context)

        torch.testing.assert_close(cache1.s_base, cache2.s_base)
        torch.testing.assert_close(cache1.z, cache2.z)

    def test_conditioning_uses_log_sigma_scaled_fourier_input(self, device, context):
        diff_cond = DiffusionConditioning().to(device)
        spy = SpyFourierEmbedding(diff_cond.fourier_embedding.c).to(device)
        diff_cond.fourier_embedding = spy

        n_atoms = context.reference.ref_pos.shape[-2]
        batch_size = context.token.residue_index.shape[0]
        noise_level = torch.tensor([[16.0, 160.0], [1.6, 0.16]], device=device)
        r_noisy = torch.randn(batch_size, 2, n_atoms, 3, device=device)

        diff_cond(r_noisy=r_noisy, noise_level=noise_level, context=context)

        expected = torch.log(noise_level / SIGMA_DATA) / 4
        torch.testing.assert_close(spy.seen_input, expected)


class TestDiffusionModule:
    """Tests for DiffusionModule."""

    @pytest.fixture
    def diffusion_module(self, device):
        """Create a DiffusionModule with minimal configuration for testing."""
        config = DiffusionModuleConfig(
            c_z=128,
            c_token=64,  # Smaller for faster testing
            c_atom=128,
            c_s=384,
            transformer=DiffusionTransformerConfig(num_blocks=2, num_heads=4),
        )
        return DiffusionModule(config=config).to(device)

    def test_forward_pass(self, diffusion_module, context, device):
        """Test the complete forward pass."""
        n_atoms = context.reference.ref_pos.shape[-2]
        batch_size = context.token.residue_index.shape[0]
        n_samples = 1

        # Create input tensors
        r_noisy = torch.randn(batch_size, n_samples, n_atoms, 3).to(device)
        noise_level = torch.rand(batch_size, n_samples).to(device) * 16 + 0.1

        # Run forward pass
        r_pred = diffusion_module(
            r_noisy=r_noisy,
            noise_level=noise_level,
            context=context,
            inplace_safe=False,
        )

        assert r_pred.shape == (batch_size, n_samples, n_atoms, 3)
        assert not torch.isnan(r_pred).any()

    def test_as_predictor(self, diffusion_module, context, device):
        """Test that as_predictor returns a callable that produces denoised coordinates."""
        n_atoms = context.reference.ref_pos.shape[-2]
        batch_size = context.token.residue_index.shape[0]
        n_samples = 1

        # Create predictor
        predictor = diffusion_module.as_predictor(context=context, inplace_safe=False)

        # Create input tensors
        x_noisy = torch.randn(batch_size, n_samples, n_atoms, 3).to(device)
        noise_level = torch.rand(batch_size, n_samples).to(device) * 16 + 0.1

        # Call predictor
        denoised = predictor(x_noisy, noise_level)

        # Check output shapes
        assert denoised.shape == (batch_size, n_samples, n_atoms, 3)
        assert not torch.isnan(denoised).any()

    def test_edm_preconditioning_factors(self, diffusion_module, device):
        noise_level = torch.tensor([0.0, 16.0, 32.0], device=device)
        sigma_data = SIGMA_DATA

        expected_c_in = 1 / torch.sqrt(sigma_data**2 + noise_level**2)
        expected_c_skip = sigma_data**2 / (sigma_data**2 + noise_level**2)
        expected_c_out = (
            sigma_data * noise_level / torch.sqrt(sigma_data**2 + noise_level**2)
        )

        torch.testing.assert_close(diffusion_module.c_in(noise_level), expected_c_in)
        torch.testing.assert_close(
            diffusion_module.c_skip(noise_level), expected_c_skip
        )
        torch.testing.assert_close(diffusion_module.c_out(noise_level), expected_c_out)

    def test_prev_pred_is_scaled_by_sigma_data(
        self, diffusion_module, context, device, monkeypatch
    ):
        """Self-conditioning receives denoised coords scaled by SIGMA_DATA."""
        n_atoms = context.reference.ref_pos.shape[-2]
        batch_size = context.token.residue_index.shape[0]
        captured = {}

        def fake_f_forward(
            self,
            r_noisy,
            noise_level,
            context,
            cache=None,
            inplace_safe=False,
            prev_pred=None,
        ):
            captured["prev_pred"] = prev_pred.detach().clone()
            return torch.zeros_like(r_noisy)

        monkeypatch.setattr(DiffusionModule, "f_forward", fake_f_forward)

        r_noisy = torch.randn(batch_size, 1, n_atoms, 3, device=device)
        noise_level = torch.full((batch_size, 1), 32.0, device=device)
        prev_pred = torch.randn(batch_size, 1, n_atoms, 3, device=device)

        diffusion_module(
            r_noisy=r_noisy,
            noise_level=noise_level,
            context=context,
            prev_pred=prev_pred,
        )

        expected = prev_pred / SIGMA_DATA
        old_noise_dependent_scaling = prev_pred * diffusion_module.c_in(
            noise_level[..., None, None]
        )
        torch.testing.assert_close(captured["prev_pred"], expected)
        assert not torch.allclose(captured["prev_pred"], old_noise_dependent_scaling)

    def test_zero_initialized_readout_returns_skip_scaled_noisy_coords(
        self, diffusion_module, context, device
    ):
        n_atoms = context.reference.ref_pos.shape[-2]
        batch_size = context.token.residue_index.shape[0]

        r_noisy = torch.randn(batch_size, 1, n_atoms, 3, device=device)
        noise_level = torch.full((batch_size, 1), 16.0, device=device)

        diffusion_module.eval()
        with torch.no_grad():
            r_pred = diffusion_module(
                r_noisy=r_noisy,
                noise_level=noise_level,
                context=context,
                inplace_safe=False,
            )

        expected = r_noisy * diffusion_module.c_skip(noise_level[..., None, None])
        torch.testing.assert_close(r_pred, expected)


class TestSelfConditioning:
    """Tests for self-conditioning in DiffusionModule."""

    @pytest.fixture
    def diffusion_module(self, device):
        config = DiffusionModuleConfig(
            c_z=128,
            c_token=64,
            c_atom=128,
            c_s=384,
            transformer=DiffusionTransformerConfig(num_blocks=2, num_heads=4),
        )
        return DiffusionModule(config=config).to(device)

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
        noise_level = torch.rand(batch_size, 1, device=device) * 16 + 0.1
        prev_pred = torch.randn(batch_size, 1, n_atoms, 3, device=device) * 10.0

        diffusion_module.eval()
        with torch.no_grad():
            out_without = diffusion_module(
                r_noisy=r_noisy, noise_level=noise_level, context=context
            )
            out_with = diffusion_module(
                r_noisy=r_noisy,
                noise_level=noise_level,
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
        noise_level = torch.full((batch_size, 1), 16.0, device=device)

        with torch.no_grad():
            # First call: no prev_pred available
            pred1 = predictor(x, noise_level)

            # Second call: should use prev_pred from first call
            pred2 = predictor(x, noise_level)

        # Same input but different output because second call has self-conditioning
        assert not torch.allclose(pred1, pred2, atol=1e-5)

    def test_sampler_runs_with_self_conditioning_enabled(
        self, diffusion_module, context, device, monkeypatch
    ):
        """The EDM sample loop should feed previous denoised coords into later steps."""
        calls = {"align": 0}

        def recording_align(pred_pose, true_pose, atom_mask):
            calls["align"] += 1
            assert pred_pose.shape == true_pose.shape
            assert atom_mask.shape == true_pose.shape[:-1]
            return pred_pose, None, None

        monkeypatch.setattr(
            "artios.model.modules.diffusion_backbone.align_pred_to_true",
            recording_align,
        )

        diffusion_module.eval()
        scheduler = EDMInferenceNoiseScheduler(
            s_max=1.0,
            s_min=0.1,
            rho=1.0,
        )

        with torch.no_grad():
            result = diffusion_module.sample(
                context=context,
                noise_scheduler=scheduler,
                n_steps=2,
                n_sample=1,
                centre_coords=False,
                self_conditioning=True,
            )

        n_atoms = context.reference.ref_pos.shape[-2]
        batch_size = context.token.residue_index.shape[0]
        assert result.final.shape == (batch_size, 1, n_atoms, 3)
        assert torch.isfinite(result.final).all()
        assert calls["align"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
