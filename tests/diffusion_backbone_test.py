"""Tests for the refactored diffusion_backbone module using FullContext.

This test file uses the new typed FullContext dataclass instead of raw dicts.
Test data is loaded from tests/testdata/sample_batch_full_context.pt.
"""

import os

import pytest
import torch

from protenix.data.features import FullContext
from protenix.model.modules.diffusion_backbone import (
    DiffusionConditioning,
    DiffusionModule,
)


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def context(device) -> FullContext:
    """Load test context from saved batch data."""
    test_data_path = os.path.join(
        os.path.dirname(__file__), "testdata", "sample_batch_full_context.pt"
    )
    data = torch.load(test_data_path, weights_only=False)
    ctx = FullContext.from_dict(data)

    # Move all tensors to device
    return _move_context_to_device(ctx, device)


def _move_context_to_device(ctx: FullContext, device: torch.device) -> FullContext:
    """Move all tensors in a FullContext to the specified device."""
    from dataclasses import fields

    def move_context(context):
        kwargs = {}
        for field in fields(context):
            value = getattr(context, field.name)
            if torch.is_tensor(value):
                kwargs[field.name] = value.to(device)
            else:
                kwargs[field.name] = value
        return type(context)(**kwargs)

    return FullContext(
        token=move_context(ctx.token),
        reference=move_context(ctx.reference),
        chain_perm=move_context(ctx.chain_perm),
        mask=move_context(ctx.mask),
        motif=move_context(ctx.motif),
        index=move_context(ctx.index),
        entity=move_context(ctx.entity),
        coordinate=move_context(ctx.coordinate),
        basic=ctx.basic,  # BasicContext has no tensors
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

    def test_gradient_flow(self, diffusion_module, context, device):
        """Test that gradients flow through the model correctly."""
        n_atoms = context.reference.ref_pos.shape[-2]
        n_samples = 1

        # Use single batch item for gradient test
        single_context = _slice_context(context, 0)
        single_context = _move_context_to_device(single_context, device)

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


def _slice_context(ctx: FullContext, idx: int) -> FullContext:
    """Slice a batched FullContext to get a single item with batch dim preserved."""
    from dataclasses import fields

    from protenix.data.features import BasicContext

    def slice_context(context, idx):
        kwargs = {}
        for field in fields(context):
            value = getattr(context, field.name)
            if torch.is_tensor(value):
                kwargs[field.name] = value[idx : idx + 1]
            else:
                kwargs[field.name] = value
        return type(context)(**kwargs)

    # Handle BasicContext separately since it has list fields
    basic_kwargs = {
        "atom_array": [ctx.basic.atom_array[idx]] if ctx.basic.atom_array else None,
        "token_array": [ctx.basic.token_array[idx]] if ctx.basic.token_array else None,
        "pdb_id": [ctx.basic.pdb_id[idx]]
        if isinstance(ctx.basic.pdb_id, list)
        else ctx.basic.pdb_id,
        "bioassembly_dict_fpath": [ctx.basic.bioassembly_dict_fpath[idx]]
        if isinstance(ctx.basic.bioassembly_dict_fpath, list)
        else ctx.basic.bioassembly_dict_fpath,
        "n_asym": [ctx.basic.n_asym[idx]]
        if isinstance(ctx.basic.n_asym, list)
        else ctx.basic.n_asym,
        "n_token": [ctx.basic.n_token[idx]]
        if isinstance(ctx.basic.n_token, list)
        else ctx.basic.n_token,
        "n_atom": [ctx.basic.n_atom[idx]]
        if isinstance(ctx.basic.n_atom, list)
        else ctx.basic.n_atom,
    }
    basic = BasicContext(**basic_kwargs)

    return FullContext(
        token=slice_context(ctx.token, idx),
        reference=slice_context(ctx.reference, idx),
        chain_perm=slice_context(ctx.chain_perm, idx),
        mask=slice_context(ctx.mask, idx),
        motif=slice_context(ctx.motif, idx),
        index=slice_context(ctx.index, idx),
        entity=slice_context(ctx.entity, idx),
        coordinate=slice_context(ctx.coordinate, idx),
        basic=basic,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
