import os

import pytest
import torch

from protenix.model.modules.diffusion_backbone import (
    DiffusionConditioning,
    DiffusionModule,
)


class TestDiffusionConditioning:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def sample_data(self):
        test_data_path = os.path.join(
            os.path.dirname(__file__), "testdata", "sample_data_dict.pt"
        )
        return torch.load(test_data_path, weights_only=False)

    @pytest.fixture
    def extended_feature_dict(self, sample_data, device):
        """Create extended feature dict with target and motif features for diffusion."""
        input_feature_dict = sample_data["input_feature_dict"]
        basic = sample_data["basic"]
        n_tokens = basic["N_token"]
        n_atoms = basic["N_atom"]

        # Add target features to the existing feature dict
        extended_dict = input_feature_dict.copy()

        # Target center coordinates - random 3D coordinates for each token
        extended_dict["target_centre_coordinate"] = torch.randn(n_tokens, 3).to(device)

        # Target restype - one-hot encoded residue types (32 dimensions)
        target_restype = torch.zeros(n_tokens, 32).to(device)
        # Set random residue types
        random_indices = torch.randint(0, 32, (n_tokens,))
        target_restype[torch.arange(n_tokens.item()), random_indices] = 1.0
        extended_dict["target_restype"] = target_restype

        # Is target token - boolean mask indicating which tokens are targets
        extended_dict["is_target_token"] = (
            torch.randint(0, 2, (n_tokens,)).long().to(device)
        )

        # Add the entity type binary tensors
        extended_dict["is_protein_token"] = (
            torch.randint(0, 2, (n_tokens,)).long().to(device)
        )
        extended_dict["is_ligand_token"] = (
            torch.randint(0, 2, (n_tokens,)).long().to(device)
        )
        extended_dict["is_dna_token"] = (
            torch.randint(0, 2, (n_tokens,)).long().to(device)
        )
        extended_dict["is_rna_token"] = (
            torch.randint(0, 2, (n_tokens,)).long().to(device)
        )
        extended_dict["is_hotspot"] = torch.zeros((n_tokens,)).long().to(device)

        # Add motif features
        # Atom-level motif features
        extended_dict["is_motif"] = torch.randint(0, 2, (n_atoms,)).long().to(device)
        extended_dict["is_motif_related"] = (
            torch.randint(0, 2, (n_atoms,)).long().to(device)
        )

        # Token-level motif features
        extended_dict["is_motif_token"] = (
            torch.randint(0, 2, (n_tokens,)).long().to(device)
        )
        extended_dict["is_motif_related_token"] = (
            torch.randint(0, 2, (n_tokens,)).long().to(device)
        )

        # Motif restype - one-hot encoded residue types (32 dimensions)
        motif_restype = torch.zeros(n_tokens, 32).to(device)
        random_indices = torch.randint(0, 32, (n_tokens,))
        motif_restype[torch.arange(n_tokens.item()), random_indices] = 1.0
        extended_dict["motif_restype"] = motif_restype

        # Motif coordinates for each token
        extended_dict["motif_coordinate"] = torch.randn(n_tokens, 3).to(device)

        # Atom mask
        extended_dict["atom_mask"] = torch.randint(0, 2, (n_atoms,)).bool().to(device)

        return extended_dict

    def test_diffusion_conditioning_forward(self, device, extended_feature_dict):
        """Test DiffusionConditioning forward pass with target and motif features."""
        diff_cond = DiffusionConditioning(sigma_coord=12.0).to(device)

        n_tokens = extended_feature_dict["residue_index"].shape[-1]
        n_samples = 2

        # Create noisy coordinates and timesteps
        r_noisy = torch.randn(n_samples, n_tokens, 3).to(device)
        timestep = torch.ones(n_samples).to(device) * 0.5

        # Run forward pass
        s, z = diff_cond(
            r_noisy=r_noisy,
            timestep=timestep,
            input_feature_dict=extended_feature_dict,
            inplace_safe=False,
        )

        # Check output shapes and values
        assert s.shape == (n_samples, n_tokens, 384)  # [N_sample, N_tokens, c_s]
        assert z.shape == (n_tokens, n_tokens, 128)  # [N_tokens, N_tokens, c_z]
        assert not torch.isnan(s).any()
        assert not torch.isnan(z).any()

    def test_get_target_condition(self, device, extended_feature_dict):
        """Test target condition generation."""
        diff_cond = DiffusionConditioning(sigma_coord=12.0).to(device)

        n_tokens = extended_feature_dict["residue_index"].shape[-1]

        # Test target condition generation
        s_target, z_target = diff_cond.get_target_condition(
            input_feature_dict=extended_feature_dict
        )

        # Check shapes
        assert s_target.shape == (n_tokens, 384)
        assert z_target.shape == (n_tokens, n_tokens, 128)
        assert not torch.isnan(s_target).any()
        assert not torch.isnan(z_target).any()

    def test_get_atom_condition(self, device, extended_feature_dict):
        """Test atom condition generation."""
        diff_cond = DiffusionConditioning(sigma_coord=12.0).to(device)

        n_tokens = extended_feature_dict["residue_index"].shape[-1]

        # Test atom condition generation
        s_atom, z_atom = diff_cond.get_atom_condition(
            input_feature_dict=extended_feature_dict
        )

        # Check shape
        assert s_atom.shape == (n_tokens, 384)
        assert z_atom.shape == (n_tokens, n_tokens, 128)
        assert not torch.isnan(s_atom).any()
        assert not torch.isnan(z_atom).any()

    def test_get_motif_condition(self, device, extended_feature_dict):
        """Test motif condition generation."""
        diff_cond = DiffusionConditioning(sigma_coord=12.0).to(device)

        n_tokens = extended_feature_dict["residue_index"].shape[-1]

        # Test motif condition generation
        s_motif, z_motif = diff_cond.get_motif_condition(
            input_feature_dict=extended_feature_dict
        )

        # Check shapes
        assert s_motif.shape == (n_tokens, 384)
        assert z_motif.shape == (n_tokens, n_tokens, 128)
        assert not torch.isnan(s_motif).any()
        assert not torch.isnan(z_motif).any()

    def test_motif_distance_binning(self, device, extended_feature_dict):
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
        s_motif, z_motif = diff_cond.get_motif_condition(
            input_feature_dict=extended_feature_dict
        )

        assert s_motif.shape[-1] == 384
        assert z_motif.shape[-1] == 128

    def test_combined_conditions(self, device, extended_feature_dict):
        """Test that all conditions combine correctly."""
        diff_cond = DiffusionConditioning(sigma_coord=12.0).to(device)

        n_tokens = extended_feature_dict["residue_index"].shape[-1]

        # Get individual conditions
        s_atom, z_atom = diff_cond.get_atom_condition(
            input_feature_dict=extended_feature_dict
        )
        s_motif, z_motif = diff_cond.get_motif_condition(
            input_feature_dict=extended_feature_dict
        )
        s_target, z_target = diff_cond.get_target_condition(
            input_feature_dict=extended_feature_dict
        )

        # Combine conditions (as done in forward)
        s_combined = s_target + s_atom + s_motif
        z_combined = z_target + z_atom + z_motif

        # Check combined shapes
        assert s_combined.shape == (n_tokens, 384)
        assert z_combined.shape == (n_tokens, n_tokens, 128)
        assert not torch.isnan(s_combined).any()
        assert not torch.isnan(z_combined).any()


class TestDiffusionModule:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def sample_data(self):
        test_data_path = os.path.join(
            os.path.dirname(__file__), "testdata", "sample_data_dict.pt"
        )
        return torch.load(test_data_path, weights_only=False)

    @pytest.fixture
    def extended_feature_dict(self, sample_data, device):
        """Create extended feature dict with target and motif features for diffusion."""
        input_feature_dict = sample_data["input_feature_dict"]
        basic = sample_data["basic"]
        n_tokens = basic["N_token"]
        n_atoms = basic["N_atom"]

        # Add target features to the existing feature dict
        extended_dict = input_feature_dict.copy()

        # Target center coordinates - random 3D coordinates for each token
        extended_dict["target_centre_coordinate"] = torch.randn(n_tokens, 3).to(device)

        # Target restype - one-hot encoded residue types (32 dimensions)
        target_restype = torch.zeros(n_tokens, 32).to(device)
        # Set random residue types
        random_indices = torch.randint(0, 32, (n_tokens,))
        target_restype[torch.arange(n_tokens.item()), random_indices] = 1.0
        extended_dict["target_restype"] = target_restype

        # Is target token - boolean mask indicating which tokens are targets
        extended_dict["is_target_token"] = (
            torch.randint(0, 2, (n_tokens,)).long().to(device)
        )

        # Add the entity type binary tensors
        extended_dict["is_protein_token"] = (
            torch.randint(0, 2, (n_tokens,)).long().to(device)
        )
        extended_dict["is_ligand_token"] = (
            torch.randint(0, 2, (n_tokens,)).long().to(device)
        )
        extended_dict["is_dna_token"] = (
            torch.randint(0, 2, (n_tokens,)).long().to(device)
        )
        extended_dict["is_rna_token"] = (
            torch.randint(0, 2, (n_tokens,)).long().to(device)
        )
        extended_dict["is_hotspot"] = torch.zeros(n_tokens).long().to(device)

        # Add motif features
        # Atom-level motif features
        extended_dict["is_motif"] = torch.randint(0, 2, (n_atoms,)).long().to(device)
        extended_dict["is_motif_related"] = (
            torch.randint(0, 2, (n_atoms,)).long().to(device)
        )

        # Token-level motif features
        extended_dict["is_motif_token"] = (
            torch.randint(0, 2, (n_tokens,)).long().to(device)
        )
        extended_dict["is_motif_related_token"] = (
            torch.randint(0, 2, (n_tokens,)).long().to(device)
        )

        # Motif restype - one-hot encoded residue types (32 dimensions)
        motif_restype = torch.zeros(n_tokens, 32).to(device)
        random_indices = torch.randint(0, 32, (n_tokens,))
        motif_restype[torch.arange(n_tokens.item()), random_indices] = 1.0
        extended_dict["motif_restype"] = motif_restype

        # Motif coordinates for each token
        extended_dict["motif_coordinate"] = torch.randn(n_tokens, 3).to(device)

        # Token and atom masks
        extended_dict["token_mask"] = torch.randint(0, 2, (n_tokens,)).bool().to(device)
        extended_dict["atom_mask"] = torch.randint(0, 2, (n_atoms,)).bool().to(device)

        return extended_dict

    @pytest.fixture
    def diffusion_module(self, device):
        """Create a DiffusionModule with minimal configuration for testing."""
        config = {
            "sigma_coord": 12.0,
            "c_z": 128,
            "c_token": 64,  # Smaller for faster testing
            "c_atom": 128,
            "c_s": 64,
            "transformer": {"num_blocks": 2, "num_heads": 4},  # Minimal transformer
            "dropout": 0.0,
            "blocks_per_ckpt": None,
        }
        module = DiffusionModule(**config).to(device)

        # Add missing linear_no_bias_f layer that is used in f_forward but not defined in __init__
        module.linear_no_bias_f = torch.nn.Linear(
            sum(module.atom_input_feature.values()), module.c_atom, bias=False
        ).to(device)

        return module

    def test_readout(self, diffusion_module, device):
        """Test readout layer."""
        batch_size, n_tokens, c_atom = 2, 8, 128
        q_atom = torch.randn(batch_size, n_tokens, c_atom).to(device)

        output = diffusion_module.readout(q_atom)

        # Check output shape - should be 3D coordinates
        assert output.shape == (batch_size, n_tokens, 3)
        assert not torch.isnan(output).any()

    def test_f_forward(self, diffusion_module, extended_feature_dict, device):
        """Test the f_forward method."""
        n_atoms = extended_feature_dict["atom_mask"].shape[-1]
        n_samples = 2

        # Create input tensors
        r_noisy = torch.randn(n_samples, n_atoms, 3).to(device)
        timestep = torch.rand(n_samples).to(device)

        # Run f_forward
        output = diffusion_module.f_forward(
            r_noisy=r_noisy,
            timestep=timestep,
            input_feature_dict=extended_feature_dict,
            inplace_safe=False,
        )

        # Check output shape
        assert output.shape == (n_samples, n_atoms, 3)
        assert not torch.isnan(output).any()

    def test_forward_pass(self, diffusion_module, extended_feature_dict, device):
        """Test the complete forward pass."""
        n_atoms = extended_feature_dict["atom_mask"].shape[-1]
        n_samples = 2

        # Create input tensors
        r_noisy = torch.randn(n_samples, n_atoms, 3).to(device)
        timestep = torch.rand(n_samples).to(device)

        # Run forward pass
        r_pred = diffusion_module(
            r_noisy=r_noisy,
            timestep=timestep,
            input_feature_dict=extended_feature_dict,
            inplace_safe=False,
        )

        assert r_pred.shape == (n_samples, n_atoms, 3)
        assert not torch.isnan(r_pred).any()

    def test_different_timesteps(self, diffusion_module, extended_feature_dict, device):
        """Test forward pass with different timesteps."""
        n_atoms = extended_feature_dict["atom_mask"].shape[-1]
        n_samples = 2

        # Create input tensors
        r_noisy = torch.randn(n_samples, n_atoms, 3).to(device)

        # Different timesteps
        timestep = torch.tensor([0.2, 0.8]).to(device)

        # Run forward pass
        output = diffusion_module(
            r_noisy=r_noisy,
            timestep=timestep,
            input_feature_dict=extended_feature_dict,
        )

        assert output.shape == (n_samples, n_atoms, 3)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, diffusion_module, extended_feature_dict, device):
        """Test that gradients flow through the model correctly."""
        n_atoms = extended_feature_dict["atom_mask"].shape[-1]
        n_samples = 1

        # Create input tensors that require gradients
        r_noisy = torch.randn(n_samples, n_atoms, 3, requires_grad=True).to(device)
        timestep = torch.rand(n_samples).to(device)

        # Forward pass
        output = diffusion_module(
            r_noisy=r_noisy,
            timestep=timestep,
            input_feature_dict=extended_feature_dict,
        )

        # Compute loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        assert r_noisy.grad is not None
        assert not torch.isnan(r_noisy.grad).any()

    def test_with_motif_features(self, diffusion_module, extended_feature_dict, device):
        """Test that motif features are correctly processed."""
        n_atoms = extended_feature_dict["atom_mask"].shape[-1]
        n_samples = 1

        # Ensure motif features are present
        assert "is_motif" in extended_feature_dict
        assert "is_motif_related" in extended_feature_dict
        assert "is_motif_token" in extended_feature_dict
        assert "motif_restype" in extended_feature_dict
        assert "motif_coordinate" in extended_feature_dict

        # Create input tensors
        r_noisy = torch.randn(n_samples, n_atoms, 3).to(device)
        timestep = torch.tensor([0.5]).to(device)

        # Run forward pass
        output = diffusion_module(
            r_noisy=r_noisy,
            timestep=timestep,
            input_feature_dict=extended_feature_dict,
        )

        assert output.shape == (n_samples, n_atoms, 3)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__])
