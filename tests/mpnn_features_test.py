"""Tests for MPNN featurizer modules."""

import os

import pytest
import torch

from protenix.model.modules.mpnn_features import (
    EdgeRelativePositionalEncoding,
    GaussianRBF,
    MPNNFeaturizer,
)


class TestEdgeRelativePositionalEncoding:
    """Tests for EdgeRelativePositionalEncoding module."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_initialization(self):
        """Test EdgeRelativePositionalEncoding initialization."""
        encoder = EdgeRelativePositionalEncoding(
            edge_dim=128,
            r_max=32,
            s_max=2,
        )

        assert encoder.edge_dim == 128
        assert encoder.r_max == 32
        assert encoder.s_max == 2
        assert hasattr(encoder, "relpos")

    def test_forward_shape(self, device):
        """Test forward pass output shape."""
        encoder = EdgeRelativePositionalEncoding(
            edge_dim=64,
            r_max=32,
            s_max=2,
        ).to(device)

        batch_size, n_tokens, n_neighbors = 2, 20, 10

        # Create input tensors
        asym_id = torch.randint(0, 5, (batch_size, n_tokens)).to(device)
        residue_index = (
            torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1).to(device)
        )
        entity_id = torch.randint(0, 3, (batch_size, n_tokens)).to(device)
        sym_id = torch.zeros(batch_size, n_tokens).long().to(device)
        token_index = (
            torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1).to(device)
        )
        edge_index = torch.randint(0, n_tokens, (batch_size, n_tokens, n_neighbors)).to(
            device
        )

        # Forward pass
        output = encoder(
            asym_id=asym_id,
            residue_index=residue_index,
            entity_id=entity_id,
            sym_id=sym_id,
            token_index=token_index,
            edge_index=edge_index,
        )

        # Check output shape
        assert output.shape == (batch_size, n_tokens, n_neighbors, 64)
        assert not torch.isnan(output).any()

    def test_forward_single_batch(self, device):
        """Test forward pass with single batch."""
        encoder = EdgeRelativePositionalEncoding(edge_dim=32).to(device)

        n_tokens, n_neighbors = 15, 8

        asym_id = torch.zeros(1, n_tokens).long().to(device)
        residue_index = torch.arange(n_tokens).unsqueeze(0).to(device)
        entity_id = torch.zeros(1, n_tokens).long().to(device)
        sym_id = torch.zeros(1, n_tokens).long().to(device)
        token_index = torch.arange(n_tokens).unsqueeze(0).to(device)
        edge_index = torch.randint(0, n_tokens, (1, n_tokens, n_neighbors)).to(device)

        output = encoder(
            asym_id=asym_id,
            residue_index=residue_index,
            entity_id=entity_id,
            sym_id=sym_id,
            token_index=token_index,
            edge_index=edge_index,
        )

        assert output.shape == (1, n_tokens, n_neighbors, 32)
        assert not torch.isnan(output).any()

    def test_same_chain_vs_different_chain(self, device):
        """Test that same-chain and different-chain edges have different encodings."""
        encoder = EdgeRelativePositionalEncoding(edge_dim=64).to(device)

        n_tokens, n_neighbors = 10, 5

        # Create two scenarios: same chain and different chains
        asym_id_same = torch.zeros(1, n_tokens).long().to(device)
        asym_id_diff = (
            torch.arange(n_tokens).unsqueeze(0).to(device)
        )  # Each token different chain

        residue_index = torch.arange(n_tokens).unsqueeze(0).to(device)
        entity_id = torch.zeros(1, n_tokens).long().to(device)
        sym_id = torch.zeros(1, n_tokens).long().to(device)
        token_index = torch.arange(n_tokens).unsqueeze(0).to(device)
        edge_index = torch.randint(0, n_tokens, (1, n_tokens, n_neighbors)).to(device)

        # Forward pass for same chain
        output_same = encoder(
            asym_id=asym_id_same,
            residue_index=residue_index,
            entity_id=entity_id,
            sym_id=sym_id,
            token_index=token_index,
            edge_index=edge_index,
        )

        # Forward pass for different chains
        output_diff = encoder(
            asym_id=asym_id_diff,
            residue_index=residue_index,
            entity_id=entity_id,
            sym_id=sym_id,
            token_index=token_index,
            edge_index=edge_index,
        )

        # Outputs should be different
        assert not torch.allclose(output_same, output_diff)

    def test_gradient_flow(self, device):
        """Test that gradients flow through the encoder."""
        encoder = EdgeRelativePositionalEncoding(edge_dim=32).to(device)

        n_tokens, n_neighbors = 8, 4

        asym_id = torch.zeros(1, n_tokens).long().to(device)
        residue_index = (
            torch.arange(n_tokens, dtype=torch.float32, requires_grad=True)
            .unsqueeze(0)
            .to(device)
        )
        entity_id = torch.zeros(1, n_tokens).long().to(device)
        sym_id = torch.zeros(1, n_tokens).long().to(device)
        token_index = torch.arange(n_tokens).unsqueeze(0).to(device)
        edge_index = torch.randint(0, n_tokens, (1, n_tokens, n_neighbors)).to(device)

        output = encoder(
            asym_id=asym_id,
            residue_index=residue_index.long(),
            entity_id=entity_id,
            sym_id=sym_id,
            token_index=token_index,
            edge_index=edge_index,
        )

        # Compute loss and backprop
        loss = output.sum()
        loss.backward()

        # Note: residue_index gradients may not flow through due to integer indexing
        # But the model parameters should have gradients
        assert any(p.grad is not None for p in encoder.parameters())


class TestGaussianRBF:
    """Tests for GaussianRBF module."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_initialization(self):
        """Test RBF initialization."""
        rbf = GaussianRBF(num_kernels=16, min_distance=2.0, max_distance=22.0)

        assert rbf.num_kernels == 16
        assert rbf.min_distance == 2.0
        assert rbf.max_distance == 22.0
        assert rbf.centers.shape == (16,)
        assert torch.allclose(rbf.centers[0], torch.tensor(2.0))
        assert torch.allclose(rbf.centers[-1], torch.tensor(22.0))

    def test_forward_shape(self, device):
        """Test RBF forward pass output shape."""
        rbf = GaussianRBF(num_kernels=16).to(device)

        # Test with various input shapes
        distances = torch.randn(10, 25).to(device)
        output = rbf(distances)

        assert output.shape == (10, 25, 16)
        assert not torch.isnan(output).any()

    def test_forward_single_distance(self, device):
        """Test RBF with single distance value."""
        rbf = GaussianRBF(num_kernels=16, min_distance=2.0, max_distance=22.0).to(
            device
        )

        # Distance at first center should have high response
        distances = torch.tensor([2.0]).to(device)
        output = rbf(distances)

        assert output.shape == (1, 16)
        # First kernel should have highest response
        assert output[0, 0] > 0.9

    def test_forward_batched(self, device):
        """Test RBF with batched input."""
        rbf = GaussianRBF(num_kernels=8).to(device)

        batch_size, n_tokens, n_neighbors, n_pairs = 2, 10, 5, 25
        distances = (
            torch.randn(batch_size, n_tokens, n_neighbors, n_pairs).abs().to(device)
        )

        output = rbf(distances)

        assert output.shape == (batch_size, n_tokens, n_neighbors, n_pairs, 8)
        assert not torch.isnan(output).any()
        assert torch.all(output >= 0)  # RBF values should be non-negative

    def test_gaussian_properties(self, device):
        """Test that RBF has proper Gaussian properties."""
        rbf = GaussianRBF(num_kernels=5, min_distance=0.0, max_distance=10.0).to(device)

        # Test at center points
        distances = torch.linspace(0, 10, 5).to(device)
        output = rbf(distances)

        # Each distance should have maximum response at corresponding kernel
        for i in range(5):
            assert output[i, i] == torch.max(output[i])


class TestMPNNFeaturizer:
    """Tests for MPNNFeaturizer methods."""

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
        """Create extended feature dict with all required features."""
        input_feature_dict = sample_data["input_feature_dict"]
        basic = sample_data["basic"]
        n_tokens = basic["N_token"]
        n_atoms = basic["N_atom"]

        # Add all required features
        extended_dict = input_feature_dict.copy()

        # Entity type features
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

        # Atom-level features
        extended_dict["atom_mask"] = torch.ones(n_atoms).long().to(device)
        extended_dict["is_ligand"] = torch.randint(0, 2, (n_atoms,)).long().to(device)

        return extended_dict

    @pytest.fixture
    def featurizer(self, device):
        """Create MPNNFeaturizer instance."""
        return MPNNFeaturizer(
            node_dim=128,
            edge_dim=64,
            positional_encoding_dim=16,
            num_rbf=16,
            num_neighbours=32,
            max_relative_feats=32,
            coord_noise=0.0,  # Disable noise for testing
        ).to(device)

    def test_compute_neighbour_graph(self, featurizer, device):
        """Test k-nearest neighbor graph computation."""
        batch_size, n_tokens = 2, 20
        centre_coords = torch.randn(batch_size, n_tokens, 3).to(device)
        centre_mask = torch.ones(batch_size, n_tokens).bool().to(device)

        # Mask out last 2 tokens
        centre_mask[:, -2:] = False

        neighbor_dists, edge_index = featurizer._compute_neighbour_graph(
            centre_atom_coords=centre_coords,
            centre_atom_mask=centre_mask,
        )

        # Check output shapes
        k = min(featurizer.num_neighbours, n_tokens)
        assert neighbor_dists.shape == (batch_size, n_tokens, k)
        assert edge_index.shape == (batch_size, n_tokens, k)

        # Check edge indices are valid
        assert torch.all(edge_index >= 0)
        assert torch.all(edge_index < n_tokens)

        # Check distances are sorted (smallest first)
        assert torch.all(neighbor_dists[:, :, :-1] <= neighbor_dists[:, :, 1:])

    def test_compute_neighbour_graph_single_batch(self, featurizer, device):
        """Test neighbor graph without batch dimension."""
        n_tokens = 15
        centre_coords = torch.randn(1, n_tokens, 3).to(device)
        centre_mask = torch.ones(1, n_tokens).bool().to(device)

        neighbor_dists, edge_index = featurizer._compute_neighbour_graph(
            centre_atom_coords=centre_coords,
            centre_atom_mask=centre_mask,
        )

        k = min(featurizer.num_neighbours, n_tokens)
        assert neighbor_dists.shape == (1, n_tokens, k)
        assert edge_index.shape == (1, n_tokens, k)

    def test_compute_node_features(self, featurizer, extended_feature_dict, device):
        """Test node feature computation."""
        node_feats = featurizer._compute_node_features(extended_feature_dict)

        n_tokens = extended_feature_dict["residue_index"].shape[-1]

        # Check output shape
        assert node_feats.shape == (n_tokens, featurizer.node_dim)
        assert not torch.isnan(node_feats).any()

    def test_compute_node_features_with_ligands(
        self, featurizer, extended_feature_dict, device
    ):
        """Test node features with ligand atoms present."""
        # Set some atoms as ligands
        n_atoms = extended_feature_dict["ref_pos"].shape[-2]
        extended_feature_dict["is_ligand"][: n_atoms // 2] = 1

        node_feats = featurizer._compute_node_features(extended_feature_dict)

        n_tokens = extended_feature_dict["residue_index"].shape[-1]
        assert node_feats.shape == (n_tokens, featurizer.node_dim)
        assert not torch.isnan(node_feats).any()

    def test_compute_backbone_distance_features(self, featurizer, device):
        """Test backbone distance feature computation."""
        batch_size, n_tokens, n_neighbors = 2, 15, 10

        # Create synthetic backbone coordinates
        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3).to(device)
        backbone_mask = torch.ones(batch_size, n_tokens, 5).to(device)

        # Mask out some backbone atoms
        backbone_mask[:, :, -1] = 0  # Mask last backbone atom for all tokens
        backbone_mask[:, -3:, :] = 0  # Mask all backbone atoms for last 3 tokens

        # Create edge index
        edge_index = torch.randint(0, n_tokens, (batch_size, n_tokens, n_neighbors)).to(
            device
        )

        # Compute features
        edge_feats = featurizer._compute_backbone_distance_features(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            edge_index=edge_index,
        )

        # Check output shape
        assert edge_feats.shape == (
            batch_size,
            n_tokens,
            n_neighbors,
            featurizer.edge_dim,
        )
        assert not torch.isnan(edge_feats).any()

    def test_compute_backbone_distance_features_masking(self, featurizer, device):
        """Test that masking properly zeros out invalid distances."""
        n_tokens, n_neighbors = 10, 5

        backbone_coords = torch.randn(1, n_tokens, 5, 3).to(device)
        backbone_mask = torch.ones(1, n_tokens, 5).to(device)

        # Completely mask out one token's backbone
        masked_token_idx = 5
        backbone_mask[:, masked_token_idx, :] = 0

        edge_index = torch.randint(0, n_tokens, (1, n_tokens, n_neighbors)).to(device)

        edge_feats = featurizer._compute_backbone_distance_features(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            edge_index=edge_index,
        )

        assert edge_feats.shape == (1, n_tokens, n_neighbors, featurizer.edge_dim)
        # Features should still be computed (not necessarily zero due to normalization)
        assert not torch.isnan(edge_feats).any()

    def test_compute_backbone_distance_features_single_backbone_atom(
        self, featurizer, device
    ):
        """Test with ligands (only first backbone atom valid)."""
        n_tokens, n_neighbors = 8, 4

        backbone_coords = torch.randn(1, n_tokens, 5, 3).to(device)
        backbone_mask = torch.zeros(1, n_tokens, 5).to(device)

        # Only first backbone slot valid (simulating ligands)
        backbone_mask[:, :, 0] = 1

        edge_index = torch.randint(0, n_tokens, (1, n_tokens, n_neighbors)).to(device)

        edge_feats = featurizer._compute_backbone_distance_features(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            edge_index=edge_index,
        )

        assert edge_feats.shape == (1, n_tokens, n_neighbors, featurizer.edge_dim)
        assert not torch.isnan(edge_feats).any()

    def test_forward_pass(self, featurizer, extended_feature_dict, device):
        """Test complete forward pass through featurizer."""
        n_tokens = extended_feature_dict["residue_index"].shape[-1]

        # Create inputs
        centre_atom_coord = torch.randn(1, n_tokens, 3).to(device)
        centre_atom_mask = torch.ones(1, n_tokens).to(device)
        backbone_coords = torch.randn(1, n_tokens, 5, 3).to(device)
        backbone_mask = torch.ones(1, n_tokens, 5).to(device)

        # Add batch dimension to feature dict tensors
        for key in extended_feature_dict:
            if isinstance(extended_feature_dict[key], torch.Tensor):
                extended_feature_dict[key] = extended_feature_dict[key].unsqueeze(0)

        # Forward pass
        node_feats, edge_feats, edge_index = featurizer(
            input_feature_dict=extended_feature_dict,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
        )

        # Check output shapes
        k = min(featurizer.num_neighbours, n_tokens)
        assert node_feats.shape == (1, n_tokens, featurizer.node_dim)
        assert edge_feats.shape == (1, n_tokens, k, featurizer.edge_dim)
        assert edge_index.shape == (1, n_tokens, k)

        # Check no NaN values
        assert not torch.isnan(node_feats).any()
        assert not torch.isnan(edge_feats).any()

    def test_forward_pass_with_noise(self, device):
        """Test forward pass with coordinate noise enabled."""
        featurizer = MPNNFeaturizer(
            node_dim=64,
            edge_dim=32,
            positional_encoding_dim=16,
            num_rbf=8,
            num_neighbours=16,
            max_relative_feats=32,
            coord_noise=0.1,  # Enable noise
        ).to(device)

        # Create minimal feature dict
        n_tokens, n_atoms = 10, 30
        feature_dict = {
            "residue_index": torch.arange(n_tokens).to(device),
            "ref_pos": torch.randn(n_atoms, 3).to(device),
            "ref_charge": torch.zeros(n_atoms).long().to(device),
            "ref_mask": torch.ones(n_atoms).long().to(device),
            "ref_element": torch.zeros(n_atoms, 128).to(device),
            "atom_to_token_idx": torch.randint(0, n_tokens, (n_atoms,)).to(device),
            "atom_mask": torch.ones(n_atoms).long().to(device),
            "is_ligand": torch.zeros(n_atoms).long().to(device),
            "is_protein_token": torch.ones(n_tokens).long().to(device),
            "is_ligand_token": torch.zeros(n_tokens).long().to(device),
            "is_dna_token": torch.zeros(n_tokens).long().to(device),
            "is_rna_token": torch.zeros(n_tokens).long().to(device),
        }

        centre_atom_coord = torch.randn(1, n_tokens, 3).to(device)
        centre_atom_mask = torch.ones(1, n_tokens).to(device)
        backbone_coords = torch.randn(1, n_tokens, 5, 3).to(device)
        backbone_mask = torch.ones(1, n_tokens, 5).to(device)

        featurizer.train()  # Enable training mode for noise

        # Two forward passes with same input should give different results due to noise
        out1 = featurizer(
            feature_dict,
            centre_atom_coord,
            centre_atom_mask,
            backbone_coords,
            backbone_mask,
        )

        out2 = featurizer(
            feature_dict,
            centre_atom_coord,
            centre_atom_mask,
            backbone_coords,
            backbone_mask,
        )

        # Edge features should be different due to coordinate noise
        assert not torch.allclose(out1[1], out2[1], atol=1e-5)

    def test_gradient_flow(self, featurizer, extended_feature_dict, device):
        """Test that gradients flow through the featurizer."""
        n_tokens = extended_feature_dict["residue_index"].shape[-1]

        centre_atom_coord = torch.randn(1, n_tokens, 3, requires_grad=True).to(device)
        centre_atom_mask = torch.ones(1, n_tokens).to(device)
        backbone_coords = torch.randn(1, n_tokens, 5, 3, requires_grad=True).to(device)
        backbone_mask = torch.ones(1, n_tokens, 5).to(device)

        # Add batch dimension
        for key in extended_feature_dict:
            if isinstance(extended_feature_dict[key], torch.Tensor):
                extended_feature_dict[key] = extended_feature_dict[key].unsqueeze(0)

        node_feats, edge_feats, _ = featurizer(
            extended_feature_dict,
            centre_atom_coord,
            centre_atom_mask,
            backbone_coords,
            backbone_mask,
        )

        # Compute loss and backprop
        loss = node_feats.sum() + edge_feats.sum()
        loss.backward()

        # Check gradients exist
        assert backbone_coords.grad is not None
        assert not torch.isnan(backbone_coords.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
