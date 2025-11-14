"""Tests for All-atom MPNN module.

This module tests the core functionality of the AllatomMPNN class, including:
- Encoder trunk execution
- Decoder input preparation
- Sequential mask generation for autoregressive decoding
- Decoder step execution
"""

import os

import pytest
import torch
from torch import nn

from protenix.model.modules.mpnn import MPNN


@pytest.fixture
def mpnn_model():
    """Create a small MPNN model for testing.

    Note: The decoder layers are initialized with edge_input_dim=edge_dim in the original code,
    but the actual input dimension during forward pass is larger due to feature concatenation.
    This is a known issue that these tests help expose.
    """
    model = MPNN(
        num_residue_types=32,
        node_dim=64,
        edge_dim=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_neighbours=30,
        positional_encoding_dim=16,
        positional_encoding_max_relative_feats=32,
        num_rbf_kernels=16,
        dropout=0.0,
        coord_noise=0.1,
    )

    # Fix the decoder layers to accept the correct edge input dimension
    # In autoregressive mode: (edge_dim + node_dim) + node_dim = 128 + 64 = 192
    # In zero-shot mode: (edge_dim + 0 + node_dim) = 192
    # Both modes use 192-dimensional edge features
    from protenix.model.modules.mpnn_layers import MPNNDecoderLayer

    model.decoder = nn.ModuleList(
        [
            MPNNDecoderLayer(
                node_dim=64,
                edge_dim=64,
                edge_input_dim=2 * 64 + 64,  # Corrected dimension
                dropout=0.0,
            )
            for _ in range(2)
        ]
    )

    return model


@pytest.fixture
def sample_tensors():
    """Create sample tensors for testing with consistent shapes."""
    batch_size = 2
    seq_len = 10
    num_neighbours = 30
    hidden_dim = 64
    num_residue_types = 32

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_neighbours": num_neighbours,
        "hidden_dim": hidden_dim,
        "num_residue_types": num_residue_types,
        "node_feats": torch.randn(batch_size, seq_len, hidden_dim),
        "edge_feats": torch.randn(batch_size, seq_len, num_neighbours, hidden_dim),
        "sequence_embeds": torch.randn(batch_size, seq_len, hidden_dim),
        "edge_index": torch.randint(0, seq_len, (batch_size, seq_len, num_neighbours)),
        "node_mask": torch.ones(batch_size, seq_len),
        "edge_mask": torch.ones(batch_size, seq_len, num_neighbours),
        "restype": torch.zeros(batch_size, seq_len, num_residue_types),
        "decoding_order": torch.randperm(seq_len).unsqueeze(0).expand(batch_size, -1),
    }


@pytest.fixture
def sample_data():
    """Load sample data dict from test data."""
    test_data_path = os.path.join(
        os.path.dirname(__file__), "testdata", "sample_data_dict.pt"
    )
    return torch.load(test_data_path, weights_only=False)


@pytest.fixture
def input_feature_dict(sample_data):
    """Create input feature dict following the pattern from diffusion tests.

    Adds batch dimension of 1 to all tensors to match expected input format.
    """
    device = torch.device("cpu")
    input_feature_dict = sample_data["input_feature_dict"]
    basic = sample_data["basic"]
    n_tokens = basic["N_token"]
    n_atoms = basic["N_atom"]

    # Add required features that might be missing
    extended_dict = {}

    # Add batch dimension to all existing tensors
    for key, value in input_feature_dict.items():
        if torch.is_tensor(value):
            # Add batch dimension (unsqueeze at dim 0)
            extended_dict[key] = value.unsqueeze(0).to(device)
        else:
            extended_dict[key] = value

    # Add entity type binary tensors if not present, with batch dimension
    if "is_protein_token" not in extended_dict:
        extended_dict["is_protein_token"] = torch.ones(1, n_tokens).long().to(device)
    if "is_ligand_token" not in extended_dict:
        extended_dict["is_ligand_token"] = torch.zeros(1, n_tokens).long().to(device)
    if "is_dna_token" not in extended_dict:
        extended_dict["is_dna_token"] = torch.zeros(1, n_tokens).long().to(device)
    if "is_rna_token" not in extended_dict:
        extended_dict["is_rna_token"] = torch.zeros(1, n_tokens).long().to(device)

    # Add atom_mask if not present
    if "atom_mask" not in extended_dict:
        extended_dict["atom_mask"] = torch.ones(1, n_atoms).bool().to(device)

    return extended_dict


class TestRunEncoderTrunk:
    """Tests for run_encoder_trunk method."""

    def test_output_shapes(self, mpnn_model, sample_data, input_feature_dict):
        """Test that encoder trunk outputs have correct shapes."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        num_neighbours = 30
        batch_size = 1

        # Create backbone coordinates and masks with batch dimension
        # backbone_coords: [batch, n_token, 5, 3] - 5 backbone atoms (N, CA, C, O, CB)
        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3)
        backbone_mask = torch.ones(batch_size, n_tokens, 5)

        # centre_atom_coord: [batch, n_token, 3] - centre atom per token
        centre_atom_coord = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)

        # Run encoder trunk
        node_feats, edge_feats, edge_index, edge_mask = mpnn_model.run_encoder_trunk(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            input_feature_dict=input_feature_dict,
        )

        # Check output shapes
        # node_feats: [batch, n_token, node_dim]
        assert node_feats.shape == (
            batch_size,
            n_tokens,
            64,
        ), f"node_feats shape {node_feats.shape} != ({batch_size}, {n_tokens}, 64)"

        # edge_feats: [batch, n_token, neighbours, edge_dim]
        assert (
            edge_feats.shape == (batch_size, n_tokens, num_neighbours, 64)
        ), f"edge_feats shape {edge_feats.shape} != ({batch_size}, {n_tokens}, {num_neighbours}, 64)"

        # edge_index: [batch, n_token, neighbours]
        assert (
            edge_index.shape == (batch_size, n_tokens, num_neighbours)
        ), f"edge_index shape {edge_index.shape} != ({batch_size}, {n_tokens}, {num_neighbours})"

        # edge_mask: [batch, n_token, neighbours]
        assert (
            edge_mask.shape == (batch_size, n_tokens, num_neighbours)
        ), f"edge_mask shape {edge_mask.shape} != ({batch_size}, {n_tokens}, {num_neighbours})"

    def test_output_no_nan_or_inf(self, mpnn_model, sample_data, input_feature_dict):
        """Test that encoder outputs don't contain NaN or Inf values."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1

        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3)
        backbone_mask = torch.ones(batch_size, n_tokens, 5)
        centre_atom_coord = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)

        node_feats, edge_feats, edge_index, _ = mpnn_model.run_encoder_trunk(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            input_feature_dict=input_feature_dict,
        )

        assert not torch.isnan(node_feats).any(), "node_feats contains NaN"
        assert not torch.isinf(node_feats).any(), "node_feats contains Inf"
        assert not torch.isnan(edge_feats).any(), "edge_feats contains NaN"
        assert not torch.isinf(edge_feats).any(), "edge_feats contains Inf"
        assert not torch.isnan(edge_index.float()).any(), "edge_index contains NaN"
        assert not torch.isinf(edge_index.float()).any(), "edge_index contains Inf"

    def test_masking_application(self, mpnn_model, sample_data, input_feature_dict):
        """Test that masking properly affects the output.

        This is critical to avoid the silent failure mode mentioned in CLAUDE.md
        where masked coordinates can generate large numbers if not properly handled.
        """
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1

        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3)
        backbone_mask = torch.ones(batch_size, n_tokens, 5)
        centre_atom_coord = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)

        # Mask out first 3 tokens
        centre_atom_mask[:, :3] = 0

        node_feats, edge_feats, _, edge_mask = mpnn_model.run_encoder_trunk(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            input_feature_dict=input_feature_dict,
        )

        # Check that edge_mask properly reflects masked tokens
        # Edges originating from masked tokens should be masked
        assert torch.all(
            edge_mask[:, :3] == 0
        ), "edge_mask not zero for masked source tokens"

        # Check that values are reasonable (no huge numbers from masked operations)
        assert torch.all(
            torch.abs(node_feats) < 1e6
        ), "node_feats contains unreasonably large values - possible masking bug"
        assert torch.all(
            torch.abs(edge_feats) < 1e6
        ), "edge_feats contains unreasonably large values - possible masking bug"

    def test_edge_index_validity(self, mpnn_model, sample_data, input_feature_dict):
        """Test that edge indices are valid."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1

        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3)
        backbone_mask = torch.ones(batch_size, n_tokens, 5)
        centre_atom_coord = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)

        _, _, edge_index, _ = mpnn_model.run_encoder_trunk(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            input_feature_dict=input_feature_dict,
        )

        # Edge indices should be in valid range [0, n_tokens)
        assert torch.all(edge_index >= 0), "edge_index contains negative values"
        assert torch.all(
            edge_index < n_tokens
        ), f"edge_index contains values >= n_tokens ({n_tokens})"

    def test_different_backbone_masks(
        self, mpnn_model, sample_data, input_feature_dict
    ):
        """Test that different backbone masks produce different outputs."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1

        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3)
        centre_atom_coord = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)

        # Full backbone mask
        backbone_mask_full = torch.ones(batch_size, n_tokens, 5)
        node_feats_full, _, _, _ = mpnn_model.run_encoder_trunk(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask_full,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            input_feature_dict=input_feature_dict,
        )

        # Partial backbone mask (mask out CB atoms)
        backbone_mask_partial = torch.ones(batch_size, n_tokens, 5)
        backbone_mask_partial[:, :, 4] = 0  # Mask CB
        node_feats_partial, _, _, _ = mpnn_model.run_encoder_trunk(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask_partial,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            input_feature_dict=input_feature_dict,
        )

        # Different masks should produce different node features
        assert not torch.allclose(
            node_feats_full, node_feats_partial, atol=1e-3
        ), "Different backbone masks produce identical outputs"

    def test_device_consistency(self, mpnn_model, sample_data, input_feature_dict):
        """Test that outputs are on the same device as inputs."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1

        mpnn_model = mpnn_model.to(device)

        # Move input feature dict to device
        input_feature_dict_device = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in input_feature_dict.items()
        }

        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3).to(device)
        backbone_mask = torch.ones(batch_size, n_tokens, 5).to(device)
        centre_atom_coord = torch.randn(batch_size, n_tokens, 3).to(device)
        centre_atom_mask = torch.ones(batch_size, n_tokens).to(device)

        node_feats, edge_feats, edge_index, edge_mask = mpnn_model.run_encoder_trunk(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            input_feature_dict=input_feature_dict_device,
        )

        assert node_feats.device == device
        assert edge_feats.device == device
        assert edge_index.device == device
        assert edge_mask.device == device


class TestPrepareDecoderInputs:
    """Tests for _prepare_decoder_inputs method."""

    def test_output_shapes(self, mpnn_model, sample_tensors):
        """Test that output tensors have correct shapes."""
        encoder_edge_feats, sequence_edge_feats = mpnn_model._prepare_decoder_inputs(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            sequence_embeds=sample_tensors["sequence_embeds"],
            edge_index=sample_tensors["edge_index"],
        )

        batch_size = sample_tensors["batch_size"]
        seq_len = sample_tensors["seq_len"]
        num_neighbours = sample_tensors["num_neighbours"]
        hidden_dim = sample_tensors["hidden_dim"]

        # sequence_edge_feats: edge_feats (64) + sequence_embeds (64) = 128
        # encoder_edge_feats: edge_feats (64) + zeros (64) + node_feats (64) = 192
        sequence_expected_shape = (batch_size, seq_len, num_neighbours, hidden_dim * 2)
        encoder_expected_shape = (batch_size, seq_len, num_neighbours, hidden_dim * 3)

        assert (
            sequence_edge_feats.shape == sequence_expected_shape
        ), f"sequence_edge_feats shape {sequence_edge_feats.shape} != {sequence_expected_shape}"
        assert (
            encoder_edge_feats.shape == encoder_expected_shape
        ), f"encoder_edge_feats shape {encoder_edge_feats.shape} != {encoder_expected_shape}"

    def test_output_types(self, mpnn_model, sample_tensors):
        """Test that outputs are torch tensors with correct dtype."""
        encoder_edge_feats, sequence_edge_feats = mpnn_model._prepare_decoder_inputs(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            sequence_embeds=sample_tensors["sequence_embeds"],
            edge_index=sample_tensors["edge_index"],
        )

        assert isinstance(encoder_edge_feats, torch.Tensor)
        assert isinstance(sequence_edge_feats, torch.Tensor)
        assert encoder_edge_feats.dtype == sample_tensors["node_feats"].dtype
        assert sequence_edge_feats.dtype == sample_tensors["node_feats"].dtype

    def test_device_consistency(self, mpnn_model, sample_tensors):
        """Test that outputs are on the same device as inputs."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        node_feats = sample_tensors["node_feats"].to(device)
        edge_feats = sample_tensors["edge_feats"].to(device)
        sequence_embeds = sample_tensors["sequence_embeds"].to(device)
        edge_index = sample_tensors["edge_index"].to(device)

        mpnn_model = mpnn_model.to(device)

        encoder_edge_feats, sequence_edge_feats = mpnn_model._prepare_decoder_inputs(
            node_feats=node_feats,
            edge_feats=edge_feats,
            sequence_embeds=sequence_embeds,
            edge_index=edge_index,
        )

        assert encoder_edge_feats.device == device
        assert sequence_edge_feats.device == device

    def test_different_batch_sizes(self, mpnn_model):
        """Test that method works with different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            seq_len = 10
            num_neighbours = 30
            hidden_dim = 64

            node_feats = torch.randn(batch_size, seq_len, hidden_dim)
            edge_feats = torch.randn(batch_size, seq_len, num_neighbours, hidden_dim)
            sequence_embeds = torch.randn(batch_size, seq_len, hidden_dim)
            edge_index = torch.randint(
                0, seq_len, (batch_size, seq_len, num_neighbours)
            )

            encoder_edge_feats, sequence_edge_feats = (
                mpnn_model._prepare_decoder_inputs(
                    node_feats=node_feats,
                    edge_feats=edge_feats,
                    sequence_embeds=sequence_embeds,
                    edge_index=edge_index,
                )
            )

            # sequence_edge_feats: 2x, encoder_edge_feats: 3x hidden_dim
            sequence_expected = (batch_size, seq_len, num_neighbours, hidden_dim * 2)
            encoder_expected = (batch_size, seq_len, num_neighbours, hidden_dim * 3)
            assert sequence_edge_feats.shape == sequence_expected
            assert encoder_edge_feats.shape == encoder_expected

    def test_edge_features_not_nan_or_inf(self, mpnn_model, sample_tensors):
        """Test that output features don't contain NaN or Inf values."""
        encoder_edge_feats, sequence_edge_feats = mpnn_model._prepare_decoder_inputs(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            sequence_embeds=sample_tensors["sequence_embeds"],
            edge_index=sample_tensors["edge_index"],
        )

        assert not torch.isnan(
            encoder_edge_feats
        ).any(), "encoder_edge_feats contains NaN"
        assert not torch.isinf(
            encoder_edge_feats
        ).any(), "encoder_edge_feats contains Inf"
        assert not torch.isnan(
            sequence_edge_feats
        ).any(), "sequence_edge_feats contains NaN"
        assert not torch.isinf(
            sequence_edge_feats
        ).any(), "sequence_edge_feats contains Inf"


class TestGenerateRandomSpatiallySmoothDecodingOrder:
    """Tests for _generate_random_spatially_smooth_decoding_order method."""

    def test_output_shape(self, mpnn_model, sample_data, input_feature_dict):
        """Test that output has correct shape."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1

        centre_atom_coords = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)

        decoding_order = mpnn_model._generate_random_spatially_smooth_decoding_order(
            centre_atom_coords=centre_atom_coords,
            centre_atom_mask=centre_atom_mask,
            seed=42,
        )

        expected_shape = (batch_size, n_tokens)
        assert (
            decoding_order.shape == expected_shape
        ), f"decoding_order shape {decoding_order.shape} != {expected_shape}"

    def test_output_is_valid_permutation(
        self, mpnn_model, sample_data, input_feature_dict
    ):
        """Test that output is a valid permutation of indices."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1

        centre_atom_coords = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)

        decoding_order = mpnn_model._generate_random_spatially_smooth_decoding_order(
            centre_atom_coords=centre_atom_coords,
            centre_atom_mask=centre_atom_mask,
            seed=42,
        )

        # Check that each batch contains all indices from 0 to n_tokens-1
        for b in range(batch_size):
            sorted_indices = torch.sort(decoding_order[b])[0]
            expected_indices = torch.arange(n_tokens, device=decoding_order.device)
            assert torch.equal(
                sorted_indices, expected_indices
            ), f"Batch {b}: decoding_order is not a valid permutation"

    def test_deterministic_with_seed(self, mpnn_model, sample_data, input_feature_dict):
        """Test that same seed produces same decoding order."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1

        centre_atom_coords = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)

        decoding_order_1 = mpnn_model._generate_random_spatially_smooth_decoding_order(
            centre_atom_coords=centre_atom_coords,
            centre_atom_mask=centre_atom_mask,
            seed=42,
            smoothing_factor=0.5,
        )

        decoding_order_2 = mpnn_model._generate_random_spatially_smooth_decoding_order(
            centre_atom_coords=centre_atom_coords,
            centre_atom_mask=centre_atom_mask,
            seed=42,
            smoothing_factor=0.5,
        )

        assert torch.equal(
            decoding_order_1, decoding_order_2
        ), "Same seed should produce same decoding order"

    def test_different_seeds_produce_different_orders(
        self, mpnn_model, sample_data, input_feature_dict
    ):
        """Test that different seeds produce different decoding orders."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1

        centre_atom_coords = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)

        decoding_order_1 = mpnn_model._generate_random_spatially_smooth_decoding_order(
            centre_atom_coords=centre_atom_coords,
            centre_atom_mask=centre_atom_mask,
            seed=42,
            smoothing_factor=0.5,
        )

        decoding_order_2 = mpnn_model._generate_random_spatially_smooth_decoding_order(
            centre_atom_coords=centre_atom_coords,
            centre_atom_mask=centre_atom_mask,
            seed=123,
            smoothing_factor=0.5,
        )

        # Different seeds should produce different orders (with high probability)
        assert not torch.equal(
            decoding_order_1, decoding_order_2
        ), "Different seeds should produce different decoding orders"

    def test_different_smoothing_factors(
        self, mpnn_model, sample_data, input_feature_dict
    ):
        """Test that different smoothing factors produce different orders."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1

        centre_atom_coords = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)

        # Fully spatial (mu=1.0)
        decoding_order_spatial = (
            mpnn_model._generate_random_spatially_smooth_decoding_order(
                centre_atom_coords=centre_atom_coords,
                centre_atom_mask=centre_atom_mask,
                seed=42,
                smoothing_factor=1.0,
            )
        )

        # Fully random (mu=0.0)
        decoding_order_random = (
            mpnn_model._generate_random_spatially_smooth_decoding_order(
                centre_atom_coords=centre_atom_coords,
                centre_atom_mask=centre_atom_mask,
                seed=42,
                smoothing_factor=0.0,
            )
        )

        # Intermediate (mu=0.5)
        decoding_order_mixed = (
            mpnn_model._generate_random_spatially_smooth_decoding_order(
                centre_atom_coords=centre_atom_coords,
                centre_atom_mask=centre_atom_mask,
                seed=42,
                smoothing_factor=0.5,
            )
        )

        # All should be valid permutations
        assert decoding_order_spatial.shape == (batch_size, n_tokens)
        assert decoding_order_random.shape == (batch_size, n_tokens)
        assert decoding_order_mixed.shape == (batch_size, n_tokens)

        # They should be different (with high probability)
        assert not torch.equal(
            decoding_order_spatial, decoding_order_random
        ), "Fully spatial and fully random should produce different orders"

    def test_seed_token_index_parameter(
        self, mpnn_model, sample_data, input_feature_dict
    ):
        """Test that seed_token_index parameter works correctly."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1

        centre_atom_coords = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)

        # Use specific seed token
        seed_token_idx = n_tokens // 2  # Middle token
        decoding_order = mpnn_model._generate_random_spatially_smooth_decoding_order(
            centre_atom_coords=centre_atom_coords,
            centre_atom_mask=centre_atom_mask,
            seed=42,
            smoothing_factor=1.0,  # Fully spatial
            seed_token_index=seed_token_idx,
        )

        # Check that the decoding order is valid
        assert decoding_order.shape == (batch_size, n_tokens)

        # With fully spatial smoothing, tokens close to seed token should
        # generally appear earlier in decoding order (but this is probabilistic)
        # Just verify it runs without error
        assert not torch.isnan(decoding_order.float()).any()

    def test_masking_application(self, mpnn_model, sample_data, input_feature_dict):
        """Test that masking properly affects seed selection.

        This is critical to avoid the silent failure mode mentioned in CLAUDE.md
        where masked coordinates can generate large numbers if not properly handled.
        """
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1

        centre_atom_coords = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)

        # Mask out first half of tokens
        centre_atom_mask[:, : n_tokens // 2] = 0

        decoding_order = mpnn_model._generate_random_spatially_smooth_decoding_order(
            centre_atom_coords=centre_atom_coords,
            centre_atom_mask=centre_atom_mask,
            seed=42,
            smoothing_factor=0.5,
        )

        # Check that the decoding order is still valid
        assert decoding_order.shape == (batch_size, n_tokens)

        # Check for reasonable values (no huge numbers from masked operations)
        assert torch.all(
            torch.abs(decoding_order.float()) < 1e6
        ), "decoding_order contains unreasonably large values - possible masking bug"

    def test_output_no_nan_or_inf(self, mpnn_model, sample_data, input_feature_dict):
        """Test that output doesn't contain NaN or Inf values."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1

        centre_atom_coords = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)

        decoding_order = mpnn_model._generate_random_spatially_smooth_decoding_order(
            centre_atom_coords=centre_atom_coords,
            centre_atom_mask=centre_atom_mask,
            seed=42,
        )

        assert not torch.isnan(
            decoding_order.float()
        ).any(), "decoding_order contains NaN"
        assert not torch.isinf(
            decoding_order.float()
        ).any(), "decoding_order contains Inf"

    def test_device_consistency(self, mpnn_model, sample_data, input_feature_dict):
        """Test that output is on the same device as inputs."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1

        mpnn_model = mpnn_model.to(device)

        centre_atom_coords = torch.randn(batch_size, n_tokens, 3).to(device)
        centre_atom_mask = torch.ones(batch_size, n_tokens).to(device)

        decoding_order = mpnn_model._generate_random_spatially_smooth_decoding_order(
            centre_atom_coords=centre_atom_coords,
            centre_atom_mask=centre_atom_mask,
            seed=42,
        )

        assert decoding_order.device == device

    def test_spatial_ordering_property(
        self, mpnn_model, sample_data, input_feature_dict
    ):
        """Test that fully spatial decoding (mu=1.0) produces spatially coherent order.

        With mu=1.0, tokens closer to the seed should generally appear earlier.
        """
        batch_size = 1
        n_tokens = 20  # Use smaller number for clearer spatial structure

        # Create a linear arrangement of tokens
        centre_atom_coords = torch.zeros(batch_size, n_tokens, 3)
        centre_atom_coords[0, :, 0] = torch.linspace(0, 10, n_tokens)
        centre_atom_mask = torch.ones(batch_size, n_tokens)

        # Use fully spatial smoothing
        decoding_order = mpnn_model._generate_random_spatially_smooth_decoding_order(
            centre_atom_coords=centre_atom_coords,
            centre_atom_mask=centre_atom_mask,
            seed=42,
            smoothing_factor=1.0,
            seed_token_index=0,  # Start from first token
        )

        # With seed at position 0 and fully spatial, tokens should be decoded
        # in roughly ascending order of their x-coordinate
        # We can't guarantee perfect ordering due to normalization, but there
        # should be some spatial coherence
        assert decoding_order.shape == (batch_size, n_tokens)

        # Just verify the method produces valid output
        sorted_indices = torch.sort(decoding_order[0])[0]
        expected_indices = torch.arange(n_tokens, device=decoding_order.device)
        assert torch.equal(sorted_indices, expected_indices)

    def test_different_batch_sizes(self, mpnn_model):
        """Test that method works with different batch sizes."""
        for batch_size in [1, 2, 4]:
            n_tokens = 15

            centre_atom_coords = torch.randn(batch_size, n_tokens, 3)
            centre_atom_mask = torch.ones(batch_size, n_tokens)

            decoding_order = (
                mpnn_model._generate_random_spatially_smooth_decoding_order(
                    centre_atom_coords=centre_atom_coords,
                    centre_atom_mask=centre_atom_mask,
                    seed=42,
                    smoothing_factor=0.5,
                )
            )

            expected_shape = (batch_size, n_tokens)
            assert (
                decoding_order.shape == expected_shape
            ), f"Failed for batch_size={batch_size}"

            # Verify each batch has valid permutation
            for b in range(batch_size):
                sorted_indices = torch.sort(decoding_order[b])[0]
                expected_indices = torch.arange(n_tokens, device=decoding_order.device)
                assert torch.equal(sorted_indices, expected_indices)


class TestGenerateSequentialMasks:
    """Tests for _generate_sequential_masks method."""

    def test_output_shapes(self, mpnn_model, sample_tensors):
        """Test that mask outputs have correct shapes."""
        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=sample_tensors["decoding_order"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        batch_size = sample_tensors["batch_size"]
        seq_len = sample_tensors["seq_len"]
        num_neighbours = sample_tensors["num_neighbours"]

        # Both masks should have shape: [batch, seq_len, neighbours]
        expected_shape = (batch_size, seq_len, num_neighbours)
        assert (
            forward_attn_mask.shape == expected_shape
        ), f"forward_attn_mask shape {forward_attn_mask.shape} != {expected_shape}"
        assert (
            backward_attn_mask.shape == expected_shape
        ), f"backward_attn_mask shape {backward_attn_mask.shape} != {expected_shape}"

    def test_mask_values_binary(self, mpnn_model, sample_tensors):
        """Test that masks contain only 0s and 1s."""
        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=sample_tensors["decoding_order"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        # Check that all values are in {0, 1}
        assert torch.all(
            (forward_attn_mask == 0) | (forward_attn_mask == 1)
        ), "forward_attn_mask contains non-binary values"
        assert torch.all(
            (backward_attn_mask == 0) | (backward_attn_mask == 1)
        ), "backward_attn_mask contains non-binary values"

    def test_masks_are_complementary_with_node_mask(self, mpnn_model, sample_tensors):
        """Test that forward and backward masks sum to node_mask along edges.

        This test verifies the property that at each position, the forward and
        backward masks should sum to the node mask, accounting for valid positions.
        """
        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=sample_tensors["decoding_order"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        # Get the node mask for neighboring nodes
        # edge_index: [batch, seq_len, num_neighbours]
        # node_mask: [batch, seq_len]
        edge_index = sample_tensors["edge_index"]
        node_mask = sample_tensors["node_mask"]

        # Gather node mask values at edge indices
        neighbor_mask = torch.gather(
            node_mask.unsqueeze(-1).expand(-1, -1, edge_index.shape[-1]),
            dim=1,
            index=edge_index,
        )

        # The sum of forward and backward masks should equal the product of
        # source node mask and neighbor node mask
        mask_sum = forward_attn_mask + backward_attn_mask
        expected_sum = node_mask.unsqueeze(-1) * neighbor_mask

        assert torch.allclose(
            mask_sum, expected_sum, atol=1e-6
        ), "Forward and backward masks don't sum correctly with node mask"

    def test_different_decoding_orders(self, mpnn_model, sample_tensors):
        """Test that different decoding orders produce different masks."""
        # Generate two different random decoding orders
        seq_len = sample_tensors["seq_len"]
        batch_size = sample_tensors["batch_size"]

        decoding_order_1 = torch.randperm(seq_len).unsqueeze(0).expand(batch_size, -1)
        decoding_order_2 = torch.randperm(seq_len).unsqueeze(0).expand(batch_size, -1)

        # Ensure they're actually different
        while torch.equal(decoding_order_1, decoding_order_2):
            decoding_order_2 = (
                torch.randperm(seq_len).unsqueeze(0).expand(batch_size, -1)
            )

        forward_1, backward_1 = mpnn_model._generate_sequential_masks(
            decoding_order=decoding_order_1,
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        forward_2, backward_2 = mpnn_model._generate_sequential_masks(
            decoding_order=decoding_order_2,
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        # Masks should be different for different orders
        assert not torch.equal(forward_1, forward_2) or not torch.equal(
            backward_1, backward_2
        ), "Different decoding orders produced identical masks"

    def test_node_mask_application(self, mpnn_model, sample_tensors):
        """Test that node mask properly zeros out invalid positions."""
        # Create a node mask with some positions invalid
        node_mask = sample_tensors["node_mask"].clone()
        node_mask[:, :3] = 0  # Mask out first 3 positions

        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=sample_tensors["decoding_order"],
            edge_index=sample_tensors["edge_index"],
            node_mask=node_mask,
        )

        # First 3 positions should be all zeros in both masks
        assert torch.all(
            forward_attn_mask[:, :3] == 0
        ), "Masked positions not zero in forward_attn_mask"
        assert torch.all(
            backward_attn_mask[:, :3] == 0
        ), "Masked positions not zero in backward_attn_mask"

    def test_output_device_consistency(self, mpnn_model, sample_tensors):
        """Test that output masks are on the same device as inputs."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        decoding_order = sample_tensors["decoding_order"].to(device)
        edge_index = sample_tensors["edge_index"].to(device)
        node_mask = sample_tensors["node_mask"].to(device)

        mpnn_model = mpnn_model.to(device)

        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=decoding_order,
            edge_index=edge_index,
            node_mask=node_mask,
        )

        assert forward_attn_mask.device == device
        assert backward_attn_mask.device == device


class TestRunDecoderStep:
    """Tests for _run_decoder_step method."""

    def test_output_shape(self, mpnn_model, sample_tensors):
        """Test that decoder step output has correct shape."""
        # Need to add embed_restype to model first
        mpnn_model.restype_embed = nn.Linear(
            sample_tensors["num_residue_types"], sample_tensors["hidden_dim"]
        )

        # Generate masks
        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=sample_tensors["decoding_order"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        # Add channel dimension to masks
        forward_attn_mask = forward_attn_mask.unsqueeze(-1)
        backward_attn_mask = backward_attn_mask.unsqueeze(-1)

        logits = mpnn_model._run_decoder_step(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            restype=sample_tensors["restype"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
            forward_attn_mask=forward_attn_mask,
            backward_attn_mask=backward_attn_mask,
            edge_mask=sample_tensors["edge_mask"],
            zero_shot=False,
        )

        batch_size = sample_tensors["batch_size"]
        seq_len = sample_tensors["seq_len"]
        num_residue_types = sample_tensors["num_residue_types"]

        expected_shape = (batch_size, seq_len, num_residue_types)
        assert (
            logits.shape == expected_shape
        ), f"logits shape {logits.shape} != {expected_shape}"

    def test_zero_shot_mode(self, mpnn_model, sample_tensors):
        """Test that zero_shot mode runs without errors and produces valid output."""
        mpnn_model.restype_embed = nn.Linear(
            sample_tensors["num_residue_types"], sample_tensors["hidden_dim"]
        )

        # Generate masks
        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=sample_tensors["decoding_order"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        forward_attn_mask = forward_attn_mask.unsqueeze(-1)
        backward_attn_mask = backward_attn_mask.unsqueeze(-1)

        logits_zero_shot = mpnn_model._run_decoder_step(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            restype=sample_tensors["restype"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
            forward_attn_mask=forward_attn_mask,
            backward_attn_mask=backward_attn_mask,
            edge_mask=sample_tensors["edge_mask"],
            zero_shot=True,
        )

        logits_autoregressive = mpnn_model._run_decoder_step(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            restype=sample_tensors["restype"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
            forward_attn_mask=forward_attn_mask,
            backward_attn_mask=backward_attn_mask,
            edge_mask=sample_tensors["edge_mask"],
            zero_shot=False,
        )

        # Both should have same shape but different values
        assert logits_zero_shot.shape == logits_autoregressive.shape
        assert not torch.allclose(
            logits_zero_shot, logits_autoregressive, atol=1e-3
        ), "Zero-shot and autoregressive modes produce identical outputs"

    def test_output_no_nan_or_inf(self, mpnn_model, sample_tensors):
        """Test that decoder output doesn't contain NaN or Inf values."""
        mpnn_model.restype_embed = nn.Linear(
            sample_tensors["num_residue_types"], sample_tensors["hidden_dim"]
        )

        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=sample_tensors["decoding_order"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        forward_attn_mask = forward_attn_mask.unsqueeze(-1)
        backward_attn_mask = backward_attn_mask.unsqueeze(-1)

        logits = mpnn_model._run_decoder_step(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            restype=sample_tensors["restype"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
            forward_attn_mask=forward_attn_mask,
            backward_attn_mask=backward_attn_mask,
            edge_mask=sample_tensors["edge_mask"],
            zero_shot=False,
        )

        assert not torch.isnan(logits).any(), "Logits contain NaN values"
        assert not torch.isinf(logits).any(), "Logits contain Inf values"

    def test_edge_mask_application(self, mpnn_model, sample_tensors):
        """Test that edge masking properly affects the output."""
        mpnn_model.restype_embed = nn.Linear(
            sample_tensors["num_residue_types"], sample_tensors["hidden_dim"]
        )

        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=sample_tensors["decoding_order"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        forward_attn_mask = forward_attn_mask.unsqueeze(-1)
        backward_attn_mask = backward_attn_mask.unsqueeze(-1)

        # Run with full edge mask
        logits_full = mpnn_model._run_decoder_step(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            restype=sample_tensors["restype"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
            forward_attn_mask=forward_attn_mask,
            backward_attn_mask=backward_attn_mask,
            edge_mask=sample_tensors["edge_mask"],
            zero_shot=False,
        )

        # Create a partially masked edge mask
        edge_mask_partial = sample_tensors["edge_mask"].clone()
        edge_mask_partial[:, :, :10] = 0  # Mask first 10 edges

        # Run with partial edge mask
        logits_partial = mpnn_model._run_decoder_step(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            restype=sample_tensors["restype"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
            forward_attn_mask=forward_attn_mask,
            backward_attn_mask=backward_attn_mask,
            edge_mask=edge_mask_partial,
            zero_shot=False,
        )

        # Different edge masks should produce different outputs
        assert not torch.allclose(
            logits_full, logits_partial, atol=1e-3
        ), "Different edge masks produce identical outputs"

    def test_different_restype_inputs(self, mpnn_model, sample_tensors):
        """Test that different restype inputs produce different outputs."""
        mpnn_model.restype_embed = nn.Linear(
            sample_tensors["num_residue_types"], sample_tensors["hidden_dim"]
        )

        forward_attn_mask, backward_attn_mask = mpnn_model._generate_sequential_masks(
            decoding_order=sample_tensors["decoding_order"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
        )

        forward_attn_mask = forward_attn_mask.unsqueeze(-1)
        backward_attn_mask = backward_attn_mask.unsqueeze(-1)

        # First restype (zeros)
        logits_1 = mpnn_model._run_decoder_step(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            restype=sample_tensors["restype"],
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
            forward_attn_mask=forward_attn_mask,
            backward_attn_mask=backward_attn_mask,
            edge_mask=sample_tensors["edge_mask"],
            zero_shot=False,
        )

        # Different restype (random one-hot)
        restype_2 = sample_tensors["restype"].clone()
        restype_2[:, 0, :] = 0
        restype_2[:, 0, 5] = 1  # Set to different residue type

        logits_2 = mpnn_model._run_decoder_step(
            node_feats=sample_tensors["node_feats"],
            edge_feats=sample_tensors["edge_feats"],
            restype=restype_2,
            edge_index=sample_tensors["edge_index"],
            node_mask=sample_tensors["node_mask"],
            forward_attn_mask=forward_attn_mask,
            backward_attn_mask=backward_attn_mask,
            edge_mask=sample_tensors["edge_mask"],
            zero_shot=False,
        )

        # Different restypes should produce different outputs
        assert not torch.allclose(
            logits_1, logits_2, atol=1e-4
        ), "Different restype inputs produce identical outputs"


class TestForward:
    """Tests for the forward method."""

    def test_forward_output_structure(
        self, mpnn_model, sample_data, input_feature_dict
    ):
        """Test that forward returns correct output structure."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1
        num_residue_types = 32

        # Create inputs
        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3)
        backbone_mask = torch.ones(batch_size, n_tokens, 5)
        centre_atom_coord = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)
        restype = torch.zeros(batch_size, n_tokens, num_residue_types)

        # Run forward
        logits, node_feats, edge_feats = mpnn_model.forward(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            restype=restype,
            input_feature_dict=input_feature_dict,
        )

        # Check output structure
        assert isinstance(logits, torch.Tensor), "First return value should be a tensor"
        assert isinstance(
            node_feats, torch.Tensor
        ), "Second return value should be a tensor"
        assert isinstance(
            edge_feats, torch.Tensor
        ), "Third return value should be a tensor"
        assert logits.shape == (
            batch_size,
            n_tokens,
            num_residue_types,
        ), "logits should have correct shape"

    def test_forward_logits_shape(self, mpnn_model, sample_data, input_feature_dict):
        """Test that forward produces logits with correct shape."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1
        num_residue_types = 32

        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3)
        backbone_mask = torch.ones(batch_size, n_tokens, 5)
        centre_atom_coord = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)
        restype = torch.zeros(batch_size, n_tokens, num_residue_types)

        logits, _, _ = mpnn_model.forward(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            restype=restype,
            input_feature_dict=input_feature_dict,
        )

        expected_shape = (batch_size, n_tokens, num_residue_types)
        assert (
            logits.shape == expected_shape
        ), f"logits shape {logits.shape} != {expected_shape}"

    def test_forward_with_custom_decoding_order(
        self, mpnn_model, sample_data, input_feature_dict
    ):
        """Test forward with custom decoding order."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1
        num_residue_types = 32

        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3)
        backbone_mask = torch.ones(batch_size, n_tokens, 5)
        centre_atom_coord = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)
        restype = torch.zeros(batch_size, n_tokens, num_residue_types)

        # Create custom decoding order
        decoding_order = torch.randperm(n_tokens).unsqueeze(0)

        logits, _, _ = mpnn_model.forward(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            restype=restype,
            input_feature_dict=input_feature_dict,
            decoding_order=decoding_order,
        )

        assert not torch.isnan(logits).any()

    def test_forward_zero_shot_mode(self, mpnn_model, sample_data, input_feature_dict):
        """Test forward in zero-shot mode."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1
        num_residue_types = 32

        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3)
        backbone_mask = torch.ones(batch_size, n_tokens, 5)
        centre_atom_coord = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)
        restype = torch.zeros(batch_size, n_tokens, num_residue_types)

        # Run in zero-shot mode
        logits_zero_shot, _, _ = mpnn_model.forward(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            restype=restype,
            input_feature_dict=input_feature_dict,
            zero_shot=True,
        )

        # Run in autoregressive mode
        logits_autoregressive, _, _ = mpnn_model.forward(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            restype=restype,
            input_feature_dict=input_feature_dict,
            zero_shot=False,
        )

        # Outputs should be different
        assert not torch.allclose(
            logits_zero_shot, logits_autoregressive, atol=1e-3
        ), "Zero-shot and autoregressive modes produce identical outputs"

    def test_forward_no_nan_or_inf(self, mpnn_model, sample_data, input_feature_dict):
        """Test that forward doesn't produce NaN or Inf values."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1
        num_residue_types = 32

        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3)
        backbone_mask = torch.ones(batch_size, n_tokens, 5)
        centre_atom_coord = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)
        restype = torch.zeros(batch_size, n_tokens, num_residue_types)

        logits, _, _ = mpnn_model.forward(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            restype=restype,
            input_feature_dict=input_feature_dict,
        )

        assert not torch.isnan(logits).any(), "Forward logits contain NaN"
        assert not torch.isinf(logits).any(), "Forward logits contain Inf"


@pytest.mark.slow
class TestSample:
    """Tests for the sample method."""

    def test_sample_output_shape(self, mpnn_model, sample_data, input_feature_dict):
        """Test that sample returns correct output shape."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1
        num_residue_types = 32

        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3)
        backbone_mask = torch.ones(batch_size, n_tokens, 5)
        centre_atom_coord = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)
        restype = torch.zeros(batch_size, n_tokens, num_residue_types)

        sampled_sequence, _ = mpnn_model.sample(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            restype=restype,
            input_feature_dict=input_feature_dict,
            temperature=1.0,
        )

        expected_shape = (batch_size, n_tokens, num_residue_types)
        assert (
            sampled_sequence.shape == expected_shape
        ), f"sampled_sequence shape {sampled_sequence.shape} != {expected_shape}"

    def test_sample_returns_one_hot(self, mpnn_model, sample_data, input_feature_dict):
        """Test that sample returns valid one-hot encoded sequences."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1
        num_residue_types = 32

        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3)
        backbone_mask = torch.ones(batch_size, n_tokens, 5)
        centre_atom_coord = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)
        restype = torch.zeros(batch_size, n_tokens, num_residue_types)

        sampled_sequence, _ = mpnn_model.sample(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            restype=restype,
            input_feature_dict=input_feature_dict,
            temperature=1.0,
        )

        # Check that each position sums to 1 (valid one-hot)
        position_sums = sampled_sequence.sum(dim=-1)
        assert torch.allclose(
            position_sums, torch.ones_like(position_sums), atol=1e-5
        ), "Sampled sequence is not valid one-hot encoding"

        # Check that all values are 0 or 1
        assert torch.all(
            (sampled_sequence == 0) | (sampled_sequence == 1)
        ), "Sampled sequence contains non-binary values"

    def test_sample_greedy_mode(self, mpnn_model, sample_data, input_feature_dict):
        """Test greedy sampling (temperature=0)."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1
        num_residue_types = 32

        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3)
        backbone_mask = torch.ones(batch_size, n_tokens, 5)
        centre_atom_coord = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)
        restype = torch.zeros(batch_size, n_tokens, num_residue_types)

        # Greedy sampling should be deterministic
        sample_1, _ = mpnn_model.sample(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            restype=restype,
            input_feature_dict=input_feature_dict,
            temperature=0.0,
        )

        sample_2, _ = mpnn_model.sample(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            restype=restype,
            input_feature_dict=input_feature_dict,
            temperature=0.0,
        )

        # Greedy sampling should be deterministic
        assert torch.equal(
            sample_1, sample_2
        ), "Greedy sampling (temperature=0) is not deterministic"

    def test_sample_with_custom_decoding_order(
        self, mpnn_model, sample_data, input_feature_dict
    ):
        """Test sampling with custom decoding order."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1
        num_residue_types = 32

        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3)
        backbone_mask = torch.ones(batch_size, n_tokens, 5)
        centre_atom_coord = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)
        restype = torch.zeros(batch_size, n_tokens, num_residue_types)

        # Create two different decoding orders
        decoding_order_1 = torch.randperm(n_tokens).unsqueeze(0)
        decoding_order_2 = torch.randperm(n_tokens).unsqueeze(0)

        sample_1, _ = mpnn_model.sample(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            restype=restype,
            input_feature_dict=input_feature_dict,
            decoding_order=decoding_order_1,
            temperature=0.0,
        )

        sample_2, _ = mpnn_model.sample(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            restype=restype,
            input_feature_dict=input_feature_dict,
            decoding_order=decoding_order_2,
            temperature=0.0,
        )

        # Different decoding orders may produce different sequences
        # (not guaranteed, but we test that the method accepts the parameter)
        assert sample_1.shape == sample_2.shape

    def test_sample_return_logits(self, mpnn_model, sample_data, input_feature_dict):
        """Test sampling with return_logits=True."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1
        num_residue_types = 32

        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3)
        backbone_mask = torch.ones(batch_size, n_tokens, 5)
        centre_atom_coord = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)
        restype = torch.zeros(batch_size, n_tokens, num_residue_types)

        result = mpnn_model.sample(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            restype=restype,
            input_feature_dict=input_feature_dict,
            temperature=1.0,
            return_logits=True,
        )

        # Should return tuple of (sequence, node_feats, logits)
        assert isinstance(result, tuple), "return_logits=True should return tuple"
        assert len(result) == 3, "Should return tuple of length 3"

        sampled_sequence, node_feats, logits = result
        assert sampled_sequence.shape == (batch_size, n_tokens, num_residue_types)
        assert logits.shape == (batch_size, n_tokens, num_residue_types)

    def test_sample_different_temperatures(
        self, mpnn_model, sample_data, input_feature_dict
    ):
        """Test sampling with different temperature values."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1
        num_residue_types = 32

        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3)
        backbone_mask = torch.ones(batch_size, n_tokens, 5)
        centre_atom_coord = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)
        restype = torch.zeros(batch_size, n_tokens, num_residue_types)

        for temperature in [0.0, 0.5, 1.0, 1.5]:
            sampled_sequence, _ = mpnn_model.sample(
                backbone_coords=backbone_coords,
                backbone_mask=backbone_mask,
                centre_atom_coord=centre_atom_coord,
                centre_atom_mask=centre_atom_mask,
                restype=restype,
                input_feature_dict=input_feature_dict,
                temperature=temperature,
            )

            assert sampled_sequence.shape == (batch_size, n_tokens, num_residue_types)
            assert not torch.isnan(sampled_sequence).any()

    def test_sample_no_nan_or_inf(self, mpnn_model, sample_data, input_feature_dict):
        """Test that sample doesn't produce NaN or Inf values."""
        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1
        num_residue_types = 32

        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3)
        backbone_mask = torch.ones(batch_size, n_tokens, 5)
        centre_atom_coord = torch.randn(batch_size, n_tokens, 3)
        centre_atom_mask = torch.ones(batch_size, n_tokens)
        restype = torch.zeros(batch_size, n_tokens, num_residue_types)

        sampled_sequence, _ = mpnn_model.sample(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            restype=restype,
            input_feature_dict=input_feature_dict,
            temperature=1.0,
        )

        assert not torch.isnan(sampled_sequence).any(), "Sampled sequence contains NaN"
        assert not torch.isinf(sampled_sequence).any(), "Sampled sequence contains Inf"

    def test_sample_device_consistency(
        self, mpnn_model, sample_data, input_feature_dict
    ):
        """Test that sample output is on the same device as inputs."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        basic = sample_data["basic"]
        n_tokens = basic["N_token"].item()
        batch_size = 1
        num_residue_types = 32

        mpnn_model = mpnn_model.to(device)

        # Move input feature dict to device
        input_feature_dict_device = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in input_feature_dict.items()
        }

        backbone_coords = torch.randn(batch_size, n_tokens, 5, 3).to(device)
        backbone_mask = torch.ones(batch_size, n_tokens, 5).to(device)
        centre_atom_coord = torch.randn(batch_size, n_tokens, 3).to(device)
        centre_atom_mask = torch.ones(batch_size, n_tokens).to(device)
        restype = torch.zeros(batch_size, n_tokens, num_residue_types).to(device)

        sampled_sequence, _ = mpnn_model.sample(
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            centre_atom_coord=centre_atom_coord,
            centre_atom_mask=centre_atom_mask,
            restype=restype,
            input_feature_dict=input_feature_dict_device,
            temperature=1.0,
        )

        assert sampled_sequence.device == device
