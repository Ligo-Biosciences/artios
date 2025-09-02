import os

import pytest
import torch

from protenix.model.modules.atom_attention import AtomAttentionEncoder


class TestAtomAttentionEncoder:
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
    def atom_attention_encoder(self, device):
        """Create AtomAttentionEncoder with default parameters."""
        encoder = AtomAttentionEncoder(
            c_token=384,
            has_coords=True,
            has_s=True,
            has_z=True,
            c_atom=128,
            c_atompair=16,
            c_s=384,
            c_z=128,
            num_blocks=3,
            num_heads=4,
            num_queries=32,
            num_keys=128,
        )
        return encoder.to(device)

    def test_per_atom_conditioning_shapes(
        self, atom_attention_encoder, sample_data, device
    ):
        """Test that _per_atom_conditioning produces correct output shapes."""
        input_feature_dict = sample_data["input_feature_dict"]
        basic = sample_data["basic"]
        n_atoms = basic["N_atom"]

        # Move input data to device
        input_feature_dict = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in input_feature_dict.items()
        }

        # Call the method
        c_l, p_lm = atom_attention_encoder._per_atom_conditioning(input_feature_dict)

        # Check c_l (atom single conditioning) shape
        expected_c_l_shape = (
            *input_feature_dict["ref_pos"].shape[:-1],
            atom_attention_encoder.c_atom,
        )
        assert (
            c_l.shape == expected_c_l_shape
        ), f"Expected c_l shape {expected_c_l_shape}, got {c_l.shape}"

        # Check p_lm (atom pair conditioning) shape
        # Should be [..., n_blocks, n_queries, n_keys, c_atompair]
        n_blocks = (
            n_atoms + atom_attention_encoder.num_queries - 1
        ) // atom_attention_encoder.num_queries
        expected_p_lm_shape = (
            *input_feature_dict["ref_pos"].shape[:-2],
            n_blocks,
            atom_attention_encoder.num_queries,
            atom_attention_encoder.num_keys,
            atom_attention_encoder.c_atompair,
        )
        assert (
            p_lm.shape == expected_p_lm_shape
        ), f"Expected p_lm shape {expected_p_lm_shape}, got {p_lm.shape}"

    def test_per_atom_conditioning_no_nans(
        self, atom_attention_encoder, sample_data, device
    ):
        """Test that _per_atom_conditioning produces no NaN values."""
        input_feature_dict = sample_data["input_feature_dict"]

        # Move input data to device
        input_feature_dict = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in input_feature_dict.items()
        }

        # Call the method
        c_l, p_lm = atom_attention_encoder._per_atom_conditioning(input_feature_dict)

        # Check for NaN values
        assert not torch.isnan(c_l).any(), "c_l contains NaN values"
        assert not torch.isnan(p_lm).any(), "p_lm contains NaN values"

    def test_per_atom_conditioning_mask_application(
        self, atom_attention_encoder, sample_data, device
    ):
        """Test that mask is properly applied to c_l."""
        input_feature_dict = sample_data["input_feature_dict"]

        # Move input data to device
        input_feature_dict = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in input_feature_dict.items()
        }

        # Call the method
        c_l, _ = atom_attention_encoder._per_atom_conditioning(input_feature_dict)

        # Check that masked positions are zero
        ref_mask = input_feature_dict["ref_mask"].unsqueeze(-1)
        masked_positions = ~ref_mask.bool()

        if masked_positions.any():
            # Check that masked positions in c_l are zero
            assert torch.allclose(
                c_l[masked_positions.expand_as(c_l)],
                torch.zeros_like(c_l[masked_positions.expand_as(c_l)]),
            ), "Masked positions in c_l should be zero"


if __name__ == "__main__":
    pytest.main([__file__])
