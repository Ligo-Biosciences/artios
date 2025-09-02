import os

import pytest
import torch

from protenix.data.padding_utils import pad_features_dict
from protenix.data.utils import get_data_shape_dict


class TestPaddingUtils:
    """Test padding utilities for feature dictionaries."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def sample_data(self):
        data_path = os.path.join(
            os.path.dirname(__file__), "testdata", "sample_data_dict.pt"
        )
        if not os.path.exists(data_path):
            pytest.skip(f"Test data not found at {data_path}")
        return torch.load(data_path, weights_only=False)

    @pytest.fixture
    def input_feature_dict(self, sample_data):
        return sample_data["input_feature_dict"]

    def test_pad_features_dict_basic(self, input_feature_dict):
        """Test basic padding functionality."""
        # Get current dimensions from a sample feature
        current_n_atom = input_feature_dict["ref_pos"].shape[0]
        current_n_token = input_feature_dict["restype"].shape[0]

        # Assume some default values for other dimensions
        current_n_msa = 32
        current_n_template = 4
        current_n_pocket = 1

        # Set target dimensions (larger than current)
        max_n_atom = current_n_atom + 10
        max_n_token = current_n_token + 5
        max_n_msa = current_n_msa + 8
        max_n_template = current_n_template + 2
        max_n_pocket = current_n_pocket + 1

        # Create a subset of features to test
        test_features = {
            "ref_pos": input_feature_dict["ref_pos"],
            "ref_mask": input_feature_dict["ref_mask"],
            "restype": input_feature_dict["restype"],
            "atom_to_token_idx": input_feature_dict.get(
                "atom_to_token_idx", torch.zeros(current_n_atom, dtype=torch.long)
            ),
        }

        # Pad the features
        padded_features = pad_features_dict(
            test_features,
            current_n_atom=current_n_atom,
            max_n_atom=max_n_atom,
            current_n_token=current_n_token,
            max_n_token=max_n_token,
            current_n_msa=current_n_msa,
            max_n_msa=max_n_msa,
            current_n_template=current_n_template,
            max_n_template=max_n_template,
            current_n_pocket=current_n_pocket,
            max_n_pocket=max_n_pocket,
        )

        # Verify shapes are correctly padded
        assert padded_features["ref_pos"].shape == (max_n_atom, 3)
        assert padded_features["ref_mask"].shape == (max_n_atom,)
        assert padded_features["restype"].shape == (max_n_token, 32)
        assert padded_features["atom_to_token_idx"].shape == (max_n_atom,)

        # Verify original data is preserved
        assert torch.equal(
            padded_features["ref_pos"][:current_n_atom], test_features["ref_pos"]
        )
        assert torch.equal(
            padded_features["ref_mask"][:current_n_atom], test_features["ref_mask"]
        )
        assert torch.equal(
            padded_features["restype"][:current_n_token], test_features["restype"]
        )

        # Verify padding values
        # ref_pos and ref_mask should be padded with 0 (default)
        assert torch.all(padded_features["ref_pos"][current_n_atom:] == 0)
        assert torch.all(padded_features["ref_mask"][current_n_atom:] == 0)
        assert torch.all(padded_features["restype"][current_n_token:] == 0)
        assert torch.all(padded_features["atom_to_token_idx"][current_n_atom:] == 0)

    def test_pad_features_dict_custom_padding_values(self, input_feature_dict):
        """Test padding with custom padding values."""
        current_n_atom = input_feature_dict["ref_pos"].shape[0]
        current_n_token = input_feature_dict["restype"].shape[0]

        max_n_atom = current_n_atom + 5
        max_n_token = current_n_token + 3

        test_features = {
            "ref_pos": input_feature_dict["ref_pos"],
            "restype": input_feature_dict["restype"],
        }

        custom_padding_values = {
            "ref_pos": -999.0,
            "restype": 31,  # Gap token
        }

        padded_features = pad_features_dict(
            test_features,
            current_n_atom=current_n_atom,
            max_n_atom=max_n_atom,
            current_n_token=current_n_token,
            max_n_token=max_n_token,
            current_n_msa=32,
            max_n_msa=40,
            current_n_template=4,
            max_n_template=6,
            current_n_pocket=1,
            max_n_pocket=2,
            padding_values=custom_padding_values,
        )

        # Verify custom padding values are used
        assert torch.all(padded_features["ref_pos"][current_n_atom:] == -999.0)
        assert torch.all(padded_features["restype"][current_n_token:] == 31)

    def test_pad_features_dict_shape_validation(self, input_feature_dict):
        """Test that shape validation works correctly."""
        current_n_atom = input_feature_dict["ref_pos"].shape[0]
        current_n_token = input_feature_dict["restype"].shape[0]

        # Create a feature with incorrect shape
        test_features = {
            "ref_pos": torch.randn(current_n_atom + 5, 3),  # Wrong size
        }

        with pytest.raises(ValueError, match="has shape"):
            pad_features_dict(
                test_features,
                current_n_atom=current_n_atom,
                max_n_atom=current_n_atom + 10,
                current_n_token=current_n_token,
                max_n_token=current_n_token + 5,
                current_n_msa=32,
                max_n_msa=40,
                current_n_template=4,
                max_n_template=6,
                current_n_pocket=1,
                max_n_pocket=2,
            )

    def test_pad_features_dict_unknown_feature(self):
        """Test handling of unknown features."""
        test_features = {
            "unknown_feature": torch.randn(10, 5),
        }

        with pytest.raises(AssertionError, match="Unknown feature"):
            pad_features_dict(
                test_features,
                current_n_atom=50,
                max_n_atom=60,
                current_n_token=30,
                max_n_token=35,
                current_n_msa=32,
                max_n_msa=40,
                current_n_template=4,
                max_n_template=6,
                current_n_pocket=1,
                max_n_pocket=2,
            )

    def test_pad_features_dict_cannot_shrink(self, input_feature_dict):
        """Test that features cannot be padded to smaller dimensions."""
        current_n_atom = input_feature_dict["ref_pos"].shape[0]
        current_n_token = input_feature_dict["restype"].shape[0]

        test_features = {
            "ref_pos": input_feature_dict["ref_pos"],
        }

        # Test assertion for atom shrinking
        with pytest.raises(AssertionError, match="Need at least 1 padding token"):
            pad_features_dict(
                test_features,
                current_n_atom=current_n_atom,
                max_n_atom=current_n_atom - 5,  # Smaller than current
                current_n_token=current_n_token,
                max_n_token=current_n_token + 5,
                current_n_msa=32,
                max_n_msa=40,
                current_n_template=4,
                max_n_template=6,
                current_n_pocket=1,
                max_n_pocket=2,
            )

        # Test assertion for token shrinking
        with pytest.raises(AssertionError, match="Need at least 1 padding token"):
            pad_features_dict(
                test_features,
                current_n_atom=current_n_atom,
                max_n_atom=current_n_atom + 5,
                current_n_token=current_n_token,
                max_n_token=current_n_token - 3,  # Smaller than current
                current_n_msa=32,
                max_n_msa=40,
                current_n_template=4,
                max_n_template=6,
                current_n_pocket=1,
                max_n_pocket=2,
            )

    def test_pad_features_dict_non_tensor_values(self, input_feature_dict):
        """Test that non-tensor values are preserved as-is."""
        current_n_atom = input_feature_dict["ref_pos"].shape[0]
        current_n_token = input_feature_dict["restype"].shape[0]

        test_features = {
            "ref_pos": input_feature_dict["ref_pos"],
            "scalar_value": torch.tensor(42.0),  # Single element tensor
            "non_tensor": "some_string",  # Non-tensor value
        }

        padded_features = pad_features_dict(
            test_features,
            current_n_atom=current_n_atom,
            max_n_atom=current_n_atom + 5,
            current_n_token=current_n_token,
            max_n_token=current_n_token + 3,
            current_n_msa=32,
            max_n_msa=40,
            current_n_template=4,
            max_n_template=6,
            current_n_pocket=1,
            max_n_pocket=2,
        )

        # Non-tensor and single-element tensors should be preserved
        assert padded_features["scalar_value"] == torch.tensor(42.0)
        assert padded_features["non_tensor"] == "some_string"

    def test_pad_features_dict_comprehensive(self, input_feature_dict):
        """Test padding with a comprehensive set of features."""
        current_n_atom = input_feature_dict["ref_pos"].shape[0]
        current_n_token = input_feature_dict["restype"].shape[0]

        max_n_atom = current_n_atom + 8
        max_n_token = current_n_token + 4
        max_n_msa = 40
        max_n_template = 6
        max_n_pocket = 2

        # Test with multiple features of different shapes
        test_features = {
            "ref_pos": input_feature_dict["ref_pos"],
            "ref_mask": input_feature_dict["ref_mask"],
            "ref_element": input_feature_dict.get(
                "ref_element", torch.zeros(current_n_atom, 128)
            ),
            "restype": input_feature_dict["restype"],
            "atom_to_token_idx": input_feature_dict.get(
                "atom_to_token_idx", torch.zeros(current_n_atom, dtype=torch.long)
            ),
        }

        padded_features = pad_features_dict(
            test_features,
            current_n_atom=current_n_atom,
            max_n_atom=max_n_atom,
            current_n_token=current_n_token,
            max_n_token=max_n_token,
            current_n_msa=32,
            max_n_msa=max_n_msa,
            current_n_template=4,
            max_n_template=max_n_template,
            current_n_pocket=1,
            max_n_pocket=max_n_pocket,
        )

        # Verify all shapes are correct
        expected_shapes, _ = get_data_shape_dict(
            num_token=max_n_token,
            num_atom=max_n_atom,
            num_msa=max_n_msa,
            num_templ=max_n_template,
            num_pocket=max_n_pocket,
        )

        for feature_name, feature_tensor in padded_features.items():
            if isinstance(feature_tensor, torch.Tensor) and feature_tensor.numel() > 1:
                expected_shape = expected_shapes[feature_name]
                assert (
                    feature_tensor.shape == expected_shape
                ), f"Feature {feature_name} has shape {feature_tensor.shape}, expected {expected_shape}"

        # Verify original data preservation
        for feature_name, original_tensor in test_features.items():
            if (
                isinstance(original_tensor, torch.Tensor)
                and original_tensor.numel() > 1
            ):
                padded_tensor = padded_features[feature_name]

                # Extract original portion based on feature type
                if "atom" in feature_name or feature_name in [
                    "ref_pos",
                    "ref_mask",
                    "ref_element",
                    "atom_to_token_idx",
                ]:
                    original_portion = padded_tensor[:current_n_atom]
                elif "token" in feature_name or feature_name in ["restype"]:
                    original_portion = padded_tensor[:current_n_token]
                else:
                    original_portion = padded_tensor

                if original_portion.shape == original_tensor.shape:
                    assert torch.equal(
                        original_portion, original_tensor
                    ), f"Original data not preserved for feature {feature_name}"


if __name__ == "__main__":
    pytest.main([__file__])
