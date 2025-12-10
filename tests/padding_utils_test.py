import os

import pytest
import torch

from protenix.data.padding_utils import pad_features_dict, pad_tensor
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


class TestPadTensor:
    """Test pad_tensor function for Context tensor padding."""

    # Typical dimensions during training
    N_TOKEN = 128
    N_ATOM = 512
    BATCH_SIZE = 4
    PAD_TOKEN = 16
    PAD_ATOM = 32

    # ==========================================================================
    # Unbatched tests (num_batch_dims=0)
    # ==========================================================================

    def test_pad_token_1d_unbatched(self):
        """Test padding 1D token tensor [n_token] without batch."""
        tensor = torch.randn(self.N_TOKEN)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_TOKEN,), num_batch_dims=0)

        assert padded.shape == (self.N_TOKEN + self.PAD_TOKEN,)
        assert torch.equal(padded[: self.N_TOKEN], tensor)
        assert torch.all(padded[self.N_TOKEN :] == 0)

    def test_pad_token_2d_unbatched(self):
        """Test padding 2D token tensor [n_token, 32] (restype) without batch."""
        tensor = torch.randn(self.N_TOKEN, 32)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_TOKEN,), num_batch_dims=0)

        assert padded.shape == (self.N_TOKEN + self.PAD_TOKEN, 32)
        assert torch.equal(padded[: self.N_TOKEN], tensor)
        assert torch.all(padded[self.N_TOKEN :] == 0)

    def test_pad_token_pair_unbatched(self):
        """Test padding token pair tensor [n_token, n_token] (token_bonds) without batch."""
        tensor = torch.randn(self.N_TOKEN, self.N_TOKEN)
        padded = pad_tensor(
            tensor, pad_sizes=(self.PAD_TOKEN, self.PAD_TOKEN), num_batch_dims=0
        )

        expected_size = self.N_TOKEN + self.PAD_TOKEN
        assert padded.shape == (expected_size, expected_size)
        assert torch.equal(padded[: self.N_TOKEN, : self.N_TOKEN], tensor)
        # Check padding regions are zeros
        assert torch.all(padded[self.N_TOKEN :, :] == 0)
        assert torch.all(padded[:, self.N_TOKEN :] == 0)

    def test_pad_atom_1d_unbatched(self):
        """Test padding 1D atom tensor [n_atom] without batch."""
        tensor = torch.randn(self.N_ATOM)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_ATOM,), num_batch_dims=0)

        assert padded.shape == (self.N_ATOM + self.PAD_ATOM,)
        assert torch.equal(padded[: self.N_ATOM], tensor)
        assert torch.all(padded[self.N_ATOM :] == 0)

    def test_pad_atom_2d_unbatched(self):
        """Test padding 2D atom tensor [n_atom, 3] (ref_pos, coordinate) without batch."""
        tensor = torch.randn(self.N_ATOM, 3)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_ATOM,), num_batch_dims=0)

        assert padded.shape == (self.N_ATOM + self.PAD_ATOM, 3)
        assert torch.equal(padded[: self.N_ATOM], tensor)
        assert torch.all(padded[self.N_ATOM :] == 0)

    def test_pad_atom_3d_unbatched(self):
        """Test padding 3D atom tensor [n_atom, 4, 64] (ref_atom_name_chars) without batch."""
        tensor = torch.randn(self.N_ATOM, 4, 64)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_ATOM,), num_batch_dims=0)

        assert padded.shape == (self.N_ATOM + self.PAD_ATOM, 4, 64)
        assert torch.equal(padded[: self.N_ATOM], tensor)
        assert torch.all(padded[self.N_ATOM :] == 0)

    def test_pad_backbone_coords_unbatched(self):
        """Test padding backbone coords [n_token, 5, 3] without batch."""
        tensor = torch.randn(self.N_TOKEN, 5, 3)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_TOKEN,), num_batch_dims=0)

        assert padded.shape == (self.N_TOKEN + self.PAD_TOKEN, 5, 3)
        assert torch.equal(padded[: self.N_TOKEN], tensor)
        assert torch.all(padded[self.N_TOKEN :] == 0)

    # ==========================================================================
    # Batched tests (num_batch_dims=1)
    # ==========================================================================

    def test_pad_token_1d_batched(self):
        """Test padding 1D token tensor [batch, n_token] with batch dim."""
        tensor = torch.randn(self.BATCH_SIZE, self.N_TOKEN)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_TOKEN,), num_batch_dims=1)

        assert padded.shape == (self.BATCH_SIZE, self.N_TOKEN + self.PAD_TOKEN)
        assert torch.equal(padded[:, : self.N_TOKEN], tensor)
        assert torch.all(padded[:, self.N_TOKEN :] == 0)

    def test_pad_token_2d_batched(self):
        """Test padding 2D token tensor [batch, n_token, 32] (restype) with batch dim."""
        tensor = torch.randn(self.BATCH_SIZE, self.N_TOKEN, 32)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_TOKEN,), num_batch_dims=1)

        assert padded.shape == (self.BATCH_SIZE, self.N_TOKEN + self.PAD_TOKEN, 32)
        assert torch.equal(padded[:, : self.N_TOKEN, :], tensor)
        assert torch.all(padded[:, self.N_TOKEN :, :] == 0)

    def test_pad_token_pair_batched(self):
        """Test padding token pair [batch, n_token, n_token] (token_bonds) with batch dim."""
        tensor = torch.randn(self.BATCH_SIZE, self.N_TOKEN, self.N_TOKEN)
        padded = pad_tensor(
            tensor, pad_sizes=(self.PAD_TOKEN, self.PAD_TOKEN), num_batch_dims=1
        )

        expected_size = self.N_TOKEN + self.PAD_TOKEN
        assert padded.shape == (self.BATCH_SIZE, expected_size, expected_size)
        assert torch.equal(padded[:, : self.N_TOKEN, : self.N_TOKEN], tensor)
        assert torch.all(padded[:, self.N_TOKEN :, :] == 0)
        assert torch.all(padded[:, :, self.N_TOKEN :] == 0)

    def test_pad_atom_1d_batched(self):
        """Test padding 1D atom tensor [batch, n_atom] with batch dim."""
        tensor = torch.randn(self.BATCH_SIZE, self.N_ATOM)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_ATOM,), num_batch_dims=1)

        assert padded.shape == (self.BATCH_SIZE, self.N_ATOM + self.PAD_ATOM)
        assert torch.equal(padded[:, : self.N_ATOM], tensor)
        assert torch.all(padded[:, self.N_ATOM :] == 0)

    def test_pad_atom_2d_batched(self):
        """Test padding 2D atom tensor [batch, n_atom, 3] (ref_pos) with batch dim."""
        tensor = torch.randn(self.BATCH_SIZE, self.N_ATOM, 3)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_ATOM,), num_batch_dims=1)

        assert padded.shape == (self.BATCH_SIZE, self.N_ATOM + self.PAD_ATOM, 3)
        assert torch.equal(padded[:, : self.N_ATOM, :], tensor)
        assert torch.all(padded[:, self.N_ATOM :, :] == 0)

    def test_pad_atom_3d_batched(self):
        """Test padding 3D atom tensor [batch, n_atom, 4, 64] (ref_atom_name_chars) with batch dim."""
        tensor = torch.randn(self.BATCH_SIZE, self.N_ATOM, 4, 64)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_ATOM,), num_batch_dims=1)

        assert padded.shape == (self.BATCH_SIZE, self.N_ATOM + self.PAD_ATOM, 4, 64)
        assert torch.equal(padded[:, : self.N_ATOM, :, :], tensor)
        assert torch.all(padded[:, self.N_ATOM :, :, :] == 0)

    def test_pad_backbone_coords_batched(self):
        """Test padding backbone coords [batch, n_token, 5, 3] with batch dim."""
        tensor = torch.randn(self.BATCH_SIZE, self.N_TOKEN, 5, 3)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_TOKEN,), num_batch_dims=1)

        assert padded.shape == (self.BATCH_SIZE, self.N_TOKEN + self.PAD_TOKEN, 5, 3)
        assert torch.equal(padded[:, : self.N_TOKEN, :, :], tensor)
        assert torch.all(padded[:, self.N_TOKEN :, :, :] == 0)

    # ==========================================================================
    # Edge cases
    # ==========================================================================

    def test_pad_zero_padding(self):
        """Test that zero padding returns original tensor."""
        tensor = torch.randn(self.N_TOKEN, 32)
        padded = pad_tensor(tensor, pad_sizes=(0,), num_batch_dims=0)

        assert padded.shape == tensor.shape
        assert torch.equal(padded, tensor)

    def test_pad_empty_pad_sizes(self):
        """Test that empty pad_sizes returns original tensor."""
        tensor = torch.randn(self.N_TOKEN, 32)
        padded = pad_tensor(tensor, pad_sizes=(), num_batch_dims=0)

        assert padded.shape == tensor.shape
        assert torch.equal(padded, tensor)

    def test_pad_preserves_dtype(self):
        """Test that padding preserves tensor dtype."""
        for dtype in [torch.float32, torch.float64, torch.int64, torch.bool]:
            if dtype == torch.bool:
                tensor = torch.randint(0, 2, (self.N_TOKEN,)).bool()
            elif dtype in [torch.int64]:
                tensor = torch.randint(0, 100, (self.N_TOKEN,), dtype=dtype)
            else:
                tensor = torch.randn(self.N_TOKEN, dtype=dtype)

            padded = pad_tensor(tensor, pad_sizes=(self.PAD_TOKEN,), num_batch_dims=0)
            assert padded.dtype == dtype, f"dtype mismatch for {dtype}"

    def test_pad_preserves_device(self):
        """Test that padding preserves tensor device."""
        tensor = torch.randn(self.N_TOKEN)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_TOKEN,), num_batch_dims=0)
        assert padded.device == tensor.device

    def test_pad_assertion_on_insufficient_dims(self):
        """Test that assertion fires when pad_sizes exceeds available dims."""
        tensor = torch.randn(self.BATCH_SIZE, self.N_TOKEN)  # 2D tensor

        with pytest.raises(AssertionError, match="pad_sizes .* exceed available"):
            # Trying to pad 3 dims but only 1 non-batch dim available
            pad_tensor(tensor, pad_sizes=(10, 10, 10), num_batch_dims=1)

    def test_pad_ref_element_128_unbatched(self):
        """Test padding ref_element [n_atom, 128] without batch."""
        tensor = torch.randn(self.N_ATOM, 128)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_ATOM,), num_batch_dims=0)

        assert padded.shape == (self.N_ATOM + self.PAD_ATOM, 128)
        assert torch.equal(padded[: self.N_ATOM], tensor)

    def test_pad_ref_element_128_batched(self):
        """Test padding ref_element [batch, n_atom, 128] with batch dim."""
        tensor = torch.randn(self.BATCH_SIZE, self.N_ATOM, 128)
        padded = pad_tensor(tensor, pad_sizes=(self.PAD_ATOM,), num_batch_dims=1)

        assert padded.shape == (self.BATCH_SIZE, self.N_ATOM + self.PAD_ATOM, 128)
        assert torch.equal(padded[:, : self.N_ATOM, :], tensor)


if __name__ == "__main__":
    pytest.main([__file__])
