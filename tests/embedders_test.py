import os

import pytest
import torch

from protenix.model.modules.embedders import (
    FourierEmbedding,
    InputFeatureEmbedder,
    RelativePositionEncoding,
)


class TestRelativePositionEncoding:
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
        return torch.load(data_path, weights_only=False)["input_feature_dict"]

    def test_relative_position_encoding(self, device, sample_data):
        r_max = 32
        s_max = 2
        c_z = 128

        encoder = RelativePositionEncoding(r_max=r_max, s_max=s_max, c_z=c_z).to(device)

        # Move sample data to device
        input_dict = {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in sample_data.items()
        }

        out = encoder(input_dict)

        # Check output shape
        n_tokens = input_dict["asym_id"].shape[-1]
        assert out.shape[-3:] == (n_tokens, n_tokens, c_z)
        assert out.shape[:-3] == input_dict["asym_id"].shape[:-1]
        assert not torch.isnan(out).any()


class TestFourierEmbedding:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_fourier_embedding(self, device):
        batch_size = 32
        n_samples = 16
        c = 64
        seed = 42

        embedder = FourierEmbedding(c=c, seed=seed).to(device)

        # Create random noise levels between 0 and 1
        t_hat = torch.rand(batch_size, n_samples).to(device)

        out = embedder(t_hat)

        # Check output shape and values
        assert out.shape == (batch_size, n_samples, c)
        assert not torch.isnan(out).any()
        assert (out >= -1).all() and (out <= 1).all()  # cos output range

        # Test reproducibility with same seed
        embedder2 = FourierEmbedding(c=c, seed=seed).to(device)
        out2 = embedder2(t_hat)
        assert torch.allclose(out, out2)

        # Test different output with different seed
        embedder3 = FourierEmbedding(c=c, seed=seed + 1).to(device)
        out3 = embedder3(t_hat)
        assert not torch.allclose(out, out3)


class TestInputFeatureEmbedder:
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

    def test_input_feature_embedder(self, device, sample_data):
        c_atom = 128
        c_atompair = 16
        c_token = 384
        n_blocks = 3

        embedder = InputFeatureEmbedder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            n_blocks=n_blocks,
        ).to(device)

        # Move sample data to device
        n_tokens = sample_data["basic"]["N_token"][0]
        input_dict = sample_data["input_feature_dict"]
        input_dict = {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in input_dict.items()
        }

        # Test forward pass
        output = embedder(input_dict)

        # Check output shape and values
        assert output.shape == (n_tokens, c_token)
        assert not torch.isnan(output).any()

    def test_input_feature_embedder_with_chunk_size(self, device, sample_data):
        """Test InputFeatureEmbedder with chunk_size parameter"""
        c_atom = 128
        c_atompair = 16
        c_token = 384
        n_blocks = 3

        embedder = InputFeatureEmbedder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            n_blocks=n_blocks,
        ).to(device)

        # Move sample data to device
        n_tokens = sample_data["basic"]["N_token"][0]
        input_dict = sample_data["input_feature_dict"]
        input_dict = {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in input_dict.items()
        }

        # Test forward pass with chunk_size
        output = embedder(input_dict, chunk_size=64)

        # Check output shape and values
        assert output.shape == (n_tokens, c_token)
        assert not torch.isnan(output).any()

    def test_input_feature_embedder_different_sizes(self, device, sample_data):
        """Test InputFeatureEmbedder with different embedding dimensions"""
        c_atom = 64
        c_atompair = 8
        c_token = 256
        n_blocks = 2

        embedder = InputFeatureEmbedder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            n_blocks=n_blocks,
        ).to(device)

        # Move sample data to device
        n_tokens = sample_data["basic"]["N_token"][0]
        input_dict = sample_data["input_feature_dict"]
        input_dict = {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in input_dict.items()
        }

        # Test forward pass
        output = embedder(input_dict)

        # Check output shape matches the specified c_token
        assert output.shape == (n_tokens, c_token)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__])
