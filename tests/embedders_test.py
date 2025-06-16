import pytest
import torch
import os

from protenix.model.modules.embedders import (
    InputFeatureEmbedder, RelativePositionEncoding, FourierEmbedding
)


class TestInputFeatureEmbedder:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def sample_data(self):
        data_path = os.path.join(os.path.dirname(__file__), "testdata", "sample_data_dict.pt") #data_path = "/Users/baharalan/Developer/artios-ligand/tests/testdata/sample_data_dict.pt" 

        if not os.path.exists(data_path):
            pytest.skip(f"Test data not found at {data_path}")
        return torch.load(data_path)["input_feature_dict"]

    def test_input_feature_embedder(self, device, sample_data):
        c_atom = 128
        c_atompair = 16 
        c_token = 384

        embedder = InputFeatureEmbedder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token
        ).to(device)

        # Move sample data to device
        input_dict = {k: v.to(device) if torch.is_tensor(v) else v 
                     for k, v in sample_data.items()}

        out = embedder(input_dict)

        # Check output shape
        expected_dim = c_token + 32 # 449 as per docstring TODO: update since removed certain restypes 
        assert out.shape[-1] == expected_dim
        assert out.shape[:-1] == input_dict["restype"].shape[:-1]
        assert not torch.isnan(out).any()


class TestRelativePositionEncoding:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def sample_data(self):
        data_path = os.path.join(os.path.dirname(__file__), "testdata", "sample_data_dict.pt")
        if not os.path.exists(data_path):
            pytest.skip(f"Test data not found at {data_path}")
        return torch.load(data_path)["input_feature_dict"]

    def test_relative_position_encoding(self, device, sample_data):
        r_max = 32
        s_max = 2
        c_z = 128

        encoder = RelativePositionEncoding(
            r_max=r_max,
            s_max=s_max,
            c_z=c_z
        ).to(device)

        # Move sample data to device
        input_dict = {k: v.to(device) if torch.is_tensor(v) else v 
                     for k, v in sample_data.items()}

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
        embedder3 = FourierEmbedding(c=c, seed=seed+1).to(device)
        out3 = embedder3(t_hat)
        assert not torch.allclose(out, out3)


if __name__ == "__main__":
    pytest.main([__file__])

