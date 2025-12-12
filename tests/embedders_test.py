import os

import pytest
import torch

from protenix.model.modules.embedders import (
    FourierEmbedding,
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

        # Call with keyword arguments as expected by the new signature
        out = encoder(
            asym_id=input_dict["asym_id"],
            residue_index=input_dict["residue_index"],
            entity_id=input_dict["entity_id"],
            sym_id=input_dict["sym_id"],
            token_index=input_dict["token_index"],
        )

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


if __name__ == "__main__":
    pytest.main([__file__])
