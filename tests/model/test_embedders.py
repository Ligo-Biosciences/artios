import pytest
import torch

from artios.data.features.featurizer import ContextGenerator
from artios.model.modules.embedders import (
    FourierEmbedding,
    RelativePositionEncoding,
)


class TestRelativePositionEncoding:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def token_context(self, load_structure, device):
        """Token-level IDs from a real featurized structure."""
        structure = load_structure("2g1k")
        ctx = ContextGenerator(structure, pdb_id="2g1k").get_full_context().to(device)
        return ctx.token

    def test_relative_position_encoding(self, device, token_context):
        c_z = 128
        encoder = RelativePositionEncoding(r_max=32, s_max=2, c_z=c_z).to(device)
        out = encoder(
            asym_id=token_context.asym_id,
            residue_index=token_context.residue_index,
            entity_id=token_context.entity_id,
            sym_id=token_context.sym_id,
            token_index=token_context.token_index,
        )
        n_tokens = token_context.asym_id.shape[-1]
        assert out.shape[-3:] == (n_tokens, n_tokens, c_z)
        assert out.shape[:-3] == token_context.asym_id.shape[:-1]
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
