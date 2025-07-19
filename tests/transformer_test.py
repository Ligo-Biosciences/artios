import os

import pytest
import torch

from protenix.model.modules.transformer import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    AtomTransformer,
    AttentionPairBias,
    ConditionedTransitionBlock,
    Transformer,
    TransformerBlock,
    DiffusionTransformer,
    DiffusionTransformerBlock,
)


class TestAttentionPairBias:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def sample_data(self):
        data_path = os.path.join(os.path.dirname(__file__), "testdata", "sample_data_dict.pt")
        if not os.path.exists(data_path):
            pytest.skip(f"Test data not found at {data_path}")
        return torch.load(data_path, weights_only=False)["input_feature_dict"]

    def test_attention_pair_bias(self, device, sample_data):
        n_heads = 16
        c_a = 768
        c_s = 384
        c_z = 128

        model = AttentionPairBias(has_s=True, n_heads=n_heads, c_a=c_a, c_s=c_s, c_z=c_z).to(device)

        batch_size = 2
        n_tokens = 32

        # Create dummy input tensors
        a = torch.randn(batch_size, n_tokens, c_a).to(device)
        s = torch.randn(batch_size, n_tokens, c_s).to(device)
        z = torch.randn(batch_size, n_tokens, n_tokens, c_z).to(device)

        out = model(a, s, z)

        # Check output shape and values
        assert out.shape == (batch_size, n_tokens, c_a)
        assert not torch.isnan(out).any()


class TestTransformerBlock:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_transformer_block(self, device):
        c_a = 768
        c_z = 128
        n_heads = 16

        model = TransformerBlock(c_a=c_a, c_z=c_z, n_heads=n_heads).to(device)

        batch_size = 2
        n_tokens = 32

        # Create dummy input tensors
        a = torch.randn(batch_size, n_tokens, c_a).to(device)
        z = torch.randn(batch_size, n_tokens, n_tokens, c_z).to(device)

        # Keep copies to check if z is modified
        z_orig = z.clone()

        out_a, out_z = model(a, z)

        # Check output shape and values
        assert out_a.shape == (batch_size, n_tokens, c_a)
        assert out_z.shape == (batch_size, n_tokens, n_tokens, c_z)
        assert not torch.isnan(out_a).any()
        assert not torch.isnan(out_z).any()

        # Check z is not modified
        assert torch.allclose(z, z_orig)


class TestTransformer:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_transformer(self, device):
        c_a = 768
        c_z = 128
        n_blocks = 3
        n_heads = 16

        model = Transformer(
            c_a=c_a, c_z=c_z, n_blocks=n_blocks, n_heads=n_heads
        ).to(device)

        batch_size = 2
        n_tokens = 32

        # Create dummy input tensors
        a = torch.randn(batch_size, n_tokens, c_a).to(device)
        z = torch.randn(batch_size, n_tokens, n_tokens, c_z).to(device)

        out = model(a, z)

        # Check output shape and values
        assert out.shape == (batch_size, n_tokens, c_a)
        assert not torch.isnan(out).any()


class TestDiffusionTransformerBlock:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_diffusion_transformer_block(self, device):
        c_a = 768
        c_s = 384
        c_z = 128
        n_heads = 16

        model = DiffusionTransformerBlock(c_a=c_a, c_s=c_s, c_z=c_z, n_heads=n_heads).to(device)

        batch_size = 2
        n_tokens = 32

        # Create dummy input tensors
        a = torch.randn(batch_size, n_tokens, c_a).to(device)
        s = torch.randn(batch_size, n_tokens, c_s).to(device)
        z = torch.randn(batch_size, n_tokens, n_tokens, c_z).to(device)

        # Keep copies to check if s and z are modified
        s_orig = s.clone()
        z_orig = z.clone()

        out_a, out_s, out_z = model(a, s, z)

        # Check output shape and values
        assert out_a.shape == (batch_size, n_tokens, c_a)
        assert out_s.shape == (batch_size, n_tokens, c_s)
        assert out_z.shape == (batch_size, n_tokens, n_tokens, c_z)
        assert not torch.isnan(out_a).any()
        assert not torch.isnan(out_s).any()
        assert not torch.isnan(out_z).any()

        # Check s and z are not modified
        assert torch.allclose(s, s_orig)
        assert torch.allclose(z, z_orig)


class TestDiffusionTransformer:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_diffusion_transformer(self, device):
        c_a = 768
        c_s = 384
        c_z = 128
        n_blocks = 3
        n_heads = 16

        model = DiffusionTransformer(
            c_a=c_a, c_s=c_s, c_z=c_z, n_blocks=n_blocks, n_heads=n_heads
        ).to(device)

        batch_size = 2
        n_tokens = 32

        # Create dummy input tensors
        a = torch.randn(batch_size, n_tokens, c_a).to(device)
        s = torch.randn(batch_size, n_tokens, c_s).to(device)
        z = torch.randn(batch_size, n_tokens, n_tokens, c_z).to(device)

        out = model(a, s, z)

        # Check output shape and values
        assert out.shape == (batch_size, n_tokens, c_a)
        assert not torch.isnan(out).any()


class TestAtomTransformer:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def sample_data(self):
        data_path = os.path.join(os.path.dirname(__file__), "testdata", "sample_data_dict.pt")
        if not os.path.exists(data_path):
            pytest.skip(f"Test data not found at {data_path}")
        return torch.load(data_path, weights_only=False)

    def test_atom_transformer(self, device, sample_data):
        c_atom = 128
        c_atompair = 16
        n_heads = 4
        n_queries = 32
        n_keys = 128
        n_blocks = 3

        model = AtomTransformer(
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_queries=n_queries,
            n_keys=n_keys,
        ).to(device)

        batch_size = 2
        n_atoms = 256

        # Create dummy input tensors
        q = torch.randn(batch_size, n_atoms, c_atom).to(device)
        c = torch.randn(batch_size, n_atoms, c_atom).to(device)
        p = torch.randn(batch_size, n_atoms // n_queries, n_queries, n_keys, c_atompair).to(device)

        out = model(q, c, p)

        # Check output shape and values
        assert out.shape == (batch_size, n_atoms, c_atom)
        assert not torch.isnan(out).any()


class TestConditionedTransitionBlock:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_conditioned_transition_block(self, device):
        c_a = 128
        c_s = 128
        n = 2
        biasinit = -2.0

        model = ConditionedTransitionBlock(c_a=c_a, c_s=c_s, n=n, biasinit=biasinit).to(device)

        batch_size = 2
        n_tokens = 256

        # Create dummy input tensors
        a = torch.randn(batch_size, n_tokens, c_a).to(device)
        s = torch.randn(batch_size, n_tokens, c_s).to(device)

        out = model(a, s)

        # Check output shape and values
        assert out.shape == (batch_size, n_tokens, c_a)
        assert not torch.isnan(out).any()


class TestAtomAttentionDecoder:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def sample_data(self):
        data_path = os.path.join(os.path.dirname(__file__), "testdata", "sample_data_dict.pt")
        if not os.path.exists(data_path):
            pytest.skip(f"Test data not found at {data_path}")
        return torch.load(data_path, weights_only=False)

    def test_atom_attention_decoder(self, device, sample_data):
        n_blocks = 3
        n_heads = 4
        c_token = 384
        c_atom = 128
        c_atompair = 16
        n_queries = 32
        n_keys = 128

        model = AtomAttentionDecoder(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_queries=n_queries,
            n_keys=n_keys,
        ).to(device)

        # Move sample data to device
        n_tokens = sample_data["basic"]["N_token"][0]
        n_atoms = sample_data["basic"]["N_atom"][0]
        sample_data = sample_data["input_feature_dict"]
        input_dict = {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample_data.items()}
        batch_size = 2

        # Create dummy input tensors
        a = torch.randn(batch_size, n_tokens, c_token).to(device)
        q_skip = torch.randn(batch_size, n_atoms, c_atom).to(device)
        c_skip = torch.randn(batch_size, n_atoms, c_atom).to(device)
        p_skip = torch.randn(batch_size, n_atoms // n_queries, n_queries, n_keys, c_atompair).to(
            device
        )

        out = model(input_dict, a, q_skip, c_skip, p_skip)

        # Check output shape and values
        assert out.shape == (batch_size, n_atoms, 3)  # Output should be 3D coordinates
        assert not torch.isnan(out).any()

    def test_atom_attention_encoder(self, device, sample_data):
        has_coords = True
        n_blocks = 3
        n_heads = 4
        c_token = 384
        c_atom = 128
        c_atompair = 16
        c_s = 384
        c_z = 128
        n_queries = 32
        n_keys = 128

        model = AtomAttentionEncoder(
            has_coords=has_coords,
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_s=c_s,
            c_z=c_z,
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_queries=n_queries,
            n_keys=n_keys,
        ).to(device)

        # Move sample data to device
        n_tokens = sample_data["basic"]["N_token"][0]
        n_atoms = sample_data["basic"]["N_atom"][0]
        sample_data = sample_data["input_feature_dict"]
        input_dict = {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample_data.items()}
        batch_size = 2

        # Create dummy input tensors
        r_l = torch.randn(batch_size, n_atoms, 3).to(device)
        s = torch.randn(batch_size, n_tokens, c_s).to(device)
        z = torch.randn(batch_size, n_tokens, n_tokens, c_z).to(device)

        a, q_l, c_l, p_lm = model(input_dict, r_l, s, z)

        # Check output shapes and values
        assert a.shape == (batch_size, n_tokens, c_token)
        assert q_l.shape == (batch_size, n_atoms, c_atom)
        assert c_l.shape == (batch_size, n_atoms, c_atom)
        assert p_lm.shape == (
            batch_size,
            n_atoms // n_queries,
            n_queries,
            n_keys,
            c_atompair,
        )
        assert not torch.isnan(a).any()
        assert not torch.isnan(q_l).any()
        assert not torch.isnan(c_l).any()
        assert not torch.isnan(p_lm).any()


if __name__ == "__main__":
    pytest.main([__file__])
