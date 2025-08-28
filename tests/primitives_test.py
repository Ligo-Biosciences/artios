import pytest
import torch

from protenix.model.modules.primitives import (
    AdaptiveLayerNorm,
    Attention,
    BiasInitLinear,
    LayerNorm,
    Linear,
    LinearNoBias,
    SwiGLU,
    Transition,
    _calculate_fan,
    attention,
    bias_init_fn,
    create_local_attn_bias,
    final_init_,
    gather_pair_embedding_in_dense_trunk,
    gating_init_,
    glorot_uniform_init_,
    he_normal_init_,
    init_bias_to_zero_,
    init_weight_to_zero_,
    lecun_normal_init_,
    normal_init_,
    optimized_concat_split,
    rearrange_qk_to_dense_trunk,
    rearrange_to_dense_trunk,
    trunc_normal_init_,
)


class TestInitializationFunctions:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_calculate_fan(self):
        shape = (10, 20)
        assert _calculate_fan(shape, "fan_in") == 20
        assert _calculate_fan(shape, "fan_out") == 10
        assert _calculate_fan(shape, "fan_avg") == 15

        with pytest.raises(ValueError):
            _calculate_fan(shape, "invalid_fan")

    def test_trunc_normal_init(self):
        weight = torch.empty(10, 20)
        bias = torch.empty(10)

        trunc_normal_init_(weight, bias)

        assert not torch.isnan(weight).any()
        assert not torch.isnan(bias).any()
        assert bias.abs().sum() == 0

    def test_init_bias_zero(self):
        bias = torch.ones(10)
        init_bias_to_zero_(bias)
        assert torch.all(bias == 0)

    def test_init_weight_zero(self):
        weight = torch.ones(10, 20)
        init_weight_to_zero_(weight)
        assert torch.all(weight == 0)

    def test_initialization_functions(self):
        weight = torch.empty(10, 20)
        bias = torch.empty(10)

        for init_fn in [
            lecun_normal_init_,
            he_normal_init_,
            glorot_uniform_init_,
            final_init_,
            gating_init_,
            normal_init_,
        ]:
            init_fn(weight, bias)
            assert not torch.isnan(weight).any()
            assert not torch.isnan(bias).any()

    def test_bias_init_fn(self):
        weight = torch.empty(10, 20)
        bias = torch.empty(10)
        biasinit = 1.0

        bias_init_fn(weight, bias, biasinit)
        assert torch.all(weight == 0)
        assert torch.all(bias == biasinit)


class TestLayers:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_linear_layer(self, device):
        batch_size = 32
        in_features = 64
        out_features = 128

        init_methods = ["default", "relu", "glorot", "gating", "normal", "zeros"]

        for init in init_methods:
            layer = Linear(in_features, out_features, init=init).to(device)
            x = torch.randn(batch_size, in_features).to(device)

            out = layer(x)
            assert out.shape == (batch_size, out_features)
            assert not torch.isnan(out).any()

        with pytest.raises(ValueError):
            Linear(in_features, out_features, init="invalid_init")

    def test_linear_no_bias(self, device):
        batch_size = 32
        in_features = 64
        out_features = 128

        layer = LinearNoBias(in_features, out_features).to(device)
        x = torch.randn(batch_size, in_features).to(device)

        out = layer(x)
        assert out.shape == (batch_size, out_features)
        assert not torch.isnan(out).any()
        assert layer.bias is None

    def test_bias_init_linear(self, device):
        batch_size = 32
        in_features = 64
        out_features = 128
        biasinit = 1.0

        layer = BiasInitLinear(in_features, out_features, biasinit=biasinit).to(device)
        x = torch.randn(batch_size, in_features).to(device)

        out = layer(x)
        assert out.shape == (batch_size, out_features)
        assert not torch.isnan(out).any()
        assert torch.all(layer.bias == biasinit)

    def test_layer_norm(self, device):
        batch_size = 32
        features = 64

        layer = LayerNorm(features).to(device)
        x = torch.randn(batch_size, features).to(device)

        out = layer(x)
        assert out.shape == (batch_size, features)
        assert not torch.isnan(out).any()
        assert torch.all(layer.bias == 0)

    def test_swiglu(self, device):
        batch_size = 32
        c_in = 64
        c_out = 128

        layer = SwiGLU(c_in, c_out).to(device)
        x = torch.randn(batch_size, c_in).to(device)

        out = layer(x)
        assert out.shape == (batch_size, c_out)
        assert not torch.isnan(out).any()

    def test_adaptive_layer_norm(self, device):
        batch_size = 32
        n_tokens = 16
        c_a = 768
        c_s = 384

        layer = AdaptiveLayerNorm(c_a, c_s).to(device)
        a = torch.randn(batch_size, n_tokens, c_a).to(device)
        s = torch.randn(batch_size, n_tokens, c_s).to(device)

        out = layer(a, s)
        assert out.shape == (batch_size, n_tokens, c_a)
        assert not torch.isnan(out).any()

    def test_transition(self, device):
        batch_size = 32
        c_in = 64
        n = 4

        layer = Transition(c_in, n).to(device)
        x = torch.randn(batch_size, c_in).to(device)

        out = layer(x)
        assert out.shape == (batch_size, c_in)
        assert not torch.isnan(out).any()


class TestAttention:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.mark.parametrize("use_efficient_implementation", [True, False])
    def test_attention(self, device, use_efficient_implementation):
        batch_size = 2
        seq_len = 8
        c_q = c_k = c_v = 32
        c_hidden = 64
        num_heads = 4

        attn = Attention(
            c_q=c_q,
            c_k=c_k,
            c_v=c_v,
            c_hidden=c_hidden,
            num_heads=num_heads,
            use_efficient_implementation=use_efficient_implementation,
        ).to(device)

        q_x = torch.randn(batch_size, seq_len, c_q).to(device)
        kv_x = torch.randn(batch_size, seq_len, c_k).to(device)

        out = attn(q_x, kv_x)
        assert out.shape == (batch_size, seq_len, c_q)
        assert not torch.isnan(out).any()

        attn_bias = torch.randn(batch_size, seq_len, seq_len).to(device)
        out = attn(q_x, kv_x, attn_bias=attn_bias)
        assert out.shape == (batch_size, seq_len, c_q)
        assert not torch.isnan(out).any()


class TestAttentionHelpers:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_attention_core(self, device):
        batch_size = 2
        seq_len = 8
        hidden_dim = 32

        q = torch.randn(batch_size, seq_len, hidden_dim).to(device)
        k = torch.randn(batch_size, seq_len, hidden_dim).to(device)
        v = torch.randn(batch_size, seq_len, hidden_dim).to(device)

        out = attention(q, k, v)
        assert out.shape == (batch_size, seq_len, hidden_dim)
        assert not torch.isnan(out).any()

    def test_rearrange_qk_to_dense_trunk(self):
        batch_size = 2
        seq_len = 16
        hidden_dim = 32
        n_queries = 4
        n_keys = 8

        q = torch.randn(batch_size, seq_len, hidden_dim)
        k = torch.randn(batch_size, seq_len, hidden_dim)

        q_trunked, k_trunked, padding_info = rearrange_qk_to_dense_trunk(
            q=q, k=k, dim_q=-2, dim_k=-2, n_queries=n_queries, n_keys=n_keys
        )

        n_trunks = (seq_len + n_queries - 1) // n_queries
        assert q_trunked.shape == (batch_size, n_trunks, n_queries, hidden_dim)
        assert k_trunked.shape == (batch_size, n_trunks, n_keys, hidden_dim)

        assert all(
            k in padding_info
            for k in ["mask_trunked", "q_pad", "k_pad_left", "k_pad_right"]
        )

    def test_optimized_concat_split(self):
        batch_size = 2
        D = 16
        E = 8
        n_queries = 4

        attn_bias = torch.randn(batch_size, D, E)
        out = optimized_concat_split(attn_bias, n_queries)

        assert out.shape == (batch_size, n_queries, D // n_queries * E)
        assert not torch.isnan(out).any()

    def test_rearrange_to_dense_trunk(self):
        batch_size = 2
        seq_len = 16
        hidden_dim = 32
        n_queries = 4
        n_keys = 8

        q = torch.randn(batch_size, seq_len, hidden_dim)
        k = torch.randn(batch_size, seq_len, hidden_dim)
        v = torch.randn(batch_size, seq_len, hidden_dim)

        (
            q_trunked,
            k_trunked,
            v_trunked,
            attn_bias_trunked,
            q_pad_length,
        ) = rearrange_to_dense_trunk(q, k, v, n_queries, n_keys)

        n_trunks = (seq_len + n_queries - 1) // n_queries
        assert q_trunked.shape == (batch_size, n_trunks, n_queries, hidden_dim)
        assert k_trunked.shape == (batch_size, n_trunks, n_keys, hidden_dim)
        assert v_trunked.shape == (batch_size, n_trunks, n_keys, hidden_dim)

    def test_create_local_attn_bias(self):
        n = 16
        n_queries = 4
        n_keys = 8

        attn_bias = create_local_attn_bias(n, n_queries, n_keys)
        assert attn_bias.shape == (n, n)
        assert not torch.isnan(attn_bias).any()

    def test_gather_pair_embedding(self):
        # Setup basic parameters
        batch_size = 2
        n_b = 4
        seq_len = 16
        n_q = 8
        n_k = 8
        hidden_dim = 32

        # Create a tensor 'x' of shape [batch_size, seq_len, seq_len, hidden_dim]
        x = torch.randn(batch_size, seq_len, seq_len, hidden_dim)
        # Randomly generate indices for query and key selection
        idx_q = torch.randint(0, seq_len, (n_b, n_q))
        idx_k = torch.randint(0, seq_len, (n_b, n_k))

        # -------------------------
        # Basic test using integer indices
        out = gather_pair_embedding_in_dense_trunk(x, idx_q, idx_k)
        # Expected shape: [batch_size, n_b, n_q, n_k, hidden_dim]
        assert out.shape == (
            batch_size,
            n_b,
            n_q,
            n_k,
            hidden_dim,
        ), f"Expected shape {(batch_size, n_b, n_q, n_k, hidden_dim)}, got {out.shape}"
        # Verify that each gathered element matches the corresponding element from 'x'
        for b in range(batch_size):
            for b_idx in range(n_b):
                for i in range(n_q):
                    for j in range(n_k):
                        expected = x[b, idx_q[b_idx, i], idx_k[b_idx, j]]
                        actual = out[b, b_idx, i, j]
                        assert torch.allclose(
                            expected, actual
                        ), f"Mismatch at batch {b}, chain index {b_idx}, q index {i}, k index {j}"

        # -------------------------
        # Extended test: Include extra leading dimensions
        extra_dims = (3, 4)
        x_extra = torch.randn(*extra_dims, batch_size, seq_len, seq_len, hidden_dim)
        out_extra = gather_pair_embedding_in_dense_trunk(x_extra, idx_q, idx_k)
        expected_shape = extra_dims + (batch_size, n_b, n_q, n_k, hidden_dim)
        assert (
            out_extra.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {out_extra.shape}"
        # Validate individual entries with extra dimensions
        for d0 in range(extra_dims[0]):
            for d1 in range(extra_dims[1]):
                for b in range(batch_size):
                    for b_idx in range(n_b):
                        for i in range(n_q):
                            for j in range(n_k):
                                expected = x_extra[
                                    d0, d1, b, idx_q[b_idx, i], idx_k[b_idx, j]
                                ]
                                actual = out_extra[d0, d1, b, b_idx, i, j]
                                assert torch.allclose(
                                    expected, actual
                                ), f"Mismatch at extra dims ({d0},{d1}), batch {b}, chain index {b_idx}, q {i}, k {j}"

        # -------------------------
        # Test type conversion: indices provided as float should be handled correctly
        idx_q_float = idx_q.float()
        idx_k_float = idx_k.float()
        out_float = gather_pair_embedding_in_dense_trunk(x, idx_q_float, idx_k_float)
        assert torch.allclose(
            out, out_float
        ), "Output differs when indices are provided as floats compared to integers"


if __name__ == "__main__":
    pytest.main([__file__])
