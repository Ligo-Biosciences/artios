import pytest
import torch
import numpy as np
from protenix.model.modules.primitives import (
    _calculate_fan,
    trunc_normal_init_,
    Linear,
    LayerNorm,
    SwiGLU,
    Attention,
    _attention,
    rearrange_qk_to_dense_trunk,
)


class TestPrimitives:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test _calculate_fan
    def test_calculate_fan(self):
        shape = (10, 20)
        assert _calculate_fan(shape, "fan_in") == 20
        assert _calculate_fan(shape, "fan_out") == 10
        assert _calculate_fan(shape, "fan_avg") == 15
        
        with pytest.raises(ValueError):
            _calculate_fan(shape, "invalid_fan")

    # Test trunc_normal_init_
    def test_trunc_normal_init(self):
        weight = torch.empty(10, 20)
        bias = torch.empty(10)
        
        trunc_normal_init_(weight, bias)
        
        assert not torch.isnan(weight).any()
        assert not torch.isnan(bias).any()
        assert bias.abs().sum() == 0  # Bias should be initialized to zero

    # Test Linear layer
    def test_linear_layer(self, device):
        batch_size = 32
        in_features = 64
        out_features = 128
        
        # Test different initialization methods
        init_methods = ["default", "relu", "glorot", "gating", "normal", "zeros"]
        
        for init in init_methods:
            layer = Linear(in_features, out_features, init=init).to(device)
            x = torch.randn(batch_size, in_features).to(device)
            
            out = layer(x)
            assert out.shape == (batch_size, out_features)
            assert not torch.isnan(out).any()

        # Test invalid init method
        with pytest.raises(ValueError):
            Linear(in_features, out_features, init="invalid_init")

    # Test LayerNorm
    def test_layer_norm(self, device):
        batch_size = 32
        features = 64
        
        layer = LayerNorm(features).to(device)
        x = torch.randn(batch_size, features).to(device)
        
        out = layer(x)
        assert out.shape == (batch_size, features)
        assert not torch.isnan(out).any()
        
        # Test that bias is False by default
        assert layer.bias is None

    # Test SwiGLU
    def test_swiglu(self, device):
        batch_size = 32
        c_in = 64
        c_out = 128
        
        layer = SwiGLU(c_in, c_out).to(device)
        x = torch.randn(batch_size, c_in).to(device)
        
        out = layer(x)
        assert out.shape == (batch_size, c_out)
        assert not torch.isnan(out).any()

    # Test Attention
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
            use_efficient_implementation=use_efficient_implementation
        ).to(device)
        
        q_x = torch.randn(batch_size, seq_len, c_q).to(device)
        kv_x = torch.randn(batch_size, seq_len, c_k).to(device)
        
        # Test without attention bias
        out = attn(q_x, kv_x)
        assert out.shape == (batch_size, seq_len, c_q)
        assert not torch.isnan(out).any()
        
        # Test with attention bias
        attn_bias = torch.randn(batch_size, seq_len, seq_len).to(device)
        out = attn(q_x, kv_x, attn_bias=attn_bias)
        assert out.shape == (batch_size, seq_len, c_q)
        assert not torch.isnan(out).any()

    # Test rearrange_qk_to_dense_trunk
    def test_rearrange_qk_to_dense_trunk(self):
        batch_size = 2
        seq_len = 16
        hidden_dim = 32
        n_queries = 4
        n_keys = 8
        
        q = torch.randn(batch_size, seq_len, hidden_dim)
        k = torch.randn(batch_size, seq_len, hidden_dim)
        
        q_trunked, k_trunked, padding_info = rearrange_qk_to_dense_trunk(
            q=q,
            k=k,
            dim_q=-2,
            dim_k=-2,
            n_queries=n_queries,
            n_keys=n_keys
        )
        
        # Check shapes
        n_trunks = (seq_len + n_queries - 1) // n_queries
        assert q_trunked.shape == (batch_size, n_trunks, n_queries, hidden_dim)
        assert k_trunked.shape == (batch_size, n_trunks, n_keys, hidden_dim)
        
        # Check padding info
        assert "mask_trunked" in padding_info
        assert "q_pad" in padding_info
        assert "k_pad_left" in padding_info
        assert "k_pad_right" in padding_info


if __name__ == "__main__":
    pytest.main([__file__])