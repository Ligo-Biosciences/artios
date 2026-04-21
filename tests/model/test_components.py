import pytest
import torch
import torch.nn.functional as F

from artios.model.modules.components import (
    AdaptiveLayerNorm,
    Attention,
    SwiGLU,
    Transition,
)


class TestSwiGLU:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_swiglu_different_shapes(self, device):
        """Test SwiGLU with different input shapes."""
        c_in, c_out = 48, 96
        layer = SwiGLU(c_in, c_out).to(device)

        # Test 2D input
        x_2d = torch.randn(16, c_in, device=device)
        out_2d = layer(x_2d)
        assert out_2d.shape == (16, c_out)

        # Test 3D input
        x_3d = torch.randn(4, 8, c_in, device=device)
        out_3d = layer(x_3d)
        assert out_3d.shape == (4, 8, c_out)

        # Test 4D input
        x_4d = torch.randn(2, 4, 6, c_in, device=device)
        out_4d = layer(x_4d)
        assert out_4d.shape == (2, 4, 6, c_out)

    def test_swiglu_activation_function(self, device):
        """Test that SwiGLU applies SiLU activation correctly."""
        c_in, c_out = 16, 32
        layer = SwiGLU(c_in, c_out).to(device)

        # Test with known input
        x = torch.ones(1, c_in, device=device)
        output = layer(x)

        # Manual computation for verification
        linear_out = layer.linear(x)
        a, b = linear_out.chunk(2, dim=-1)
        expected = F.silu(a) * b

        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-7)


class TestAdaptiveLayerNorm:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_adaptive_layer_norm_different_shapes(self, device):
        """Test AdaptiveLayerNorm with different input shapes."""
        c_a, c_s = 64, 32
        layer = AdaptiveLayerNorm(c_a=c_a, c_s=c_s).to(device)

        # Test 2D inputs
        a_2d = torch.randn(8, c_a, device=device)
        s_2d = torch.randn(8, c_s, device=device)
        out_2d = layer(a_2d, s_2d)
        assert out_2d.shape == (8, c_a)

        # Test 3D inputs
        a_3d = torch.randn(4, 12, c_a, device=device)
        s_3d = torch.randn(4, 12, c_s, device=device)
        out_3d = layer(a_3d, s_3d)
        assert out_3d.shape == (4, 12, c_a)

    def test_adaptive_layer_norm_computation(self, device):
        """Test AdaptiveLayerNorm computation matches expected formula."""
        c_a, c_s = 32, 16
        layer = AdaptiveLayerNorm(c_a=c_a, c_s=c_s).to(device)

        a = torch.randn(2, 8, c_a, device=device)
        s = torch.randn(2, 8, c_s, device=device)

        output = layer(a, s)

        # Manual computation
        a_norm = layer.layernorm_a(a)
        gate = torch.sigmoid(layer.linear_s(s))
        shift = layer.linear_nobias_s(s)
        expected = gate * a_norm + shift

        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-7)


class TestTransition:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_transition_different_multipliers(self, device):
        """Test Transition with different multiplier values."""
        c_in = 96

        for n in [1, 2, 4, 8]:
            layer = Transition(c_in=c_in, n=n).to(device)
            x = torch.randn(8, c_in, device=device)
            output = layer(x)
            assert output.shape == (8, c_in)

    def test_transition_computation_flow(self, device):
        """Test Transition computation flow."""
        c_in, n = 32, 2
        layer = Transition(c_in=c_in, n=n).to(device)

        x = torch.randn(4, c_in, device=device)
        output = layer(x)

        # Manual computation
        x_norm = layer.layernorm(x)
        swiglu_out = layer.swiglu(x_norm)
        expected = layer.linear_no_bias(swiglu_out)

        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-7)

    def test_transition_different_shapes(self, device):
        """Test Transition with different input shapes."""
        c_in, n = 48, 3
        layer = Transition(c_in=c_in, n=n).to(device)

        # Test 3D input
        x_3d = torch.randn(2, 8, c_in, device=device)
        out_3d = layer(x_3d)
        assert out_3d.shape == (2, 8, c_in)

        # Test 4D input
        x_4d = torch.randn(2, 4, 6, c_in, device=device)
        out_4d = layer(x_4d)
        assert out_4d.shape == (2, 4, 6, c_in)

    def test_transition_output_init_schemes(self, device):
        """``output_linear_init='zeros'`` must zero the output linear so the
        whole block returns zeros pre-training; ``'normal'`` must not."""
        c_in, n = 64, 2
        x = torch.randn(4, c_in, device=device)

        zeros_layer = Transition(c_in=c_in, n=n, output_linear_init="zeros").to(device)
        zeros_out = zeros_layer(x)
        assert torch.equal(
            zeros_out, torch.zeros_like(zeros_out)
        ), "zeros init must produce an all-zero output"

        normal_layer = Transition(c_in=c_in, n=n, output_linear_init="normal").to(
            device
        )
        normal_out = normal_layer(x)
        assert normal_out.abs().sum() > 0, "normal init must produce a non-zero output"


class TestAttention:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_attention_gating_flag_controls_params(self, device):
        """``gating=True`` must add a gating projection — so the layer has
        strictly more parameters than ``gating=False``. Also confirms the
        non-gating forward pass still runs end-to-end."""
        kwargs = dict(q_dim=96, kv_dim=96, head_dim=24, num_heads=4)
        gated = Attention(**kwargs, gating=True).to(device)
        ungated = Attention(**kwargs, gating=False).to(device)

        gated_params = sum(p.numel() for p in gated.parameters())
        ungated_params = sum(p.numel() for p in ungated.parameters())
        assert gated_params > ungated_params, (
            f"gating=True should add parameters; "
            f"got {gated_params} vs {ungated_params}"
        )

        batch_size, seq_len = 2, 12
        q_x = torch.randn(batch_size, seq_len, kwargs["q_dim"], device=device)
        kv_x = torch.randn(batch_size, seq_len, kwargs["kv_dim"], device=device)
        pair_bias = torch.randn(
            batch_size, kwargs["num_heads"], seq_len, seq_len, device=device
        )
        pair_mask = torch.ones(batch_size, seq_len, seq_len, device=device).bool()
        output = ungated(q_x, kv_x, pair_bias, pair_mask)
        assert output.shape == (batch_size, seq_len, kwargs["q_dim"])

    def test_attention_different_q_kv_dims(self, device):
        """Test Attention with different query and key/value dimensions."""
        q_dim, kv_dim, head_dim, num_heads = 128, 256, 32, 4
        batch_size, q_len, kv_len = 2, 10, 14

        layer = Attention(
            q_dim=q_dim, kv_dim=kv_dim, head_dim=head_dim, num_heads=num_heads
        ).to(device)

        q_x = torch.randn(batch_size, q_len, q_dim, device=device)
        kv_x = torch.randn(batch_size, kv_len, kv_dim, device=device)
        pair_bias = torch.randn(batch_size, num_heads, q_len, kv_len, device=device)
        pair_mask = torch.ones(batch_size, q_len, kv_len, device=device).bool()

        output = layer(q_x, kv_x, pair_bias, pair_mask)

        assert output.shape == (batch_size, q_len, q_dim)
