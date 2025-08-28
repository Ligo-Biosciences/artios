import pytest
import torch
import torch.nn.functional as F

from protenix.model.modules.components import (
    AdaptiveLayerNorm,
    Attention,
    SwiGLU,
    Transition,
)


class TestSwiGLU:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_swiglu_forward(self, device):
        """Test SwiGLU forward pass with basic functionality."""
        c_in, c_out = 64, 128
        batch_size = 32

        layer = SwiGLU(c_in, c_out).to(device)
        x = torch.randn(batch_size, c_in, device=device)

        output = layer(x)

        assert output.shape == (batch_size, c_out)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_swiglu_initialization(self, device):
        """Test SwiGLU layer initialization."""
        c_in, c_out = 32, 64
        layer = SwiGLU(c_in, c_out).to(device)

        assert layer.c_in == c_in
        assert layer.c_out == c_out
        assert layer.linear.in_features == c_in
        assert layer.linear.out_features == c_out * 2

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

    def test_swiglu_zero_input(self, device):
        """Test SwiGLU behavior with zero input."""
        c_in, c_out = 32, 64
        layer = SwiGLU(c_in, c_out).to(device)

        x = torch.zeros(8, c_in, device=device)
        output = layer(x)

        assert output.shape == (8, c_out)
        assert not torch.isnan(output).any()


class TestAdaptiveLayerNorm:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_adaptive_layer_norm_forward(self, device):
        """Test AdaptiveLayerNorm forward pass."""
        c_a, c_s = 256, 128
        batch_size, seq_len = 4, 16

        layer = AdaptiveLayerNorm(c_a=c_a, c_s=c_s).to(device)
        a = torch.randn(batch_size, seq_len, c_a, device=device)
        s = torch.randn(batch_size, seq_len, c_s, device=device)

        output = layer(a, s)

        assert output.shape == (batch_size, seq_len, c_a)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_adaptive_layer_norm_initialization(self, device):
        """Test AdaptiveLayerNorm initialization."""
        c_a, c_s = 128, 64
        layer = AdaptiveLayerNorm(c_a=c_a, c_s=c_s).to(device)

        # Check layer norm has no affine parameters
        assert not layer.layernorm_a.elementwise_affine

        # Check linear layer dimensions
        assert layer.linear_s.in_features == c_s
        assert layer.linear_s.out_features == c_a
        assert layer.linear_nobias_s.in_features == c_s
        assert layer.linear_nobias_s.out_features == c_a

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

    def test_transition_forward(self, device):
        """Test Transition forward pass."""
        c_in, n = 128, 4
        batch_size = 16

        layer = Transition(c_in=c_in, n=n).to(device)
        x = torch.randn(batch_size, c_in, device=device)

        output = layer(x)

        assert output.shape == (batch_size, c_in)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_transition_initialization(self, device):
        """Test Transition layer initialization."""
        c_in, n = 64, 2
        layer = Transition(c_in=c_in, n=n, output_linear_init="zeros").to(device)

        assert layer.c_in == c_in
        assert layer.n == n
        assert layer.output_linear_init == "zeros"
        assert layer.swiglu.c_in == c_in
        assert layer.swiglu.c_out == n * c_in
        assert layer.linear_no_bias.in_features == n * c_in
        assert layer.linear_no_bias.out_features == c_in

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
        """Test Transition with different output initialization schemes."""
        c_in, n = 64, 2

        for init_scheme in ["zeros", "normal"]:
            layer = Transition(c_in=c_in, n=n, output_linear_init=init_scheme).to(
                device
            )
            x = torch.randn(4, c_in, device=device)
            output = layer(x)
            assert output.shape == (4, c_in)
            assert not torch.isnan(output).any()


class TestAttention:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_attention_forward(self, device):
        """Test Attention forward pass."""
        q_dim, kv_dim, head_dim, num_heads = 128, 128, 32, 4
        batch_size, seq_len = 2, 16

        layer = Attention(
            q_dim=q_dim, kv_dim=kv_dim, head_dim=head_dim, num_heads=num_heads
        ).to(device)

        q_x = torch.randn(batch_size, seq_len, q_dim, device=device)
        kv_x = torch.randn(batch_size, seq_len, kv_dim, device=device)
        pair_bias = torch.randn(batch_size, num_heads, seq_len, seq_len, device=device)
        pair_mask = torch.ones(batch_size, seq_len, seq_len, device=device).bool()

        output = layer(q_x, kv_x, pair_bias, pair_mask)

        assert output.shape == (batch_size, seq_len, q_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_attention_initialization(self, device):
        """Test Attention layer initialization."""
        q_dim, kv_dim, head_dim, num_heads = 64, 128, 16, 8

        layer = Attention(
            q_dim=q_dim,
            kv_dim=kv_dim,
            head_dim=head_dim,
            num_heads=num_heads,
            gating=True,
            final_linear_init="zeros",
        ).to(device)

        assert layer.q_dim == q_dim
        assert layer.kv_dim == kv_dim
        assert layer.head_dim == head_dim
        assert layer.num_heads == num_heads
        assert layer.hidden_dim == num_heads * head_dim
        assert layer.gating is True
        assert layer.final_linear_init == "zeros"

    def test_attention_without_gating(self, device):
        """Test Attention without gating mechanism."""
        q_dim, kv_dim, head_dim, num_heads = 96, 96, 24, 4
        batch_size, seq_len = 2, 12

        layer = Attention(
            q_dim=q_dim,
            kv_dim=kv_dim,
            head_dim=head_dim,
            num_heads=num_heads,
            gating=False,
        ).to(device)

        q_x = torch.randn(batch_size, seq_len, q_dim, device=device)
        kv_x = torch.randn(batch_size, seq_len, kv_dim, device=device)
        pair_bias = torch.randn(batch_size, num_heads, seq_len, seq_len, device=device)
        pair_mask = torch.ones(batch_size, seq_len, seq_len, device=device).bool()

        output = layer(q_x, kv_x, pair_bias, pair_mask)

        assert output.shape == (batch_size, seq_len, q_dim)
        assert not torch.isnan(output).any()

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

    def test_attention_masking(self, device):
        """Test Attention with different masking patterns."""
        q_dim, kv_dim, head_dim, num_heads = 64, 64, 16, 4
        batch_size, seq_len = 2, 8

        layer = Attention(
            q_dim=q_dim, kv_dim=kv_dim, head_dim=head_dim, num_heads=num_heads
        ).to(device)

        q_x = torch.randn(batch_size, seq_len, q_dim, device=device)
        kv_x = torch.randn(batch_size, seq_len, kv_dim, device=device)
        pair_bias = torch.randn(batch_size, num_heads, seq_len, seq_len, device=device)

        # Test causal mask
        causal_mask = torch.tril(
            torch.ones(batch_size, seq_len, seq_len, device=device)
        ).bool()
        output_causal = layer(q_x, kv_x, pair_bias, causal_mask)
        assert output_causal.shape == (batch_size, seq_len, q_dim)

        # Test random mask
        random_mask = torch.randint(
            0, 2, (batch_size, seq_len, seq_len), device=device
        ).bool()
        output_random = layer(q_x, kv_x, pair_bias, random_mask)
        assert output_random.shape == (batch_size, seq_len, q_dim)

    def test_attention_pair_bias_shape_assertion(self, device):
        """Test that pair_bias shape assertion works correctly."""
        q_dim, kv_dim, head_dim, num_heads = 64, 64, 16, 4
        batch_size, q_len, kv_len = 2, 8, 10

        layer = Attention(
            q_dim=q_dim, kv_dim=kv_dim, head_dim=head_dim, num_heads=num_heads
        ).to(device)

        q_x = torch.randn(batch_size, q_len, q_dim, device=device)
        kv_x = torch.randn(batch_size, kv_len, kv_dim, device=device)
        pair_mask = torch.ones(batch_size, q_len, kv_len, device=device).bool()

        # Correct pair_bias shape
        pair_bias_correct = torch.randn(
            batch_size, num_heads, q_len, kv_len, device=device
        )
        output = layer(q_x, kv_x, pair_bias_correct, pair_mask)
        assert output.shape == (batch_size, q_len, q_dim)

        # Incorrect pair_bias shape should raise assertion error
        pair_bias_wrong = torch.randn(
            batch_size, num_heads, q_len + 1, kv_len, device=device
        )
        with pytest.raises(AssertionError):
            layer(q_x, kv_x, pair_bias_wrong, pair_mask)
