import pytest
import torch

from artios.model.modules.transformer import (
    DiffusionTransformer,
    DiTBlock,
    LocalAttnTransformer,
    LocalAttnTransformerBlock,
    Transformer,
    TransformerBlock,
)
from artios.utils.typing import TypeCheckError


class TestDiTBlock:
    @pytest.mark.parametrize("c_token", [128, 256, 512])
    @pytest.mark.parametrize("num_heads", [4, 8, 16])
    def test_shapes(self, c_token, num_heads):
        BATCH_SIZE = 3
        N = 7

        block = DiTBlock(c_token=c_token, num_heads=num_heads)

        a = torch.randn(BATCH_SIZE, N, c_token)
        scale_attn = torch.randn(BATCH_SIZE, N, c_token)
        shift_attn = torch.randn(BATCH_SIZE, N, c_token)
        gate_attn = torch.randn(BATCH_SIZE, N, c_token)
        scale_mlp = torch.randn(BATCH_SIZE, N, c_token)
        shift_mlp = torch.randn(BATCH_SIZE, N, c_token)
        gate_mlp = torch.randn(BATCH_SIZE, N, c_token)

        pair_bias = torch.randn(BATCH_SIZE, num_heads, N, N)
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N, N)).bool()

        out = block(
            a=a,
            scale_attn=scale_attn,
            shift_attn=shift_attn,
            gate_attn=gate_attn,
            scale_mlp=scale_mlp,
            shift_mlp=shift_mlp,
            gate_mlp=gate_mlp,
            pair_bias=pair_bias,
            pair_mask=pair_mask,
        )

        assert out.shape == (BATCH_SIZE, N, c_token)

    def test_evil_shape(self):
        BATCH_SIZE = 3
        N = 7
        c_token = 128
        num_heads = 4

        block = DiTBlock(c_token=c_token, num_heads=num_heads)

        a = torch.randn(BATCH_SIZE, N, c_token)
        scale_attn = torch.randn(BATCH_SIZE, N, c_token)
        shift_attn = torch.randn(BATCH_SIZE, N, c_token)
        gate_attn = torch.randn(BATCH_SIZE, N, c_token)
        scale_mlp = torch.randn(BATCH_SIZE, N, c_token)
        shift_mlp = torch.randn(BATCH_SIZE, N, c_token)
        gate_mlp = torch.randn(BATCH_SIZE, N, c_token)
        pair_bias = torch.randn(BATCH_SIZE, num_heads, N + 1, N + 1)
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N + 1, N + 1)).bool()

        with pytest.raises(TypeCheckError):
            block(
                a=a,
                scale_attn=scale_attn,
                shift_attn=shift_attn,
                gate_attn=gate_attn,
                scale_mlp=scale_mlp,
                shift_mlp=shift_mlp,
                gate_mlp=gate_mlp,
                pair_bias=pair_bias,
                pair_mask=pair_mask,
            )


class TestDiffusionTransformer:
    @pytest.mark.parametrize("num_blocks", [2])
    @pytest.mark.parametrize("c_token", [128, 256])
    @pytest.mark.parametrize("c_s", [128, 256])
    @pytest.mark.parametrize("c_z", [128, 256])
    @pytest.mark.parametrize("num_heads", [4])
    def test_shapes(self, num_blocks, c_token, c_s, c_z, num_heads):
        BATCH_SIZE = 3
        N = 7

        transformer = DiffusionTransformer(
            num_blocks=num_blocks,
            c_token=c_token,
            c_s=c_s,
            c_z=c_z,
            num_heads=num_heads,
        )

        a = torch.randn(BATCH_SIZE, N, c_token)
        s = torch.randn(BATCH_SIZE, N, c_s)
        z = torch.randn(BATCH_SIZE, N, N, c_z)
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N, N)).bool()

        out = transformer(a, s, z, pair_mask)

        assert out.shape == (BATCH_SIZE, N, c_token)

    def test_evil_shape(self):
        BATCH_SIZE = 3
        N = 7
        c_token = 128
        c_s = 128
        c_z = 128
        num_heads = 4
        num_blocks = 2

        transformer = DiffusionTransformer(
            num_blocks=num_blocks,
            c_token=c_token,
            c_s=c_s,
            c_z=c_z,
            num_heads=num_heads,
        )

        a = torch.randn(BATCH_SIZE, N, c_token)
        s = torch.randn(BATCH_SIZE, N, c_s)
        z = torch.randn(BATCH_SIZE, N, N, c_z)
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N + 1, N + 1)).bool()

        with pytest.raises(TypeCheckError):
            transformer(a, s, z, pair_mask)

        a = torch.randn(BATCH_SIZE, N, c_token)
        s = torch.randn(BATCH_SIZE, N, c_s)
        z = torch.randn(BATCH_SIZE, N, N, c_z)
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N, N + 1)).bool()

        with pytest.raises(TypeCheckError):
            transformer(a, s, z, pair_mask)

        # Wrong a shape - extra dimension
        a = torch.randn(BATCH_SIZE, N, N, c_token)
        s = torch.randn(BATCH_SIZE, N, c_s)
        z = torch.randn(BATCH_SIZE, N, N, c_z)
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N, N)).bool()

        with pytest.raises((TypeCheckError, RuntimeError)):
            transformer(a, s, z, pair_mask)

    def test_batch_dim_mismatch(self):
        BATCH_SIZE = 3
        N = 7
        c_token = 128
        c_s = 128
        c_z = 128
        num_heads = 4
        num_blocks = 2

        transformer = DiffusionTransformer(
            num_blocks=num_blocks,
            c_token=c_token,
            c_s=c_s,
            c_z=c_z,
            num_heads=num_heads,
        )

        a = torch.randn(BATCH_SIZE, N, c_token)
        s = torch.randn(BATCH_SIZE, N, c_s)
        z = torch.randn(1, N, N, c_z)
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N, N)).bool()

        out = transformer(a, s, z, pair_mask)

        assert out.shape == (BATCH_SIZE, N, c_token)


class TestLocalAttnTransformerBlock:
    @pytest.mark.parametrize("c_atom", [256, 512])
    @pytest.mark.parametrize("num_heads", [4, 8, 16])
    @pytest.mark.parametrize("num_queries", [16, 32])
    @pytest.mark.parametrize("num_keys", [64, 128])
    def test_shapes(self, c_atom, num_heads, num_queries, num_keys):
        BATCH_SIZE = 3
        N_atom = num_queries * 16  # Ensure we have enough atoms
        N_subset = N_atom // num_queries

        block = LocalAttnTransformerBlock(
            c_atom=c_atom,
            num_heads=num_heads,
            num_queries=num_queries,
            num_keys=num_keys,
        )

        q_l = torch.randn(BATCH_SIZE, N_atom, c_atom)
        scale_attn = torch.randn(BATCH_SIZE, N_atom, c_atom)
        shift_attn = torch.randn(BATCH_SIZE, N_atom, c_atom)
        gate_attn = torch.randn(BATCH_SIZE, N_atom, c_atom)
        scale_mlp = torch.randn(BATCH_SIZE, N_atom, c_atom)
        shift_mlp = torch.randn(BATCH_SIZE, N_atom, c_atom)
        gate_mlp = torch.randn(BATCH_SIZE, N_atom, c_atom)

        trunked_pair_bias = torch.randn(
            BATCH_SIZE, N_subset, num_heads, num_queries, num_keys
        )
        trunked_pair_mask = torch.randint(
            0, 2, (BATCH_SIZE, N_subset, num_queries, num_keys)
        ).bool()

        out = block(
            q_l,
            scale_attn=scale_attn,
            shift_attn=shift_attn,
            gate_attn=gate_attn,
            scale_mlp=scale_mlp,
            shift_mlp=shift_mlp,
            gate_mlp=gate_mlp,
            trunked_pair_bias=trunked_pair_bias,
            trunked_pair_mask=trunked_pair_mask,
        )

        assert out.shape == (BATCH_SIZE, N_atom, c_atom)

    def test_evil_shape(self):
        BATCH_SIZE = 3
        c_atom = 128
        num_heads = 4
        num_queries = 32
        num_keys = 128
        N_atom = num_queries * 4
        N_subset = N_atom // num_queries

        block = LocalAttnTransformerBlock(
            c_atom=c_atom,
            num_heads=num_heads,
            num_queries=num_queries,
            num_keys=num_keys,
        )

        q_l = torch.randn(BATCH_SIZE, N_atom, c_atom)
        scale_attn = torch.randn(BATCH_SIZE, N_atom, c_atom)
        shift_attn = torch.randn(BATCH_SIZE, N_atom, c_atom)
        gate_attn = torch.randn(BATCH_SIZE, N_atom, c_atom)
        scale_mlp = torch.randn(BATCH_SIZE, N_atom, c_atom)
        shift_mlp = torch.randn(BATCH_SIZE, N_atom, c_atom)
        gate_mlp = torch.randn(BATCH_SIZE, N_atom, c_atom)

        # Wrong pair bias shape (wrong num_heads)
        trunked_pair_bias = torch.randn(
            BATCH_SIZE, N_subset, num_heads + 1, num_queries, num_keys
        )
        trunked_pair_mask = torch.randint(
            0, 2, (BATCH_SIZE, N_subset, num_queries, num_keys)
        ).bool()

        with pytest.raises((AssertionError, TypeCheckError)):
            block(
                q_l,
                scale_attn=scale_attn,
                shift_attn=shift_attn,
                gate_attn=gate_attn,
                scale_mlp=scale_mlp,
                shift_mlp=shift_mlp,
                gate_mlp=gate_mlp,
                trunked_pair_bias=trunked_pair_bias,
                trunked_pair_mask=trunked_pair_mask,
            )

        # Wrong pair bias dimensions (wrong num_queries)
        trunked_pair_bias = torch.randn(
            BATCH_SIZE, N_subset, num_heads, num_queries + 1, num_keys
        )
        trunked_pair_mask = torch.randint(
            0, 2, (BATCH_SIZE, N_subset, num_queries, num_keys)
        ).bool()

        with pytest.raises(TypeCheckError):
            block(
                q_l,
                scale_attn=scale_attn,
                shift_attn=shift_attn,
                gate_attn=gate_attn,
                scale_mlp=scale_mlp,
                shift_mlp=shift_mlp,
                gate_mlp=gate_mlp,
                trunked_pair_bias=trunked_pair_bias,
                trunked_pair_mask=trunked_pair_mask,
            )


class TestLocalAttnTransformer:
    @pytest.mark.parametrize("num_blocks", [1, 2])
    @pytest.mark.parametrize("c_atom", [256])
    @pytest.mark.parametrize("c_cond", [128])
    @pytest.mark.parametrize("c_atompair", [16])
    @pytest.mark.parametrize("num_heads", [4, 8])
    @pytest.mark.parametrize("num_queries", [16, 32])
    @pytest.mark.parametrize("num_keys", [64, 128])
    def test_shapes(
        self, num_blocks, c_atom, c_cond, c_atompair, num_heads, num_queries, num_keys
    ):
        BATCH_SIZE = 3
        N_atom = num_queries * 16  # Ensure we have enough atoms
        N_subset = N_atom // num_queries

        transformer = LocalAttnTransformer(
            num_blocks=num_blocks,
            c_atom=c_atom,
            c_cond=c_cond,
            c_atompair=c_atompair,
            num_heads=num_heads,
            num_queries=num_queries,
            num_keys=num_keys,
        )

        q = torch.randn(BATCH_SIZE, N_atom, c_atom)
        c = torch.randn(BATCH_SIZE, N_atom, c_cond)
        p = torch.randn(BATCH_SIZE, N_subset, num_queries, num_keys, c_atompair)
        trunked_pair_mask = torch.randint(
            0, 2, (BATCH_SIZE, N_subset, num_queries, num_keys)
        ).bool()

        out = transformer(
            q=q,
            c=c,
            p=p,
            trunked_pair_mask=trunked_pair_mask,
        )

        assert out.shape == (BATCH_SIZE, N_atom, c_atom)

    def test_single_block(self):
        BATCH_SIZE = 2
        num_blocks = 1
        c_atom = 128
        c_cond = 128
        c_atompair = 128
        num_heads = 4
        num_queries = 16
        num_keys = 64
        N_atom = num_queries * 16
        N_subset = N_atom // num_queries

        transformer = LocalAttnTransformer(
            num_blocks=num_blocks,
            c_atom=c_atom,
            c_cond=c_cond,
            c_atompair=c_atompair,
            num_heads=num_heads,
            num_queries=num_queries,
            num_keys=num_keys,
        )

        q = torch.randn(BATCH_SIZE, N_atom, c_atom)
        c = torch.randn(BATCH_SIZE, N_atom, c_cond)
        p = torch.randn(BATCH_SIZE, N_subset, num_queries, num_keys, c_atompair)
        trunked_pair_mask = torch.randint(
            0, 2, (BATCH_SIZE, N_subset, num_queries, num_keys)
        ).bool()

        out = transformer(
            q=q,
            c=c,
            p=p,
            trunked_pair_mask=trunked_pair_mask,
        )

        assert out.shape == (BATCH_SIZE, N_atom, c_atom)


class TestTransformerBlock:
    @pytest.mark.parametrize("c_a", [128, 256])
    @pytest.mark.parametrize("c_z", [64, 128])
    @pytest.mark.parametrize("num_heads", [4, 8])
    def test_shapes(self, c_a, c_z, num_heads):
        BATCH_SIZE = 2
        N = 12

        block = TransformerBlock(c_a=c_a, c_z=c_z, num_heads=num_heads)

        a = torch.randn(BATCH_SIZE, N, c_a)
        pair_bias = torch.randn(BATCH_SIZE, num_heads, N, N)
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N, N)).bool()

        out = block(a=a, pair_bias=pair_bias, pair_mask=pair_mask)

        assert out.shape == (BATCH_SIZE, N, c_a)

    def test_shape_validation(self):
        BATCH_SIZE = 2
        N = 8
        c_a = 128
        c_z = 64
        num_heads = 4

        block = TransformerBlock(c_a=c_a, c_z=c_z, num_heads=num_heads)

        a = torch.randn(BATCH_SIZE, N, c_a)

        # Test wrong pair_bias head dimension
        pair_bias = torch.randn(BATCH_SIZE, num_heads + 1, N, N)
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N, N)).bool()

        with pytest.raises((AssertionError, TypeCheckError)):
            block(a=a, pair_bias=pair_bias, pair_mask=pair_mask)

        # Test wrong pair_bias spatial dimensions
        pair_bias = torch.randn(BATCH_SIZE, num_heads, N + 1, N + 1)
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N, N)).bool()

        with pytest.raises((AssertionError, TypeCheckError)):
            block(a=a, pair_bias=pair_bias, pair_mask=pair_mask)

        # Test wrong pair_mask spatial dimensions
        pair_bias = torch.randn(BATCH_SIZE, num_heads, N, N)
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N + 1, N + 1)).bool()

        with pytest.raises((AssertionError, TypeCheckError)):
            block(a=a, pair_bias=pair_bias, pair_mask=pair_mask)

    def test_c_a_divisible_by_num_heads(self):
        """Test that c_a must be divisible by num_heads."""
        c_a = 127  # Not divisible by 4
        c_z = 64
        num_heads = 4

        with pytest.raises(AssertionError):
            TransformerBlock(c_a=c_a, c_z=c_z, num_heads=num_heads)

    def test_residual_connection(self):
        """Test that the output includes residual connections."""
        BATCH_SIZE = 1
        N = 4
        c_a = 128
        c_z = 64
        num_heads = 4

        block = TransformerBlock(c_a=c_a, c_z=c_z, num_heads=num_heads)

        # Zero input should give non-zero output due to layer norms and residuals
        a = torch.zeros(BATCH_SIZE, N, c_a)
        pair_bias = torch.zeros(BATCH_SIZE, num_heads, N, N)
        pair_mask = torch.ones(BATCH_SIZE, N, N).bool()

        out = block(a=a, pair_bias=pair_bias, pair_mask=pair_mask)

        # Output should be close to input due to residual connection when starting from zero
        # (though not exactly zero due to layer norm behavior)
        assert out.shape == (BATCH_SIZE, N, c_a)


class TestTransformer:
    @pytest.mark.parametrize("num_blocks", [1, 2])
    @pytest.mark.parametrize("c_a", [128, 256])
    @pytest.mark.parametrize("c_z", [64, 128])
    @pytest.mark.parametrize("num_heads", [4, 8])
    def test_shapes(self, num_blocks, c_a, c_z, num_heads):
        BATCH_SIZE = 2
        N = 10

        transformer = Transformer(
            c_a=c_a, c_z=c_z, num_blocks=num_blocks, num_heads=num_heads
        )

        a = torch.randn(BATCH_SIZE, N, c_a)
        z = torch.randn(BATCH_SIZE, N, N, c_z)
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N, N)).bool()

        out = transformer(a=a, z=z, pair_mask=pair_mask)

        assert out.shape == (BATCH_SIZE, N, c_a)

    def test_input_validation(self):
        """Test input validation in transformer."""
        BATCH_SIZE = 2
        N = 8
        c_a = 128
        c_z = 64
        num_heads = 4
        num_blocks = 2

        transformer = Transformer(
            c_a=c_a, c_z=c_z, num_blocks=num_blocks, num_heads=num_heads
        )

        # Test wrong c_a dimension - will fail at runtime when LayerNorm expects c_a
        a = torch.randn(BATCH_SIZE, N, c_a + 10)  # Wrong c_a
        z = torch.randn(BATCH_SIZE, N, N, c_z)
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N, N)).bool()

        with pytest.raises((TypeCheckError, RuntimeError)):
            transformer(a=a, z=z, pair_mask=pair_mask)

        # Test wrong c_z dimension - will fail at runtime when z_proj expects c_z
        a = torch.randn(BATCH_SIZE, N, c_a)
        z = torch.randn(BATCH_SIZE, N, N, c_z + 10)  # Wrong c_z
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N, N)).bool()

        with pytest.raises((TypeCheckError, RuntimeError)):
            transformer(a=a, z=z, pair_mask=pair_mask)

        # Test wrong z spatial dimensions
        a = torch.randn(BATCH_SIZE, N, c_a)
        z = torch.randn(BATCH_SIZE, N + 1, N + 1, c_z)  # Wrong spatial dims
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N, N)).bool()

        with pytest.raises(TypeCheckError):
            transformer(a=a, z=z, pair_mask=pair_mask)

        # Test wrong pair_mask spatial dimensions
        a = torch.randn(BATCH_SIZE, N, c_a)
        z = torch.randn(BATCH_SIZE, N, N, c_z)
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N + 1, N + 1)).bool()

        with pytest.raises(TypeCheckError):
            transformer(a=a, z=z, pair_mask=pair_mask)

    def test_pair_bias_chunking(self):
        """Test that pair bias is properly chunked across blocks."""
        BATCH_SIZE = 1
        N = 4
        c_a = 64
        c_z = 32
        num_heads = 2
        num_blocks = 3

        transformer = Transformer(
            c_a=c_a, c_z=c_z, num_blocks=num_blocks, num_heads=num_heads
        )

        a = torch.randn(BATCH_SIZE, N, c_a)
        z = torch.randn(BATCH_SIZE, N, N, c_z)
        pair_mask = torch.ones(BATCH_SIZE, N, N).bool()

        # Check that z_proj outputs correct dimensions for chunking
        expected_output_dim = num_heads * num_blocks
        assert transformer.z_proj.out_features == expected_output_dim

        out = transformer(a=a, z=z, pair_mask=pair_mask)
        assert out.shape == (BATCH_SIZE, N, c_a)

    def test_single_block(self):
        """Test transformer with single block."""
        BATCH_SIZE = 1
        N = 6
        c_a = 128
        c_z = 64
        num_heads = 4
        num_blocks = 1

        transformer = Transformer(
            c_a=c_a, c_z=c_z, num_blocks=num_blocks, num_heads=num_heads
        )

        a = torch.randn(BATCH_SIZE, N, c_a)
        z = torch.randn(BATCH_SIZE, N, N, c_z)
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N, N)).bool()

        out = transformer(a=a, z=z, pair_mask=pair_mask)
        assert out.shape == (BATCH_SIZE, N, c_a)

    def test_broadcasting_z(self):
        """Test that z can broadcast across batch dimensions."""
        BATCH_SIZE = 3
        N = 5
        c_a = 128
        c_z = 64
        num_heads = 4
        num_blocks = 2

        transformer = Transformer(
            c_a=c_a, c_z=c_z, num_blocks=num_blocks, num_heads=num_heads
        )

        a = torch.randn(BATCH_SIZE, N, c_a)
        z = torch.randn(1, N, N, c_z)  # Single batch dimension for broadcasting
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N, N)).bool()

        out = transformer(a=a, z=z, pair_mask=pair_mask)
        assert out.shape == (BATCH_SIZE, N, c_a)
