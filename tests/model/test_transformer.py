import torch

from artios.model.modules.transformer import (
    DiffusionTransformer,
    DiTBlock,
    LocalAttnTransformer,
    LocalAttnTransformerBlock,
)


class TestDiTBlock:
    def test_forward_runs(self):
        BATCH_SIZE, N, c_token, num_heads = 3, 7, 128, 4

        block = DiTBlock(c_token=c_token, num_heads=num_heads)

        a = torch.randn(BATCH_SIZE, N, c_token)
        conditioning = {
            k: torch.randn(BATCH_SIZE, N, c_token)
            for k in (
                "scale_attn",
                "shift_attn",
                "gate_attn",
                "scale_mlp",
                "shift_mlp",
                "gate_mlp",
            )
        }
        pair_bias = torch.randn(BATCH_SIZE, num_heads, N, N)
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N, N)).bool()

        out = block(a=a, **conditioning, pair_bias=pair_bias, pair_mask=pair_mask)
        assert out.shape == (BATCH_SIZE, N, c_token)


class TestDiffusionTransformer:
    def test_z_broadcasts_across_batch(self):
        """``z`` with batch=1 must broadcast across an ``a`` with batch=B."""
        BATCH_SIZE, N = 3, 7
        c_token, c_s, c_z, num_heads, num_blocks = 128, 128, 128, 4, 2

        transformer = DiffusionTransformer(
            num_blocks=num_blocks,
            c_token=c_token,
            c_s=c_s,
            c_z=c_z,
            num_heads=num_heads,
        )

        a = torch.randn(BATCH_SIZE, N, c_token)
        s = torch.randn(BATCH_SIZE, N, c_s)
        z = torch.randn(1, N, N, c_z)  # broadcast across batch
        pair_mask = torch.randint(0, 2, (BATCH_SIZE, N, N)).bool()

        out = transformer(a, s, z, pair_mask)
        assert out.shape == (BATCH_SIZE, N, c_token)


class TestLocalAttnTransformerBlock:
    def test_forward_runs(self):
        BATCH_SIZE, c_atom, num_heads, num_queries, num_keys = 3, 256, 4, 16, 64
        N_atom = num_queries * 16
        N_subset = N_atom // num_queries

        block = LocalAttnTransformerBlock(
            c_atom=c_atom,
            num_heads=num_heads,
            num_queries=num_queries,
            num_keys=num_keys,
        )

        q_l = torch.randn(BATCH_SIZE, N_atom, c_atom)
        conditioning = {
            k: torch.randn(BATCH_SIZE, N_atom, c_atom)
            for k in (
                "scale_attn",
                "shift_attn",
                "gate_attn",
                "scale_mlp",
                "shift_mlp",
                "gate_mlp",
            )
        }
        trunked_pair_bias = torch.randn(
            BATCH_SIZE, N_subset, num_heads, num_queries, num_keys
        )
        trunked_pair_mask = torch.randint(
            0, 2, (BATCH_SIZE, N_subset, num_queries, num_keys)
        ).bool()

        out = block(
            q_l,
            **conditioning,
            trunked_pair_bias=trunked_pair_bias,
            trunked_pair_mask=trunked_pair_mask,
        )
        assert out.shape == (BATCH_SIZE, N_atom, c_atom)


class TestLocalAttnTransformer:
    def test_forward_runs(self):
        BATCH_SIZE = 2
        num_blocks, c_atom, c_cond, c_atompair = 2, 256, 128, 16
        num_heads, num_queries, num_keys = 4, 16, 64
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

        out = transformer(q=q, c=c, p=p, trunked_pair_mask=trunked_pair_mask)
        assert out.shape == (BATCH_SIZE, N_atom, c_atom)
