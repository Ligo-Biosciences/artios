import pytest
import torch

from protenix.data.features import FullContext
from protenix.model.modules.atom_attention import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    AtomAttentionSkipConnections,
)
from protenix.model.modules.layout import (
    rearrange_to_key_subset,
    rearrange_to_query_subset,
)
from tests.context_utils import load_sample_context


class TestAtomAttentionEncoder:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def context(self, device) -> FullContext:
        """Load sample data as FullContext."""
        return load_sample_context(device=device)

    @pytest.fixture
    def atom_attention_encoder(self, device):
        """Create AtomAttentionEncoder with default parameters."""
        encoder = AtomAttentionEncoder(
            c_token=384,
            has_coords=True,
            has_s=True,
            has_z=True,
            c_atom=128,
            c_atompair=16,
            c_s=384,
            c_z=128,
            num_blocks=3,
            num_heads=4,
            num_queries=32,
            num_keys=128,
        )
        return encoder.to(device)

    def test_per_atom_conditioning_shapes(
        self, atom_attention_encoder, context, device
    ):
        """Test that _per_atom_conditioning produces correct output shapes."""
        n_atoms = context.basic.n_atom

        # Call the method with ReferenceConformerContext
        c_l, p_lm = atom_attention_encoder._per_atom_conditioning(context.reference)

        # Check c_l (atom single conditioning) shape
        expected_c_l_shape = (n_atoms, atom_attention_encoder.c_atom)
        assert (
            c_l.shape == expected_c_l_shape
        ), f"Expected c_l shape {expected_c_l_shape}, got {c_l.shape}"

        # Check p_lm (atom pair conditioning) shape
        # Should be [..., n_blocks, n_queries, n_keys, c_atompair]
        n_blocks = (
            n_atoms + atom_attention_encoder.num_queries - 1
        ) // atom_attention_encoder.num_queries
        expected_p_lm_shape = (
            n_blocks,
            atom_attention_encoder.num_queries,
            atom_attention_encoder.num_keys,
            atom_attention_encoder.c_atompair,
        )
        assert (
            p_lm.shape == expected_p_lm_shape
        ), f"Expected p_lm shape {expected_p_lm_shape}, got {p_lm.shape}"

    def test_per_atom_conditioning_no_nans(
        self, atom_attention_encoder, context, device
    ):
        """Test that _per_atom_conditioning produces no NaN values."""
        # Call the method with ReferenceConformerContext
        c_l, p_lm = atom_attention_encoder._per_atom_conditioning(context.reference)

        # Check for NaN values
        assert not torch.isnan(c_l).any(), "c_l contains NaN values"
        assert not torch.isnan(p_lm).any(), "p_lm contains NaN values"

    def test_per_atom_conditioning_mask_application(
        self, atom_attention_encoder, context, device
    ):
        """Test that mask is properly applied to c_l."""
        # Call the method with ReferenceConformerContext
        c_l, _ = atom_attention_encoder._per_atom_conditioning(context.reference)

        # Check that masked positions are zero
        ref_mask = context.reference.ref_mask.unsqueeze(-1)
        masked_positions = ~ref_mask.bool()

        if masked_positions.any():
            # Check that masked positions in c_l are zero
            assert torch.allclose(
                c_l[masked_positions.expand_as(c_l)],
                torch.zeros_like(c_l[masked_positions.expand_as(c_l)]),
            ), "Masked positions in c_l should be zero"

    def test_atom_attention_encoder_forward_pass(
        self, atom_attention_encoder, context, device
    ):
        """Test the complete forward pass of AtomAttentionEncoder."""
        n_atoms = context.basic.n_atom
        n_tokens = context.basic.n_token
        n_samples = 2

        # Create optional inputs for testing different branches
        r_l = torch.randn(n_samples, n_atoms, 3, device=device)
        s = torch.randn(n_samples, n_tokens, atom_attention_encoder.c_s, device=device)
        z = torch.randn(
            1, n_tokens, n_tokens, atom_attention_encoder.c_z, device=device
        )

        # Test full forward pass with FullContext
        a, skip = atom_attention_encoder(context=context, r_l=r_l, s=s, z=z)

        # Check output shapes
        assert a.shape == (n_samples, n_tokens, atom_attention_encoder.c_token)
        assert skip.q.shape == (n_samples, n_atoms, atom_attention_encoder.c_atom)
        assert skip.c.shape == (n_samples, n_atoms, atom_attention_encoder.c_atom)

        # Check skip.p shape (local attention format)
        n_blocks = (
            n_atoms + atom_attention_encoder.num_queries - 1
        ) // atom_attention_encoder.num_queries
        expected_p_shape = (
            1,
            n_blocks,
            atom_attention_encoder.num_queries,
            atom_attention_encoder.num_keys,
            atom_attention_encoder.c_atompair,
        )
        assert skip.p.shape == expected_p_shape

        # Check for NaN/Inf values
        assert torch.isfinite(a).all(), "a contains non-finite values"
        assert torch.isfinite(skip.q).all(), "skip.q contains non-finite values"
        assert torch.isfinite(skip.c).all(), "skip.c contains non-finite values"
        assert torch.isfinite(skip.p).all(), "skip.p contains non-finite values"

    def test_atom_attention_encoder_without_optional_inputs(self, context, device):
        """Test forward pass without optional inputs (r_l, s, z)."""
        n_atoms = context.basic.n_atom
        n_tokens = context.basic.n_token

        atom_attention_encoder = AtomAttentionEncoder(
            c_token=384,
            has_coords=False,
            has_s=False,
            has_z=False,
            c_atom=128,
            c_atompair=16,
            c_s=384,
            c_z=128,
            num_blocks=3,
            num_heads=4,
            num_queries=32,
            num_keys=128,
        )

        # Test forward pass with FullContext, without optional inputs
        a, skip = atom_attention_encoder(context=context)

        # Check output shapes
        assert a.shape == (1, n_tokens, atom_attention_encoder.c_token)
        assert skip.q.shape == (1, n_atoms, atom_attention_encoder.c_atom)
        assert skip.c.shape == (1, n_atoms, atom_attention_encoder.c_atom)

        # Check for finite values
        assert torch.isfinite(a).all(), "a contains non-finite values"
        assert torch.isfinite(skip.q).all(), "skip.q contains non-finite values"
        assert torch.isfinite(skip.c).all(), "skip.c contains non-finite values"
        assert torch.isfinite(skip.p).all(), "skip.p contains non-finite values"


class TestAtomAttentionDecoder:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def context(self, device) -> FullContext:
        """Load sample data as FullContext."""
        return load_sample_context(device=device)

    @pytest.fixture
    def atom_attention_decoder(self, device):
        """Create AtomAttentionDecoder with default parameters."""
        decoder = AtomAttentionDecoder(
            c_token=384,
            c_atom=128,
            c_atompair=16,
            num_blocks=3,
            num_heads=4,
            num_queries=32,
            num_keys=128,
        )
        return decoder.to(device)

    def _generate_trunked_pair_mask(self, atom_mask, num_queries, num_keys):
        """Generate trunked pair mask using the same pattern as AtomAttentionEncoder."""
        # Convert to bool if not already
        atom_mask_bool = (
            atom_mask.bool() if atom_mask.dtype != torch.bool else atom_mask
        )
        queries_mask = rearrange_to_query_subset(
            atom_mask_bool, rearrange_dim=-1, num_queries=num_queries
        )  # [..., n_blocks, n_queries]
        keys_mask = rearrange_to_key_subset(
            atom_mask_bool,
            rearrange_dim=-1,
            num_keys=num_keys,
            num_queries=num_queries,
        )  # [..., n_blocks, n_keys]
        trunked_pair_mask = queries_mask[..., :, None] & keys_mask[..., None, :]
        return trunked_pair_mask

    def test_atom_attention_decoder_forward_pass(
        self, atom_attention_decoder, context, device
    ):
        """Test the complete forward pass of AtomAttentionDecoder."""
        n_atoms = context.basic.n_atom
        n_tokens = context.basic.n_token
        N_sample = 2

        # Create input tensors
        a = torch.randn(
            N_sample, n_tokens, atom_attention_decoder.c_token, device=device
        )

        # Calculate n_blocks for skip tensors
        n_blocks = (
            n_atoms + atom_attention_decoder.num_queries - 1
        ) // atom_attention_decoder.num_queries

        # Create atom_to_token_idx and atom_mask
        atom_to_token_idx = context.index.atom_to_token_idx
        atom_mask = context.mask.atom_mask

        # Generate local_attn_mask using the same pattern as encoder
        local_attn_mask = self._generate_trunked_pair_mask(
            atom_mask,
            atom_attention_decoder.num_queries,
            atom_attention_decoder.num_keys,
        )

        # Create skip connections dataclass
        skip = AtomAttentionSkipConnections(
            q=torch.randn(
                N_sample, n_atoms, atom_attention_decoder.c_atom, device=device
            ),
            c=torch.randn(1, n_atoms, atom_attention_decoder.c_atom, device=device),
            p=torch.randn(
                1,
                n_blocks,
                atom_attention_decoder.num_queries,
                atom_attention_decoder.num_keys,
                atom_attention_decoder.c_atompair,
                device=device,
            ),
            local_attn_mask=local_attn_mask.unsqueeze(-4),
        )

        # Forward pass
        r_update = atom_attention_decoder(
            a=a,
            skip=skip,
            atom_to_token_idx=atom_to_token_idx,
            atom_mask=atom_mask,
        )

        # Check output shape
        expected_shape = (N_sample, n_atoms, 3)
        assert (
            r_update.shape == expected_shape
        ), f"Expected r_update shape {expected_shape}, got {r_update.shape}"

        # Check for finite values
        assert torch.isfinite(r_update).all(), "r_update contains non-finite values"

    def test_atom_attention_decoder_with_batch_dims(
        self, atom_attention_decoder, context, device
    ):
        """Test AtomAttentionDecoder with batch dimensions."""
        n_atoms = context.basic.n_atom
        n_tokens = context.basic.n_token
        n_sample = 1
        batch_size = 2

        # Create batched input tensors
        a = torch.randn(
            batch_size,
            n_sample,
            n_tokens,
            atom_attention_decoder.c_token,
            device=device,
        )

        # Calculate n_blocks for skip tensors
        n_blocks = (
            n_atoms + atom_attention_decoder.num_queries - 1
        ) // atom_attention_decoder.num_queries

        # Create batched atom_to_token_idx and atom_mask
        atom_to_token_idx = context.index.atom_to_token_idx.unsqueeze(0).expand(
            batch_size, -1
        )
        atom_mask = context.mask.atom_mask.unsqueeze(0).expand(batch_size, -1)

        # Generate batched local_attn_mask
        local_attn_mask = self._generate_trunked_pair_mask(
            atom_mask,
            atom_attention_decoder.num_queries,
            atom_attention_decoder.num_keys,
        )

        # Create batched skip connections dataclass
        skip = AtomAttentionSkipConnections(
            q=torch.randn(
                batch_size,
                n_sample,
                n_atoms,
                atom_attention_decoder.c_atom,
                device=device,
            ),
            c=torch.randn(
                batch_size,
                n_sample,
                n_atoms,
                atom_attention_decoder.c_atom,
                device=device,
            ),
            p=torch.randn(
                batch_size,
                1,
                n_blocks,
                atom_attention_decoder.num_queries,
                atom_attention_decoder.num_keys,
                atom_attention_decoder.c_atompair,
                device=device,
            ),
            local_attn_mask=local_attn_mask.unsqueeze(-4),
        )

        # Forward pass
        r_update = atom_attention_decoder(
            a=a,
            skip=skip,
            atom_to_token_idx=atom_to_token_idx,
            atom_mask=atom_mask,
        )

        # Check output shape
        expected_shape = (batch_size, n_sample, n_atoms, 3)
        assert (
            r_update.shape == expected_shape
        ), f"Expected r_update shape {expected_shape}, got {r_update.shape}"

        # Check for finite values
        assert torch.isfinite(r_update).all(), "r_update contains non-finite values"


if __name__ == "__main__":
    pytest.main([__file__])
