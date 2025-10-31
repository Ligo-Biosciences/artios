import os

import pytest
import torch

from protenix.model.modules.atom_attention import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    SidechainDecoder,
)
from protenix.model.modules.layout import (
    rearrange_to_key_subset,
    rearrange_to_query_subset,
)


class TestAtomAttentionEncoder:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def sample_data(self):
        test_data_path = os.path.join(
            os.path.dirname(__file__), "testdata", "sample_data_dict.pt"
        )
        sample_data = torch.load(test_data_path, weights_only=False)

        atom_mask = torch.randint(
            low=0, high=2, size=(sample_data["input_feature_dict"]["ref_mask"].shape)
        )
        sample_data["input_feature_dict"]["atom_mask"] = atom_mask
        return sample_data

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
        self, atom_attention_encoder, sample_data, device
    ):
        """Test that _per_atom_conditioning produces correct output shapes."""
        input_feature_dict = sample_data["input_feature_dict"]
        basic = sample_data["basic"]
        n_atoms = basic["N_atom"]

        # Move input data to device
        input_feature_dict = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in input_feature_dict.items()
        }

        # Call the method
        c_l, p_lm = atom_attention_encoder._per_atom_conditioning(input_feature_dict)

        # Check c_l (atom single conditioning) shape
        expected_c_l_shape = (
            *input_feature_dict["ref_pos"].shape[:-1],
            atom_attention_encoder.c_atom,
        )
        assert (
            c_l.shape == expected_c_l_shape
        ), f"Expected c_l shape {expected_c_l_shape}, got {c_l.shape}"

        # Check p_lm (atom pair conditioning) shape
        # Should be [..., n_blocks, n_queries, n_keys, c_atompair]
        n_blocks = (
            n_atoms + atom_attention_encoder.num_queries - 1
        ) // atom_attention_encoder.num_queries
        expected_p_lm_shape = (
            *input_feature_dict["ref_pos"].shape[:-2],
            n_blocks,
            atom_attention_encoder.num_queries,
            atom_attention_encoder.num_keys,
            atom_attention_encoder.c_atompair,
        )
        assert (
            p_lm.shape == expected_p_lm_shape
        ), f"Expected p_lm shape {expected_p_lm_shape}, got {p_lm.shape}"

    def test_per_atom_conditioning_no_nans(
        self, atom_attention_encoder, sample_data, device
    ):
        """Test that _per_atom_conditioning produces no NaN values."""
        input_feature_dict = sample_data["input_feature_dict"]

        # Move input data to device
        input_feature_dict = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in input_feature_dict.items()
        }

        # Call the method
        c_l, p_lm = atom_attention_encoder._per_atom_conditioning(input_feature_dict)

        # Check for NaN values
        assert not torch.isnan(c_l).any(), "c_l contains NaN values"
        assert not torch.isnan(p_lm).any(), "p_lm contains NaN values"

    def test_per_atom_conditioning_mask_application(
        self, atom_attention_encoder, sample_data, device
    ):
        """Test that mask is properly applied to c_l."""
        input_feature_dict = sample_data["input_feature_dict"]

        # Move input data to device
        input_feature_dict = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in input_feature_dict.items()
        }

        # Call the method
        c_l, _ = atom_attention_encoder._per_atom_conditioning(input_feature_dict)

        # Check that masked positions are zero
        ref_mask = input_feature_dict["ref_mask"].unsqueeze(-1)
        masked_positions = ~ref_mask.bool()

        if masked_positions.any():
            # Check that masked positions in c_l are zero
            assert torch.allclose(
                c_l[masked_positions.expand_as(c_l)],
                torch.zeros_like(c_l[masked_positions.expand_as(c_l)]),
            ), "Masked positions in c_l should be zero"

    def test_atom_attention_encoder_forward_pass(
        self, atom_attention_encoder, sample_data, device
    ):
        """Test the complete forward pass of AtomAttentionEncoder."""
        input_feature_dict = sample_data["input_feature_dict"]
        basic = sample_data["basic"]
        n_atoms = basic["N_atom"]
        n_tokens = basic["N_token"]
        n_samples = 2

        # Move input data to device
        input_feature_dict = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in input_feature_dict.items()
        }

        # Create optional inputs for testing different branches
        # Test with r_l (noisy coordinates)
        r_l = torch.randn(n_samples, n_atoms, 3, device=device)

        # Test with s (single embedding)
        s = torch.randn(n_samples, n_tokens, atom_attention_encoder.c_s, device=device)

        # Test with z (pair embedding)
        z = torch.randn(
            1, n_tokens, n_tokens, atom_attention_encoder.c_z, device=device
        )

        # Test full forward pass
        a, q_l, c_l, p_lm = atom_attention_encoder(
            input_feature_dict=input_feature_dict, r_l=r_l, s=s, z=z
        )

        # Check output shapes
        assert (
            a.shape == (n_samples, n_tokens, atom_attention_encoder.c_token)
        ), f"Expected a shape ({n_samples}, {n_tokens}, {atom_attention_encoder.c_token}), got {a.shape}"

        assert (
            q_l.shape == (n_samples, n_atoms, atom_attention_encoder.c_atom)
        ), f"Expected q_l shape ({n_samples}, {n_atoms}, {atom_attention_encoder.c_atom}), got {q_l.shape}"

        assert (
            c_l.shape == (n_samples, n_atoms, atom_attention_encoder.c_atom)
        ), f"Expected c_l shape ({n_samples}, {n_atoms}, {atom_attention_encoder.c_atom}), got {c_l.shape}"

        # Check p_lm shape (local attention format)
        n_blocks = (
            n_atoms + atom_attention_encoder.num_queries - 1
        ) // atom_attention_encoder.num_queries
        expected_p_lm_shape = (
            n_samples,
            n_blocks,
            atom_attention_encoder.num_queries,
            atom_attention_encoder.num_keys,
            atom_attention_encoder.c_atompair,
        )
        assert (
            p_lm.shape == expected_p_lm_shape
        ), f"Expected p_lm shape {expected_p_lm_shape}, got {p_lm.shape}"

        # Check for NaN/Inf values
        assert torch.isfinite(a).all(), "a contains non-finite values"
        assert torch.isfinite(q_l).all(), "q_l contains non-finite values"
        assert torch.isfinite(c_l).all(), "c_l contains non-finite values"
        assert torch.isfinite(p_lm).all(), "p_lm contains non-finite values"

    def test_atom_attention_encoder_without_optional_inputs(self, sample_data, device):
        """Test forward pass without optional inputs (r_l, s, z)."""
        input_feature_dict = sample_data["input_feature_dict"]
        basic = sample_data["basic"]
        n_atoms = basic["N_atom"]
        n_tokens = basic["N_token"]

        # Move input data to device
        input_feature_dict = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in input_feature_dict.items()
        }
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

        # Test forward pass without optional inputs
        a, q_l, c_l, p_lm = atom_attention_encoder(
            input_feature_dict=input_feature_dict, r_l=None, s=None, z=None
        )

        # Check that outputs are still valid
        assert (
            a.shape == (1, n_tokens, atom_attention_encoder.c_token)
        ), f"Expected a shape (..., {n_tokens}, {atom_attention_encoder.c_token}), got {a.shape}"

        assert (
            q_l.shape == (1, n_atoms, atom_attention_encoder.c_atom)
        ), f"Expected q_l shape (..., {n_atoms}, {atom_attention_encoder.c_atom}), got {q_l.shape}"

        assert (
            c_l.shape == (1, n_atoms, atom_attention_encoder.c_atom)
        ), f"Expected c_l shape (..., {n_atoms}, {atom_attention_encoder.c_atom}), got {c_l.shape}"

        # Check for finite values
        assert torch.isfinite(a).all(), "a contains non-finite values"
        assert torch.isfinite(q_l).all(), "q_l contains non-finite values"
        assert torch.isfinite(c_l).all(), "c_l contains non-finite values"
        assert torch.isfinite(p_lm).all(), "p_lm contains non-finite values"


class TestAtomAttentionDecoder:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def sample_data(self):
        test_data_path = os.path.join(
            os.path.dirname(__file__), "testdata", "sample_data_dict.pt"
        )
        sample_data = torch.load(test_data_path, weights_only=False)

        # Add atom_mask if not present
        atom_mask = torch.randint(
            low=0, high=2, size=(sample_data["input_feature_dict"]["ref_mask"].shape)
        )
        sample_data["input_feature_dict"]["atom_mask"] = atom_mask
        return sample_data

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
        queries_mask = rearrange_to_query_subset(
            atom_mask, rearrange_dim=-1, num_queries=num_queries
        )  # [..., n_blocks, n_queries]
        keys_mask = rearrange_to_key_subset(
            atom_mask,
            rearrange_dim=-1,
            num_keys=num_keys,
            num_queries=num_queries,
        )  # [..., n_blocks, n_keys]
        trunked_pair_mask = queries_mask[..., :, None] * keys_mask[..., None, :]
        return trunked_pair_mask

    def test_atom_attention_decoder_forward_pass(
        self, atom_attention_decoder, sample_data, device
    ):
        """Test the complete forward pass of AtomAttentionDecoder."""
        basic = sample_data["basic"]
        n_atoms = basic["N_atom"]
        n_tokens = basic["N_token"]
        N_sample = 2

        # Create input tensors
        a = torch.randn(
            N_sample, n_tokens, atom_attention_decoder.c_token, device=device
        )
        q_skip = torch.randn(
            N_sample, n_atoms, atom_attention_decoder.c_atom, device=device
        )
        c_skip = torch.randn(1, n_atoms, atom_attention_decoder.c_atom, device=device)

        # Calculate n_blocks for p_skip shape
        n_blocks = (
            n_atoms + atom_attention_decoder.num_queries - 1
        ) // atom_attention_decoder.num_queries
        p_skip = torch.randn(
            1,
            n_blocks,
            atom_attention_decoder.num_queries,
            atom_attention_decoder.num_keys,
            atom_attention_decoder.c_atompair,
            device=device,
        )

        # Create atom_to_token_idx and atom_mask
        atom_to_token_idx = torch.randint(0, n_tokens, (n_atoms,), device=device)
        atom_mask = torch.ones(n_atoms, device=device)

        # Generate trunked_pair_mask using the same pattern as encoder
        trunked_pair_mask = self._generate_trunked_pair_mask(
            atom_mask,
            atom_attention_decoder.num_queries,
            atom_attention_decoder.num_keys,
        )

        # Forward pass
        r_update = atom_attention_decoder(
            a=a,
            q_skip=q_skip,
            c_skip=c_skip,
            p_skip=p_skip,
            atom_to_token_idx=atom_to_token_idx,
            atom_mask=atom_mask,
            trunked_pair_mask=trunked_pair_mask,
        )

        # Check output shape
        expected_shape = (N_sample, n_atoms, 3)
        assert (
            r_update.shape == expected_shape
        ), f"Expected r_update shape {expected_shape}, got {r_update.shape}"

        # Check for finite values
        assert torch.isfinite(r_update).all(), "r_update contains non-finite values"

    def test_atom_attention_decoder_with_batch_dims(
        self, atom_attention_decoder, sample_data, device
    ):
        """Test AtomAttentionDecoder with batch dimensions."""
        basic = sample_data["basic"]
        n_atoms = basic["N_atom"]
        n_tokens = basic["N_token"]
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
        q_skip = torch.randn(
            batch_size, n_sample, n_atoms, atom_attention_decoder.c_atom, device=device
        )
        c_skip = torch.randn(
            batch_size, n_sample, n_atoms, atom_attention_decoder.c_atom, device=device
        )

        # Calculate n_blocks for p_skip shape
        n_blocks = (
            n_atoms + atom_attention_decoder.num_queries - 1
        ) // atom_attention_decoder.num_queries
        p_skip = torch.randn(
            batch_size,
            1,
            n_blocks,
            atom_attention_decoder.num_queries,
            atom_attention_decoder.num_keys,
            atom_attention_decoder.c_atompair,
            device=device,
        )

        # Create batched atom_to_token_idx and atom_mask
        atom_to_token_idx = torch.randint(
            0, n_tokens, (batch_size, n_atoms), device=device
        )
        atom_mask = torch.ones(batch_size, n_atoms, device=device)

        # Generate batched trunked_pair_mask
        trunked_pair_mask = self._generate_trunked_pair_mask(
            atom_mask,
            atom_attention_decoder.num_queries,
            atom_attention_decoder.num_keys,
        )

        # Forward pass
        r_update = atom_attention_decoder(
            a=a,
            q_skip=q_skip,
            c_skip=c_skip,
            p_skip=p_skip,
            atom_to_token_idx=atom_to_token_idx,
            atom_mask=atom_mask,
            trunked_pair_mask=trunked_pair_mask,
        )

        # Check output shape
        expected_shape = (batch_size, n_sample, n_atoms, 3)
        assert (
            r_update.shape == expected_shape
        ), f"Expected r_update shape {expected_shape}, got {r_update.shape}"

        # Check for finite values
        assert torch.isfinite(r_update).all(), "r_update contains non-finite values"


class TestSidechainDecoder:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def sample_data(self):
        test_data_path = os.path.join(
            os.path.dirname(__file__), "testdata", "sample_data_dict.pt"
        )
        sample_data = torch.load(test_data_path, weights_only=False)

        # Add atom_mask if not present
        atom_mask = torch.randint(
            low=0, high=2, size=(sample_data["input_feature_dict"]["ref_mask"].shape)
        )
        sample_data["input_feature_dict"]["atom_mask"] = atom_mask
        return sample_data

    @pytest.fixture
    def sidechain_decoder(self, device):
        """Create SidechainDecoder with default parameters."""
        decoder = SidechainDecoder(
            node_dim=384,
            c_atom=128,
            c_atompair=16,
            num_blocks=2,
            num_heads=4,
            num_queries=32,
            num_keys=128,
        )
        return decoder.to(device)

    def test_sidechain_decoder_forward_pass(
        self, sidechain_decoder, sample_data, device
    ):
        """Test the complete forward pass of SidechainDecoder.

        Note: This test patches a bug in SidechainDecoder where z is not
        unsqueezed before passing to AtomAttentionEncoder.
        """
        input_feature_dict = sample_data["input_feature_dict"]
        basic = sample_data["basic"]
        n_atoms = basic["N_atom"]
        n_tokens = basic["N_token"]
        n_samples = 2

        # Move input data to device
        input_feature_dict = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in input_feature_dict.items()
        }

        # Create input tensors
        r_l = torch.randn(n_samples, n_atoms, 3, device=device)
        node_emb = torch.randn(n_tokens, 384, device=device)

        # Forward pass
        denoised_atoms = sidechain_decoder(
            input_feature_dict=input_feature_dict,
            r_l=r_l,
            node_emb=node_emb,
        )

        # Check output shape
        expected_shape = (n_samples, n_atoms, 3)
        assert (
            denoised_atoms.shape == expected_shape
        ), f"Expected denoised_atoms shape {expected_shape}, got {denoised_atoms.shape}"

        # Check for finite values
        assert torch.isfinite(
            denoised_atoms
        ).all(), "denoised_atoms contains non-finite values"


if __name__ == "__main__":
    pytest.main([__file__])
