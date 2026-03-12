"""Tests for the ArtiosPotts module.

Tests cover the forward pass, main_train_loop, compute_potts_single_logits,
compute_potts_joint_logits, and compute_loss for both pseudo and composite likelihood modes.
"""

import pytest
import torch
from ml_collections import ConfigDict

from artios.model.artios_potts import ArtiosPotts, PottsLossConfig
from artios.model.modules.potts import (
    compute_potts_joint_logits,
    compute_potts_single_logits,
)
from artios.model.output import PottsOutput
from artios.model.utils import Mode
from tests.context_utils import create_test_context

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_configs(
    loss_type: str = "pseudo",
    label_smoothing: float = 0.0,
    symmetric_J: bool = False,
    parameterisation: str = "linear",
) -> ConfigDict:
    """Create test configs for ArtiosPotts."""
    return ConfigDict(
        {
            "loss": {
                "potts": {
                    "loss_type": loss_type,
                    "label_smoothing": label_smoothing,
                },
            },
            "artios_mpnn": {
                "model": {
                    "num_residue_types": 32,
                    "node_dim": 64,
                    "edge_dim": 64,
                    "num_encoder_layers": 2,
                    "num_decoder_layers": 2,
                    "num_neighbours": 30,
                    "positional_encoding_dim": 16,
                    "positional_encoding_max_relative_feats": 32,
                    "num_rbf_kernels": 16,
                    "dropout": 0.0,
                    "coord_noise": 0.1,
                    "checkpoint_grads": False,
                },
                "potts": {
                    "symmetric_J": symmetric_J,
                    "parameterisation": parameterisation,
                    "sampling": {
                        "proposal": "gibbs",
                        "n_sweeps": 5,
                        "temperature": 0.1,
                        "temperature_init": 10.0,
                        "annealing_fraction": 0.5,
                        "dt": 0.5,
                        "rejection_step": True,
                        "complexity_penalty_weight": 0.0,
                        "complexity_window_size": 30,
                        "complexity_entropy_min": 2.32,
                    },
                },
            },
        }
    )


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def context(device):
    """Create test FullContext with valid one-hot restypes."""
    ctx = create_test_context(
        batch_size=2,
        n_tokens=32,
        n_atoms=64,
        device=device,
    )
    # Set valid one-hot restypes (create_test_context leaves them as zeros)
    B, N = 2, 32
    indices = torch.randint(0, 32, (B, N), device=device)
    restype = torch.zeros(B, N, 32, dtype=torch.long, device=device)
    restype.scatter_(-1, indices.unsqueeze(-1), 1)
    ctx.token.restype = restype
    return ctx


@pytest.fixture
def pseudo_configs():
    return _make_configs(loss_type="pseudo")


@pytest.fixture
def composite_configs():
    return _make_configs(loss_type="composite")


@pytest.fixture
def symmetric_configs():
    return _make_configs(loss_type="composite", symmetric_J=True)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestArtiosPottsInit:
    """Tests for ArtiosPotts initialization."""

    def test_model_initialization(self, pseudo_configs, device):
        model = ArtiosPotts(pseudo_configs).to(device)

        assert hasattr(model, "mpnn")
        assert hasattr(model, "potts")
        assert hasattr(model, "masking_sampler")
        assert hasattr(model, "loss_config")

    def test_loss_config_pseudo(self, pseudo_configs, device):
        model = ArtiosPotts(pseudo_configs).to(device)

        assert model.loss_config.loss_type == "pseudo"
        assert model.loss_config.label_smoothing == 0.0

    def test_loss_config_composite(self, composite_configs, device):
        model = ArtiosPotts(composite_configs).to(device)

        assert model.loss_config.loss_type == "composite"

    def test_symmetric_J_config(self, symmetric_configs, device):
        model = ArtiosPotts(symmetric_configs).to(device)

        assert model.potts.symmetric_J is True

    def test_potts_loss_config_frozen(self):
        cfg = PottsLossConfig(loss_type="pseudo", label_smoothing=0.1)
        with pytest.raises(AttributeError):
            cfg.loss_type = "composite"


# ---------------------------------------------------------------------------
# Forward / train loop tests
# ---------------------------------------------------------------------------


class TestArtiosPottsForward:
    """Tests for the forward pass and main_train_loop."""

    @pytest.fixture
    def model(self, pseudo_configs, device):
        return ArtiosPotts(pseudo_configs).to(device)

    def test_main_train_loop_output_type(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)

        assert isinstance(output, PottsOutput)

    def test_main_train_loop_output_shapes(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)

        B, N = 2, 32
        R = 32
        K = output.edge_index.shape[-1]

        assert output.h.shape == (B, N, R)
        assert output.J.shape == (B, N, K, R, R)
        assert output.edge_index.shape == (B, N, K)
        assert output.edge_mask.shape == (B, N, K)
        assert output.is_masked.shape == (B, N)
        assert output.timestep.shape == (B,)

    def test_main_train_loop_finite_values(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)

        assert torch.isfinite(output.h).all(), "h contains non-finite values"
        assert torch.isfinite(output.J).all(), "J contains non-finite values"

    def test_forward_train_mode(self, model, context):
        model.train()
        output = model.forward(context=context, mode=Mode.TRAIN, seed=42)

        assert isinstance(output, PottsOutput)

    def test_forward_eval_mode(self, model, context):
        model.eval()
        output = model.forward(context=context, mode=Mode.EVAL, seed=42)

        assert isinstance(output, PottsOutput)

    def test_forward_inference(self, model, context):
        model.eval()
        with torch.no_grad():
            output = model.forward(context=context, mode=Mode.INFERENCE, seed=42)
        assert isinstance(output, PottsOutput)
        assert output.sampled_sequence is not None
        assert output.energy is not None

    def test_train_eval_modes_equivalent(self, model, context):
        """Both Mode.TRAIN and Mode.EVAL route to main_train_loop."""
        model.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            out_train = model.forward(context=context, mode=Mode.TRAIN, seed=42)
            torch.manual_seed(0)
            out_eval = model.forward(context=context, mode=Mode.EVAL, seed=42)

        assert torch.allclose(out_train.h, out_eval.h)
        assert torch.allclose(out_train.J, out_eval.J)

    def test_reproducibility_with_seed(self, pseudo_configs, context, device):
        m1 = ArtiosPotts(pseudo_configs).to(device)
        m2 = ArtiosPotts(pseudo_configs).to(device)
        m2.load_state_dict(m1.state_dict())

        m1.eval()
        m2.eval()

        with torch.no_grad():
            torch.manual_seed(0)
            o1 = m1.main_train_loop(context=context, seed=42)
            torch.manual_seed(0)
            o2 = m2.main_train_loop(context=context, seed=42)

        assert torch.allclose(o1.h, o2.h)
        assert torch.allclose(o1.J, o2.J)


# ---------------------------------------------------------------------------
# get_logits tests
# ---------------------------------------------------------------------------


class TestComputePottsSingleLogits:
    """Tests for compute_potts_single_logits (pseudo-likelihood)."""

    @pytest.fixture
    def output_and_seq(self, pseudo_configs, context, device):
        model = ArtiosPotts(pseudo_configs).to(device)
        model.train()
        output = model.main_train_loop(context=context, seed=42)
        seq = context.token.restype.argmax(dim=-1)
        return output, seq

    def test_output_shape(self, output_and_seq):
        output, seq = output_and_seq
        logits = compute_potts_single_logits(
            seq, output.h, output.J, output.edge_index, output.edge_mask
        )

        B, N, R = 2, 32, 32
        assert logits.shape == (B, N, R)

    def test_output_finite(self, output_and_seq):
        output, seq = output_and_seq
        logits = compute_potts_single_logits(
            seq, output.h, output.J, output.edge_index, output.edge_mask
        )

        assert torch.isfinite(logits).all(), "Logits contain non-finite values"

    def test_logits_are_valid_distributions(self, output_and_seq):
        output, seq = output_and_seq
        logits = compute_potts_single_logits(
            seq, output.h, output.J, output.edge_index, output.edge_mask
        )

        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(
            probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-5
        )
        assert (probs >= 0).all()


# ---------------------------------------------------------------------------
# compute_potts_joint_logits tests
# ---------------------------------------------------------------------------


class TestComputePottsJointLogits:
    """Tests for compute_potts_joint_logits (composite likelihood)."""

    @pytest.fixture
    def output_and_seq(self, pseudo_configs, context, device):
        model = ArtiosPotts(pseudo_configs).to(device)
        model.train()
        output = model.main_train_loop(context=context, seed=42)
        seq = context.token.restype.argmax(dim=-1)
        return output, seq

    def test_output_shapes(self, output_and_seq):
        output, seq = output_and_seq
        joint_logits, edge_mask = compute_potts_joint_logits(
            seq, output.h, output.J, output.edge_index, output.edge_mask
        )

        B, N = 2, 32
        R = 32
        K = output.edge_index.shape[-1]

        assert joint_logits.shape == (B, N, K, R, R)
        assert edge_mask.shape == (B, N, K)

    def test_output_finite(self, output_and_seq):
        output, seq = output_and_seq
        joint_logits, _ = compute_potts_joint_logits(
            seq, output.h, output.J, output.edge_index, output.edge_mask
        )

        assert torch.isfinite(
            joint_logits
        ).all(), "Joint logits contain non-finite values"

    def test_edge_mask_excludes_self_edges(self, output_and_seq):
        """Self-edges (i->i) should be masked out."""
        output, seq = output_and_seq
        _, edge_mask = compute_potts_joint_logits(
            seq, output.h, output.J, output.edge_index, output.edge_mask
        )

        N = output.edge_index.shape[1]
        ii = torch.arange(N, device=output.edge_index.device)
        is_self = output.edge_index == ii[None, :, None]

        # Where edge is self-edge, mask should be False
        assert not (edge_mask & is_self).any(), "Self-edges should be masked"

    def test_joint_logits_normalize_over_rr(self, output_and_seq):
        """Softmax over R*R should sum to 1 for valid edges."""
        output, seq = output_and_seq
        joint_logits, edge_mask = compute_potts_joint_logits(
            seq, output.h, output.J, output.edge_index, output.edge_mask
        )

        R = joint_logits.shape[-1]
        logits_flat = joint_logits.reshape(*joint_logits.shape[:3], R * R)
        probs = torch.softmax(logits_flat, dim=-1)

        # Check normalization for a few valid edges
        valid = edge_mask.nonzero(as_tuple=False)[:10]
        for idx in valid:
            b, n, k = idx[0], idx[1], idx[2]
            assert torch.allclose(
                probs[b, n, k].sum(),
                torch.tensor(1.0, device=probs.device),
                atol=1e-5,
            )


# ---------------------------------------------------------------------------
# compute_loss tests (pseudo)
# ---------------------------------------------------------------------------


class TestComputeLossPseudo:
    """Tests for compute_loss with loss_type='pseudo'."""

    @pytest.fixture
    def model(self, pseudo_configs, device):
        return ArtiosPotts(pseudo_configs).to(device)

    def test_loss_is_scalar(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)
        loss, _ = model.compute_loss(output, context)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_loss_non_negative(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)
        loss, _ = model.compute_loss(output, context)

        assert loss >= 0

    def test_metrics_structure(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)
        _, metrics = model.compute_loss(output, context)

        assert "weighted_nll_loss" in metrics
        assert "accuracy" in metrics
        assert "nll_protein_loss" in metrics
        assert "accuracy_protein" in metrics
        assert "nll_ligand_loss" in metrics
        assert "accuracy_ligand" in metrics

    def test_accuracy_in_range(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)
        _, metrics = model.compute_loss(output, context)

        accuracy = metrics["accuracy"]
        valid = accuracy[~torch.isnan(accuracy)]
        assert ((valid >= 0) & (valid <= 1)).all()

    def test_gradient_flow(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)
        loss, _ = model.compute_loss(output, context)
        loss.backward()

        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                assert torch.isfinite(
                    param.grad
                ).all(), f"Non-finite gradient for {name}"

        assert has_grad, "No gradients computed"

    def test_label_smoothing(self, context, device):
        """Different label smoothing should produce different losses."""
        m0 = ArtiosPotts(_make_configs(label_smoothing=0.0)).to(device)
        m1 = ArtiosPotts(_make_configs(label_smoothing=0.1)).to(device)
        m1.load_state_dict(m0.state_dict())

        m0.eval()
        m1.eval()

        with torch.no_grad():
            out = m0.main_train_loop(context=context, seed=42)
            loss0, _ = m0.compute_loss(out, context)
            loss1, _ = m1.compute_loss(out, context)

        assert not torch.allclose(loss0, loss1), "Label smoothing should change loss"


# ---------------------------------------------------------------------------
# compute_loss tests (composite)
# ---------------------------------------------------------------------------


class TestComputeLossComposite:
    """Tests for compute_loss with loss_type='composite'."""

    @pytest.fixture
    def model(self, composite_configs, device):
        return ArtiosPotts(composite_configs).to(device)

    def test_loss_is_scalar(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)
        loss, _ = model.compute_loss(output, context)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_loss_non_negative(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)
        loss, _ = model.compute_loss(output, context)

        assert loss >= 0

    def test_metrics_structure(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)
        _, metrics = model.compute_loss(output, context)

        assert "weighted_nll_loss" in metrics
        assert "accuracy" in metrics
        assert "nll_protein_loss" in metrics
        assert "accuracy_protein" in metrics

    def test_accuracy_in_range(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)
        _, metrics = model.compute_loss(output, context)

        accuracy = metrics["accuracy"]
        valid = accuracy[~torch.isnan(accuracy)]
        assert ((valid >= 0) & (valid <= 1)).all()

    def test_gradient_flow(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)
        loss, _ = model.compute_loss(output, context)
        loss.backward()

        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                assert torch.isfinite(
                    param.grad
                ).all(), f"Non-finite gradient for {name}"

        assert has_grad, "No gradients computed"

    def test_composite_differs_from_pseudo(
        self, pseudo_configs, composite_configs, context, device
    ):
        """Pseudo and composite losses should differ (same weights, different loss)."""
        m_pseudo = ArtiosPotts(pseudo_configs).to(device)
        m_composite = ArtiosPotts(composite_configs).to(device)
        m_composite.load_state_dict(m_pseudo.state_dict())

        m_pseudo.eval()
        m_composite.eval()

        with torch.no_grad():
            out = m_pseudo.main_train_loop(context=context, seed=42)
            loss_pseudo, _ = m_pseudo.compute_loss(out, context)
            loss_composite, _ = m_composite.compute_loss(out, context)

        assert not torch.allclose(
            loss_pseudo, loss_composite
        ), "Pseudo and composite losses should differ"


# ---------------------------------------------------------------------------
# compute_loss with symmetric J
# ---------------------------------------------------------------------------


class TestComputeLossSymmetric:
    """Tests for compute_loss with symmetric_J=True."""

    @pytest.fixture
    def model(self, symmetric_configs, device):
        return ArtiosPotts(symmetric_configs).to(device)

    def test_loss_is_finite(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)
        loss, _ = model.compute_loss(output, context)

        assert torch.isfinite(loss)

    def test_gradient_flow(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)
        loss, _ = model.compute_loss(output, context)
        loss.backward()

        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                assert torch.isfinite(
                    param.grad
                ).all(), f"Non-finite gradient for {name}"

        assert has_grad, "No gradients computed"


# ---------------------------------------------------------------------------
# Parameterisation variants
# ---------------------------------------------------------------------------


class TestParameterisationVariants:
    """Tests for factor and score parameterisations through ArtiosPotts."""

    @pytest.fixture(
        params=["linear", "factor", "score"],
        ids=["linear", "factor", "score"],
    )
    def model(self, request, device):
        configs = _make_configs(
            loss_type="pseudo",
            parameterisation=request.param,
        )
        return ArtiosPotts(configs).to(device)

    def test_train_loop_output_shapes(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)

        B, N, R = 2, 32, 32
        K = output.edge_index.shape[-1]

        assert output.h.shape == (B, N, R)
        assert output.J.shape == (B, N, K, R, R)

    def test_train_loop_finite_values(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)

        assert torch.isfinite(output.h).all()
        assert torch.isfinite(output.J).all()

    def test_loss_is_finite(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)
        loss, _ = model.compute_loss(output, context)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_gradient_flow(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)
        loss, _ = model.compute_loss(output, context)
        loss.backward()

        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                assert torch.isfinite(
                    param.grad
                ).all(), f"Non-finite gradient for {name}"

        assert has_grad, "No gradients computed"


class TestParameterisationWithCompositeLoss:
    """Tests that all parameterisations work with composite likelihood loss."""

    @pytest.fixture(
        params=["linear", "factor", "score"],
        ids=["linear", "factor", "score"],
    )
    def model(self, request, device):
        configs = _make_configs(
            loss_type="composite",
            parameterisation=request.param,
        )
        return ArtiosPotts(configs).to(device)

    def test_composite_loss_is_finite(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)
        loss, _ = model.compute_loss(output, context)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_composite_gradient_flow(self, model, context):
        model.train()
        output = model.main_train_loop(context=context, seed=42)
        loss, _ = model.compute_loss(output, context)
        loss.backward()

        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                assert torch.isfinite(
                    param.grad
                ).all(), f"Non-finite gradient for {name}"

        assert has_grad, "No gradients computed"


# ---------------------------------------------------------------------------
# Invalid config
# ---------------------------------------------------------------------------


class TestInvalidConfig:
    """Tests for invalid configurations."""

    def test_invalid_loss_type_raises(self, context, device):
        configs = _make_configs(loss_type="invalid")
        model = ArtiosPotts(configs).to(device)
        model.train()
        output = model.main_train_loop(context=context, seed=42)

        with pytest.raises(ValueError, match="Unknown loss_type"):
            model.compute_loss(output, context)
