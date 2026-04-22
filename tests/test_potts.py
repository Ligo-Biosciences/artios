"""Tests for the Potts model module.

Tests GraphPotts parameterization and J symmetrization.
"""

import pytest
import torch

from artios.model.modules.penalty import (
    AROMATIC_INDICES,
    aromatic_enrichment_penalty,
)
from artios.model.modules.potts import (
    GraphPotts,
    GraphPottsParameterisation,
    _symmetrize_J,
    compute_potts_energy,
)
from artios.model.output import PottsOutput

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(
    params=[
        (False, GraphPottsParameterisation.LINEAR),
        (True, GraphPottsParameterisation.LINEAR),
        (False, GraphPottsParameterisation.FACTOR),
        (True, GraphPottsParameterisation.FACTOR),
        (False, GraphPottsParameterisation.SCORE),
        (True, GraphPottsParameterisation.SCORE),
    ],
    ids=[
        "asymmetric-linear",
        "symmetric-linear",
        "asymmetric-factor",
        "symmetric-factor",
        "asymmetric-score",
        "symmetric-score",
    ],
)
def graph_potts(request, device):
    """Create a GraphPotts model, parametrized over symmetric_J and parameterisation."""
    symmetric_J, parameterisation = request.param
    return GraphPotts(
        node_dim=64,
        edge_dim=128,
        num_residue_types=8,
        parameterisation=parameterisation,
        symmetric_J=symmetric_J,
    ).to(device)


@pytest.fixture
def graph_inputs(device):
    """Create synthetic inputs for GraphPotts.forward."""
    B, N, K = 2, 16, 6
    node_dim, edge_dim = 64, 128

    node_feats = torch.randn(B, N, node_dim, device=device)
    edge_feats = torch.randn(B, N, K, edge_dim, device=device)

    # Build a valid edge_index: for each node, pick K random neighbors
    edge_index = torch.stack([torch.randint(0, N, (B, N, K), device=device)]).squeeze(0)
    # Ensure at least one self-edge per node for realism
    edge_index[:, :, 0] = torch.arange(N, device=device).unsqueeze(0)

    edge_mask = torch.ones(B, N, K, dtype=torch.bool, device=device)
    token_mask = torch.ones(B, N, dtype=torch.bool, device=device)

    return node_feats, edge_feats, edge_index, edge_mask, token_mask


@pytest.fixture
def reciprocal_edge_index(device):
    """Create edge_index where every edge (i,j) has a reverse edge (j,i).

    Uses a simple chain graph: each node connects to its two neighbors.
    This guarantees all reverse edges exist, which is needed for
    _symmetrize_J to produce fully symmetric results.
    """
    B, N, K = 2, 8, 3
    edge_index = torch.zeros(B, N, K, dtype=torch.long, device=device)

    for i in range(N):
        # neighbors: self, left, right (wrapping)
        edge_index[:, i, 0] = i
        edge_index[:, i, 1] = (i - 1) % N
        edge_index[:, i, 2] = (i + 1) % N

    edge_mask = torch.ones(B, N, K, dtype=torch.bool, device=device)
    return edge_index, edge_mask, B, N, K


# ---------------------------------------------------------------------------
# GraphPotts tests
# ---------------------------------------------------------------------------


class TestGraphPottsOutput:
    """Tests for GraphPotts output shapes and values."""

    def test_output_shapes(self, graph_potts, graph_inputs):
        """Test that h and J have correct shapes."""
        node_feats, edge_feats, edge_index, edge_mask, token_mask = graph_inputs
        B, N, K = edge_index.shape
        R = graph_potts.num_residue_types

        h, J = graph_potts(node_feats, edge_feats, edge_index, edge_mask, token_mask)

        assert h.shape == (B, N, R)
        assert J.shape == (B, N, K, R, R)

    def test_output_no_nan_or_inf(self, graph_potts, graph_inputs):
        """Test that outputs don't contain NaN or Inf."""
        h, J = graph_potts(*graph_inputs)

        assert torch.isfinite(h).all(), "h contains non-finite values"
        assert torch.isfinite(J).all(), "J contains non-finite values"

    def test_token_mask_zeros_h(self, graph_potts, graph_inputs):
        """Test that masked tokens have zero h."""
        node_feats, edge_feats, edge_index, edge_mask, token_mask = graph_inputs
        token_mask = token_mask.clone()
        token_mask[:, 5:10] = False

        h, _ = graph_potts(node_feats, edge_feats, edge_index, edge_mask, token_mask)

        assert (h[:, 5:10] == 0).all(), "Masked token h should be zero"
        assert not (h[:, :5] == 0).all(), "Unmasked token h should be non-zero"

    def test_edge_mask_zeros_J(self, graph_potts, graph_inputs):
        """Test that masked edges have zero J."""
        node_feats, edge_feats, edge_index, edge_mask, token_mask = graph_inputs
        edge_mask = edge_mask.clone()
        edge_mask[:, :, 3:] = False

        _, J = graph_potts(node_feats, edge_feats, edge_index, edge_mask, token_mask)

        assert (J[:, :, 3:] == 0).all(), "Masked edge J should be zero"


class TestGraphPottsGradients:
    """Tests for gradient flow through GraphPotts."""

    def test_gradients_flow(self, graph_potts, graph_inputs):
        """Test that gradients flow through h and J."""
        h, J = graph_potts(*graph_inputs)

        loss = h.sum() + J.sum()
        loss.backward()

        for name, param in graph_potts.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(
                    param.grad
                ).all(), f"Non-finite gradient for {name}"


# ---------------------------------------------------------------------------
# _symmetrize_J tests
# ---------------------------------------------------------------------------


class TestSymmetrizeJ:
    """Tests for the _symmetrize_J function."""

    def test_output_shape(self, device):
        """Test that symmetrized J has the same shape as input."""
        B, N, K, R = 2, 8, 4, 6
        J = torch.randn(B, N, K, R, R, device=device)
        edge_index = torch.randint(0, N, (B, N, K), device=device)
        edge_mask = torch.ones(B, N, K, dtype=torch.bool, device=device)

        J_symm = _symmetrize_J(J, edge_index, edge_mask)

        assert J_symm.shape == J.shape

    def test_output_finite(self, device):
        """Test that symmetrized J contains no NaN or Inf."""
        B, N, K, R = 2, 8, 4, 6
        J = torch.randn(B, N, K, R, R, device=device)
        edge_index = torch.randint(0, N, (B, N, K), device=device)
        edge_mask = torch.ones(B, N, K, dtype=torch.bool, device=device)

        J_symm = _symmetrize_J(J, edge_index, edge_mask)

        assert torch.isfinite(J_symm).all()

    def test_symmetry_with_reciprocal_edges(self, reciprocal_edge_index, device):
        """Test that J_ij(a,b) == J_ji(b,a) after symmetrization.

        Uses a graph where every edge (i,j) has a reverse edge (j,i),
        so symmetrization should produce perfect symmetry.
        """
        edge_index, edge_mask, B, N, K = reciprocal_edge_index
        R = 4
        J = torch.randn(B, N, K, R, R, device=device)

        J_symm = _symmetrize_J(J, edge_index, edge_mask)

        # For each non-self edge (i, k) -> j, find the reverse edge at j
        # and verify J_ij(a,b) == J_ji(b,a)
        for b in range(B):
            for i in range(N):
                for k in range(K):
                    j = edge_index[b, i, k].item()
                    if i == j:
                        continue  # skip self-edges

                    # Find reverse: which slot k' at j points to i?
                    j_neighbors = edge_index[b, j]  # [K]
                    reverse_slots = (j_neighbors == i).nonzero(as_tuple=False)
                    if len(reverse_slots) == 0:
                        continue
                    k_rev = reverse_slots[0].item()

                    J_ij = J_symm[b, i, k]  # [R, R]
                    J_ji = J_symm[b, j, k_rev]  # [R, R]

                    assert torch.allclose(
                        J_ij, J_ji.T, atol=1e-5
                    ), f"Symmetry violated: J[{b},{i},{k}] != J[{b},{j},{k_rev}].T"

    def test_already_symmetric_unchanged(self, reciprocal_edge_index, device):
        """Test that symmetrizing already-symmetric J is a fixpoint."""
        edge_index, edge_mask, B, N, K = reciprocal_edge_index
        R = 4
        J = torch.randn(B, N, K, R, R, device=device)

        J_symm1 = _symmetrize_J(J, edge_index, edge_mask)
        J_symm2 = _symmetrize_J(J_symm1, edge_index, edge_mask)

        assert torch.allclose(
            J_symm1, J_symm2, atol=1e-5
        ), "Symmetrizing twice should give the same result"

    def test_masked_edges_stay_zero(self, device):
        """Test that masked edges remain zero after symmetrization."""
        B, N, K, R = 1, 6, 3, 4
        J = torch.randn(B, N, K, R, R, device=device)

        # Chain graph
        edge_index = torch.zeros(B, N, K, dtype=torch.long, device=device)
        for i in range(N):
            edge_index[:, i, 0] = i
            edge_index[:, i, 1] = (i - 1) % N
            edge_index[:, i, 2] = (i + 1) % N

        edge_mask = torch.ones(B, N, K, dtype=torch.bool, device=device)
        # Mask out the last edge slot for all nodes
        edge_mask[:, :, 2] = False

        J_symm = _symmetrize_J(J, edge_index, edge_mask)

        assert (J_symm[:, :, 2] == 0).all(), "Masked edges should remain zero"

    def test_self_edges_handled(self, reciprocal_edge_index, device):
        """Test that self-edges don't cause issues."""
        edge_index, edge_mask, B, N, K = reciprocal_edge_index
        R = 4
        J = torch.randn(B, N, K, R, R, device=device)

        # Slot 0 is self-edge in our fixture
        J_symm = _symmetrize_J(J, edge_index, edge_mask)

        # Self-edges should be symmetric with themselves (J_ii = J_ii^T averaged)
        for b in range(B):
            for i in range(N):
                J_self = J_symm[b, i, 0]  # self-edge slot
                assert torch.allclose(
                    J_self, J_self.T, atol=1e-5
                ), f"Self-edge J[{b},{i},0] should be symmetric"

    def test_gradient_flows(self, reciprocal_edge_index, device):
        """Test that gradients flow through _symmetrize_J."""
        edge_index, edge_mask, B, N, K = reciprocal_edge_index
        R = 4
        J = torch.randn(B, N, K, R, R, device=device, requires_grad=True)

        J_symm = _symmetrize_J(J, edge_index, edge_mask)
        loss = J_symm.sum()
        loss.backward()

        assert J.grad is not None, "No gradient for J"
        assert torch.isfinite(J.grad).all(), "Non-finite gradient for J"


class TestGraphPottsSymmetric:
    """Tests specific to GraphPotts with symmetric_J=True."""

    def test_symmetric_flag_stored(self):
        """Test that symmetric_J flag is stored."""
        model = GraphPotts(
            32, 64, 8, GraphPottsParameterisation.LINEAR, symmetric_J=True
        )
        assert model.symmetric_J is True

        model = GraphPotts(
            32, 64, 8, GraphPottsParameterisation.LINEAR, symmetric_J=False
        )
        assert model.symmetric_J is False

    def test_symmetric_differs_from_asymmetric(self, graph_inputs, device):
        """Test that symmetric and asymmetric produce different J."""
        model_asym = GraphPotts(
            64, 128, 8, GraphPottsParameterisation.LINEAR, symmetric_J=False
        ).to(device)
        model_sym = GraphPotts(
            64, 128, 8, GraphPottsParameterisation.LINEAR, symmetric_J=True
        ).to(device)

        # Share weights
        model_sym.load_state_dict(model_asym.state_dict())

        _, J_asym = model_asym(*graph_inputs)
        _, J_sym = model_sym(*graph_inputs)

        assert not torch.allclose(
            J_asym, J_sym, atol=1e-5
        ), "Symmetric and asymmetric J should differ"


# ---------------------------------------------------------------------------
# Parameterisation-specific tests
# ---------------------------------------------------------------------------


class TestZeroSumGauge:
    """Tests that the factor parameterisation applies zero-sum gauge.

    Note: Score also applies zero-sum gauge internally but then adds
    background components that break the zero-sum property on the output.
    """

    @pytest.fixture
    def gauge_model(self, device):
        return GraphPotts(
            node_dim=64,
            edge_dim=128,
            num_residue_types=8,
            parameterisation=GraphPottsParameterisation.FACTOR,
            symmetric_J=False,
        ).to(device)

    def test_h_zero_mean(self, gauge_model, graph_inputs):
        """h should have approximately zero mean across residue types."""
        node_feats, edge_feats, edge_index, edge_mask, token_mask = graph_inputs
        h, _ = gauge_model(node_feats, edge_feats, edge_index, edge_mask, token_mask)

        # Only check unmasked positions
        h_mean = h[token_mask].mean(dim=-1)
        assert torch.allclose(
            h_mean, torch.zeros_like(h_mean), atol=1e-5
        ), "h should have zero mean across residue types after zero-sum gauge"

    def test_J_zero_sum(self, gauge_model, graph_inputs):
        """J should have zero row and column means after zero-sum gauge."""
        node_feats, edge_feats, edge_index, edge_mask, token_mask = graph_inputs
        _, J = gauge_model(node_feats, edge_feats, edge_index, edge_mask, token_mask)

        # Check a few valid edges
        valid = edge_mask.nonzero(as_tuple=False)[:10]
        for idx in valid:
            b, n, k = idx[0], idx[1], idx[2]
            J_ij = J[b, n, k]  # [R, R]
            row_mean = J_ij.mean(dim=-1)
            col_mean = J_ij.mean(dim=-2)
            assert torch.allclose(
                row_mean, torch.zeros_like(row_mean), atol=1e-5
            ), "J row means should be zero after zero-sum gauge"
            assert torch.allclose(
                col_mean, torch.zeros_like(col_mean), atol=1e-5
            ), "J column means should be zero after zero-sum gauge"


class TestScoreBackground:
    """Tests for background components in the score parameterisation."""

    def test_background_adds_uniform_shift_to_h(self, device):
        """The background h component adds a state-independent offset."""
        model = GraphPotts(
            node_dim=64,
            edge_dim=128,
            num_residue_types=8,
            parameterisation=GraphPottsParameterisation.SCORE,
            symmetric_J=False,
        ).to(device)

        B, N, K = 2, 16, 6
        node_feats = torch.randn(B, N, 64, device=device)
        edge_feats = torch.randn(B, N, K, 128, device=device)
        edge_index = torch.randint(0, N, (B, N, K), device=device)
        edge_mask = torch.ones(B, N, K, dtype=torch.bool, device=device)
        token_mask = torch.ones(B, N, dtype=torch.bool, device=device)

        h, _ = model(node_feats, edge_feats, edge_index, edge_mask, token_mask)

        # h mean across residue types should NOT be zero (background breaks zero-sum)
        h_mean = h.mean(dim=-1)
        assert not torch.allclose(
            h_mean, torch.zeros_like(h_mean), atol=1e-5
        ), "Score h should have non-zero mean due to background"


class TestParameterisationEnum:
    """Tests for GraphPottsParameterisation enum."""

    def test_enum_values(self):
        assert GraphPottsParameterisation.LINEAR.value == "linear"
        assert GraphPottsParameterisation.FACTOR.value == "factor"
        assert GraphPottsParameterisation.SCORE.value == "score"

    def test_construct_from_string(self):
        assert GraphPottsParameterisation("linear") == GraphPottsParameterisation.LINEAR
        assert GraphPottsParameterisation("factor") == GraphPottsParameterisation.FACTOR
        assert GraphPottsParameterisation("score") == GraphPottsParameterisation.SCORE

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            GraphPottsParameterisation("invalid")

    def test_invalid_parameterisation_raises(self):
        """GraphPotts should reject invalid parameterisation values."""
        with pytest.raises(ValueError):
            GraphPottsParameterisation("not_a_param")


# ---------------------------------------------------------------------------
# compute_potts_energy tests
# ---------------------------------------------------------------------------


class TestComputePottsEnergy:
    """Tests for the compute_potts_energy function."""

    def test_default_returns_per_node(self, device):
        """Default (collate_nodes=False) returns per-node energy [B, N]."""
        B, N, K, R = 2, 16, 6, 8
        S = torch.randint(0, R, (B, N), device=device)
        h = torch.randn(B, N, R, device=device)
        J = torch.randn(B, N, K, R, R, device=device)
        edge_index = torch.randint(0, N, (B, N, K), device=device)
        edge_mask = torch.ones(B, N, K, dtype=torch.bool, device=device)

        per_node = compute_potts_energy(S, h, J, edge_index, edge_mask)
        assert per_node.shape == (B, N)

    def test_collate_nodes_returns_batch(self, device):
        """collate_nodes=True returns the per-sample scalar [B]."""
        B, N, K, R = 2, 16, 6, 8
        S = torch.randint(0, R, (B, N), device=device)
        h = torch.randn(B, N, R, device=device)
        J = torch.randn(B, N, K, R, R, device=device)
        edge_index = torch.randint(0, N, (B, N, K), device=device)
        edge_mask = torch.ones(B, N, K, dtype=torch.bool, device=device)

        total = compute_potts_energy(S, h, J, edge_index, edge_mask, collate_nodes=True)
        assert total.shape == (B,)

    def test_collate_equivalent_to_sum(self, device):
        """collate_nodes=True must equal collate_nodes=False summed over nodes.

        This is the load-bearing invariant: the per-node 0.5-coupling tensor
        sums over the node axis to the total Potts energy with each undirected
        edge counted exactly once.
        """
        B, N, K, R = 2, 16, 6, 8
        S = torch.randint(0, R, (B, N), device=device)
        h = torch.randn(B, N, R, device=device)
        J = torch.randn(B, N, K, R, R, device=device)
        edge_index = torch.randint(0, N, (B, N, K), device=device)
        edge_mask = torch.ones(B, N, K, dtype=torch.bool, device=device)

        per_node = compute_potts_energy(S, h, J, edge_index, edge_mask)
        total = compute_potts_energy(S, h, J, edge_index, edge_mask, collate_nodes=True)

        assert torch.allclose(per_node.sum(-1), total, atol=1e-5)


# ---------------------------------------------------------------------------
# PottsOutput tests
# ---------------------------------------------------------------------------


class TestPottsOutput:
    """Tests for the PottsOutput dataclass."""

    def test_construction(self, device):
        """Test that PottsOutput can be constructed with correct types."""
        B, N, K, R = 2, 16, 6, 8
        output = PottsOutput(
            h=torch.randn(B, N, R, device=device),
            J=torch.randn(B, N, K, R, R, device=device),
            edge_index=torch.randint(0, N, (B, N, K), device=device),
            edge_mask=torch.ones(B, N, K, dtype=torch.bool, device=device),
            is_masked=torch.ones(B, N, dtype=torch.bool, device=device),
            timestep=torch.rand(B, device=device),
        )

        assert output.h.shape == (B, N, R)
        assert output.J.shape == (B, N, K, R, R)
        assert output.edge_index.shape == (B, N, K)
        assert output.edge_mask.shape == (B, N, K)
        assert output.is_masked.shape == (B, N)
        assert output.timestep.shape == (B,)


# ---------------------------------------------------------------------------
# Aromatic enrichment penalty tests
# ---------------------------------------------------------------------------


class TestAromaticEnrichmentPenalty:
    """Tests for the aromatic enrichment penalty function."""

    @pytest.fixture
    def setup(self, device):
        B, N, R = 2, 10, 20
        return B, N, R, device

    def test_zero_penalty_when_above_target(self, setup):
        """Penalty should be 0 when aromatic fraction >= target."""
        B, N, R, device = setup
        target = 0.25

        # Build S where all active site positions are aromatic (fraction=1.0)
        S = torch.zeros(B, N, R, device=device)
        S[:, :, 0] = 1.0  # default to ALA
        active_site_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        active_site_mask[:, :4] = True  # 4 active site positions

        # Set active site positions to PHE (index 13)
        S[:, :4, 0] = 0.0
        S[:, :4, 13] = 1.0

        penalty = aromatic_enrichment_penalty(
            S, active_site_mask, target_fraction=target
        )
        assert torch.allclose(
            penalty, torch.zeros(B, device=device)
        ), "Penalty should be 0 when aromatic fraction >= target"

    def test_positive_penalty_when_below_target(self, setup):
        """Penalty should be positive when aromatic fraction < target."""
        B, N, R, device = setup
        target = 0.25

        # Build S where no active site positions are aromatic (fraction=0.0)
        S = torch.zeros(B, N, R, device=device)
        S[:, :, 0] = 1.0  # all ALA
        active_site_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        active_site_mask[:, :4] = True

        penalty = aromatic_enrichment_penalty(
            S, active_site_mask, target_fraction=target
        )
        assert (penalty > 0).all(), "Penalty should be positive when below target"

    def test_gradient_nonzero_only_at_active_aromatic(self, setup):
        """Gradient w.r.t. S should be nonzero only at active site aromatic indices."""
        B, N, R, device = setup
        target = 0.5  # high target to ensure penalty is active

        S = torch.zeros(B, N, R, device=device, requires_grad=True)
        # Initialize to ALA everywhere (no aromatics)
        with torch.no_grad():
            S.data[:, :, 0] = 1.0

        active_site_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        active_site_mask[:, 2:5] = True  # positions 2, 3, 4

        penalty = aromatic_enrichment_penalty(
            S, active_site_mask, target_fraction=target
        )
        penalty.sum().backward()

        grad = S.grad  # [B, N, R]

        # Non-active-site positions should have zero gradient
        non_active = ~active_site_mask
        assert (
            grad[:, non_active[0], :] == 0
        ).all(), "Gradient should be zero outside active site"

        # Non-aromatic indices at active site should have zero gradient
        non_aromatic_mask = torch.ones(R, dtype=torch.bool)
        non_aromatic_mask[AROMATIC_INDICES] = False
        assert (
            grad[:, active_site_mask[0], :][:, :, non_aromatic_mask] == 0
        ).all(), "Gradient should be zero for non-aromatic indices"

        # Aromatic indices at active site should have nonzero gradient
        assert (
            grad[:, active_site_mask[0], :][:, :, AROMATIC_INDICES] != 0
        ).any(), "Gradient should be nonzero for aromatic indices in active site"

    def test_no_active_site_returns_zero(self, setup):
        """Penalty should be 0 when there are no active site residues."""
        B, N, R, device = setup

        S = torch.randn(B, N, R, device=device)
        active_site_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        penalty = aromatic_enrichment_penalty(S, active_site_mask)
        assert torch.allclose(
            penalty, torch.zeros(B, device=device)
        ), "Penalty should be 0 with empty active site"

    def test_exact_target_gives_zero(self, setup):
        """Penalty should be 0 when aromatic fraction exactly equals target."""
        B, N, R, device = setup
        target = 0.25

        S = torch.zeros(B, N, R, device=device)
        active_site_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        active_site_mask[:, :4] = True

        # 1 of 4 active site positions is aromatic → fraction = 0.25
        S[:, :3, 0] = 1.0  # ALA
        S[:, 3, 13] = 1.0  # PHE

        penalty = aromatic_enrichment_penalty(
            S, active_site_mask, target_fraction=target
        )
        assert torch.allclose(
            penalty, torch.zeros(B, device=device), atol=1e-6
        ), "Penalty should be 0 when fraction equals target"
