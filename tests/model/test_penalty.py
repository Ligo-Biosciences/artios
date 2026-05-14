"""Tests for ``artios.model.modules.penalty.net_charge_penalty`` and
``artios.model.modules.penalty.hbond_density_penalty``.

The penalties are differentiable scoring functions used in Potts MCMC
sampling. The bugs we care about catching:

- Charge/index drift: someone renames PROTEIN_TYPES order or swaps the
  pos/neg index tuples and silently changes what "net charge" means.
- Chain-count confounding: the per-residue formulation was specifically
  introduced to make a monomer and a multimer at the same charge density
  score identically — a regression to total-charge would silently re-break
  that.
- token_mask leakage: padded positions are counted toward the residue
  count or the charge tally.
- H-bond gate: cross-chain pairs slip past the chain filter, or
  near-sequence pairs slip past ``sequence_separation``.
- ÷N attenuation regression: the open-ended encourager form dropped the
  per-residue normalization so the per-position gradient is O(neighbors)
  rather than O(neighbors/N). A regression to "÷N" would silently make
  the penalty too weak to steer sampling on realistic protein lengths.
- Soft-S gradients: the penalties are linear (charge) and bilinear (hbond)
  in S, so autograd should give finite, signed gradients — a refactor that
  detaches anywhere would silently break gradient-based sampling.
"""

import pytest
import torch

from artios.common import residue_names
from artios.model.modules.penalty import (
    _aa_pair_propensity_tensor,
    _aa_reach_tensor,
    hbond_density_penalty,
    net_charge_penalty,
)

_AA = residue_names.PROTEIN_TYPES
_IDX = {r: i for i, r in enumerate(_AA)}


def _one_hot(seq: list[str], n_res: int = 20) -> torch.Tensor:
    """Build a [1, len(seq), n_res] one-hot tensor."""
    n = len(seq)
    S = torch.zeros(1, n, n_res, dtype=torch.float32)
    for i, aa in enumerate(seq):
        S[0, i, _IDX[aa]] = 1.0
    return S


# ---------------------------------------------------------------------------
# net_charge_penalty
# ---------------------------------------------------------------------------


class TestNetChargePenalty:
    @pytest.mark.parametrize(
        "seq, target, expected_q",
        [
            # 4 K, 2 D, 4 ALA -> (4 - 2)/10 = 0.2
            (["LYS"] * 4 + ["ASP"] * 2 + ["ALA"] * 4, 0.0, 0.2),
            # 2 R, 2 E, 2 ALA -> (2 - 2)/6 = 0.0
            (["ARG"] * 2 + ["GLU"] * 2 + ["ALA"] * 2, 0.0, 0.0),
            # 1 K, 1 R, 3 D, 2 E, 5 ALA -> (2 - 5)/12 = -0.25
            (
                ["LYS", "ARG", "ASP", "ASP", "ASP", "GLU", "GLU"] + ["ALA"] * 5,
                -0.013,
                -0.25,
            ),
        ],
    )
    def test_matches_analytic_formula(self, seq, target, expected_q):
        """((K+R-D-E)/N - target)² for explicit one-hot inputs."""
        S = _one_hot(seq)
        mask = torch.ones(1, len(seq), dtype=torch.bool)
        penalty = net_charge_penalty(S, mask, target_per_residue=target)
        expected = (expected_q - target) ** 2
        torch.testing.assert_close(
            penalty,
            torch.tensor([expected], dtype=torch.float32),
            rtol=1e-5,
            atol=1e-7,
        )

    def test_chain_count_invariance(self):
        """A monomer and a dimer at the same charge density score identically.

        The whole reason the function is per-residue rather than total —
        if someone reverts to total charge this test fails.
        """
        small = ["LYS"] * 2 + ["ASP"] + ["ALA"] * 7  # 10 res, density 0.1
        large = ["LYS"] * 20 + ["ASP"] * 10 + ["ALA"] * 70  # 100 res, density 0.1

        S_s = _one_hot(small)
        S_l = _one_hot(large)
        m_s = torch.ones(1, len(small), dtype=torch.bool)
        m_l = torch.ones(1, len(large), dtype=torch.bool)

        p_s = net_charge_penalty(S_s, m_s, target_per_residue=0.0)
        p_l = net_charge_penalty(S_l, m_l, target_per_residue=0.0)
        torch.testing.assert_close(p_s, p_l, rtol=1e-5, atol=1e-7)

    def test_token_mask_excludes_padding(self):
        """Charges at masked-out positions must not enter the tally or the
        denominator."""
        seq_real = ["LYS"] * 3 + ["ASP"] + ["ALA"] * 6  # 10 valid residues
        # Pad with charged residues that should be invisible.
        seq_padded = seq_real + ["LYS"] * 5 + ["GLU"] * 5
        mask = torch.cat(
            [torch.ones(1, 10, dtype=torch.bool), torch.zeros(1, 10, dtype=torch.bool)],
            dim=1,
        )

        S_real = _one_hot(seq_real)
        S_padded = _one_hot(seq_padded)
        m_real = torch.ones(1, 10, dtype=torch.bool)

        p_real = net_charge_penalty(S_real, m_real, target_per_residue=0.0)
        p_padded = net_charge_penalty(S_padded, mask, target_per_residue=0.0)
        torch.testing.assert_close(p_real, p_padded, rtol=1e-5, atol=1e-7)

    def test_zero_when_no_valid_tokens(self):
        """All-False mask returns exactly 0 (not target², not NaN)."""
        S = _one_hot(["LYS"] * 5)
        mask = torch.zeros(1, 5, dtype=torch.bool)
        penalty = net_charge_penalty(S, mask, target_per_residue=-0.013)
        torch.testing.assert_close(
            penalty,
            torch.zeros(1),
            rtol=0.0,
            atol=0.0,
        )

    def test_only_kred_are_charged(self):
        """HIS, neutral residues, and aromatics must not contribute.

        Catches a refactor that accidentally extends the pos/neg index
        tuples to include HIS (which is sometimes counted as +ve charge
        at lower pH but is deliberately excluded here).
        """
        base = ["ALA"] * 20
        S_base = _one_hot(base)
        mask = torch.ones(1, 20, dtype=torch.bool)
        p_base = net_charge_penalty(S_base, mask, target_per_residue=0.0)

        # Replace ALA with each non-charged amino acid and assert no change.
        non_charged = [r for r in _AA if r not in ("LYS", "ARG", "ASP", "GLU")]
        for aa in non_charged:
            seq = [aa] * 20
            S = _one_hot(seq)
            p = net_charge_penalty(S, mask, target_per_residue=0.0)
            torch.testing.assert_close(
                p,
                p_base,
                rtol=1e-5,
                atol=1e-7,
                msg=f"{aa} unexpectedly contributed to net charge",
            )

    def test_linear_gradient_is_analytic(self):
        """Penalty is quadratic in q, which is linear in S — autograd must
        produce the closed-form gradient. A refactor that detaches counts
        or routes counts through a non-differentiable op breaks this.
        """
        seq = ["LYS"] * 3 + ["ASP"] * 2 + ["ALA"] * 5
        S = _one_hot(seq).requires_grad_(True)
        mask = torch.ones(1, len(seq), dtype=torch.bool)
        target = -0.05

        penalty = net_charge_penalty(S, mask, target_per_residue=target).sum()
        penalty.backward()

        n = len(seq)
        q = (3 - 2) / n  # 0.1
        coef = 2.0 * (q - target) / n  # dpenalty/d(count_share) at +/- 1 sign
        # Position 0 holds LYS=1.0; dpenalty/dS[0, LYS] = +coef.
        # dpenalty/dS[0, ASP] = -coef. Non-charge channels: 0.
        grad = S.grad[0, 0]
        torch.testing.assert_close(
            grad[_IDX["LYS"]],
            torch.tensor(+coef),
            rtol=1e-5,
            atol=1e-7,
        )
        torch.testing.assert_close(
            grad[_IDX["ARG"]],
            torch.tensor(+coef),
            rtol=1e-5,
            atol=1e-7,
        )
        torch.testing.assert_close(
            grad[_IDX["ASP"]],
            torch.tensor(-coef),
            rtol=1e-5,
            atol=1e-7,
        )
        torch.testing.assert_close(
            grad[_IDX["GLU"]],
            torch.tensor(-coef),
            rtol=1e-5,
            atol=1e-7,
        )
        torch.testing.assert_close(
            grad[_IDX["ALA"]],
            torch.tensor(0.0),
            rtol=0.0,
            atol=1e-7,
        )
        torch.testing.assert_close(
            grad[_IDX["HIS"]],
            torch.tensor(0.0),
            rtol=0.0,
            atol=1e-7,
        )


# ---------------------------------------------------------------------------
# hbond_density_penalty
# ---------------------------------------------------------------------------


def _backbone_for_residues(positions: torch.Tensor) -> torch.Tensor:
    """Build a [1, N, 5, 3] backbone in (N, CA, C, O, CB) order. CA is the
    caller-supplied position; N and C are offset along orthogonal axes so
    the ProteinMPNN virtual-CB formula has a non-degenerate cross product.

    CB is computed in-line using the same ideal-geometry formula the data
    pipeline uses (artios.utils.geometry.compute_virtual_cb), so the test
    fixture matches what the featurizer would produce for real structures.
    """
    ca = positions  # [N, 3]
    bb_n = ca + torch.tensor([-1.0, 0.0, 0.0])
    bb_c = ca + torch.tensor([0.0, 1.0, 0.0])
    bb_o = ca + torch.tensor([0.0, 0.0, 1.0])
    n_to_ca = ca - bb_n
    ca_to_c = bb_c - ca
    cb = (
        ca
        + 0.568 * n_to_ca
        - 0.54068 * ca_to_c
        - 0.5827 * torch.cross(n_to_ca, ca_to_c, dim=-1)
    )
    return torch.stack([bb_n, ca, bb_c, bb_o, cb], dim=-2).unsqueeze(0)


class TestHbondDensityPenalty:
    """The penalty is an open-ended encourager: ``U = −Σ chem_pair · gate``
    over intra-chain upper-tri pairs. More compatible neighbors in range
    ⇒ lower (more negative) penalty; no target, no tolerance, no dead-zone."""

    def test_zero_when_no_aa_tokens(self):
        """Empty/non-AA samples return 0 (no pairs in pair_mask)."""
        N = 5
        S = torch.zeros(1, N, 20)  # no AA channel set anywhere
        bb = _backbone_for_residues(torch.zeros(N, 3))
        chain = torch.zeros(1, N, dtype=torch.long)
        mask = torch.ones(1, N, dtype=torch.bool)

        out = hbond_density_penalty(S, bb, chain, mask)
        torch.testing.assert_close(out, torch.zeros(1), rtol=0.0, atol=0.0)

    def test_all_ala_yields_zero(self):
        """ALA has zero pair propensity, so chem_pair = 0 everywhere and
        the penalty is 0 even when residues are packed inside the distance
        gate. Pins that chem_pair (not the distance gate) is what gates
        H-bond compatibility."""
        seq = ["ALA"] * 10
        S = _one_hot(seq)
        positions = torch.zeros(len(seq), 3)
        bb = _backbone_for_residues(positions)
        chain = torch.zeros(1, len(seq), dtype=torch.long)
        mask = torch.ones(1, len(seq), dtype=torch.bool)

        out = hbond_density_penalty(S, bb, chain, mask)
        torch.testing.assert_close(out, torch.zeros(1), rtol=0.0, atol=1e-7)

    def test_cross_chain_pairs_excluded(self):
        """A SER-SER pair on the same chain produces a strictly negative
        penalty (encouragement); the same pair on different chains
        produces 0 (no cross-chain credit)."""
        seq = ["SER", "SER"]
        S = _one_hot(seq)
        positions = torch.tensor([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
        bb = _backbone_for_residues(positions)
        mask = torch.ones(1, 2, dtype=torch.bool)

        same_chain = torch.zeros(1, 2, dtype=torch.long)
        diff_chain = torch.tensor([[0, 1]], dtype=torch.long)

        p_same = hbond_density_penalty(S, bb, same_chain, mask)
        p_diff = hbond_density_penalty(S, bb, diff_chain, mask)

        torch.testing.assert_close(p_diff, torch.zeros(1), rtol=0.0, atol=1e-7)
        assert (
            p_same.item() < -1e-6
        ), f"same-chain SER-SER should encourage, got {p_same.item()}"

    def test_sequence_separation_excludes_near_neighbours(self):
        """Raising ``sequence_separation`` above the chain length empties
        the upper-tri pair set, so the penalty collapses to 0 even though
        the residues are H-bond capable and within distance range."""
        seq = ["SER"] * 4
        S = _one_hot(seq)
        positions = torch.tensor([[i * 4.0, 0.0, 0.0] for i in range(len(seq))])
        bb = _backbone_for_residues(positions)
        chain = torch.zeros(1, len(seq), dtype=torch.long)
        mask = torch.ones(1, len(seq), dtype=torch.bool)

        p_close = hbond_density_penalty(S, bb, chain, mask, sequence_separation=1)
        p_far = hbond_density_penalty(S, bb, chain, mask, sequence_separation=len(seq))

        torch.testing.assert_close(p_far, torch.zeros(1), rtol=0.0, atol=1e-7)
        assert p_close.item() < -1e-6

    def test_distance_gate_is_monotonic(self):
        """Same SER-SER pair moved apart must produce a monotonically
        non-decreasing penalty (encouragement weakens as the soft gate
        closes), converging to 0 at large separation."""
        seq = ["SER", "SER"]
        S = _one_hot(seq)
        chain = torch.zeros(1, 2, dtype=torch.long)
        mask = torch.ones(1, 2, dtype=torch.bool)

        penalties = []
        for d in [3.0, 8.0, 15.0, 25.0]:
            positions = torch.tensor([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
            bb = _backbone_for_residues(positions)
            p = hbond_density_penalty(S, bb, chain, mask).item()
            penalties.append(p)

        assert all(
            penalties[i] <= penalties[i + 1] + 1e-7 for i in range(len(penalties) - 1)
        ), f"penalty not monotonic in distance: {penalties}"
        # At 25Å the sigmoid is ~0 so the encouragement vanishes.
        assert abs(penalties[-1]) < 1e-3

    def test_token_mask_excludes_invalid_positions(self):
        """Padded positions must drop out of the pair sum entirely — same
        chain + same residues + same coords with 3 valid vs 3 valid + 3
        masked (far away) should produce identical penalty."""
        S_short = _one_hot(["SER"] * 3)
        pos_short = torch.tensor([[i * 4.0, 0.0, 0.0] for i in range(3)])
        bb_short = _backbone_for_residues(pos_short)
        chain_short = torch.zeros(1, 3, dtype=torch.long)
        mask_short = torch.ones(1, 3, dtype=torch.bool)

        S_long = _one_hot(["SER"] * 6)
        pos_long = torch.cat(
            [pos_short, torch.tensor([[100.0, 100.0, 100.0]] * 3)],
            dim=0,
        )
        bb_long = _backbone_for_residues(pos_long)
        chain_long = torch.zeros(1, 6, dtype=torch.long)
        mask_long = torch.tensor(
            [[True, True, True, False, False, False]],
        )

        p_short = hbond_density_penalty(S_short, bb_short, chain_short, mask_short)
        p_long = hbond_density_penalty(S_long, bb_long, chain_long, mask_long)
        torch.testing.assert_close(p_short, p_long, rtol=1e-5, atol=1e-7)

    def test_gradient_flows_on_soft_inputs(self):
        """Autograd through the bilinear chem term must produce finite,
        signed gradients. Catches any silent detach in the chem matmul or
        the virtual-CB construction."""
        S = torch.zeros(1, 4, 20)
        for i in range(4):
            S[0, i, _IDX["SER"]] = 0.5
            S[0, i, _IDX["TYR"]] = 0.5
        S.requires_grad_(True)

        positions = torch.tensor(
            [[i * 4.0, 0.0, 0.0] for i in range(4)],
            dtype=torch.float32,
        )
        bb = _backbone_for_residues(positions)
        chain = torch.zeros(1, 4, dtype=torch.long)
        mask = torch.ones(1, 4, dtype=torch.bool)

        loss = hbond_density_penalty(S, bb, chain, mask).sum()
        loss.backward()

        assert torch.isfinite(S.grad).all(), "non-finite gradient on soft input"
        assert S.grad.abs().sum().item() > 1e-6

    def test_more_compatible_neighbors_lower_penalty(self):
        """Open-ended encourager invariant: adding an in-range H-bond-capable
        neighbor must strictly lower (more negative) the penalty. The old
        squared/dead-zone form could saturate; the linear form should not."""
        chain = torch.zeros(1, 4, dtype=torch.long)
        mask = torch.ones(1, 4, dtype=torch.bool)
        # Four residues in a tight line so every pair is within the gate.
        positions = torch.tensor([[i * 4.0, 0.0, 0.0] for i in range(4)])
        bb = _backbone_for_residues(positions)

        # Mix in increasingly many SER, replacing ALA (zero propensity).
        scores = []
        for k in range(5):  # 0..4 SER among 4 residues
            seq = ["SER"] * k + ["ALA"] * (4 - k)
            S = _one_hot(seq)
            scores.append(hbond_density_penalty(S, bb, chain, mask).item())

        # Strictly monotonically decreasing as more SER are added.
        assert all(
            scores[i + 1] < scores[i] - 1e-6 for i in range(len(scores) - 1)
        ), f"penalty not strictly decreasing with more H-bond-capable residues: {scores}"

    def test_grad_magnitude_independent_of_chain_length(self):
        """Regression for the ÷N attenuation that motivated this rewrite.

        Doubling the chain (with the same per-position neighborhood) must
        not shrink the per-position gradient — that was the dominant scale
        bug in the old form. Compare max|∂U/∂S| across N=10 vs N=20 of the
        same residue type; they should be within a small constant factor."""
        chain_10 = torch.zeros(1, 10, dtype=torch.long)
        mask_10 = torch.ones(1, 10, dtype=torch.bool)
        positions_10 = torch.tensor([[i * 4.0, 0.0, 0.0] for i in range(10)])
        bb_10 = _backbone_for_residues(positions_10)
        S_10 = _one_hot(["SER"] * 10).requires_grad_(True)
        hbond_density_penalty(S_10, bb_10, chain_10, mask_10).sum().backward()
        # Middle-positioned residue has both flanks fully populated.
        max_grad_10 = S_10.grad[0, 5].abs().max().item()

        chain_20 = torch.zeros(1, 20, dtype=torch.long)
        mask_20 = torch.ones(1, 20, dtype=torch.bool)
        positions_20 = torch.tensor([[i * 4.0, 0.0, 0.0] for i in range(20)])
        bb_20 = _backbone_for_residues(positions_20)
        S_20 = _one_hot(["SER"] * 20).requires_grad_(True)
        hbond_density_penalty(S_20, bb_20, chain_20, mask_20).sum().backward()
        max_grad_20 = S_20.grad[0, 10].abs().max().item()

        # The N=20 case has more neighbors in range, so its per-position
        # gradient should be ≥ the N=10 case — NOT smaller by ~N as the
        # old ÷n_aa_residues form caused.
        assert max_grad_20 >= 0.5 * max_grad_10, (
            f"per-position gradient shrank with N: N=10 → {max_grad_10:.4g}, "
            f"N=20 → {max_grad_20:.4g}; ÷N attenuation may have returned"
        )

    def test_matches_manual_two_residue_score(self):
        """End-to-end formula pin: a single SER-TYR pair at distance d on
        the same chain should yield exactly
        ``−M[SER, TYR] · sigmoid((cutoff − tip_d)/sharpness)``. Catches
        off-by-one in upper-tri counting (a symmetric loop would
        double-count this pair), wrong matrix lookup, and any stray ÷N."""
        cutoff = 15.0
        sharpness = 2.0
        d = 6.0

        S = _one_hot(["SER", "TYR"])
        chain = torch.zeros(1, 2, dtype=torch.long)
        mask = torch.ones(1, 2, dtype=torch.bool)

        ser_reach = _aa_reach_tensor(torch.device("cpu"), torch.float32)[
            _IDX["SER"]
        ].item()
        tyr_reach = _aa_reach_tensor(torch.device("cpu"), torch.float32)[
            _IDX["TYR"]
        ].item()

        positions = torch.tensor([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
        bb = _backbone_for_residues(positions)

        # Reproduce the function's virtual-tip math to get the exact gate input.
        ca = bb[..., 1, :]
        cb = bb[..., 4, :]
        cb_dir = (cb - ca) / (cb - ca).norm(dim=-1, keepdim=True)
        tip_0 = ca[0, 0] + ser_reach * cb_dir[0, 0]
        tip_1 = ca[0, 1] + tyr_reach * cb_dir[0, 1]
        actual_d = (tip_0 - tip_1).norm().item()

        matrix = _aa_pair_propensity_tensor(torch.device("cpu"), torch.float32)
        gate = torch.sigmoid(torch.tensor((cutoff - actual_d) / sharpness)).item()
        m_st = matrix[_IDX["SER"], _IDX["TYR"]].item()
        expected = -m_st * gate  # single upper-tri pair, no ÷N, no dead-zone

        out = hbond_density_penalty(
            S,
            bb,
            chain,
            mask,
            distance_cutoff=cutoff,
            sharpness=sharpness,
            sequence_separation=1,
        )
        torch.testing.assert_close(
            out,
            torch.tensor([expected], dtype=torch.float32),
            rtol=1e-5,
            atol=1e-6,
        )
