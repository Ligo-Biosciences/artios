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
- H-bond density gate: cross-chain pairs slip past the chain filter, or
  near-sequence pairs slip past ``sequence_separation``.
- Phantom-target bug: an empty-AA sample returns ``target²`` instead of 0
  (the explicit ``(n_aa_residues > 0)`` multiplier in the implementation).
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
    """Build a [1, N, 4, 3] backbone where each residue's CA is at the given
    position, with N and C placed ±1Å in x/y so that the virtual-CB direction
    is well-defined (non-degenerate cross product).
    """
    ca = positions  # [N, 3]
    n_off = torch.tensor([-1.0, 0.0, 0.0])
    c_off = torch.tensor([0.0, 1.0, 0.0])
    bb_n = ca + n_off
    bb_c = ca + c_off
    bb_o = ca + torch.tensor([0.0, 0.0, 1.0])
    return torch.stack([bb_n, ca, bb_c, bb_o], dim=-2).unsqueeze(0)


class TestHbondDensityPenalty:
    def test_zero_when_no_aa_tokens(self):
        """Empty/non-AA samples return exactly 0, not target².

        Regression pin for the explicit ``(n_aa_residues > 0)`` multiplier
        on the return.
        """
        N = 5
        S = torch.zeros(1, N, 20)  # no AA channel set anywhere
        bb = _backbone_for_residues(torch.zeros(N, 3))
        chain = torch.zeros(1, N, dtype=torch.long)
        mask = torch.ones(1, N, dtype=torch.bool)

        out = hbond_density_penalty(
            S,
            bb,
            chain,
            mask,
            target_density=0.27,
        )
        torch.testing.assert_close(out, torch.zeros(1), rtol=0.0, atol=0.0)

    def test_all_ala_yields_target_squared(self):
        """ALA has reach=0 and zero pair propensity, so density=0 →
        shortfall=target_density → penalty=target². The non-AA path
        returns 0 instead, so this also confirms we picked up the AA
        flag properly."""
        seq = ["ALA"] * 10
        S = _one_hot(seq)
        # Pack residues tightly so the distance gate would saturate if the
        # chemistry matrix weren't zero — proves chem_pair gating is doing
        # the work, not the distance gate.
        positions = torch.zeros(len(seq), 3)
        bb = _backbone_for_residues(positions)
        chain = torch.zeros(1, len(seq), dtype=torch.long)
        mask = torch.ones(1, len(seq), dtype=torch.bool)
        target = 0.27

        out = hbond_density_penalty(
            S,
            bb,
            chain,
            mask,
            target_density=target,
        )
        torch.testing.assert_close(
            out,
            torch.tensor([target * target]),
            rtol=1e-5,
            atol=1e-7,
        )

    def test_cross_chain_pairs_excluded(self):
        """Two close H-bond-capable residues on different chains must not
        contribute. Build a SER-SER pair (positive pair propensity, nonzero
        reach) and check that same-chain → penalty < target² but
        different-chain → penalty == target²."""
        seq = ["SER", "SER"]
        S = _one_hot(seq)
        # Place 8Å apart so the distance gate (cutoff 15, sharpness 2) is
        # firmly on.
        positions = torch.tensor([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
        bb = _backbone_for_residues(positions)
        mask = torch.ones(1, 2, dtype=torch.bool)
        target = 0.27

        same_chain = torch.zeros(1, 2, dtype=torch.long)
        diff_chain = torch.tensor([[0, 1]], dtype=torch.long)

        p_same = hbond_density_penalty(
            S,
            bb,
            same_chain,
            mask,
            target_density=target,
        )
        p_diff = hbond_density_penalty(
            S,
            bb,
            diff_chain,
            mask,
            target_density=target,
        )

        # Cross-chain → zero density → full target² shortfall.
        torch.testing.assert_close(
            p_diff,
            torch.tensor([target * target]),
            rtol=1e-5,
            atol=1e-7,
        )
        # Same-chain SER-SER should pull the penalty strictly below target²
        # (positive propensity contribution).
        assert p_same.item() < p_diff.item() - 1e-6

    def test_sequence_separation_excludes_near_neighbours(self):
        """With ``sequence_separation`` raised above the chain length, no
        pairs qualify and penalty falls back to target²."""
        seq = ["SER"] * 4
        S = _one_hot(seq)
        positions = torch.tensor([[i * 4.0, 0.0, 0.0] for i in range(len(seq))])
        bb = _backbone_for_residues(positions)
        chain = torch.zeros(1, len(seq), dtype=torch.long)
        mask = torch.ones(1, len(seq), dtype=torch.bool)
        target = 0.27

        # sequence_separation=1 (default) -> many pairs contribute.
        p_close = hbond_density_penalty(
            S,
            bb,
            chain,
            mask,
            target_density=target,
            sequence_separation=1,
        )
        # sequence_separation=len(seq) -> upper-tri is empty.
        p_far = hbond_density_penalty(
            S,
            bb,
            chain,
            mask,
            target_density=target,
            sequence_separation=len(seq),
        )
        torch.testing.assert_close(
            p_far,
            torch.tensor([target * target]),
            rtol=1e-5,
            atol=1e-7,
        )
        assert p_close.item() < p_far.item() - 1e-6

    def test_distance_gate_is_monotonic(self):
        """Same residue pair, moved apart, must produce a monotonically
        non-decreasing penalty (density drops as distance grows)."""
        seq = ["SER", "SER"]
        S = _one_hot(seq)
        chain = torch.zeros(1, 2, dtype=torch.long)
        mask = torch.ones(1, 2, dtype=torch.bool)
        target = 0.27

        penalties = []
        for d in [3.0, 8.0, 15.0, 25.0]:
            positions = torch.tensor([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
            bb = _backbone_for_residues(positions)
            p = hbond_density_penalty(
                S,
                bb,
                chain,
                mask,
                target_density=target,
            ).item()
            penalties.append(p)

        assert all(
            penalties[i] <= penalties[i + 1] + 1e-7 for i in range(len(penalties) - 1)
        ), f"penalty not monotonic in distance: {penalties}"
        # Sanity: at 25Å the sigmoid is ~0 so we're back at target².
        assert abs(penalties[-1] - target * target) < 1e-3

    def test_token_mask_excludes_invalid_positions(self):
        """Padded positions must drop out of both the pair sum and the
        residue-count denominator. We verify by comparing a 3-SER chain
        to a 3-SER + 3-masked-SER configuration — same density expected.
        """
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

        p_short = hbond_density_penalty(
            S_short,
            bb_short,
            chain_short,
            mask_short,
            target_density=0.27,
        )
        p_long = hbond_density_penalty(
            S_long,
            bb_long,
            chain_long,
            mask_long,
            target_density=0.27,
        )
        torch.testing.assert_close(p_short, p_long, rtol=1e-5, atol=1e-7)

    def test_gradient_flows_on_soft_inputs(self):
        """Autograd through the bilinear chem term must produce finite,
        signed gradients. Catches any silent detach in the chem matmul or
        the virtual-CB construction."""
        # Mix-soft input so chem_pair has support across multiple AA pairs.
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

        loss = hbond_density_penalty(
            S,
            bb,
            chain,
            mask,
            target_density=0.27,
        ).sum()
        loss.backward()

        assert torch.isfinite(S.grad).all(), "non-finite gradient on soft input"
        # At least one AA channel must have non-zero gradient — otherwise
        # the function is effectively constant in S and the penalty cannot
        # drive sampling.
        assert S.grad.abs().sum().item() > 1e-6

    def test_matches_manual_two_residue_density(self):
        """Pair (SER, TYR) at distance d on the same chain: density should
        equal ``M[SER, TYR] * sigmoid((cutoff-d)/sharpness) / 2``. This
        pins the formula end-to-end against the published reference
        propensity matrix and reach array, and catches off-by-one in
        upper-tri counting (a symmetric loop would double-count this pair).
        """
        target = 0.05  # below achievable so we land in the shortfall regime
        cutoff = 15.0
        sharpness = 2.0
        d = 6.0

        S = _one_hot(["SER", "TYR"])
        chain = torch.zeros(1, 2, dtype=torch.long)
        mask = torch.ones(1, 2, dtype=torch.bool)

        # Need the actual virtual-tip distance, not the CA-CA distance, to
        # match the implementation. Compute it analytically the same way
        # the function does.
        ser_reach = _aa_reach_tensor(torch.device("cpu"), torch.float32)[
            _IDX["SER"]
        ].item()
        tyr_reach = _aa_reach_tensor(torch.device("cpu"), torch.float32)[
            _IDX["TYR"]
        ].item()

        # Place SER at origin, TYR offset along x. Build matching CB dirs:
        # use the same backbone offsets as _backbone_for_residues so both
        # residues' virtual tips lie along a known axis.
        positions = torch.tensor([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
        bb = _backbone_for_residues(positions)

        # Recompute virtual-tip positions exactly as the function does.
        ca = bb[..., 1, :]
        n = bb[..., 0, :]
        c = bb[..., 2, :]
        n_to_ca = ca - n
        ca_to_c = c - ca
        cb = (
            ca
            + 0.568 * n_to_ca
            - 0.54068 * ca_to_c
            - 0.5827 * torch.cross(n_to_ca, ca_to_c, dim=-1)
        )
        cb_dir = (cb - ca) / (cb - ca).norm(dim=-1, keepdim=True)
        tip_0 = ca[0, 0] + ser_reach * cb_dir[0, 0]
        tip_1 = ca[0, 1] + tyr_reach * cb_dir[0, 1]
        actual_d = (tip_0 - tip_1).norm().item()

        matrix = _aa_pair_propensity_tensor(torch.device("cpu"), torch.float32)
        gate = torch.sigmoid(torch.tensor((cutoff - actual_d) / sharpness)).item()
        m_st = matrix[_IDX["SER"], _IDX["TYR"]].item()
        expected_density = m_st * gate / 2.0  # 2 residues
        expected_shortfall = min(expected_density - target, 0.0)
        expected = expected_shortfall**2

        out = hbond_density_penalty(
            S,
            bb,
            chain,
            mask,
            target_density=target,
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
