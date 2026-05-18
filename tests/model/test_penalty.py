"""Tests for ``artios.model.modules.penalty.net_charge_penalty``.

The penalty is a differentiable scoring function used in Potts MCMC
sampling. The bugs we care about catching:

- Charge/index drift: someone renames PROTEIN_TYPES order or swaps the
  pos/neg index tuples and silently changes what "net charge" means.
- Chain-count confounding: the per-residue formulation was specifically
  introduced to make a monomer and a multimer at the same charge density
  score identically — a regression to total-charge would silently re-break
  that.
- token_mask leakage: padded positions are counted toward the residue
  count or the charge tally.
- Soft-S gradients: the penalty is linear in S, so autograd should give
  finite, signed gradients — a refactor that detaches anywhere would
  silently break gradient-based sampling.
"""

import pytest
import torch

from artios.common import residue_names
from artios.model.modules.penalty import net_charge_penalty

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
