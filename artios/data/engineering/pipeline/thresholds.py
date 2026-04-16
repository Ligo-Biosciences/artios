"""Version-controlled default thresholds for fragment and pocket filtering.

Two presets:

    FRAGMENT_THRESHOLDS     — broad fragment quality filters (Pass 1).
    PRE_POCKET_THRESHOLDS   — stricter fragment filters applied before pocket detection (Pass 2).

Also exposes ``RG_UPPER_BOUND_COEFF`` — the coefficient for the length-dependent
radius-of-gyration upper bound ``rg <= coeff * length^0.4`` that is applied
alongside ``FragmentThresholds`` wherever fragments are filtered.

Import and use directly::

    from artios.data.engineering.pipeline.thresholds import (
        FRAGMENT_THRESHOLDS, PRE_POCKET_THRESHOLDS, RG_UPPER_BOUND_COEFF,
    )

Or override individual values::

    custom = PRE_POCKET_THRESHOLDS.replace(min_sse_entropy=1.0)
"""

from dataclasses import dataclass, replace

# Length-dependent Rg upper bound: rg <= RG_UPPER_BOUND_COEFF * length^0.4.
# Fitted to approximate the previous linear bound over lengths 150-500, but
# follows the physical scaling law for globular proteins (Rg ~ N^0.4).
# Previous linear bound: rg <= 0.029767441860465118 * length + 14.279069767441861.
RG_UPPER_BOUND_COEFF = 2.397538


@dataclass(frozen=True)
class FragmentThresholds:
    """Thresholds for fragment metric filtering.

    ``None`` means the filter is disabled for that metric.
    """

    min_plddt: float | None = 70.0
    max_plddt: float | None = None

    min_length: int | None = 150
    max_length: int | None = None

    min_rg: float | None = None
    max_rg: float | None = None

    min_sse_entropy: float | None = None
    max_sse_entropy: float | None = None

    min_repeat_score: float | None = None
    max_repeat_score: float | None = None

    min_buried_hydrophobic: float | None = None
    max_buried_hydrophobic: float | None = None

    min_hydrophobic_sasa: float | None = None
    max_hydrophobic_sasa: float | None = 0.4

    min_contact_order: float | None = None
    max_contact_order: float | None = None

    min_corrected_contact_order: float | None = 1.2
    max_corrected_contact_order: float | None = None

    min_packing_efficiency: float | None = None
    max_packing_efficiency: float | None = None

    min_corrected_packing_efficiency: float | None = 0.04
    max_corrected_packing_efficiency: float | None = None

    def replace(self, **kwargs) -> "FragmentThresholds":
        return replace(self, **kwargs)


# ── Presets ──────────────────────────────────────────────────────────────────

LIGHTWEIGHT_FRAGMENT_THRESHOLDS = FragmentThresholds(
    max_hydrophobic_sasa=None,
    min_corrected_contact_order=None,
    min_corrected_packing_efficiency=None,
)
"""Cheap-only fragment filters: min_plddt + min_length (+ the Rg length bound).

Use when fragment detection should be fast and downstream code computes the
expensive structural metrics separately for analysis."""

FRAGMENT_THRESHOLDS = FragmentThresholds()
"""Broad fragment quality filters (Pass 1)."""

PRE_POCKET_THRESHOLDS = FragmentThresholds(
    min_sse_entropy=1.2,
    min_buried_hydrophobic=0.4,
    max_hydrophobic_sasa=0.4,
    max_repeat_score=0.15,
)
"""Stricter fragment filters applied before pocket detection (Pass 2)."""
