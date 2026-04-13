"""Version-controlled default thresholds for fragment and pocket filtering.

Two presets:

    FRAGMENT_THRESHOLDS     — broad fragment quality filters (Pass 1).
    PRE_POCKET_THRESHOLDS   — stricter fragment filters applied before pocket detection (Pass 2).

Import and use directly::

    from artios.data.engineering.pipeline.thresholds import FRAGMENT_THRESHOLDS, PRE_POCKET_THRESHOLDS

Or override individual values::

    custom = PRE_POCKET_THRESHOLDS.replace(min_sse_entropy=1.0)
"""

from dataclasses import dataclass, replace


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

FRAGMENT_THRESHOLDS = FragmentThresholds()
"""Broad fragment quality filters (Pass 1)."""

PRE_POCKET_THRESHOLDS = FragmentThresholds(
    min_sse_entropy=1.2,
    min_buried_hydrophobic=0.4,
    max_hydrophobic_sasa=0.4,
)
"""Stricter fragment filters applied before pocket detection (Pass 2)."""
