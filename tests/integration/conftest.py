"""Shared fixtures and helpers for integration (parity-golden) tests.

Provides:

- The ``--regenerate-golden`` CLI flag and matching ``regenerate_golden``
  fixture, used to rewrite snapshot fixtures after an intentional change.
- ``full_context_to_flat_tensors`` / ``assert_full_context_matches`` — turn
  a :class:`FullContext` into a flat ``dict[str, np.ndarray]`` keyed
  ``"<subcontext>.<field>"``, suitable for ``np.savez`` round-tripping.

See ``tests/integration/CLAUDE.md`` for the strategy.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# CLI option + fixture
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--regenerate-golden",
        action="store_true",
        default=False,
        help=(
            "Rewrite golden parity fixtures under tests/fixtures/integration/ "
            "instead of asserting against them. Use after an intentional "
            "change to the data pipeline — then inspect the diff."
        ),
    )


@pytest.fixture
def regenerate_golden(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--regenerate-golden"))


# ---------------------------------------------------------------------------
# FullContext ↔ flat-tensor-dict helpers
# ---------------------------------------------------------------------------


def full_context_to_flat_tensors(ctx: Any) -> dict[str, np.ndarray]:
    """Flatten the tensor fields of a ``FullContext`` into a flat dict.

    Walks each sub-context dataclass field; keeps tensor fields, skips
    everything else (strings, lists, AtomArray, TokenArray). Keys are
    ``"<sub>.<field>"`` for downstream field-level diffing.
    """
    out: dict[str, np.ndarray] = {}
    for sub in fields(ctx):
        val = getattr(ctx, sub.name)
        if val is None or not is_dataclass(val):
            continue
        for f in fields(val):
            inner = getattr(val, f.name)
            if torch.is_tensor(inner):
                out[f"{sub.name}.{f.name}"] = inner.detach().cpu().numpy()
    return out


def dump_full_context(ctx: Any, path: Path) -> None:
    """Save a FullContext's tensor fields as an ``.npz`` golden fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **full_context_to_flat_tensors(ctx))


def assert_full_context_matches(
    ctx: Any,
    golden_path: Path,
    *,
    float_atol: float = 1e-5,
) -> None:
    """Assert ctx's tensor fields match the golden .npz field-for-field.

    Fails loudly on missing or extra keys so a field added/removed by the
    refactor is named in the assertion message — not silently skipped.
    """
    actual = full_context_to_flat_tensors(ctx)
    with np.load(golden_path) as golden:
        golden_keys = set(golden.files)
        actual_keys = set(actual.keys())
        missing = sorted(golden_keys - actual_keys)
        extra = sorted(actual_keys - golden_keys)
        assert (
            not missing
        ), f"{golden_path.name}: golden has fields not in actual: {missing}"
        assert (
            not extra
        ), f"{golden_path.name}: actual has fields not in golden: {extra}"

        for key in sorted(golden_keys):
            g = golden[key]
            a = actual[key]
            assert a.shape == g.shape, f"{key}: shape {a.shape} != golden {g.shape}"
            if np.issubdtype(g.dtype, np.floating):
                np.testing.assert_allclose(a, g, atol=float_atol, err_msg=key)
            else:
                np.testing.assert_array_equal(a, g, err_msg=key)
