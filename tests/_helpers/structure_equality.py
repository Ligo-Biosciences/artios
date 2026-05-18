"""Field-by-field equality check for two ``AnnotatedStructure`` instances.

Used by the CIF-parse parity tests to pin current pipeline output against
golden fixtures during the ``cif.py`` refactor. The assertion model is:

- Every field of every typed group must match.
- Float arrays compare with ``np.allclose(atol=...)``.
- Int arrays / bool masks compare with exact equality.
- String arrays (numpy ``U*`` dtype) compare with exact equality **after**
  widening both sides to the wider dtype, so a harmless ``U7 → U16`` bump
  in ``type_masks.mol_type`` does not flag.
- Bond lists compare as sorted ``(from, to, bond_type)`` triple sets.
- ``AtomArray`` annotations compare identically to the typed groups.

An ``allowlist`` dict lets callers mark specific fields as expected to
diverge with a human-readable reason. Fields in the allowlist log the diff
and pass; every other mismatch fails. Field paths use ``"<group>.<field>"``
format (e.g. ``"atoms.coord"``, ``"type_masks.mol_type"``).
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
from biotite.structure import AtomArray, BondList

from artios.data.structure import AnnotatedStructure
from artios.data.structure.annotation import _AnnotationGroup

log = logging.getLogger(__name__)


class StructureDiff(AssertionError):
    """Raised when two ``AnnotatedStructure`` instances disagree on a
    field that is not in the allowlist."""


def assert_atom_arrays_equal(
    actual: AtomArray,
    expected: AtomArray,
    *,
    allowlist: Mapping[str, str] | None = None,
    atol: float = 1e-5,
) -> None:
    """Assert two raw ``AtomArray`` instances match field-by-field.

    Used by the motif-CIF parity test, where the parser output is a raw
    ``AtomArray`` rather than a full ``AnnotatedStructure``.
    """
    allowlist = dict(allowlist or {})
    problems: list[str] = []
    _compare_atom_array(
        actual,
        expected,
        path_prefix="atoms",
        allowlist=allowlist,
        atol=atol,
        problems=problems,
    )
    if problems:
        raise StructureDiff(
            f"{len(problems)} field(s) mismatched:\n  - " + "\n  - ".join(problems)
        )


def assert_structures_equal(
    actual: AnnotatedStructure,
    expected: AnnotatedStructure,
    *,
    allowlist: Mapping[str, str] | None = None,
    atol: float = 1e-5,
) -> None:
    """Assert ``actual`` and ``expected`` match field-by-field.

    Args:
        actual: Structure produced by the current-under-test pipeline.
        expected: Golden structure loaded from a committed ``.npz``.
        allowlist: ``{field_path: reason}`` for fields that may differ.
        atol: Absolute tolerance for float array comparisons.

    Raises:
        StructureDiff: on any mismatch whose field_path is not in
            ``allowlist``. The exception message enumerates every failing
            field (not just the first) so the caller sees the full diff in
            one shot.
    """
    allowlist = dict(allowlist or {})
    problems: list[str] = []

    _compare_atom_array(
        actual.atoms,
        expected.atoms,
        path_prefix="atoms",
        allowlist=allowlist,
        atol=atol,
        problems=problems,
    )

    for field in dataclasses.fields(actual):
        name = field.name
        if name == "atoms":
            continue
        actual_group = getattr(actual, name)
        expected_group = getattr(expected, name)
        if actual_group is None and expected_group is None:
            continue
        if actual_group is None or expected_group is None:
            _register(
                problems,
                allowlist,
                f"{name}",
                f"one side is None (actual={actual_group is None}, "
                f"expected={expected_group is None})",
            )
            continue
        if isinstance(actual_group, _AnnotationGroup):
            _compare_group(
                actual_group,
                expected_group,
                path_prefix=name,
                allowlist=allowlist,
                atol=atol,
                problems=problems,
            )
        else:
            _compare_dataclass(
                actual_group,
                expected_group,
                path_prefix=name,
                allowlist=allowlist,
                atol=atol,
                problems=problems,
            )

    if problems:
        raise StructureDiff(
            f"{len(problems)} field(s) mismatched:\n  - " + "\n  - ".join(problems)
        )


def _compare_atom_array(
    actual: AtomArray,
    expected: AtomArray,
    *,
    path_prefix: str,
    allowlist: Mapping[str, str],
    atol: float,
    problems: list[str],
) -> None:
    if len(actual) != len(expected):
        _register(
            problems,
            allowlist,
            f"{path_prefix}.len",
            f"len(actual)={len(actual)} != len(expected)={len(expected)}",
        )
        return
    _compare_array(
        actual.coord,
        expected.coord,
        path=f"{path_prefix}.coord",
        allowlist=allowlist,
        atol=atol,
        problems=problems,
    )
    _compare_bonds(
        actual.bonds,
        expected.bonds,
        path=f"{path_prefix}.bonds",
        allowlist=allowlist,
        problems=problems,
    )
    actual_cats = set(actual.get_annotation_categories())
    expected_cats = set(expected.get_annotation_categories())
    missing = expected_cats - actual_cats
    extra = actual_cats - expected_cats
    if missing:
        _register(
            problems,
            allowlist,
            f"{path_prefix}.annotations",
            f"missing categories on actual: {sorted(missing)}",
        )
    if extra:
        _register(
            problems,
            allowlist,
            f"{path_prefix}.annotations",
            f"extra categories on actual: {sorted(extra)}",
        )
    for cat in actual_cats & expected_cats:
        _compare_array(
            actual.get_annotation(cat),
            expected.get_annotation(cat),
            path=f"{path_prefix}.{cat}",
            allowlist=allowlist,
            atol=atol,
            problems=problems,
        )


def _compare_group(
    actual: _AnnotationGroup,
    expected: _AnnotationGroup,
    *,
    path_prefix: str,
    allowlist: Mapping[str, str],
    atol: float,
    problems: list[str],
) -> None:
    if type(actual) is not type(expected):
        _register(
            problems,
            allowlist,
            path_prefix,
            f"group types differ: {type(actual).__name__} "
            f"vs {type(expected).__name__}",
        )
        return
    for field in dataclasses.fields(actual):
        _compare_value(
            getattr(actual, field.name),
            getattr(expected, field.name),
            path=f"{path_prefix}.{field.name}",
            allowlist=allowlist,
            atol=atol,
            problems=problems,
        )


def _compare_dataclass(
    actual: Any,
    expected: Any,
    *,
    path_prefix: str,
    allowlist: Mapping[str, str],
    atol: float,
    problems: list[str],
) -> None:
    if not dataclasses.is_dataclass(actual):
        if actual != expected:
            _register(problems, allowlist, path_prefix, f"{actual!r} != {expected!r}")
        return
    if type(actual) is not type(expected):
        _register(
            problems,
            allowlist,
            path_prefix,
            f"dataclass types differ: {type(actual).__name__} "
            f"vs {type(expected).__name__}",
        )
        return
    for field in dataclasses.fields(actual):
        _compare_value(
            getattr(actual, field.name),
            getattr(expected, field.name),
            path=f"{path_prefix}.{field.name}",
            allowlist=allowlist,
            atol=atol,
            problems=problems,
        )


def _compare_value(
    actual: Any,
    expected: Any,
    *,
    path: str,
    allowlist: Mapping[str, str],
    atol: float,
    problems: list[str],
) -> None:
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        _compare_array(
            actual,
            expected,
            path=path,
            allowlist=allowlist,
            atol=atol,
            problems=problems,
        )
        return
    if isinstance(actual, BondList) or isinstance(expected, BondList):
        _compare_bonds(
            actual, expected, path=path, allowlist=allowlist, problems=problems
        )
        return
    if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            _register(
                problems,
                allowlist,
                path,
                f"length {len(actual)} != {len(expected)}",
            )
            return
        for i, (a, e) in enumerate(zip(actual, expected)):
            _compare_value(
                a,
                e,
                path=f"{path}[{i}]",
                allowlist=allowlist,
                atol=atol,
                problems=problems,
            )
        return
    if actual != expected:
        _register(problems, allowlist, path, f"{actual!r} != {expected!r}")


def _compare_array(
    actual: np.ndarray | None,
    expected: np.ndarray | None,
    *,
    path: str,
    allowlist: Mapping[str, str],
    atol: float,
    problems: list[str],
) -> None:
    if actual is None and expected is None:
        return
    if actual is None or expected is None:
        _register(
            problems,
            allowlist,
            path,
            f"one side is None (actual={actual is None}, "
            f"expected={expected is None})",
        )
        return
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        _register(
            problems,
            allowlist,
            path,
            f"shape {actual.shape} != {expected.shape}",
        )
        return
    if actual.dtype.kind == "f" or expected.dtype.kind == "f":
        if not np.allclose(actual, expected, atol=atol, equal_nan=True):
            max_diff = float(np.nanmax(np.abs(actual - expected)))
            _register(
                problems,
                allowlist,
                path,
                f"max abs diff {max_diff:.2e} > atol {atol:.2e}",
            )
        return
    if actual.dtype.kind in ("U", "S", "O"):
        # Widen both sides to the wider dtype before comparing so a
        # deliberate U7 -> U16 bump (or equivalent) does not flag.
        a_str = actual.astype(str)
        e_str = expected.astype(str)
        if not np.array_equal(a_str, e_str):
            diff_idx = np.where(a_str != e_str)[0]
            preview = [(int(i), a_str.flat[i], e_str.flat[i]) for i in diff_idx[:3]]
            _register(
                problems,
                allowlist,
                path,
                f"{len(diff_idx)} elements differ; first few (idx, actual, expected): {preview}",
            )
        return
    if not np.array_equal(actual, expected):
        diff_idx = np.where(actual != expected)
        count = int(np.sum(actual != expected))
        first = (
            (int(diff_idx[0][0]), actual[diff_idx][0], expected[diff_idx][0])
            if count
            else None
        )
        _register(
            problems,
            allowlist,
            path,
            f"{count} elements differ; first (idx, actual, expected): {first}",
        )


def _compare_bonds(
    actual: BondList | None,
    expected: BondList | None,
    *,
    path: str,
    allowlist: Mapping[str, str],
    problems: list[str],
) -> None:
    if actual is None and expected is None:
        return
    if actual is None or expected is None:
        _register(
            problems,
            allowlist,
            path,
            f"one side is None (actual={actual is None}, "
            f"expected={expected is None})",
        )
        return
    a = actual.as_array()
    e = expected.as_array()
    # Canonicalise each bond to (min, max, type) so orientation differences
    # don't cause false negatives, then sort to remove ordering dependence.
    a_sorted = _canonical_bonds(a)
    e_sorted = _canonical_bonds(e)
    if a_sorted.shape != e_sorted.shape or not np.array_equal(a_sorted, e_sorted):
        _register(
            problems,
            allowlist,
            path,
            f"bond lists differ (actual n={len(a)}, expected n={len(e)})",
        )


def _canonical_bonds(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo = np.minimum(arr[:, 0], arr[:, 1])
    hi = np.maximum(arr[:, 0], arr[:, 1])
    canon = np.stack([lo, hi, arr[:, 2]], axis=1)
    order = np.lexsort((canon[:, 2], canon[:, 1], canon[:, 0]))
    return canon[order]


def _register(
    problems: list[str],
    allowlist: Mapping[str, str],
    path: str,
    detail: str,
) -> None:
    if path in allowlist:
        log.info(
            "allowlisted diff at %s: %s (reason: %s)", path, detail, allowlist[path]
        )
        return
    problems.append(f"{path}: {detail}")
