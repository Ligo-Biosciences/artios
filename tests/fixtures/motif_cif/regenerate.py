"""Regenerate motif-CIF parity snapshots.

Every ``.cif`` under ``evaluation/inputs/motifs/`` gets parsed via
``MmcifReader.atom_array()`` — the lightweight inference-side parse —
and saved to ``tests/fixtures/motif_cif/<domain>/<stem>.npz``. The
companion parity test
(``tests/data_pipeline/test_motif_cif_parse_parity.py``) compares fresh
parses against these snapshots field-by-field.

Run from the repo root:

    uv run python -m tests.fixtures.motif_cif.regenerate

Any refactor step that changes ``atom_array()`` output will turn the
parity test red, forcing a deliberate allowlist entry or a revert.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from artios.data.parsing.mmcif import MmcifReader
from artios.utils.file_io import dump_atomarray_to_npz

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("regenerate_motif_cif")

REPO_ROOT = Path(__file__).resolve().parents[3]
MOTIF_CIF_DIR = REPO_ROOT / "evaluation" / "inputs" / "motifs"
SNAPSHOT_DIR = REPO_ROOT / "tests" / "fixtures" / "motif_cif"


def regenerate_one(cif_path: Path) -> Path:
    """Parse ``cif_path`` with ``MmcifReader.atom_array()`` and dump
    the resulting AtomArray to the sibling snapshot location."""
    rel = cif_path.relative_to(MOTIF_CIF_DIR)
    out_path = SNAPSHOT_DIR / rel.with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    atom_array = MmcifReader(cif_path).atom_array()
    dump_atomarray_to_npz(atom_array, out_path)

    size_kb = out_path.stat().st_size / 1024
    log.info(
        "  %s  (n_atoms=%d, %.1f KB) -> %s",
        rel,
        len(atom_array),
        size_kb,
        out_path.relative_to(REPO_ROOT),
    )
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only",
        nargs="*",
        default=[],
        help=(
            "Optional list of relative CIF paths under evaluation/inputs/motifs/ "
            "to regenerate (e.g. luciferase/his_arg_4_atom_dtz.cif). If omitted, every "
            "motif CIF is regenerated."
        ),
    )
    args = ap.parse_args()

    if not MOTIF_CIF_DIR.exists():
        raise SystemExit(
            f"motif CIF directory not found: {MOTIF_CIF_DIR}. "
            "Evaluation assets not checked out?"
        )

    if args.only:
        paths = [MOTIF_CIF_DIR / p for p in args.only]
        for p in paths:
            if not p.exists():
                raise SystemExit(f"no such motif CIF: {p}")
    else:
        paths = sorted(MOTIF_CIF_DIR.rglob("*.cif"))

    if not paths:
        raise SystemExit("nothing to regenerate")

    failed: list[tuple[Path, str]] = []
    for cif_path in paths:
        try:
            regenerate_one(cif_path)
        except Exception as exc:  # noqa: BLE001
            log.error("FAILED %s: %s", cif_path, exc)
            failed.append((cif_path, str(exc)))

    if failed:
        log.error("%d failure(s):", len(failed))
        for path, err in failed:
            log.error("  %s: %s", path, err)
        return 1
    log.info("regenerated %d motif-CIF snapshot(s)", len(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
