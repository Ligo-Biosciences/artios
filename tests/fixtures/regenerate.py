"""Regenerate golden AnnotatedStructure .npz fixtures from source CIFs.

Run from the repo root:

    uv run python -m tests.fixtures.regenerate

Source CIFs live in ``tests/fixtures/cif/`` (committed). Outputs are
written to ``tests/fixtures/structures/<pdb_id>.npz``. The preset list
below drives the default run; arbitrary pairs can be passed via
``--pairs path.cif:name ...``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from artios.data.parsing.cif import MMCIFParser
from artios.data.structure import AnnotatedStructure

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("regenerate")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent
STRUCTURES_DIR = FIXTURES_DIR / "structures"
CIF_DIR = FIXTURES_DIR / "cif"


# (cif filename in CIF_DIR, output name without .npz, one-line description)
PRESET: list[tuple[str, str, str]] = [
    ("2g1k.cif", "2g1k", "smallest: 2 chains, 188 tokens, 1318 atoms"),
    ("1m2g.cif", "1m2g", "3 chains, 286 tokens, ligand + cofactor"),
    ("3iwd.cif", "3iwd", "6 chains, 310 tokens, multiple cofactor chains"),
    ("1dai.cif", "1dai", "4 chains, 480 tokens, medium enzyme"),
    ("4lbp.cif", "4lbp", "multi-chain: 8 chains, 440 tokens"),
]


def regenerate_one(cif_path: Path, out_name: str) -> Path:
    """Parse a CIF, build an AnnotatedStructure, save to structures/."""
    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STRUCTURES_DIR / f"{out_name}.npz"

    log.info("parsing %s", cif_path.name)
    parser = MMCIFParser(mmcif_file=cif_path)
    bioassembly = parser.get_bioassembly()
    atom_array = bioassembly["atom_array"]
    if atom_array is None:
        raise RuntimeError(f"{cif_path.name}: get_bioassembly returned no atom_array")

    atom_array.set_annotation("resolution", [parser.resolution] * len(atom_array))
    structure = AnnotatedStructure.from_atom_array(atom_array)

    log.info(
        "  n_atoms=%d n_tokens=%d n_chains=%d resolution=%.2f",
        len(structure.atoms),
        structure.grouping.n_tokens,
        len(set(structure.atoms.chain_id)),
        structure.resolution,
    )

    structure.save(out_path)
    size_kb = out_path.stat().st_size / 1024
    log.info("  saved %s (%.1f KB)", out_path.relative_to(REPO_ROOT), size_kb)

    # Round-trip check: load and compare basic shapes.
    loaded = AnnotatedStructure.load(out_path)
    assert len(loaded.atoms) == len(structure.atoms), "atom count mismatch on reload"
    assert (
        loaded.grouping.n_tokens == structure.grouping.n_tokens
    ), "token count mismatch"
    log.info("  round-trip OK")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pairs",
        nargs="*",
        default=[],
        help="CIF:name pairs (e.g. pdb-samples/enzyme/4lbp.cif:4lbp). "
        "If omitted, uses the PRESET list.",
    )
    args = ap.parse_args()

    if args.pairs:
        pairs: list[tuple[Path, str]] = []
        for spec in args.pairs:
            cif_str, _, name = spec.partition(":")
            if not name:
                raise SystemExit(f"bad --pairs entry '{spec}' (expected path:name)")
            pairs.append((Path(cif_str), name))
    else:
        pairs = [(CIF_DIR / cif_name, out_name) for cif_name, out_name, _ in PRESET]

    if not pairs:
        raise SystemExit("no pairs to process (PRESET empty and no --pairs given)")

    failed: list[tuple[str, str]] = []
    for cif_path, out_name in pairs:
        try:
            regenerate_one(cif_path, out_name)
        except Exception as e:  # noqa: BLE001
            log.error("FAILED %s (%s): %s", cif_path.name, out_name, e)
            failed.append((cif_path.name, str(e)))

    if failed:
        log.error("%d failures", len(failed))
        for name, err in failed:
            log.error("  %s: %s", name, err)
        return 1
    log.info("all %d fixtures regenerated", len(pairs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
