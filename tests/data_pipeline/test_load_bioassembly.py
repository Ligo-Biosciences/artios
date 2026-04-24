"""Parity tests for ``bioassembly.load_bioassembly`` against committed goldens.

Phase 2 of the ``get_bioassembly`` decomposition. Runs the new
``load_bioassembly`` free function on the same five CIF fixtures the
legacy-path parity test uses, and asserts the produced
``Bioassembly.structure`` matches the committed ``.npz`` goldens
field-by-field.

Once every caller is migrated (Phase 4) and ``MMCIFParser.get_bioassembly``
is deleted (Phase 5), the legacy-path parity test goes away and this
file becomes the canonical CIF-parse regression.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from artios.data.parsing.bioassembly import Bioassembly, load_bioassembly
from artios.data.structure import AnnotatedStructure
from tests._helpers.structure_equality import assert_structures_equal

REPO_ROOT = Path(__file__).resolve().parents[2]
CIF_DIR = REPO_ROOT / "tests" / "fixtures" / "cif"
GOLDEN_DIR = REPO_ROOT / "tests" / "fixtures" / "structures"

# Same five fixtures as test_cif_parse_parity.py. Keep in sync.
PARITY_CASES: list[tuple[str, str]] = [
    ("2g1k.cif", "2g1k"),
    ("1m2g.cif", "1m2g"),
    ("3iwd.cif", "3iwd"),
    ("1dai.cif", "1dai"),
    ("4lbp.cif", "4lbp"),
]


@pytest.mark.parametrize("cif_name,pdb_id", PARITY_CASES)
def test_load_bioassembly_matches_golden(cif_name: str, pdb_id: str) -> None:
    """``load_bioassembly(cif_path).structure`` matches the golden NPZ."""
    cif_path = CIF_DIR / cif_name
    golden_path = GOLDEN_DIR / f"{pdb_id}.npz"
    if not cif_path.exists():
        pytest.skip(f"CIF fixture missing: {cif_path}")
    if not golden_path.exists():
        pytest.skip(f"golden fixture missing: {golden_path}")

    bioassembly = load_bioassembly(cif_path)
    assert bioassembly is not None, f"{cif_name}: load_bioassembly returned None"
    assert isinstance(bioassembly, Bioassembly)
    expected = AnnotatedStructure.load(golden_path)
    assert_structures_equal(bioassembly.structure, expected)


@pytest.mark.parametrize("cif_name,pdb_id", PARITY_CASES)
def test_load_bioassembly_header_fields(cif_name: str, pdb_id: str) -> None:
    """Header carries pdb_id and entity_poly_type — the two fields downstream
    indexing/featurization actually read out of the legacy dict."""
    cif_path = CIF_DIR / cif_name
    if not cif_path.exists():
        pytest.skip(f"CIF fixture missing: {cif_path}")

    bioassembly = load_bioassembly(cif_path)
    assert bioassembly is not None
    assert bioassembly.header.pdb_id == pdb_id
    # entity_poly_type must have at least one polypeptide entity for all
    # five fixtures (verified by inspection of the fixture shapes).
    assert any("polypeptide" in t for t in bioassembly.header.entity_poly_type.values())


def test_load_bioassembly_returns_none_when_assembly_missing() -> None:
    """Requesting an assembly_id absent from the CIF yields ``None``.

    ``num_assembly_polymer_chains`` returns 0 for a non-matching
    assembly_id, triggering the early skip branch. This pins that
    semantic — callers rely on the None signal to drop unusable samples
    instead of crashing mid-pipeline.
    """
    cif_path = CIF_DIR / "2g1k.cif"
    if not cif_path.exists():
        pytest.skip(f"CIF fixture missing: {cif_path}")

    assert load_bioassembly(cif_path, assembly_id="99") is None


def test_load_bioassembly_returns_none_when_max_assembly_chains_exceeded() -> None:
    """Pinning the other skip branch — oversized assemblies return None."""
    cif_path = CIF_DIR / "4lbp.cif"  # 8 chains — so 1 chain cap is sure-to-trip
    if not cif_path.exists():
        pytest.skip(f"CIF fixture missing: {cif_path}")

    assert load_bioassembly(cif_path, max_assembly_chains=1) is None
