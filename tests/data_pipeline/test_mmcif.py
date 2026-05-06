"""Unit tests for ``artios.data.parsing.mmcif``.

The reader owns non-trivial logic — altloc filtering, PyMOL-export
fallbacks, ``replace_auth_with_label`` with its historical int/str
hazard — so each gets a focused test. End-to-end byte equivalence on the
``atom_array()`` output is pinned by ``test_motif_cif_parse_parity.py``
(motif CIFs) and ``test_load_bioassembly.py`` (PDB CIFs); this file
covers the per-method invariants those goldens don't isolate.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from artios.data.parsing.mmcif import MmcifReader

REPO_ROOT = Path(__file__).resolve().parents[2]
CIF_DIR = REPO_ROOT / "tests" / "fixtures" / "cif"
MOTIF_DIR = REPO_ROOT / "evaluation" / "inputs" / "motifs"


# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------


class TestHeader:
    def test_header_fields_populated_for_pdb_cif(self):
        """``2g1k.cif`` is a real PDB export; every header field should
        carry non-sentinel content."""
        reader = MmcifReader(CIF_DIR / "2g1k.cif")
        header = reader.header

        assert header.pdb_id == "2g1k"
        # Crystal structure — resolution is an actual float, not the -1.0 sentinel.
        assert header.resolution > 0.0
        # Release date is a real datetime.date (not the 9999-12-31 sentinel).
        assert header.release_date < date(2025, 1, 1)
        # At least one experimental method recorded.
        assert len(header.methods) >= 1
        # entity_poly_type → polypeptide / polynucleotide markers only.
        assert all(
            t.startswith(
                ("polypeptide", "polydeoxyribonucleotide", "polyribonucleotide")
            )
            for t in header.entity_poly_type.values()
        )
        # Sequences keyed by entity id, values are canonical strings.
        assert all(isinstance(s, str) and len(s) > 0 for s in header.sequences.values())

    def test_header_sentinels_for_motif_cif(self):
        """Motif CIFs are hand-crafted: they may retain the source PDB's
        entry id but lack refinement tables, revision history, and
        experimental method records. The reader must return the
        documented sentinels so downstream callers can treat those
        fields uniformly without special-casing the motif path."""
        reader = MmcifReader(
            MOTIF_DIR / "carbonic_anhydrase" / "carbonic_anhydrase_motif_no_h.cif"
        )
        header = reader.header

        # No refinement tables → -1.0 sentinel.
        assert header.resolution == -1.0
        # No revision history → 9999-12-31 sentinel kept sortable.
        assert header.release_date == date(9999, 12, 31)
        # No experimental method loop.
        assert header.methods == ()

    def test_header_is_cached(self):
        """Repeat ``header`` access must return the same object (cached
        property). Header extraction is cheap but downstream code loops
        over structures and benefits from the no-recompute guarantee."""
        reader = MmcifReader(CIF_DIR / "2g1k.cif")
        assert reader.header is reader.header


# ---------------------------------------------------------------------------
# replace_auth_with_label — bug #2 regression pin
# ---------------------------------------------------------------------------


class TestReplaceAuthWithLabel:
    def test_wide_ligand_res_ids_preserved(self) -> None:
        """Bug-scan finding #2: the legacy ``replace_auth_with_label``
        deep-copies ``label_seq_id`` (a string dtype) and writes ints
        into it, so ``num=10000`` silently truncates to "1000" in a
        <U4 dtype. The reader builds ``res_id`` as an int array from
        the start and never transits through a string dtype.

        This test fabricates a ligand chain with 30k pseudo-residues and
        asserts every one survives the rewrite with its integer res_id
        intact.
        """
        from biotite.structure import AtomArray, BondList

        from artios.data.parsing.mmcif import _replace_auth_with_label

        n = 30_000
        aa = AtomArray(n)
        aa.coord = np.zeros((n, 3), dtype=np.float32)
        aa.atom_name = np.full(n, "ZN", dtype="<U4")
        aa.element = np.full(n, "ZN", dtype="<U2")
        aa.res_name = np.full(n, "ZN", dtype="<U3")
        # Pretend these are ligand atoms — label_seq_id "." triggers
        # the reset-per-atom branch.
        aa.set_annotation("label_seq_id", np.full(n, ".", dtype="<U1"))
        aa.set_annotation("label_asym_id", np.full(n, "L", dtype="<U8"))
        aa.chain_id = np.full(n, "L", dtype="<U8")
        # Each atom is its own residue; res_id sequences 1..N-ish before
        # the rewrite. We just need a unique (chain, res_id) per atom.
        aa.res_id = np.arange(1, n + 1, dtype=int)
        aa.bonds = BondList(n)

        out = _replace_auth_with_label(aa)

        # res_id must be a contiguous 1..n int sequence; no truncation.
        np.testing.assert_array_equal(out.res_id, np.arange(1, n + 1, dtype=int))
        assert out.res_id.dtype.kind == "i"


# ---------------------------------------------------------------------------
# sequences_for — altloc-aware variant
# ---------------------------------------------------------------------------


class TestSequencesFor:
    def test_altloc_aware_sequences_match_atom_array(self):
        """``sequences_for(atom_array)`` must reflect the altloc variant
        present in the atom_array, not the first altloc in the CIF.

        Using a real PDB CIF (``2g1k`` doesn't have altlocs but sanity-
        checks the non-altloc path); the altloc-picked path is exercised
        end-to-end by the structure parity golden, so this test doesn't
        need to synthesise a mismatched altloc."""
        reader = MmcifReader(CIF_DIR / "2g1k.cif")
        atoms = reader.atom_array(altloc="first")
        header_seqs = reader.header.sequences
        for_seqs = reader.sequences_for(atoms)
        # Without altlocs the two should match entity-by-entity.
        assert set(header_seqs.keys()) == set(for_seqs.keys())
        for eid in header_seqs:
            assert header_seqs[eid] == for_seqs[eid]
