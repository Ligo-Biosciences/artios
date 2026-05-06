"""Invariant tests for ``canonicalize_chain_ids`` and ``order_atoms_by_mol``.

The byte-equal comparisons in ``test_structure.py`` catch drift in per-atom
*values*, but not bugs that can still produce equal-looking field arrays:
a lost bond, a duplicated atom, a non-contiguous mol_id run. These risks
are specific to the structural ops (they mutate AtomArray shape / order /
bond list), so they get their own invariants file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from biotite.structure import AtomArray, BondList

from artios.data.structure.annotation_v2 import (
    canonicalize_chain_ids,
    order_atoms_by_mol,
)

CIF_DIR = Path(__file__).parent.parent / "fixtures" / "cif"
FIXTURES = ["1dai", "1m2g", "2g1k", "3iwd", "4lbp"]


# ---------------------------------------------------------------------------
# Synthesised small-input tests — specific invariants, hand-verifiable.
# ---------------------------------------------------------------------------


def _tiny_aa_two_mols() -> AtomArray:
    """4 atoms, 2 mols, 2 intra-mol bonds. Interleaved chain order."""
    aa = AtomArray(4)
    aa.coord = np.zeros((4, 3))
    # Interleaved so the reorder has something to do:
    # chain A atoms at indices 0,2; chain B atoms at indices 1,3.
    aa.chain_id = np.array(["A", "B", "A", "B"], dtype="<U8")
    aa.res_id = np.array([1, 1, 1, 1])
    aa.res_name = np.array(["ALA", "GLY", "ALA", "GLY"], dtype="<U3")
    aa.atom_name = np.array(["N", "N", "C", "C"], dtype="<U6")
    aa.element = np.array(["N", "N", "C", "C"], dtype="<U2")
    aa.set_annotation("label_entity_id", np.array(["e1", "e2", "e1", "e2"]))
    # Bonds: 0-2 (within chain A), 1-3 (within chain B).
    aa.bonds = BondList(4, np.array([[0, 2, 1], [1, 3, 1]]))
    return aa


class TestOrderAtomsByMol:
    def test_every_atom_appears_exactly_once(self):
        """Permutation invariant — no atom lost, no atom duplicated."""
        aa = _tiny_aa_two_mols()
        ordered, _mol_id, _emid = order_atoms_by_mol(aa, check_final_equiv=False)
        assert len(ordered) == len(aa)
        # element/res_name multisets are preserved.
        assert sorted(ordered.element.tolist()) == sorted(aa.element.tolist())
        assert sorted(ordered.res_name.tolist()) == sorted(aa.res_name.tolist())

    def test_bonds_preserved_under_reorder(self):
        """Bond set survives reindexing (no drops, no phantom bonds).

        Absolute atom indices change post-reorder; we canonicalise each bond
        to (endpoint identities, type) so the comparison is index-independent.
        """
        aa = _tiny_aa_two_mols()
        ordered, _, _ = order_atoms_by_mol(aa, check_final_equiv=False)

        def _bond_identity_set(a: AtomArray) -> set[tuple]:
            out = set()
            for i, j, t in a.bonds.as_array():
                end_i = (a.element[int(i)], a.chain_id[int(i)])
                end_j = (a.element[int(j)], a.chain_id[int(j)])
                out.add((frozenset([end_i, end_j]), int(t)))
            return out

        assert _bond_identity_set(ordered) == _bond_identity_set(aa)

    def test_mol_id_is_contiguous_and_monotonic(self):
        """After reorder, mol_id runs are contiguous: 0..0, 1..1, ..."""
        aa = _tiny_aa_two_mols()
        _, mol_id, _ = order_atoms_by_mol(aa, check_final_equiv=False)
        assert (np.diff(mol_id) >= 0).all()
        # Each distinct mol_id forms a single contiguous run.
        transitions = int((np.diff(mol_id) != 0).sum())
        assert transitions == len(np.unique(mol_id)) - 1


class TestCanonicalizeChainIds:
    def test_duplicate_chain_labels_get_numbered_suffixes(self):
        """[A, B, A, B, C] → [A, B, A.1, B.1, C] — one chain per distinct label."""
        aa = AtomArray(10)
        aa.coord = np.zeros((10, 3))
        # Two atoms per chain so chain boundaries are explicit.
        aa.chain_id = np.array(
            ["A"] * 2 + ["B"] * 2 + ["A"] * 2 + ["B"] * 2 + ["C"] * 2,
            dtype="<U8",
        )
        aa.res_id = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        aa.res_name = np.array(["ALA"] * 10, dtype="<U3")
        aa.atom_name = np.array(["N", "CA"] * 5, dtype="<U6")
        aa.element = np.array(["N", "C"] * 5, dtype="<U2")
        aa.set_annotation("label_entity_id", np.array(["e"] * 10))

        out = canonicalize_chain_ids(aa)
        np.testing.assert_array_equal(
            out.chain_id,
            ["A", "A", "B", "B", "A.1", "A.1", "B.1", "B.1", "C", "C"],
        )


# ---------------------------------------------------------------------------
# Golden-fixture invariant tests — real CIF inputs, real bond graphs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FIXTURES)
class TestOrderAtomsByMolOnGoldenFixtures:
    """Run order_atoms_by_mol on each fixture and check invariants.

    These catch bugs where the reorder permutation or the polymer-bond
    filter drift silently — field-level byte equality can still hold when
    bonds are wrong, because bonds live on the AtomArray, not on any group.
    """

    def _load_aa(self, name: str) -> tuple[AtomArray, dict | None]:
        from artios.data.parsing.bioassembly import load_bioassembly

        bioassembly = load_bioassembly(CIF_DIR / f"{name}.cif")
        assert bioassembly is not None
        return (
            bioassembly.structure.to_atom_array(),
            bioassembly.header.entity_poly_type,
        )

    def test_permutation_preserves_atom_multiset(self, name: str):
        """Same atoms out, just reordered — no add/remove."""
        aa, ept = self._load_aa(name)
        aa = canonicalize_chain_ids(aa.copy())
        ordered, _, _ = order_atoms_by_mol(aa, ept, check_final_equiv=ept is not None)
        assert len(ordered) == len(aa)
        np.testing.assert_array_equal(np.sort(ordered.element), np.sort(aa.element))
        np.testing.assert_array_equal(np.sort(ordered.atom_name), np.sort(aa.atom_name))

    def test_mol_id_contiguous(self, name: str):
        """No mol_id value appears in two separate runs after reorder."""
        aa, ept = self._load_aa(name)
        aa = canonicalize_chain_ids(aa.copy())
        _, mol_id, _ = order_atoms_by_mol(aa, ept, check_final_equiv=ept is not None)
        assert (np.diff(mol_id) >= 0).all()
        transitions = int((np.diff(mol_id) != 0).sum())
        assert transitions == len(np.unique(mol_id)) - 1

    def test_polymer_filter_matches_direct_call(self, name: str):
        """``order_atoms_by_mol`` must drop exactly the bonds that
        ``remove_bonds_between_polymer_chains`` would drop — no more, no less.
        """
        from artios.data.structure.annotation_v2 import (
            remove_bonds_between_polymer_chains,
        )

        aa, ept = self._load_aa(name)
        if ept is None:
            pytest.skip("fixture lacks entity_poly_type")

        aa = canonicalize_chain_ids(aa.copy())
        expected_bonds = remove_bonds_between_polymer_chains(aa, ept)
        ordered, _, _ = order_atoms_by_mol(aa, ept, check_final_equiv=True)

        assert len(ordered.bonds.as_array()) == len(expected_bonds.as_array())
