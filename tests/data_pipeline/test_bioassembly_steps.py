"""Unit tests for the ``bioassembly`` pipeline step functions.

The goldens in ``test_load_bioassembly.py`` pin end-to-end parity;
these tests pin the small, non-obvious invariants each step owns — so
a refactor that preserves the end-to-end output but breaks an
intermediate contract still lights up.
"""

from __future__ import annotations

import numpy as np
from biotite.structure import AtomArray

from artios.data.parsing.bioassembly import (
    _clean_and_repair,
    _count_protein_chains,
)


def _build_aa(
    *,
    chain_id: list[str],
    res_id: list[int],
    res_name: list[str],
    atom_name: list[str],
    element: list[str],
    label_entity_id: list[str],
    label_seq_id: list[str] | None = None,
    hetero: list[bool] | None = None,
) -> AtomArray:
    """Minimal AtomArray builder. Mirrors ``test_annotation_groups._build_aa``
    with the extra ``label_seq_id`` column filters in ``clean_structure`` read.
    When unset, defaults to ``str(res_id)`` so polymer filters' int-casts work.
    """
    n = len(chain_id)
    aa = AtomArray(n)
    aa.coord = np.zeros((n, 3), dtype=np.float32)
    aa.chain_id = np.array(chain_id, dtype="<U8")
    aa.res_id = np.array(res_id, dtype=int)
    aa.res_name = np.array(res_name, dtype="<U3")
    aa.atom_name = np.array(atom_name, dtype="<U6")
    aa.element = np.array(element, dtype="<U2")
    aa.set_annotation("label_entity_id", np.array(label_entity_id))
    # label_seq_id / auth_seq_id are string columns that downstream filters
    # and add_missing_atoms_and_residues read. Default to str(res_id).
    default_seq_ids = np.array(
        label_seq_id if label_seq_id is not None else [str(r) for r in res_id]
    )
    aa.set_annotation("label_seq_id", default_seq_ids)
    aa.set_annotation("auth_seq_id", default_seq_ids.copy())
    if hetero is not None:
        aa.hetero = np.array(hetero, dtype=bool)
    return aa


# ---------------------------------------------------------------------------
# _clean_and_repair
# ---------------------------------------------------------------------------


class TestCleanAndRepair:
    """``_clean_and_repair`` composes four already-tested primitives
    (``clean_structure``, ``fix_arginine``, ``mse_to_met``,
    ``add_missing_atoms_and_residues``, ``Filter.remove_element_X``).
    The contract this step *owns* — not duplicated by the callees — is
    the empty-after-clean short-circuit: a refactor that drops the
    ``None``-passthrough would mask an empty array downstream where
    filters crash with less actionable errors. That's the one test worth
    keeping at this seam.
    """

    def test_returns_none_when_all_atoms_dropped(self):
        """Water-only input → ``clean_structure`` removes everything →
        ``_clean_and_repair`` must propagate ``None`` so ``load_bioassembly``
        reports the skip without crashing in add_missing_atoms_and_residues.
        """
        aa = _build_aa(
            chain_id=["W"] * 3,
            res_id=[1, 2, 3],
            res_name=["HOH", "HOH", "HOH"],
            atom_name=["O"] * 3,
            element=["O"] * 3,
            label_entity_id=["water"] * 3,
            hetero=[True] * 3,
        )

        out = _clean_and_repair(
            aa,
            entity_poly_type={},
            poly_res_names_fn=lambda _aa: {},
        )

        assert out is None


# ---------------------------------------------------------------------------
# _count_protein_chains
# ---------------------------------------------------------------------------


class TestCountProteinChains:
    def test_counts_polypeptide_L_chains_only(self):
        """Two polypeptide(L) chains + one polyribonucleotide chain + one
        ligand → returns 2. Tests the entity-type discrimination that
        downstream sampling weights depend on.
        """
        aa = _build_aa(
            chain_id=["A", "A", "B", "B", "C", "C", "D"],
            res_id=[1, 1, 1, 1, 1, 1, 1],
            res_name=["ALA", "ALA", "ALA", "ALA", "A", "A", "NAG"],
            atom_name=["N", "CA"] * 3 + ["C1"],
            element=["N", "C"] * 3 + ["C"],
            # Two protein chains (A,B) share polypeptide entity; RNA
            # chain (C) has a polyribonucleotide entity; ligand (D) has
            # a non-polymer entity.
            label_entity_id=["p1", "p1", "p1", "p1", "r1", "r1", "l1"],
        )
        entity_poly_type = {
            "p1": "polypeptide(L)",
            "r1": "polyribonucleotide",
            # l1 intentionally absent — non-polymer ligand.
        }
        assert _count_protein_chains(aa, entity_poly_type=entity_poly_type) == 2

    def test_counts_polypeptide_D_as_protein(self):
        """Both polypeptide(L) and polypeptide(D) contain 'polypeptide'
        — the substring match is load-bearing; switching to an exact
        equality check would silently miscount D-peptide entries.
        """
        aa = _build_aa(
            chain_id=["A", "A"],
            res_id=[1, 1],
            res_name=["ALA", "ALA"],
            atom_name=["N", "CA"],
            element=["N", "C"],
            label_entity_id=["e1", "e1"],
        )
        assert _count_protein_chains(aa, entity_poly_type={"e1": "polypeptide(D)"}) == 1

    def test_returns_zero_when_no_polymer_entities(self):
        """Nucleotide-only and ligand-only bioassemblies must return 0 —
        downstream protein-centric logic (training-weight `is_protein`
        checks) must not accidentally include them.
        """
        aa = _build_aa(
            chain_id=["A"],
            res_id=[1],
            res_name=["A"],
            atom_name=["C1'"],
            element=["C"],
            label_entity_id=["r1"],
        )
        assert (
            _count_protein_chains(aa, entity_poly_type={"r1": "polyribonucleotide"})
            == 0
        )
