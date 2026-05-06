"""Per-group ``compute()`` unit tests.

Each group's ``compute`` classmethod gets a focused test that synthesises a
minimal AtomArray, calls compute, and asserts specific field values. These
tests complement ``test_structure.py::TestGoldenRoundTrip`` (which pins
whole-pipeline output against goldens) by catching per-group logic drift
against hand-computed expectations.

Shared helper ``_build_aa`` keeps the synthesised arrays honest — each test
constructs exactly the atoms it needs and declares every annotation the
compute() under test reads.
"""

from __future__ import annotations

import numpy as np
import pytest
from biotite.structure import AtomArray

from artios.data.structure.annotation_v2 import (
    DerivedAnnotations,
    EntityIds,
    MoleculeIds,
    ReferenceConformer,
    TrainingAnnotations,
    TypeMasks,
    populate_ccd_reference,
)


def _build_aa(
    *,
    chain_id: list[str],
    res_id: list[int],
    res_name: list[str],
    atom_name: list[str],
    element: list[str],
    label_entity_id: list[str],
    is_resolved: list[bool] | None = None,
) -> AtomArray:
    """Construct a minimal AtomArray with the annotations group computes read."""
    n = len(chain_id)
    assert len(res_id) == n and len(res_name) == n
    assert len(atom_name) == n and len(element) == n
    assert len(label_entity_id) == n
    aa = AtomArray(n)
    aa.coord = np.zeros((n, 3), dtype=np.float32)
    aa.chain_id = np.array(chain_id, dtype="<U8")
    aa.res_id = np.array(res_id, dtype=int)
    aa.res_name = np.array(res_name, dtype="<U3")
    aa.atom_name = np.array(atom_name, dtype="<U6")
    aa.element = np.array(element, dtype="<U2")
    aa.set_annotation("label_entity_id", np.array(label_entity_id))
    if is_resolved is not None:
        aa.set_annotation("is_resolved", np.array(is_resolved, dtype=bool))
    return aa


# ---------------------------------------------------------------------------
# EntityIds
# ---------------------------------------------------------------------------


class TestEntityIdsCompute:
    def test_ids_distinct_per_chain_shared_per_entity(self):
        """asym_id counts chains; entity_id shared across copies; sym_id indexes copies."""
        # 5 chains (2 atoms each): A, B, A.1, B.1, C
        # Entities: A and A.1 share e1, B and B.1 share e2, C is e3.
        aa = _build_aa(
            chain_id=["A", "A", "B", "B", "A.1", "A.1", "B.1", "B.1", "C", "C"],
            res_id=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            res_name=["ALA"] * 10,
            atom_name=["N", "CA"] * 5,
            element=["N", "C"] * 5,
            label_entity_id=[
                "e1",
                "e1",
                "e2",
                "e2",
                "e1",
                "e1",
                "e2",
                "e2",
                "e3",
                "e3",
            ],
        )
        groups = EntityIds.compute(aa)

        # asym_id_int is one integer per chain, in chain order.
        np.testing.assert_array_equal(
            groups.asym_id_int, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        )
        # entity_id_int: sort unique entities → {e1:0, e2:1, e3:2}.
        np.testing.assert_array_equal(
            groups.entity_id_int, [0, 0, 1, 1, 0, 0, 1, 1, 2, 2]
        )
        # sym_id_int: running index within each entity.
        # e1: chain 0 → 0, chain 2 → 1; e2: chain 1 → 0, chain 3 → 1; e3: chain 4 → 0.
        np.testing.assert_array_equal(groups.sym_id_int, [0, 0, 0, 0, 1, 1, 1, 1, 0, 0])


# ---------------------------------------------------------------------------
# MoleculeIds
# ---------------------------------------------------------------------------


class TestMoleculeIdsCompute:
    def test_mol_atom_index_resets_per_mol(self):
        """mol_atom_index is a running count within each contiguous mol_id run."""
        mol_id = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int32)
        entity_mol_id = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int32)
        result = MoleculeIds.compute(mol_id, entity_mol_id)
        np.testing.assert_array_equal(
            result.mol_atom_index, [0, 1, 2, 0, 1, 0, 1, 2, 3]
        )
        # mol_id / entity_mol_id pass through unchanged.
        np.testing.assert_array_equal(result.mol_id, mol_id)
        np.testing.assert_array_equal(result.entity_mol_id, entity_mol_id)


# ---------------------------------------------------------------------------
# TypeMasks
# ---------------------------------------------------------------------------


class TestTypeMasksCompute:
    def test_protein_chain_and_ligand_chain(self):
        """mol_type is per-residue; is_* masks are chain-level majority-vote."""
        # Chain A: 3 protein residues (ALA, GLY, SER). Chain B: 2-atom ligand.
        aa = _build_aa(
            chain_id=["A", "A", "B", "B"],
            res_id=[1, 2, 10, 10],
            res_name=["ALA", "GLY", "HEM", "HEM"],
            atom_name=["CA", "CA", "FE", "CHA"],
            element=["C", "C", "FE", "C"],
            label_entity_id=["prot", "prot", "lig", "lig"],
        )
        groups = TypeMasks.compute(aa, sequences={"prot": "AG"})

        # Protein chain: mol_type per-residue comes from CCD.
        assert groups.mol_type[0] == "protein"
        assert groups.mol_type[1] == "protein"
        # Ligand chain: entity absent from sequences → "ligand".
        assert groups.mol_type[2] == "ligand"
        assert groups.mol_type[3] == "ligand"
        # Chain-level masks are consistent for a homogeneous chain.
        np.testing.assert_array_equal(groups.is_protein, [1, 1, 0, 0])
        np.testing.assert_array_equal(groups.is_ligand, [0, 0, 1, 1])
        np.testing.assert_array_equal(groups.is_dna, [0, 0, 0, 0])
        np.testing.assert_array_equal(groups.is_rna, [0, 0, 0, 0])


# ---------------------------------------------------------------------------
# ReferenceConformer
# ---------------------------------------------------------------------------


class TestReferenceConformerCompute:
    def test_ref_space_uid_unique_per_chain_res_pair(self):
        """Each (asym_id_int, res_id) combination gets a distinct uid."""
        aa = _build_aa(
            chain_id=["A", "A", "A", "A"],
            res_id=[1, 1, 2, 2],
            res_name=["ALA", "ALA", "GLY", "GLY"],
            atom_name=["N", "CA", "N", "CA"],
            element=["N", "C", "N", "C"],
            label_entity_id=["e1"] * 4,
        )
        entity_ids = EntityIds.compute(aa)  # asym_id = [0,0,0,0]
        result = ReferenceConformer.compute(aa, entity_ids, include_conformer=False)
        # Two distinct (asym=0, res_id=1/2) pairs → two uids, each shared by
        # its residue's atoms.
        np.testing.assert_array_equal(result.ref_space_uid, [0, 0, 1, 1])

    def test_reads_ref_annotations_off_atom_array(self):
        """``compute(..., include_conformer=True)`` reads ``ref_pos``/
        ``ref_charge``/``ref_mask`` from the atom_array annotations.

        The upstream builder (``add_reference_features`` for the inference
        path, ``add_ref_info_and_res_perm`` for CIF parse) populates them;
        ``compute`` is just responsible for plumbing them into the typed
        group plus deriving ``ref_space_uid``. The test uses synthetic
        values to confirm it doesn't silently overwrite.
        """
        aa = _build_aa(
            chain_id=["A", "A", "A"],
            res_id=[1, 1, 1],
            res_name=["ALA", "ALA", "ALA"],
            atom_name=["N", "CA", "C"],
            element=["N", "C", "C"],
            label_entity_id=["e1"] * 3,
        )
        expected_ref_pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        expected_ref_charge = np.array([-1, 0, 1], dtype=int)
        expected_ref_mask = np.array([1, 1, 0], dtype=int)
        aa.set_annotation("ref_pos", expected_ref_pos)
        aa.set_annotation("ref_charge", expected_ref_charge)
        aa.set_annotation("ref_mask", expected_ref_mask)

        entity_ids = EntityIds.compute(aa)
        result = ReferenceConformer.compute(aa, entity_ids, include_conformer=True)

        np.testing.assert_array_equal(result.ref_pos, expected_ref_pos)
        np.testing.assert_array_equal(result.ref_charge, expected_ref_charge)
        np.testing.assert_array_equal(result.ref_mask, expected_ref_mask)

    def test_raises_when_ref_annotations_missing_and_conformer_requested(self):
        """``compute(..., include_conformer=True)`` on an atom_array that
        never went through ``add_reference_features`` (or equivalent) must
        raise — silent zero-fill is what let the UNL ligand bug slip in,
        and the error message points callers at the right pre-pass.
        """
        aa = _build_aa(
            chain_id=["A"],
            res_id=[1],
            res_name=["ALA"],
            atom_name=["CA"],
            element=["C"],
            label_entity_id=["e1"],
        )
        entity_ids = EntityIds.compute(aa)
        with pytest.raises(ValueError, match="populate_ccd_reference"):
            ReferenceConformer.compute(aa, entity_ids, include_conformer=True)

    def test_zero_fallback_when_conformer_disabled(self):
        """include_conformer=False zero-fills pos/charge/mask but still computes uid."""
        aa = _build_aa(
            chain_id=["A", "A"],
            res_id=[1, 1],
            res_name=["ALA", "ALA"],
            atom_name=["N", "CA"],
            element=["N", "C"],
            label_entity_id=["e1", "e1"],
        )
        entity_ids = EntityIds.compute(aa)
        result = ReferenceConformer.compute(aa, entity_ids, include_conformer=False)
        np.testing.assert_array_equal(result.ref_pos, np.zeros((2, 3)))
        np.testing.assert_array_equal(result.ref_charge, [0, 0])
        np.testing.assert_array_equal(result.ref_mask, [0, 0])
        # uid is still computed — it's cheap and downstream needs it.
        np.testing.assert_array_equal(result.ref_space_uid, [0, 0])


# ---------------------------------------------------------------------------
# populate_ccd_reference — the pre-pass ReferenceConformer.compute requires
# ---------------------------------------------------------------------------


class TestPopulateCcdReference:
    """``populate_ccd_reference`` is the single upstream pre-pass that fills
    ``ref_pos`` / ``ref_charge`` / ``ref_mask`` / ``res_perm`` on an
    atom_array before ``AnnotatedStructure.annotate(...,
    include_ref_conformer=True)``. It subsumes both the CIF-path
    ``annotation.add_ref_info_and_res_perm`` and the inference-path
    ``array_builder.add_reference_features``; the CCD-vs-UNL dispatch lives
    here so the rest of the pipeline never has to special-case SMILES
    ligands.
    """

    def test_ccd_residue_populated_from_lookup(self):
        """For a standard residue (ALA), every atom that appears in the
        CCD component gets ref_pos / ref_charge / ref_mask copied from
        the CCD entry, keyed by atom name.
        """
        aa = _build_aa(
            chain_id=["A", "A", "A"],
            res_id=[1, 1, 1],
            res_name=["ALA", "ALA", "ALA"],
            atom_name=["N", "CA", "C"],
            element=["N", "C", "C"],
            label_entity_id=["e1"] * 3,
        )
        from artios.data.chem.ccd import get_ccd_ref_info

        ref_info = get_ccd_ref_info("ALA")
        assert ref_info is not None, "CCD data missing ALA; fixture prereq unmet"

        populate_ccd_reference(aa)

        for i, atom_name in enumerate(["N", "CA", "C"]):
            sub_idx = ref_info.atom_map[atom_name]
            np.testing.assert_array_equal(aa.ref_pos[i], ref_info.coord[sub_idx])
            assert aa.ref_charge[i] == ref_info.charge[sub_idx]
            assert aa.ref_mask[i] == ref_info.mask[sub_idx]

    def test_unl_residue_copies_coord_and_masks_resolved(self):
        """For a UNL SMILES ligand, ``ref_pos`` must equal ``atom_array.coord``
        and ``ref_mask`` must be 1 for every atom. This is the contract
        that the backbone diffusion reads for intra-ligand geometry
        conditioning; it is what the UNL ref_pos regression (commit
        8a6abcff) pinned in ``test_backbone_input_smiles_ligand.py`` —
        this test pins the same rule at the pre-pass level so migrating
        inference onto ``populate_ccd_reference`` stays safe.
        """
        aa = _build_aa(
            chain_id=["L", "L", "L"],
            res_id=[1, 1, 1],
            res_name=["UNL", "UNL", "UNL"],
            atom_name=["C1", "C2", "C3"],
            element=["C", "C", "C"],
            label_entity_id=["lig"] * 3,
        )
        aa.coord = np.array(
            [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3]], dtype=np.float32
        )

        populate_ccd_reference(aa)

        np.testing.assert_allclose(aa.ref_pos, aa.coord)
        np.testing.assert_array_equal(aa.ref_mask, [1, 1, 1])

    def test_unl_residue_reads_charge_when_present(self):
        """SMILES ligands often carry a per-atom ``charge`` annotation
        (written by ``smiles_to_atom_info``). When it exists, the UNL
        branch must read it — otherwise charged ligands lose their
        formal-charge feature in the featurizer.
        """
        aa = _build_aa(
            chain_id=["L"] * 2,
            res_id=[1, 1],
            res_name=["UNL", "UNL"],
            atom_name=["C1", "N1"],
            element=["C", "N"],
            label_entity_id=["lig"] * 2,
        )
        aa.coord = np.zeros((2, 3), dtype=np.float32)
        aa.set_annotation("charge", np.array([-1, 1], dtype=int))

        populate_ccd_reference(aa)

        np.testing.assert_array_equal(aa.ref_charge, [-1, 1])

    def test_unknown_ccd_code_leaves_zeros(self):
        """A residue whose name isn't in CCD and isn't UNL gets
        ``ref_mask = 0`` across the board. The call does NOT raise — this
        matches the legacy inference-path behaviour and downstream code
        treats ref_mask=0 atoms as unresolved.
        """
        aa = _build_aa(
            chain_id=["X"] * 2,
            res_id=[1, 1],
            res_name=["ZZZ", "ZZZ"],  # deliberately not in CCD
            atom_name=["X1", "X2"],
            element=["C", "C"],
            label_entity_id=["e1"] * 2,
        )

        populate_ccd_reference(aa)

        np.testing.assert_array_equal(aa.ref_mask, [0, 0])
        np.testing.assert_array_equal(aa.ref_pos, np.zeros((2, 3)))

    def test_unknown_atom_name_falls_through_to_ref_mask_zero(self):
        """Bug #4 from the scan: the legacy ``add_ref_feat_info`` raises
        ``KeyError`` when a residue carries an atom name that's not in the
        CCD ``atom_map`` (e.g. a modified residue with renamed atoms).
        The new implementation leaves the slot at ``ref_mask = 0`` and
        continues — the atom is simply not represented in the reference
        conformer.
        """
        aa = _build_aa(
            chain_id=["A"] * 3,
            res_id=[1, 1, 1],
            res_name=["ALA", "ALA", "ALA"],
            atom_name=["N", "CA", "CX"],  # CX isn't in the CCD ALA entry
            element=["N", "C", "C"],
            label_entity_id=["e1"] * 3,
        )

        # Must not raise.
        populate_ccd_reference(aa)

        # First two atoms matched and should have ref_mask=1; the renamed
        # CX atom falls through to ref_mask=0.
        assert aa.ref_mask[0] == 1
        assert aa.ref_mask[1] == 1
        assert aa.ref_mask[2] == 0
        np.testing.assert_array_equal(aa.ref_pos[2], np.zeros(3))

    def test_res_perm_emitted_as_underscore_encoded_strings(self):
        """``res_perm`` rides on the atom_array as a per-atom string
        annotation of the form ``"0_1_2"`` — the legacy encoding the
        featurizer's atom-permutation context reads back via ``.split("_")``.
        This test pins the shape + dtype; the exact permutation content
        is the CCD's business.
        """
        aa = _build_aa(
            chain_id=["A"] * 5,
            res_id=[1, 1, 1, 1, 1],
            res_name=["ALA", "ALA", "ALA", "ALA", "ALA"],
            atom_name=["N", "CA", "C", "O", "CB"],
            element=["N", "C", "C", "O", "C"],
            label_entity_id=["e1"] * 5,
        )

        populate_ccd_reference(aa)

        res_perm = aa.res_perm
        assert len(res_perm) == 5
        assert res_perm.dtype.kind == "U", "res_perm must be numpy string dtype"
        # Every entry decodes to a non-empty list of ints.
        for entry in res_perm:
            parts = entry.split("_")
            assert len(parts) >= 1
            for p in parts:
                int(p)  # will raise if non-integer

    def test_include_res_perm_false_skips_the_annotation(self):
        """Inference-side callers (``add_reference_features``'s
        replacement) don't need the permutation array and pass
        ``include_res_perm=False``; the ``res_perm`` annotation must then
        be absent so featurizer code gated on its presence can tell the
        two paths apart.
        """
        aa = _build_aa(
            chain_id=["A"] * 2,
            res_id=[1, 1],
            res_name=["ALA", "ALA"],
            atom_name=["N", "CA"],
            element=["N", "C"],
            label_entity_id=["e1"] * 2,
        )

        populate_ccd_reference(aa, include_res_perm=False)

        assert "res_perm" not in aa.get_annotation_categories()
        # But the other three are always written.
        assert "ref_pos" in aa.get_annotation_categories()
        assert "ref_charge" in aa.get_annotation_categories()
        assert "ref_mask" in aa.get_annotation_categories()


# ---------------------------------------------------------------------------
# DerivedAnnotations
# ---------------------------------------------------------------------------


class TestDerivedAnnotationsCompute:
    def test_modified_residue_masked_and_canonicalised(self):
        """MSE (selenomethionine) is non-standard protein: masked, canonicalised to MET."""
        # Chain A: 1 standard ALA + 1 modified MSE.
        aa = _build_aa(
            chain_id=["A"] * 6,
            res_id=[1, 1, 1, 2, 2, 2],
            res_name=["ALA", "ALA", "ALA", "MSE", "MSE", "MSE"],
            atom_name=["N", "CA", "C", "N", "CA", "C"],
            element=["N", "C", "C", "N", "C", "C"],
            label_entity_id=["e1"] * 6,
        )
        # TypeMasks needed as input: both residues are protein.
        type_masks = TypeMasks.compute(aa, sequences={"e1": "AM"})
        assert set(type_masks.mol_type.tolist()) == {"protein"}

        result = DerivedAnnotations.compute(aa, type_masks)

        # modified_res_mask: 0 for ALA (standard), 1 for MSE (non-standard, protein).
        np.testing.assert_array_equal(result.modified_res_mask, [0, 0, 0, 1, 1, 1])

        # cano_seq_resname: ALA → ALA, MSE → MET (one-letter M mapped via
        # PROT_STD_RESIDUES_ONE_TO_THREE).
        np.testing.assert_array_equal(
            result.cano_seq_resname,
            ["ALA", "ALA", "ALA", "MET", "MET", "MET"],
        )


# ---------------------------------------------------------------------------
# TrainingAnnotations
# ---------------------------------------------------------------------------


class TestTrainingAnnotationsCompute:
    def test_plddt_mask_selects_ca_for_standard_protein(self):
        """plddt_m_rep_atom_mask is 1 on CA of standard protein residues only."""
        # 2 standard protein residues (ALA N,CA,C) + 1 ligand residue (HEM FE).
        aa = _build_aa(
            chain_id=["A"] * 7,
            res_id=[1, 1, 1, 2, 2, 2, 10],
            res_name=["ALA", "ALA", "ALA", "ALA", "ALA", "ALA", "HEM"],
            atom_name=["N", "CA", "C", "N", "CA", "C", "FE"],
            element=["N", "C", "C", "N", "C", "C", "FE"],
            label_entity_id=["prot"] * 6 + ["lig"],
        )
        type_masks = TypeMasks.compute(aa, sequences={"prot": "AA"})
        result = TrainingAnnotations.compute(aa, type_masks, include_plddt_mask=True)
        # Mask fires on CA of each ALA (indices 1 and 4) — nowhere else.
        np.testing.assert_array_equal(
            result.plddt_m_rep_atom_mask, [0, 1, 0, 0, 1, 0, 0]
        )

    def test_is_resolved_defaults_to_all_true(self):
        """When atom_array has no is_resolved, compute defaults to all True."""
        aa = _build_aa(
            chain_id=["A", "A"],
            res_id=[1, 1],
            res_name=["ALA", "ALA"],
            atom_name=["N", "CA"],
            element=["N", "C"],
            label_entity_id=["e1", "e1"],
        )
        type_masks = TypeMasks.compute(aa, sequences={"e1": "A"})
        result = TrainingAnnotations.compute(aa, type_masks)
        assert result.is_resolved is not None
        assert result.is_resolved.dtype == bool
        assert result.is_resolved.all()

    def test_is_resolved_passes_through_from_atom_array(self):
        """When is_resolved is already set (e.g. low-pLDDT preprocessing), it's preserved."""
        aa = _build_aa(
            chain_id=["A", "A", "A"],
            res_id=[1, 1, 1],
            res_name=["ALA", "ALA", "ALA"],
            atom_name=["N", "CA", "C"],
            element=["N", "C", "C"],
            label_entity_id=["e1"] * 3,
            is_resolved=[True, False, True],
        )
        type_masks = TypeMasks.compute(aa, sequences={"e1": "A"})
        result = TrainingAnnotations.compute(aa, type_masks)
        np.testing.assert_array_equal(result.is_resolved, [True, False, True])


# ---------------------------------------------------------------------------
# REQUIRES contract
# ---------------------------------------------------------------------------


class TestRequiresContract:
    """``_require_annotations`` must name the group and the missing field."""

    def test_missing_required_annotation_raises_named_error(self):
        """EntityIds declares label_entity_id; omitting it yields a clear error."""
        # Build a minimal AA without label_entity_id.
        aa = AtomArray(2)
        aa.coord = np.zeros((2, 3))
        aa.chain_id = np.array(["A", "A"], dtype="<U8")
        aa.res_id = np.array([1, 1])
        aa.res_name = np.array(["ALA", "ALA"], dtype="<U3")
        aa.atom_name = np.array(["N", "CA"], dtype="<U6")
        aa.element = np.array(["N", "C"], dtype="<U2")

        with pytest.raises(ValueError) as exc:
            EntityIds.compute(aa)
        msg = str(exc.value)
        assert "EntityIds" in msg
        assert "label_entity_id" in msg
