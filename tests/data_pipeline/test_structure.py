"""Tests for AnnotatedStructure and TokenGrouping against real pipeline output.

Uses 5s8i test data: protein (128 residues, 1063 atoms) + ligand (13 atoms) = 141 tokens.
Contract tests at the bottom of this file parametrize over the golden fixtures
in ``tests/fixtures/structures/`` to catch silent serialization breakage.
"""

import dataclasses
import gzip
import pickle
from pathlib import Path

import numpy as np
import pytest

from artios.data.structure import (
    AnnotatedStructure,
    TypeMasks,
    compute_token_grouping,
    compute_token_grouping_vectorized,
)
from artios.utils.file_io import load_atomarray_from_npz

FIXTURES = Path(__file__).parent.parent / "fixtures" / "parsing"


@pytest.fixture
def atom_array():
    return load_atomarray_from_npz(FIXTURES / "5s8i.npz")


@pytest.fixture
def legacy_data():
    with gzip.open(FIXTURES / "5s8i.pkl.gz", "rb") as f:
        return pickle.load(f)


@pytest.fixture
def structure(atom_array):
    return AnnotatedStructure.from_atom_array(atom_array)


# ---------------------------------------------------------------------------
# TokenGrouping matches legacy centre_atom_mask and TokenArray
# ---------------------------------------------------------------------------


class TestTokenGrouping:
    def test_centre_atom_indices_match_legacy_token_array(self, structure, legacy_data):
        """centre_atom_indices must match what the legacy TokenArray stores."""
        legacy_ta = legacy_data["token_array"]
        legacy_centres = np.array(legacy_ta.get_annotation("centre_atom_index"))
        np.testing.assert_array_equal(structure.centre_atom_indices, legacy_centres)

    def test_atom_token_index_matches_legacy_token_atom_indices(
        self, structure, legacy_data
    ):
        """atom_token_index must be consistent with Token.atom_indices."""
        legacy_ta = legacy_data["token_array"]
        for token_idx, token in enumerate(legacy_ta):
            for atom_idx in token.atom_indices:
                assert structure.grouping.atom_token_index[atom_idx] == token_idx

    def test_standard_residue_tokens_are_multi_atom(self, structure):
        """Protein standard residues should produce multi-atom tokens."""
        ati = structure.grouping.atom_token_index
        mol_type = structure.type_masks.mol_type
        # First protein residue (token 0) should have multiple atoms
        token_0_atoms = np.where(ati == 0)[0]
        assert len(token_0_atoms) > 1
        assert mol_type[token_0_atoms[0]] == "protein"

    def test_ligand_atoms_are_single_atom_tokens(self, structure):
        """Each ligand atom should be its own token."""
        ati = structure.grouping.atom_token_index
        ligand_mask = structure.type_masks.is_ligand.astype(bool)
        ligand_tokens = ati[ligand_mask]
        # Each ligand atom should have a unique token index
        assert len(np.unique(ligand_tokens)) == ligand_mask.sum()

    def test_vectorized_matches_original(self, structure):
        """Vectorized tokenization must produce identical output to the original."""
        aa = structure.to_atom_array()
        type_masks = TypeMasks.from_atom_array(aa)

        g_orig = compute_token_grouping(aa, type_masks, motif=None)
        g_vec = compute_token_grouping_vectorized(aa, type_masks, motif=None)

        assert g_orig.n_tokens == g_vec.n_tokens
        np.testing.assert_array_equal(g_orig.atom_token_index, g_vec.atom_token_index)
        np.testing.assert_array_equal(g_orig.centre_atom_mask, g_vec.centre_atom_mask)


# ---------------------------------------------------------------------------
# AnnotatedStructure construction and validation
# ---------------------------------------------------------------------------


class TestAnnotatedStructure:
    def test_from_atom_array_shape_consistency(self, structure):
        """All per-atom arrays must have length n_atoms."""
        n = len(structure.atoms)
        assert len(structure.entity_ids.asym_id_int) == n
        assert len(structure.molecule_ids.mol_id) == n
        assert len(structure.type_masks.mol_type) == n
        assert structure.reference.ref_pos.shape == (n, 3)
        assert len(structure.grouping.atom_token_index) == n
        assert len(structure.derived.cano_seq_resname) == n
        assert len(structure.derived.tokatom_idx) == n


# ---------------------------------------------------------------------------
# Token-level cached properties
# ---------------------------------------------------------------------------


class TestTokenLevelViews:
    def test_token_is_ligand_consistent(self, structure):
        """Token-level ligand mask should reflect atom-level."""
        centre = structure.centre_atom_indices
        token_lig = structure.type_masks.is_ligand[centre]
        assert token_lig[-1] == 1
        assert token_lig[0] == 0


# ---------------------------------------------------------------------------
# Selection / Cropping
# ---------------------------------------------------------------------------


class TestSelection:
    def test_select_tokens_preserves_consistency(self, structure):
        """Selecting a subset of tokens should produce a valid structure."""
        # Select first 10 tokens
        indices = np.arange(10)
        cropped = structure.select_tokens(indices)

        assert cropped.grouping.n_tokens == 10
        assert len(cropped.atoms) < len(structure.atoms)
        # All arrays should be consistent
        n = len(cropped.atoms)
        assert len(cropped.entity_ids.asym_id_int) == n
        assert len(cropped.type_masks.mol_type) == n
        assert cropped.reference.ref_pos.shape == (n, 3)
        assert len(cropped.grouping.atom_token_index) == n

    def test_select_tokens_reindexes_correctly(self, structure):
        """Token indices should be dense 0..n-1 after selection."""
        indices = np.array([5, 10, 15])
        cropped = structure.select_tokens(indices)
        ati = cropped.grouping.atom_token_index
        assert ati.min() == 0
        assert ati.max() == 2
        assert cropped.grouping.n_tokens == 3

    def test_select_tokens_ligand_only(self, structure):
        """Selecting only ligand tokens should give per-atom tokens."""
        lig_mask = structure.type_masks.is_ligand[structure.centre_atom_indices].astype(
            bool
        )
        lig_token_indices = np.where(lig_mask)[0]
        cropped = structure.select_tokens(lig_token_indices)

        # All atoms should be ligand
        assert (cropped.type_masks.is_ligand == 1).all()
        # Each atom is its own token
        assert cropped.grouping.n_tokens == len(cropped.atoms)


# ---------------------------------------------------------------------------
# AnnotatedStructure.annotate() — end-to-end from CIF
# ---------------------------------------------------------------------------


class TestAnnotate:
    """Build AnnotatedStructure via annotate() from a clean atom_array and compare to legacy."""

    @pytest.fixture
    def annotated_from_cif(self):
        """Clean atom_array (no custom annotations) → annotate() → AnnotatedStructure."""
        from artios.data.filter import Filter, clean_structure
        from artios.data.parsing.cif import MMCIFParser

        parser = MMCIFParser(FIXTURES / "5s8i.cif")
        atom_array = parser.get_structure()
        sequences = parser.get_sequences(atom_array)

        atom_array = clean_structure(
            atom_array,
            entity_poly_type=parser.entity_poly_type,
            remove_crystallization_aids=True,
            remove_artifact_ligands=True,
            remove_unbound_metal_ions=True,
        )
        atom_array = parser.fix_arginine(atom_array)
        atom_array = parser.mse_to_met(atom_array)
        atom_array = parser.add_missing_atoms_and_residues(atom_array)
        atom_array = Filter.remove_element_X(atom_array)

        return AnnotatedStructure.annotate(
            atom_array,
            sequences,
            parser.entity_poly_type,
            include_ref_conformer=True,
            include_training_masks=True,
        )

    @pytest.fixture
    def annotated_from_npz(self, atom_array):
        """Legacy pipeline output (npz) → from_atom_array() → AnnotatedStructure."""
        return AnnotatedStructure.from_atom_array(atom_array)

    def test_annotate_matches_from_atom_array(
        self, annotated_from_cif, annotated_from_npz
    ):
        """The two construction paths must produce structures that match across
        every typed field — from_atom_array replays what annotate() emitted."""
        a, b = annotated_from_cif, annotated_from_npz

        assert len(a.atoms) == len(b.atoms)
        assert a.grouping.n_tokens == b.grouping.n_tokens

        for path, left, right in [
            (
                "grouping.centre_atom_mask",
                a.grouping.centre_atom_mask,
                b.grouping.centre_atom_mask,
            ),
            ("type_masks.mol_type", a.type_masks.mol_type, b.type_masks.mol_type),
            (
                "entity_ids.asym_id_int",
                a.entity_ids.asym_id_int,
                b.entity_ids.asym_id_int,
            ),
            (
                "entity_ids.entity_id_int",
                a.entity_ids.entity_id_int,
                b.entity_ids.entity_id_int,
            ),
            (
                "reference.ref_space_uid",
                a.reference.ref_space_uid,
                b.reference.ref_space_uid,
            ),
            (
                "derived.cano_seq_resname",
                a.derived.cano_seq_resname,
                b.derived.cano_seq_resname,
            ),
        ]:
            np.testing.assert_array_equal(left, right, err_msg=path)

        # ref_pos is float — allow rounding tolerance.
        np.testing.assert_allclose(
            a.reference.ref_pos,
            b.reference.ref_pos,
            atol=1e-6,
            err_msg="reference.ref_pos",
        )


# ---------------------------------------------------------------------------
# ContextGenerator with AnnotatedStructure matches legacy (TokenArray, AtomArray)
# ---------------------------------------------------------------------------


class TestContextGenerator:
    """Verify ContextGenerator(AnnotatedStructure) produces identical output to
    what the old (TokenArray, AtomArray) interface would produce."""

    @pytest.fixture
    def full_context(self, structure):
        from artios.data.features.featurizer import ContextGenerator

        return ContextGenerator(
            structure, pdb_id="5s8i", ref_pos_augment=False
        ).get_full_context(include_arrays=False)

    def test_atom_to_token_idx_matches_legacy(self, full_context, legacy_data):
        """atom_to_token_idx must match what legacy Token.atom_indices would produce."""
        import torch

        legacy_ta = legacy_data["token_array"]
        legacy_a2t = {}
        for idx, token in enumerate(legacy_ta.tokens):
            for atom_idx in token.atom_indices:
                legacy_a2t[atom_idx] = idx
        expected = torch.tensor(
            [legacy_a2t[i] for i in range(len(legacy_a2t))], dtype=torch.long
        )
        assert torch.equal(full_context.index.atom_to_token_idx, expected)

    def test_centre_coordinate_matches_legacy(
        self, full_context, atom_array, legacy_data
    ):
        """Centre coordinates must match legacy token centres."""
        import torch

        legacy_centres = np.array(
            legacy_data["token_array"].get_annotation("centre_atom_index")
        )
        expected = torch.from_numpy(atom_array.coord[legacy_centres].astype(np.float32))
        assert torch.allclose(full_context.coordinate.centre_coordinate, expected)

    def test_entity_masks_match_legacy(self, full_context, atom_array):
        import torch

        assert torch.equal(
            full_context.entity.is_protein,
            torch.from_numpy(atom_array.is_protein.astype(bool)),
        )
        assert torch.equal(
            full_context.entity.is_ligand,
            torch.from_numpy(atom_array.is_ligand.astype(bool)),
        )

    def test_centre_atom_mask_matches_legacy(self, full_context, atom_array):
        import torch

        expected = torch.from_numpy(atom_array.centre_atom_mask.astype(bool))
        assert torch.equal(full_context.mask.centre_atom_mask, expected)

    def test_motif_zero_when_absent(self, full_context):
        assert not full_context.motif.is_motif.any()
        assert not full_context.motif.is_motif_related.any()


# ---------------------------------------------------------------------------
# Contract tests: golden-fixture round-trips
#
# These pin the AnnotatedStructure serialization contract. If a typed field is
# ever silently dropped by save/load — or a from/to_atom_array method stops
# being invertible — these tests catch it before it lands. They use the
# parametrized ``load_structure`` fixture from conftest.py; a missing golden
# fixture will skip its parametrized instance rather than error.
# ---------------------------------------------------------------------------

GOLDEN_FIXTURES = ["2g1k", "1m2g", "3iwd", "1dai", "4lbp"]

_ANNOTATION_GROUP_FIELDS = (
    "entity_ids",
    "molecule_ids",
    "type_masks",
    "reference",
    "grouping",
    "derived",
    "training",
)


def _assert_annotation_group_equal(a, b, path: str) -> None:
    """Compare two frozen-dataclass annotation groups field-by-field."""
    assert type(a) is type(b), f"{path}: type mismatch {type(a)} vs {type(b)}"
    for f in dataclasses.fields(a):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            raise AssertionError(f"{path}.{f.name}: one is None, other isn't")
        np.testing.assert_array_equal(va, vb, err_msg=f"{path}.{f.name}")


def _assert_structures_equal(a: AnnotatedStructure, b: AnnotatedStructure) -> None:
    """Deep-compare two AnnotatedStructures.

    Frozen dataclass ``__eq__`` is unreliable with numpy arrays, so we walk
    every typed field and the underlying AtomArray explicitly.
    """
    assert len(a.atoms) == len(b.atoms), "atom count differs"

    # Typed annotation groups (required).
    for name in _ANNOTATION_GROUP_FIELDS:
        _assert_annotation_group_equal(getattr(a, name), getattr(b, name), name)

    # Optional motif.
    if (a.motif is None) != (b.motif is None):
        raise AssertionError("motif: one is None, other isn't")
    if a.motif is not None:
        _assert_annotation_group_equal(a.motif, b.motif, "motif")

    # AtomArray: coords + every remaining biotite-native annotation.
    np.testing.assert_array_equal(a.atoms.coord, b.atoms.coord, err_msg="atoms.coord")
    cats_a = set(a.atoms.get_annotation_categories())
    cats_b = set(b.atoms.get_annotation_categories())
    assert cats_a == cats_b, f"AtomArray annotation set differs: {cats_a ^ cats_b}"
    for cat in cats_a:
        np.testing.assert_array_equal(
            a.atoms.get_annotation(cat),
            b.atoms.get_annotation(cat),
            err_msg=f"atoms.{cat}",
        )


@pytest.mark.parametrize("name", GOLDEN_FIXTURES)
class TestGoldenRoundTrip:
    def test_save_load_preserves_structure(self, load_structure, name, tmp_path):
        """save(path) → load(path) must yield an equal structure."""
        original = load_structure(name)
        out = tmp_path / f"{name}.npz"
        original.save(out)
        reloaded = AnnotatedStructure.load(out)
        _assert_structures_equal(original, reloaded)

    def test_atom_array_roundtrip(self, load_structure, name):
        """from_atom_array(to_atom_array(s)) must yield an equal structure.

        Catches asymmetry between the _AnnotationGroup read/write paths — a
        field added to to_atom_array but not restored by from_atom_array will
        fail this test.
        """
        original = load_structure(name)
        aa = original.to_atom_array()
        rebuilt = AnnotatedStructure.from_atom_array(aa)
        _assert_structures_equal(original, rebuilt)


@pytest.mark.parametrize("name", GOLDEN_FIXTURES)
def test_to_atom_array_does_not_mutate_source(load_structure, name):
    """to_atom_array() must not mutate the source AnnotatedStructure's AtomArray.

    Frozen dataclass on its own doesn't protect mutable numpy arrays — if
    to_atom_array ever starts writing to ``self.atoms`` instead of a copy,
    repeated calls from a model step would silently corrupt the fixture.
    """
    s = load_structure(name)
    before_cats = set(s.atoms.get_annotation_categories())
    before_n = len(s.atoms)
    _ = s.to_atom_array()
    after_cats = set(s.atoms.get_annotation_categories())
    assert (
        before_cats == after_cats
    ), f"atoms annotations changed: {before_cats ^ after_cats}"
    assert len(s.atoms) == before_n
