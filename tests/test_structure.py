"""Tests for AnnotatedStructure and TokenGrouping against real pipeline output.

Uses 5s8i test data: protein (128 residues, 1063 atoms) + ligand (13 atoms) = 141 tokens.
"""

import gzip
import pickle
from pathlib import Path

import numpy as np
import pytest

from artios.data.structure import (
    AnnotatedStructure,
)
from artios.utils.file_io import load_atomarray_from_npz

TESTDATA = Path(__file__).parent / "testdata" / "parsing"


@pytest.fixture
def atom_array():
    return load_atomarray_from_npz(TESTDATA / "5s8i.npz")


@pytest.fixture
def legacy_data():
    with gzip.open(TESTDATA / "5s8i.pkl.gz", "rb") as f:
        return pickle.load(f)


@pytest.fixture
def structure(atom_array, legacy_data):
    return AnnotatedStructure.from_atom_array(
        atom_array, resolution=float(legacy_data["resolution"])
    )


# ---------------------------------------------------------------------------
# TokenGrouping matches legacy centre_atom_mask and TokenArray
# ---------------------------------------------------------------------------


class TestTokenGrouping:
    def test_n_tokens_matches_legacy(self, structure, legacy_data):
        """TokenGrouping.n_tokens must equal len(legacy TokenArray)."""
        legacy_token_array = legacy_data["token_array"]
        assert structure.grouping.n_tokens == len(legacy_token_array)

    def test_centre_atom_mask_matches_legacy(self, structure, atom_array):
        """centre_atom_mask must match the legacy annotation exactly."""
        legacy_mask = atom_array.centre_atom_mask.astype(bool)
        np.testing.assert_array_equal(structure.grouping.centre_atom_mask, legacy_mask)

    def test_centre_atom_indices_match_legacy_token_array(self, structure, legacy_data):
        """centre_atom_indices must match what the legacy TokenArray stores."""
        legacy_ta = legacy_data["token_array"]
        legacy_centres = np.array(legacy_ta.get_annotation("centre_atom_index"))
        np.testing.assert_array_equal(structure.centre_atom_indices, legacy_centres)

    def test_atom_token_index_covers_all_atoms(self, structure):
        """Every atom must be assigned to exactly one token."""
        ati = structure.grouping.atom_token_index
        assert (ati >= 0).all()
        assert ati.max() == structure.grouping.n_tokens - 1

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
        assert len(structure.cano_seq_resname) == n
        assert len(structure.tokatom_idx) == n

    def test_resolution_preserved(self, structure):
        assert structure.resolution == pytest.approx(1.3)

    def test_frozen(self, structure):
        """AnnotatedStructure should be immutable."""
        with pytest.raises(AttributeError):
            structure.resolution = 2.0  # type: ignore[misc]

    def test_save_load_roundtrip(self, structure, tmp_path):
        """save() then load() should produce an identical structure."""
        path = tmp_path / "test.npz"
        structure.save(path)
        loaded = AnnotatedStructure.load(path)

        assert len(loaded.atoms) == len(structure.atoms)
        assert loaded.grouping.n_tokens == structure.grouping.n_tokens
        assert loaded.resolution == pytest.approx(structure.resolution)
        np.testing.assert_array_equal(
            loaded.grouping.centre_atom_mask, structure.grouping.centre_atom_mask
        )
        np.testing.assert_array_equal(
            loaded.entity_ids.asym_id_int, structure.entity_ids.asym_id_int
        )
        np.testing.assert_allclose(
            loaded.reference.ref_pos, structure.reference.ref_pos, atol=1e-6
        )
        np.testing.assert_array_equal(
            loaded.type_masks.mol_type, structure.type_masks.mol_type
        )


# ---------------------------------------------------------------------------
# Token-level cached properties
# ---------------------------------------------------------------------------


class TestTokenLevelViews:
    def test_token_centre_coords_shape(self, structure):
        coords = structure.token_centre_coords
        assert coords.shape == (structure.grouping.n_tokens, 3)

    def test_token_chain_ids_shape(self, structure):
        chain_ids = structure.token_chain_ids
        assert len(chain_ids) == structure.grouping.n_tokens

    def test_token_is_ligand_consistent(self, structure):
        """Token-level ligand mask should reflect atom-level."""
        token_lig = structure.token_is_ligand
        # Last 13 tokens are ligand (per-atom), rest are protein
        assert token_lig[-1] == 1
        assert token_lig[0] == 0

    def test_token_ref_space_uid_shape(self, structure):
        uids = structure.token_ref_space_uid
        assert len(uids) == structure.grouping.n_tokens


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

    def test_select_atoms_preserves_consistency(self, structure):
        """Selecting a subset of atoms should produce a valid structure."""
        # Select first 50 atoms
        indices = np.arange(50)
        cropped = structure.select_atoms(indices)
        assert len(cropped.atoms) == 50
        assert cropped.reference.ref_pos.shape == (50, 3)

    def test_select_all_tokens_roundtrip(self, structure):
        """Selecting all tokens should preserve token count."""
        all_indices = np.arange(structure.grouping.n_tokens)
        cropped = structure.select_tokens(all_indices)
        assert cropped.grouping.n_tokens == structure.grouping.n_tokens
        assert len(cropped.atoms) == len(structure.atoms)

    def test_select_tokens_ligand_only(self, structure):
        """Selecting only ligand tokens should give per-atom tokens."""
        lig_mask = structure.token_is_ligand.astype(bool)
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
        from artios.data.parser import MMCIFParser

        parser = MMCIFParser(TESTDATA / "5s8i.cif")
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
            resolution=parser.resolution,
            include_ref_conformer=True,
            include_training_masks=True,
        )

    @pytest.fixture
    def annotated_from_npz(self, atom_array, legacy_data):
        """Legacy pipeline output (npz) → from_atom_array() → AnnotatedStructure."""
        return AnnotatedStructure.from_atom_array(
            atom_array, resolution=float(legacy_data["resolution"])
        )

    def test_same_n_atoms(self, annotated_from_cif, annotated_from_npz):
        assert len(annotated_from_cif.atoms) == len(annotated_from_npz.atoms)

    def test_same_n_tokens(self, annotated_from_cif, annotated_from_npz):
        assert (
            annotated_from_cif.grouping.n_tokens == annotated_from_npz.grouping.n_tokens
        )

    def test_same_centre_atom_mask(self, annotated_from_cif, annotated_from_npz):
        np.testing.assert_array_equal(
            annotated_from_cif.grouping.centre_atom_mask,
            annotated_from_npz.grouping.centre_atom_mask,
        )

    def test_same_mol_type(self, annotated_from_cif, annotated_from_npz):
        np.testing.assert_array_equal(
            annotated_from_cif.type_masks.mol_type,
            annotated_from_npz.type_masks.mol_type,
        )

    def test_same_entity_ids(self, annotated_from_cif, annotated_from_npz):
        np.testing.assert_array_equal(
            annotated_from_cif.entity_ids.asym_id_int,
            annotated_from_npz.entity_ids.asym_id_int,
        )
        np.testing.assert_array_equal(
            annotated_from_cif.entity_ids.entity_id_int,
            annotated_from_npz.entity_ids.entity_id_int,
        )

    def test_same_ref_space_uid(self, annotated_from_cif, annotated_from_npz):
        np.testing.assert_array_equal(
            annotated_from_cif.reference.ref_space_uid,
            annotated_from_npz.reference.ref_space_uid,
        )

    def test_same_ref_conformer(self, annotated_from_cif, annotated_from_npz):
        np.testing.assert_allclose(
            annotated_from_cif.reference.ref_pos,
            annotated_from_npz.reference.ref_pos,
            atol=1e-6,
        )

    def test_same_cano_seq_resname(self, annotated_from_cif, annotated_from_npz):
        np.testing.assert_array_equal(
            annotated_from_cif.cano_seq_resname,
            annotated_from_npz.cano_seq_resname,
        )

    def test_same_resolution(self, annotated_from_cif, annotated_from_npz):
        assert annotated_from_cif.resolution == pytest.approx(
            annotated_from_npz.resolution
        )


# ---------------------------------------------------------------------------
# ContextGenerator with AnnotatedStructure matches legacy (TokenArray, AtomArray)
# ---------------------------------------------------------------------------


class TestContextGenerator:
    """Verify ContextGenerator(AnnotatedStructure) produces identical output to
    what the old (TokenArray, AtomArray) interface would produce."""

    @pytest.fixture
    def full_context(self, structure):
        from artios.data.featurizer import ContextGenerator

        return ContextGenerator(
            structure, pdb_id="5s8i", ref_pos_augment=False
        ).get_full_context(include_arrays=False)

    def test_token_context_shapes(self, full_context):
        assert full_context.token.token_index.shape == (141,)
        assert full_context.token.restype.shape == (141, 32)

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

    def test_backbone_shape(self, full_context):
        assert full_context.coordinate.backbone_coordinate.shape == (141, 5, 3)
        assert full_context.mask.backbone_mask.shape == (141, 5)

    def test_resolution_is_scalar(self, full_context):
        assert full_context.coordinate.resolution == pytest.approx(1.3)

    def test_motif_zero_when_absent(self, full_context):
        assert not full_context.motif.is_motif.any()
        assert not full_context.motif.is_motif_related.any()

    def test_ref_conformer_shapes(self, full_context):
        assert full_context.reference.ref_pos.shape == (1076, 3)
        assert full_context.reference.ref_element.shape == (1076, 128)
        assert full_context.reference.ref_atom_name_chars.shape == (1076, 4, 64)
