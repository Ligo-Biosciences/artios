"""Tests for artios.data.cropping."""

import gzip
import pickle
from pathlib import Path

import numpy as np
import pytest

from artios.data.structure import AnnotatedStructure, cropping
from artios.utils.file_io import load_atomarray_from_npz

FIXTURES = Path(__file__).parent.parent / "fixtures" / "parsing"


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def structure():
    atom_array = load_atomarray_from_npz(FIXTURES / "5s8i.npz")
    with gzip.open(FIXTURES / "5s8i.pkl.gz", "rb") as f:
        legacy = pickle.load(f)
    return AnnotatedStructure.from_atom_array(
        atom_array, resolution=float(legacy["resolution"])
    )


def _centre(s: AnnotatedStructure):
    return s.centre_atom_indices


def _ref_chain_ids(structure: AnnotatedStructure) -> list[int]:
    """Pick the first chain as reference."""
    return [int(structure.entity_ids.asym_id_int[_centre(structure)[0]])]


class TestSpatialCrop:
    def test_returns_sorted_indices(self, structure):
        ids = _ref_chain_ids(structure)
        indices = cropping.spatial_crop_indices(
            structure.token_centre_coords,
            structure.training.is_resolved[_centre(structure)],
            structure.entity_ids.asym_id_int[_centre(structure)],
            structure.reference.ref_space_uid[_centre(structure)],
            ids,
            crop_size=50,
        )
        assert np.all(indices[:-1] <= indices[1:])

    def test_respects_crop_size(self, structure):
        ids = _ref_chain_ids(structure)
        indices = cropping.spatial_crop_indices(
            structure.token_centre_coords,
            structure.training.is_resolved[_centre(structure)],
            structure.entity_ids.asym_id_int[_centre(structure)],
            structure.reference.ref_space_uid[_centre(structure)],
            ids,
            crop_size=30,
        )
        assert len(indices) <= 30

    def test_complete_molecules_drops_partial(self, structure):
        ids = _ref_chain_ids(structure)
        uid = structure.reference.ref_space_uid[_centre(structure)]
        indices = cropping.spatial_crop_indices(
            structure.token_centre_coords,
            structure.training.is_resolved[_centre(structure)],
            structure.entity_ids.asym_id_int[_centre(structure)],
            uid,
            ids,
            crop_size=50,
            complete_molecules=True,
        )
        selected_uids = set(uid[indices].tolist())
        for u in selected_uids:
            all_tokens_with_uid = set(np.where(uid == u)[0].tolist())
            assert all_tokens_with_uid.issubset(set(indices.tolist()))


class TestSpatialInterfaceCrop:
    def test_returns_sorted_within_budget(self, structure):
        ids = _ref_chain_ids(structure)
        indices = cropping.spatial_interface_crop_indices(
            structure.token_centre_coords,
            structure.training.is_resolved[_centre(structure)],
            structure.entity_ids.asym_id_int[_centre(structure)],
            structure.reference.ref_space_uid[_centre(structure)],
            ids,
            crop_size=50,
        )
        assert len(indices) <= 50
        assert np.all(indices[:-1] <= indices[1:])


class TestContiguousCrop:
    def test_respects_crop_size(self, structure):
        n = structure.grouping.n_tokens
        atom_counts = cropping._compute_token_molecule_atom_counts(structure)
        indices = cropping.contiguous_crop_indices(
            n,
            structure.entity_ids.asym_id_int[_centre(structure)],
            structure.reference.ref_space_uid[_centre(structure)],
            atom_counts,
            crop_size=50,
            drop_last=True,
        )
        assert len(indices) <= 50

    def test_complete_molecules(self, structure):
        n = structure.grouping.n_tokens
        uid = structure.reference.ref_space_uid[_centre(structure)]
        atom_counts = cropping._compute_token_molecule_atom_counts(structure)
        indices = cropping.contiguous_crop_indices(
            n,
            structure.entity_ids.asym_id_int[_centre(structure)],
            uid,
            atom_counts,
            crop_size=50,
            complete_molecules=True,
            drop_last=True,
        )
        first, last = cropping._molecule_spans(uid)
        if len(indices) > 0:
            assert first[indices[0]] == indices[0]
            assert last[indices[-1]] == indices[-1]


class TestMoleculeSpans:
    def test_basic(self):
        uid = np.array([10, 10, 20, 20, 20, 30])
        first, last = cropping._molecule_spans(uid)
        np.testing.assert_array_equal(first, [0, 0, 2, 2, 2, 5])
        np.testing.assert_array_equal(last, [1, 1, 4, 4, 4, 5])


class TestDetectMetalTokens:
    def test_single_atom_single_chain(self):
        chain_ids = np.array([0, 0, 0, 1, 2])
        uid = np.array([10, 10, 10, 20, 30])
        atom_counts = np.array([3, 3, 3, 5, 1])
        metals = cropping._detect_metal_tokens(chain_ids, uid, atom_counts)
        np.testing.assert_array_equal(metals, [False, False, False, False, True])


class TestDropIncompleteMolecules:
    def test_removes_partial(self):
        selected = np.array([0, 1, 2])
        uid = np.array([10, 10, 20, 20, 20])
        result = cropping._drop_incomplete_molecules(selected, uid)
        np.testing.assert_array_equal(result, [0, 1])

    def test_keeps_all_complete(self):
        selected = np.array([0, 1, 2, 3, 4])
        uid = np.array([10, 10, 20, 20, 20])
        result = cropping._drop_incomplete_molecules(selected, uid)
        np.testing.assert_array_equal(result, selected)


class TestCropOrchestrator:
    def test_no_crash_all_methods(self, structure):
        ids = _ref_chain_ids(structure)
        for method in [
            "SpatialCropping",
            "SpatialInterfaceCropping",
            "ContiguousCropping",
        ]:
            weights = [1.0 if m == method else 0.0 for m in cropping.METHODS]
            m, indices = cropping.crop(
                structure,
                crop_size=50,
                ref_chain_ids=ids,
                method_weights=weights,
            )
            assert m == method
            assert len(indices) > 0


class TestTokenMoleculeAtomCounts:
    def test_consistent_with_structure(self, structure):
        counts = cropping._compute_token_molecule_atom_counts(structure)
        assert len(counts) == structure.grouping.n_tokens
        assert (counts > 0).all()
