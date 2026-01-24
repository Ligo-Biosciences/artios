from pathlib import Path

import numpy as np
import pytest
from biotite.structure import AtomArray, AtomArrayStack, stack
from biotite.structure.io import pdbx

from protenix.utils.file_io import (
    dump_atomarray_stack,
    dump_atomarray_to_npz,
    dump_multi_sample_atomarray,
    load_atomarray_from_npz,
    load_atomarray_stack,
    load_multi_sample_atomarray,
)

TESTDATA_DIR = Path(__file__).parent / "testdata"


@pytest.fixture
def atom_array() -> AtomArray:
    """Load a single AtomArray from a CIF file with bonds."""
    cif_file = pdbx.CIFFile.read(TESTDATA_DIR / "1bn1.cif")
    return pdbx.get_structure(cif_file, model=1, include_bonds=True)


@pytest.fixture
def atom_array_stack(atom_array: AtomArray) -> AtomArrayStack:
    """Create an AtomArrayStack by stacking perturbed copies of a single model."""
    # Create a synthetic stack with 3 models (perturbed coordinates)
    arrays = [atom_array.copy() for _ in range(3)]
    for i, arr in enumerate(arrays):
        arr.coord = arr.coord + i * 0.1  # Small perturbation
    return stack(arrays)


class TestAtomArrayNpz:
    """Tests for single AtomArray NPZ I/O."""

    def test_roundtrip(self, atom_array: AtomArray, tmp_path: Path) -> None:
        """Test that AtomArray survives a dump/load roundtrip."""
        npz_path = tmp_path / "atom_array.npz"

        dump_atomarray_to_npz(atom_array, npz_path)
        loaded = load_atomarray_from_npz(npz_path)

        assert len(loaded) == len(atom_array)
        np.testing.assert_array_almost_equal(loaded.coord, atom_array.coord)

        # Check annotations preserved
        for category in atom_array.get_annotation_categories():
            original = atom_array.get_annotation(category)
            restored = loaded.get_annotation(category)
            np.testing.assert_array_equal(restored, original)


class TestMultiSampleAtomArray:
    """Tests for legacy multi-sample AtomArray functions."""

    def test_roundtrip(self, atom_array: AtomArray, tmp_path: Path) -> None:
        """Test that multi-sample data survives a dump/load roundtrip."""
        n_samples = 5
        n_atoms = len(atom_array)

        # Create synthetic multi-sample coordinates
        coords = np.stack(
            [atom_array.coord + i * 0.1 for i in range(n_samples)],
            axis=0,
        )
        assert coords.shape == (n_samples, n_atoms, 3)

        npz_path = tmp_path / "multi_sample.npz"
        dump_multi_sample_atomarray(atom_array, coords, npz_path)

        loaded_array, loaded_coords = load_multi_sample_atomarray(npz_path)

        assert len(loaded_array) == n_atoms
        np.testing.assert_array_almost_equal(loaded_coords, coords)
        # First sample should be default coords
        np.testing.assert_array_almost_equal(loaded_array.coord, coords[0])

    def test_mismatched_atoms_raises(
        self, atom_array: AtomArray, tmp_path: Path
    ) -> None:
        """Test that mismatched atom count raises ValueError."""
        wrong_coords = np.zeros((3, len(atom_array) + 10, 3))
        npz_path = tmp_path / "bad.npz"

        with pytest.raises(ValueError, match="Coordinate n_atom"):
            dump_multi_sample_atomarray(atom_array, wrong_coords, npz_path)


class TestAtomArrayStack:
    """Tests for AtomArrayStack NPZ I/O."""

    def test_roundtrip(self, atom_array_stack: AtomArrayStack, tmp_path: Path) -> None:
        """Test that AtomArrayStack survives a dump/load roundtrip."""
        npz_path = tmp_path / "stack.npz"

        dump_atomarray_stack(atom_array_stack, npz_path)
        loaded = load_atomarray_stack(npz_path)

        # Check dimensions
        assert loaded.stack_depth() == atom_array_stack.stack_depth()
        assert loaded.array_length() == atom_array_stack.array_length()

        # Check coordinates
        np.testing.assert_array_almost_equal(loaded.coord, atom_array_stack.coord)

        # Check annotations preserved
        for category in atom_array_stack.get_annotation_categories():
            original = atom_array_stack.get_annotation(category)
            restored = loaded.get_annotation(category)
            np.testing.assert_array_equal(restored, original)

    def test_roundtrip_synthetic(self, atom_array: AtomArray, tmp_path: Path) -> None:
        """Test roundtrip with synthetically created AtomArrayStack."""
        n_models = 4
        n_atoms = len(atom_array)

        # Create stack with perturbed coordinates
        stack = AtomArrayStack(depth=n_models, length=n_atoms)
        stack.coord = np.stack(
            [atom_array.coord + i * 0.5 for i in range(n_models)],
            axis=0,
        )
        stack.bonds = atom_array.bonds

        for category in atom_array.get_annotation_categories():
            stack.set_annotation(category, atom_array.get_annotation(category))

        npz_path = tmp_path / "synthetic_stack.npz"
        dump_atomarray_stack(stack, npz_path)
        loaded = load_atomarray_stack(npz_path)

        assert loaded.stack_depth() == n_models
        assert loaded.array_length() == n_atoms
        np.testing.assert_array_almost_equal(loaded.coord, stack.coord)

    def test_indexing_yields_atomarray(
        self, atom_array_stack: AtomArrayStack, tmp_path: Path
    ) -> None:
        """Test that indexing loaded stack returns AtomArray."""
        npz_path = tmp_path / "stack.npz"
        dump_atomarray_stack(atom_array_stack, npz_path)
        loaded = load_atomarray_stack(npz_path)

        # Integer indexing should yield AtomArray
        first_model = loaded[0]
        assert isinstance(first_model, AtomArray)
        np.testing.assert_array_almost_equal(
            first_model.coord, atom_array_stack.coord[0]
        )

    def test_iteration(self, atom_array_stack: AtomArrayStack, tmp_path: Path) -> None:
        """Test that iterating over loaded stack works."""
        npz_path = tmp_path / "stack.npz"
        dump_atomarray_stack(atom_array_stack, npz_path)
        loaded = load_atomarray_stack(npz_path)

        models = list(loaded)
        assert len(models) == atom_array_stack.stack_depth()
        for i, model in enumerate(models):
            assert isinstance(model, AtomArray)
            np.testing.assert_array_almost_equal(model.coord, atom_array_stack.coord[i])
