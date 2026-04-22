from pathlib import Path

import numpy as np
import pytest
from biotite.structure import AtomArray, AtomArrayStack, stack
from biotite.structure.io import pdbx

from artios.utils.file_io import (
    dump_atomarray_stack,
    dump_atomarray_to_npz,
    load_atomarray_from_npz,
    load_atomarray_stack,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "cif"


@pytest.fixture
def atom_array() -> AtomArray:
    """Load a single AtomArray from a CIF file with bonds."""
    cif_file = pdbx.CIFFile.read(FIXTURES_DIR / "1bn1.cif")
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
