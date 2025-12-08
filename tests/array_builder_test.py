"""Tests for array_builder.py new functions."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from biotite.structure import AtomArray

from protenix.data.array_builder import (
    build_motif_ligand,
    build_motif_residue,
    build_sequence_design_structure,
)
from protenix.data.json_parser import (
    CCDLigand,
    CIFFile,
    FixedResidue,
    SequenceDesignInput,
    SMILESLigand,
)
from protenix.data.motif_parser import MotifLigand, MotifResidue


class TestBuildMotifResidue:
    """Tests for build_motif_residue function."""

    def test_build_histidine(self):
        """Test building a partial HIS sidechain."""
        coords = np.array(
            [
                [-8.390, -0.591, 11.905],  # CG
                [-7.724, -1.319, 12.926],  # CD2
                [-8.599, 0.626, 12.402],  # ND1
                [-8.111, 0.729, 13.670],  # CE1
                [-7.571, -0.471, 13.996],  # NE2
            ]
        )
        motif_residue = MotifResidue(
            res_id=94,
            res_name="HIS",
            atom_names=("CG", "CD2", "ND1", "CE1", "NE2"),
            coords=coords,
        )

        atom_array = build_motif_residue(motif_residue)

        # Check basic properties
        assert len(atom_array) > len(coords)  # Full residue has more atoms
        assert all(atom_array.res_id == 94)
        assert all(atom_array.res_name == "HIS")

        # Check is_motif annotation
        assert hasattr(atom_array, "is_motif")
        assert atom_array.is_motif.sum() == 5

        # Check is_motif_related annotation
        assert hasattr(atom_array, "is_motif_related")
        assert all(atom_array.is_motif_related)

        # Check coordinates transferred
        for i, atom_name in enumerate(["CG", "CD2", "ND1", "CE1", "NE2"]):
            idx = np.where(atom_array.atom_name == atom_name)[0][0]
            np.testing.assert_array_almost_equal(atom_array.coord[idx], coords[i])

        # Check reference features
        assert hasattr(atom_array, "ref_pos")
        assert hasattr(atom_array, "ref_charge")
        assert hasattr(atom_array, "ref_mask")

    def test_build_invalid_residue_raises(self):
        """Test that invalid residue name raises ValueError."""
        motif_residue = MotifResidue(
            res_id=1,
            res_name="INVALID",
            atom_names=("CA",),
            coords=np.array([[0.0, 0.0, 0.0]]),
        )

        # Error can come from CCD lookup or our code
        with pytest.raises(ValueError):
            build_motif_residue(motif_residue)


class TestBuildMotifLigand:
    """Tests for build_motif_ligand function."""

    def test_build_ccd_zinc(self):
        """Test building a CCD zinc ion ligand."""
        coords = np.array([[-6.844, -1.668, 15.437]])
        motif_ligand = MotifLigand(
            cif_res_name="ZN",
            atom_names=("ZN",),
            coords=coords,
            elements=("ZN",),
            full_ligand=CCDLigand(codes=["ZN"]),
        )

        atom_array = build_motif_ligand(motif_ligand)

        assert len(atom_array) == 1
        assert atom_array.atom_name[0] == "ZN"
        assert atom_array.is_motif[0]
        np.testing.assert_array_almost_equal(atom_array.coord[0], coords[0])

    def test_build_smiles_with_atom_map(self):
        """Test building a SMILES ligand with atom mapping."""
        smiles = "CC[O:1]"
        coords = np.array([[0.0, 0.0, 0.0]])
        atom_map = {"O1": 1}

        motif_ligand = MotifLigand(
            cif_res_name="ETH",
            atom_names=("O1",),
            coords=coords,
            elements=("O",),
            full_ligand=SMILESLigand(smiles=smiles, atom_map=atom_map),
        )

        atom_array = build_motif_ligand(motif_ligand)

        assert hasattr(atom_array, "is_motif")
        assert atom_array.is_motif.sum() == 1
        assert all(atom_array.is_motif_related)

    def test_build_smiles_missing_atom_map_raises(self):
        """Test that SMILES without atom_map raises ValueError."""
        motif_ligand = MotifLigand(
            cif_res_name="LIG",
            atom_names=("S1",),
            coords=np.array([[0.0, 0.0, 0.0]]),
            elements=("S",),
            full_ligand=SMILESLigand(smiles="C"),
        )

        with pytest.raises(ValueError, match="requires atom_map"):
            build_motif_ligand(motif_ligand)


class TestBuildSequenceDesignStructure:
    """Tests for build_sequence_design_structure function."""

    def test_file_not_found_raises(self, tmp_path):
        """Test that missing backbone file raises FileNotFoundError."""
        input_data = SequenceDesignInput(
            cif_file=CIFFile(path=tmp_path / "nonexistent.cif"),
        )

        with pytest.raises(FileNotFoundError, match="Backbone CIF file not found"):
            build_sequence_design_structure(input_data)

    @pytest.fixture
    def mock_mmcif_parser(self):
        """Create a mock MMCIFParser that returns a simple structure."""
        with patch("protenix.data.array_builder.MMCIFParser") as mock:
            # Create a simple mock structure with backbone + ligand
            atoms = AtomArray(5)
            atoms.atom_name = np.array(["N", "CA", "C", "O", "ZN"])
            atoms.coord = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [2.0, 1.5, 0.0],
                    [3.0, 1.5, 0.0],
                    [10.0, 10.0, 10.0],
                ]
            )
            atoms.res_id = np.array([1, 1, 1, 1, 2])
            atoms.res_name = np.array(["GLY", "GLY", "GLY", "GLY", "ZN"])
            atoms.chain_id = np.array(["A", "A", "A", "A", "B"])
            atoms.element = np.array(["N", "C", "C", "O", "ZN"])
            atoms.hetero = np.array([False, False, False, False, True])

            mock_instance = MagicMock()
            mock_instance.get_structure.return_value = atoms
            mock.return_value = mock_instance

            yield mock

    def test_fixed_residue_annotation(self, mock_mmcif_parser, tmp_path):
        """Test that fixed residues get is_fixed annotation."""
        # Create a temp CIF file (content doesn't matter since we're mocking)
        cif_file_path = tmp_path / "backbone.cif"
        cif_file_path.write_text("# Mock CIF")

        input_data = SequenceDesignInput(
            cif_file=CIFFile(path=cif_file_path),
            fixed_residues=[
                FixedResidue(chain_id="A", res_id=1, residue="GLY"),
            ],
        )

        result = build_sequence_design_structure(input_data)

        # Check is_fixed annotation exists
        assert hasattr(result, "is_fixed")

        # Check fixed residue atoms are marked
        gly_mask = result.res_name == "GLY"
        assert result.is_fixed[gly_mask].all()

        # Check ligand atoms are not marked as fixed
        zn_mask = result.res_name == "ZN"
        assert not result.is_fixed[zn_mask].any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
