"""Tests for array_builder.py new functions."""

import numpy as np
import pytest
from biotite.structure import AtomArray

from artios.data.array_builder import (
    build_motif_ligand,
    build_motif_residue,
)
from artios.data.parsing.json import (
    CCDLigand,
    SMILESLigand,
)
from artios.data.parsing.motif import MotifLigand, MotifResidue


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
            chain_id="A",
        )

        atom_array = build_motif_residue(motif_residue, entity_id=1)

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
            chain_id="A",
        )

        # Error can come from CCD lookup or our code
        with pytest.raises(ValueError):
            build_motif_residue(motif_residue, entity_id=1)


class TestBuildMotifLigand:
    """Tests for build_motif_ligand function."""

    def test_build_ccd_zinc(self):
        """Test building a CCD zinc ion ligand."""
        coords = np.array([[-6.844, -1.668, 15.437]])
        motif_ligand = MotifLigand(
            res_id=1,
            cif_res_name="ZN",
            atom_names=("ZN",),
            coords=coords,
            elements=("ZN",),
            chain_id="A",
            full_ligand=CCDLigand(codes=["ZN"]),
        )

        atom_array = build_motif_ligand(motif_ligand, entity_id=2)

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
            res_id=1,
            cif_res_name="ETH",
            atom_names=("O1",),
            coords=coords,
            elements=("O",),
            chain_id="A",
            full_ligand=SMILESLigand(smiles=smiles, atom_map=atom_map),
        )

        atom_array = build_motif_ligand(motif_ligand, entity_id=2)

        assert hasattr(atom_array, "is_motif")
        assert atom_array.is_motif.sum() == 1
        assert all(atom_array.is_motif_related)

    def test_build_smiles_substructure_match_fails_raises(self):
        """Test that SMILES substructure match failure raises ValueError."""
        # Sulfur atom can't be found in methane - substructure match will fail
        motif_ligand = MotifLigand(
            res_id=1,
            cif_res_name="LIG",
            atom_names=("S1",),
            coords=np.array([[0.0, 0.0, 0.0]]),
            elements=("S",),
            chain_id="A",
            full_ligand=SMILESLigand(smiles="C"),
        )

        with pytest.raises(ValueError, match="Substructure match failed"):
            build_motif_ligand(motif_ligand, entity_id=2)

    def test_build_smiles_without_atom_map_substructure_match(self):
        """Test building a SMILES ligand using automatic substructure matching.

        Uses the DTZ (coelenterazine) plane motif - a complex bicyclic ligand
        with 13 atoms in the motif that would be tedious to manually map.
        """
        # DTZ plane motif atoms from his_arg_plane_dtz.cif (UNL residue)
        # fmt: off
        atom_names = ("O1", "N1", "N2", "N3", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C10")
        elements = ("O", "N", "N", "N", "C", "C", "C", "C", "C", "C", "C", "C", "C")
        coords = np.array([
            [0.901, -3.206, -1.203],   # O1
            [-0.225, -1.209, -0.636],  # N1
            [1.376, 0.323, -0.864],    # N2
            [-2.116, 0.682, 0.090],    # N3
            [0.121, 0.116, -0.537],    # C1
            [1.865, -0.927, -1.160],   # C2
            [0.892, -1.894, -1.024],   # C3
            [-0.890, 1.076, -0.147],   # C4
            [3.269, -1.099, -1.564],   # C5
            [-1.519, -1.613, -0.370],  # C6
            [-2.425, -0.674, -0.015],  # C7
            [-0.560, 2.516, -0.026],   # C8
            [-3.825, -1.069, 0.259],   # C10
        ])
        # fmt: on

        # DTZ SMILES without atom map tags
        dtz_smiles = "C1=CC=C(C=C1)CC2=C(N3C=C(N=C(C3=N2)C4=CC=CC=C4)C5=CC=CC=C5)O"

        motif_ligand = MotifLigand(
            res_id=1,
            cif_res_name="DTZ",
            atom_names=atom_names,
            coords=coords,
            elements=elements,
            chain_id="A",
            full_ligand=SMILESLigand(smiles=dtz_smiles),  # No atom_map!
        )

        # This should work via substructure matching
        atom_array = build_motif_ligand(motif_ligand, entity_id=2)

        # Verify is_motif annotation
        assert hasattr(atom_array, "is_motif")
        assert atom_array.is_motif.sum() == 13  # All 13 motif atoms should be marked

        # Verify is_motif_related annotation
        assert hasattr(atom_array, "is_motif_related")
        assert all(atom_array.is_motif_related)

        # Full DTZ has 29 heavy atoms (13 motif + 16 non-motif)
        assert len(atom_array) == 29

        # Verify mapping quality by comparing motif coords to RDKit reference conformer
        # after alignment (since they're in different reference frames)
        from biotite.structure import superimpose

        from artios.data.chem.rdkit import (
            infer_atom_map_from_substructure,
            smiles_to_atom_info,
        )

        # Get RDKit reference conformer
        ref_atom_info = smiles_to_atom_info(dtz_smiles)
        ref_array = ref_atom_info.atom_array

        # Get the atom name mapping
        cif_to_full_atom_name = infer_atom_map_from_substructure(
            motif_elements=elements,
            motif_coords=coords,
            motif_atom_names=atom_names,
            full_smiles=dtz_smiles,
        )

        # Build two aligned AtomArrays for the motif atoms only
        motif_ref_coords = []
        motif_cif_coords = []
        for i, cif_atom_name in enumerate(atom_names):
            full_atom_name = cif_to_full_atom_name[cif_atom_name]
            ref_idx = np.where(ref_array.atom_name == full_atom_name)[0][0]
            motif_ref_coords.append(ref_array.coord[ref_idx])
            motif_cif_coords.append(coords[i])

        motif_ref_coords = np.array(motif_ref_coords)
        motif_cif_coords = np.array(motif_cif_coords)

        ref_atoms = AtomArray(len(atom_names))
        ref_atoms.coord = motif_ref_coords

        cif_atoms = AtomArray(len(atom_names))
        cif_atoms.coord = motif_cif_coords

        # Superimpose CIF coords onto reference coords
        mobile_onto_fixed, _ = superimpose(fixed=ref_atoms, mobile=cif_atoms)

        # Compute per-atom distances after alignment
        distances = np.linalg.norm(ref_atoms.coord - mobile_onto_fixed.coord, axis=1)

        print(
            f"Mean distance: {distances.mean():.2f} Å, Max distance: {distances.max():.2f} Å"
        )

        # After alignment, the structures should match well if mapping is correct
        assert (
            distances.mean() < 0.1
        ), f"Mean distance too large: {distances.mean():.2f} Å"
        assert (
            distances.max() < 0.25
        ), f"Max distance too large: {distances.max():.2f} Å"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
