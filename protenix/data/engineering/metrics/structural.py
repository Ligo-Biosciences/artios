"""Structural metrics calculation for protein structures."""

import io
from typing import Tuple

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np


class ProteinMetricsCalculator:
    """A class for calculating various metrics for protein structures."""

    @staticmethod
    def structure_from_pdb_string(pdb_str: str) -> struc.AtomArray:
        """Takes a PDB string and returns an AtomArray object.

        Args:
            pdb_str: The contents of the pdb file

        Returns:
            A new Biotite AtomArray parsed from the pdb contents.
        """
        pdb_file = pdb.PDBFile.read(io.StringIO(pdb_str))
        structure = pdb.get_structure(pdb_file, extra_fields=["b_factor"])
        return structure.get_array(0)  # get first model

    @staticmethod
    def calculate_radius_of_gyration(structure: struc.AtomArray) -> float:
        """Calculate radius of gyration for a protein structure.

        Args:
            structure: Biotite AtomArray object.

        Returns:
            Radius of gyration in Angstroms.
        """
        # Get coordinates of all atoms
        coords = structure.coord

        # Calculate center of mass
        center = np.mean(coords, axis=0)

        # Calculate radius of gyration
        rg = np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))
        return float(rg)

    @staticmethod
    def calculate_plddt_stats(structure: struc.AtomArray) -> Tuple[float, float]:
        """Calculate mean and standard deviation of pLDDT scores.

        Args:
            structure: Biotite AtomArray object with b_factor field containing pLDDT.

        Returns:
            Tuple of (mean_plddt, std_plddt).
        """
        plddt_scores = structure.b_factor
        return float(np.mean(plddt_scores)), float(np.std(plddt_scores))

    @staticmethod
    def calculate_average_atom_sasa(
        structure: struc.AtomArray, point_number: int = 10
    ) -> float:
        """Calculate average atom Solvent Accessible Surface Area.

        Args:
            structure: Biotite AtomArray object.
            point_number: Number of points to use for the SASA calculation.

        Returns:
            Average SASA in Angstroms squared.
        """
        atom_wise_sasa = struc.sasa(structure, point_number=point_number)
        atom_wise_sasa = np.nan_to_num(
            atom_wise_sasa, copy=False
        )  # replace NaNs inplace
        return float(np.mean(atom_wise_sasa))

    @staticmethod
    def calculate_sequence_length(structure: struc.AtomArray) -> int:
        """Calculate sequence length by counting CA atoms.

        Args:
            structure: Biotite AtomArray object.

        Returns:
            Number of residues (CA atoms).
        """
        return len(structure[structure.atom_name == "CA"])
