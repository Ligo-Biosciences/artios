"""
Motif scaffolding inference parser.

Parses MotifInput specifying:
- Motif CIF file (partial sidechains + ligand fragments with coordinates)
- Ligand specification (CCD code or SMILES)

Generates a Motif object that can be used for generating atom arrays for model inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import biotite.structure as struc
import numpy as np

from protenix.data.json_parser import CCDLigand, MotifInput, SMILESLigand
from protenix.data.parser import MMCIFParser

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MotifResidue:
    """Partial residue extracted from motif CIF file."""

    res_id: int
    res_name: str  # e.g., "HIS", "GLU", "THR"
    atom_names: tuple[str, ...]  # e.g., ("CG", "CD2", "ND1", "CE1", "NE2")
    coords: np.ndarray  # (N_atoms, 3)


@dataclass(frozen=True)
class MotifLigand:
    """Ligand specification combining motif CIF fragment with full ligand description."""

    cif_res_name: str  # Residue / CCD name in the CIF (e.g., "ZN", "AL5", "LIG")
    atom_names: tuple[str, ...]  # Atom names from CIF (e.g., ("O1A", "S1", "N21"))
    coords: np.ndarray  # (n_atoms, 3) coordinates from CIF
    elements: tuple[str, ...]  # Element types (e.g., ("O", "S", "N"))

    full_ligand: CCDLigand | SMILESLigand


@dataclass
class Motif:
    """Complete parsed motif with residues and ligands."""

    residues: list[MotifResidue]
    ligands: list[MotifLigand]

    @classmethod
    def from_input(cls, motif_input: MotifInput) -> Motif:
        """Build Motif from validated MotifInput.

        Parses the motif CIF file and creates MotifResidue objects for
        protein residues (hetero=False) and MotifLigand objects for
        heteroatoms with mappings in the input.

        Args:
            motif_input: Validated MotifInput from json_parser.py

        Returns:
            Motif object with parsed residues and ligands

        Raises:
            FileNotFoundError: If the motif CIF file doesn't exist
        """
        cif_path = motif_input.motif.cif_file.path
        if not cif_path.exists():
            raise FileNotFoundError(f"Motif CIF file not found: {cif_path}")

        parser = MMCIFParser(cif_path)
        atoms = parser.get_structure()

        # Build ligand mapping: cif_res_name -> full_ligand
        ligand_mappings = {
            mapping.cif_res_name: mapping.full_ligand
            for mapping in motif_input.motif.ligands
        }

        residues: list[MotifResidue] = []
        ligands: list[MotifLigand] = []

        # Group atoms by residue
        res_starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)

        for start, stop in zip(res_starts[:-1], res_starts[1:]):
            res_atoms = atoms[start:stop]
            res_name = res_atoms.res_name[0]
            res_id = int(res_atoms.res_id[0])
            atom_names = tuple(res_atoms.atom_name)
            coords = res_atoms.coord.copy()
            elements = tuple(res_atoms.element)
            is_hetero = res_atoms.hetero[0]

            # Protein residue (ATOM records)
            if not is_hetero:
                residues.append(
                    MotifResidue(
                        res_id=res_id,
                        res_name=res_name,
                        atom_names=atom_names,
                        coords=coords,
                    )
                )
            # Ligand (HETATM records) - requires mapping
            elif res_name in ligand_mappings:
                ligands.append(
                    MotifLigand(
                        cif_res_name=res_name,
                        atom_names=atom_names,
                        coords=coords,
                        elements=elements,
                        full_ligand=ligand_mappings[res_name],
                    )
                )
            else:
                logger.warning(
                    f"HETATM residue '{res_name}' (res_id={res_id}) has no ligand "
                    f"mapping in input. Skipping."
                )

        return cls(residues=residues, ligands=ligands)
