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
from biotite.structure import AtomArray

from protenix.data import ccd
from protenix.data.array_builder import add_reference_features, build_ligand
from protenix.data.json_parser_new import CCDLigand, MotifInput, SMILESLigand
from protenix.data.parser import MMCIFParser

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MotifResidue:
    """Partial residue extracted from motif CIF file."""

    res_id: int
    res_name: str  # e.g., "HIS", "GLU", "THR"
    atom_names: tuple[str, ...]  # e.g., ("CG", "CD2", "ND1", "CE1", "NE2")
    coords: np.ndarray  # (N_atoms, 3)

    def to_atom_array(self) -> AtomArray:
        """
        Expand partial sidechain to full residue AtomArray from CCD.

        1. Get full residue from CCD (e.g., complete HIS with all atoms)
        2. Match motif atoms by atom_name
        3. Transfer coordinates for matching atoms
        4. Add is_motif annotation (True for atoms with motif coordinates)

        Returns:
            AtomArray: Full residue with coordinates transferred where available
        """

        full_residue = ccd.get_component_atom_array(
            self.res_name,
            keep_leaving_atoms=False,
            keep_hydrogens=False,
        )

        if full_residue is None:
            raise ValueError(f"Could not find CCD component for {self.res_name}")

        is_motif = np.zeros(len(full_residue), dtype=bool)

        for i, atom_name in enumerate(self.atom_names):
            match_idx = np.where(full_residue.atom_name == atom_name)[0]
            if len(match_idx) > 0:
                full_residue.coord[match_idx[0]] = self.coords[i]
                is_motif[match_idx[0]] = True
            else:
                logger.warning(
                    f"Motif atom {atom_name} not found in CCD component {self.res_name}"
                )

        full_residue.res_id[:] = self.res_id
        full_residue.set_annotation("is_motif", is_motif)
        full_residue.set_annotation(
            "is_motif_related", np.ones_like(is_motif, dtype=bool)
        )

        # Add ref_pos, ref_charge, ref_mask annotations
        full_residue = add_reference_features(full_residue)
        return full_residue


@dataclass(frozen=True)
class MotifLigand:
    """Ligand specification combining motif CIF fragment with full ligand description."""

    cif_res_name: str  # Residue / CCD name in the CIF (e.g., "ZN", "AL5", "LIG")
    atom_names: tuple[str, ...]  # Atom names from CIF (e.g., ("O1A", "S1", "N21"))
    coords: np.ndarray  # (n_atoms, 3) coordinates from CIF
    elements: tuple[str, ...]  # Element types (e.g., ("O", "S", "N"))

    full_ligand: CCDLigand | SMILESLigand

    def _is_ccd_ligand(self) -> bool:
        """Check if full_ligand is a CCD code (vs SMILES)."""
        return isinstance(self.full_ligand, CCDLigand)

    def to_atom_array(self) -> AtomArray:
        """
        Expand motif ligand to full ligand AtomArray.

        For CCD ligands: match by atom_name
        For SMILES with atom_map: use explicit mapping via atom map numbers

        Returns:
            AtomArray: Full ligand with motif coordinates transferred
        """
        ligand_info = build_ligand(self.full_ligand)
        full_ligand_array = ligand_info.atom_array

        is_motif = np.zeros(len(full_ligand_array), dtype=bool)

        # CCD: match by atom_name
        if self._is_ccd_ligand():
            for i, atom_name in enumerate(self.atom_names):
                match_idx = np.where(full_ligand_array.atom_name == atom_name)[0]
                if len(match_idx) == 0:
                    raise ValueError(
                        f"Motif atom '{atom_name}' not found in CCD ligand "
                        f"'{self.full_ligand.codes}'. Available atoms: "
                        f"{full_ligand_array.atom_name.tolist()}"
                    )
                full_ligand_array.coord[match_idx[0]] = self.coords[i]
                is_motif[match_idx[0]] = True

        # SMILES: use atom_map for coordinate transfer
        else:
            atom_map = self.full_ligand.atom_map
            if atom_map is None:
                raise ValueError(
                    f"SMILES ligand requires atom_map for coordinate transfer. "
                    f"SMILES: {self.full_ligand.smiles}"
                )

            map_num_to_atom_name = ligand_info.atom_map_to_atom_name

            for i, cif_atom_name in enumerate(self.atom_names):
                if cif_atom_name not in atom_map:
                    raise KeyError(
                        f"CIF atom '{cif_atom_name}' not found in atom_map. "
                        f"All motif atoms must be mapped. atom_map keys: "
                        f"{list(atom_map.keys())}"
                    )

                map_num = atom_map[cif_atom_name]
                if map_num not in map_num_to_atom_name:
                    raise KeyError(
                        f"Atom map number {map_num} (for '{cif_atom_name}') not found "
                        f"in SMILES. Use [atom:{map_num}] notation in SMILES string. "
                        f"Available map numbers: {list(map_num_to_atom_name.keys())}"
                    )

                target_atom_name = map_num_to_atom_name[map_num]
                match_idx = np.where(full_ligand_array.atom_name == target_atom_name)[0]
                if len(match_idx) == 0:
                    raise ValueError(
                        f"Atom name '{target_atom_name}' not found in ligand array. "
                        f"This is an internal error."
                    )
                full_ligand_array.coord[match_idx[0]] = self.coords[i]
                is_motif[match_idx[0]] = True

        full_ligand_array.set_annotation("is_motif", is_motif)
        full_ligand_array.set_annotation(
            "is_motif_related", np.ones_like(is_motif, dtype=bool)
        )

        return full_ligand_array


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
            motif_input: Validated MotifInput from json_parser_new

        Returns:
            Motif object with parsed residues and ligands

        Raises:
            FileNotFoundError: If the motif CIF file doesn't exist
        """
        cif_path = motif_input.motif.file
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

    def to_atom_array(self) -> AtomArray:
        """Generate combined AtomArray from all residues and ligands.

        Returns:
            AtomArray with is_motif and is_motif_related annotations
        """
        arrays = []

        for residue in self.residues:
            arrays.append(residue.to_atom_array())

        for ligand in self.ligands:
            arrays.append(ligand.to_atom_array())

        if not arrays:
            raise ValueError("Motif has no residues or ligands")

        combined = arrays[0]
        for arr in arrays[1:]:
            combined += arr

        return combined
