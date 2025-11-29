"""
Motif scaffolding inference parser.

Parses JSON input specifying:
- Protein length (design backbone)
- Motif CIF file (partial sidechains + ligand fragments with coordinates)
- Ligand specification (CCD code or SMILES)

Generates an annotated AtomArray suitable for model inference.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from protenix.data import ccd
from protenix.data.json_parser import (
    add_reference_features,
    build_ligand,
)
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

    # From motif CIF file
    cif_res_name: str  # Residue / CCD name in the CIF (e.g., "ZN", "AL5", "LIG")
    atom_names: tuple[str, ...]  # Atom names from CIF (e.g., ("O1A", "S1", "N21"))
    coords: np.ndarray  # (n_atoms, 3) coordinates from CIF
    elements: tuple[str, ...]  # Element types (e.g., ("O", "S", "N"))

    # From JSON specification
    full_ligand: str  # Full ligand: "CCD_ZN", "CCD_AL5", or SMILES string

    # Optional explicit atom mapping for SMILES
    # Maps CIF atom_name -> SMILES atom map number
    atom_map: Optional[dict[str, int]] = None

    def __post_init__(self) -> None:
        """Validate that SMILES ligands have an atom_map."""
        if not self.full_ligand.startswith("CCD_") and self.atom_map is None:
            raise ValueError(
                f"SMILES ligand '{self.full_ligand}' requires atom_map. "
                f"Use SMILES atom map notation [atom:N] and provide atom_map dict "
                f"mapping CIF atom names {self.atom_names} to map numbers."
            )

    def _is_ccd_ligand(self) -> bool:
        """Check if full_ligand is a CCD code (vs SMILES)."""
        return self.full_ligand.startswith("CCD_")

    def to_atom_array(self) -> AtomArray:
        """
        Expand motif ligand to full ligand AtomArray.

        For CCD ligands: match by atom_name
        For SMILES with atom_map: use explicit mapping via atom map numbers
        For SMILES without atom_map: raises error (use atom_map for reliable matching)

        Returns:
            AtomArray: Full ligand with motif coordinates transferred
        """
        # Build full ligand using build_ligand from json_parser
        # Works for both CCD ("CCD_ZN", "CCD_AL5") and SMILES
        # TODO: we can separate ions into separate things from ligands to make it explicit.
        entity_info = {"ligand": {"ligand": self.full_ligand, "count": 1}}

        ligand_info = build_ligand(entity_info)
        full_ligand = ligand_info["atom_array"]

        is_motif = np.zeros(len(full_ligand), dtype=bool)

        # CCD
        if self._is_ccd_ligand():
            # CCD ligand: match by atom_name
            for i, atom_name in enumerate(self.atom_names):
                match_idx = np.where(full_ligand.atom_name == atom_name)[0]
                if len(match_idx) == 0:
                    raise ValueError(
                        f"Motif atom '{atom_name}' not found in CCD ligand "
                        f"'{self.full_ligand}'. Available atoms: "
                        f"{full_ligand.atom_name.tolist()}"
                    )
                full_ligand.coord[match_idx[0]] = self.coords[i]
                is_motif[match_idx[0]] = True

        # SMILES
        else:
            map_num_to_atom_name = ligand_info.get("atom_map_to_atom_name", {})
            # map number -> AtomArray atom_name

            for i, cif_atom_name in enumerate(self.atom_names):
                if cif_atom_name not in self.atom_map:
                    raise KeyError(
                        f"CIF atom '{cif_atom_name}' not found in atom_map. "
                        f"All motif atoms must be mapped. atom_map keys: "
                        f"{list(self.atom_map.keys())}"
                    )

                map_num = self.atom_map[cif_atom_name]
                if map_num not in map_num_to_atom_name:
                    raise KeyError(
                        f"Atom map number {map_num} (for '{cif_atom_name}') not found "
                        f"in SMILES. Use [atom:{map_num}] notation in SMILES string. "
                        f"Available map numbers: {list(map_num_to_atom_name.keys())}"
                    )

                target_atom_name = map_num_to_atom_name[map_num]
                match_idx = np.where(full_ligand.atom_name == target_atom_name)[0]
                if len(match_idx) == 0:
                    raise ValueError(
                        f"Atom name '{target_atom_name}' not found in ligand array. "
                        f"This is an internal error."
                    )
                full_ligand.coord[match_idx[0]] = self.coords[i]
                is_motif[match_idx[0]] = True

        full_ligand.set_annotation("is_motif", is_motif)
        full_ligand.set_annotation(
            "is_motif_related", np.ones_like(is_motif, dtype=bool)
        )

        return full_ligand


def parse_motif(
    motif: dict,
    base_path: Optional[Path] = None,
) -> tuple[list[MotifResidue], list[MotifLigand]]:
    """
    Parse motif dict to build MotifResidue and MotifLigand objects.

    Args:
        motif: Dict with "file" (CIF path) and "ligands" (list of ligand specs)
        base_path: Base path for resolving relative CIF path. Defaults to cwd.

    Returns:
        (list of MotifResidue, list of MotifLigand)
    """
    # base_path = base_path or Path.cwd()
    motif_cif_path = motif["file"]

    # Build ligand spec lookup: cif_res_name -> spec
    ligand_specs = {spec["cif_res_name"]: spec for spec in motif.get("ligands", [])}

    # Parse CIF
    parser = MMCIFParser(motif_cif_path)
    motif_array = parser.get_structure()

    residues = []
    ligands = []

    res_starts = struc.get_residue_starts(motif_array, add_exclusive_stop=True)

    for start, stop in zip(res_starts[:-1], res_starts[1:]):
        res_slice = motif_array[start:stop]
        cif_res_name = str(res_slice.res_name[0])

        if res_slice.hetero[0]:
            # Ligand
            if cif_res_name not in ligand_specs:
                raise ValueError(
                    f"Ligand '{cif_res_name}' in CIF has no matching entry in "
                    f"motif.ligands. Available: {list(ligand_specs.keys())}"
                )
            spec = ligand_specs[cif_res_name]
            ligands.append(
                MotifLigand(
                    cif_res_name=cif_res_name,
                    atom_names=tuple(res_slice.atom_name.tolist()),
                    coords=res_slice.coord.copy(),
                    elements=tuple(res_slice.element.tolist()),
                    full_ligand=spec["full_ligand"],
                    atom_map=spec.get("atom_map"),
                )
            )
        else:
            # Protein residue
            residues.append(
                MotifResidue(
                    res_id=int(res_slice.res_id[0]),
                    res_name=cif_res_name,
                    atom_names=tuple(res_slice.atom_name.tolist()),
                    coords=res_slice.coord.copy(),
                )
            )

    logger.info(
        f"Parsed motif CIF: {len(residues)} residues, {len(ligands)} ligand fragments"
    )
    return residues, ligands
