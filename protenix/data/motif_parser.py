"""
Motif scaffolding inference parser.

Parses JSON input specifying:
- Protein length (design backbone)
- Motif CIF file (partial sidechains + ligand fragments with coordinates)
- Ligand specification (CCD code or SMILES)

Generates an annotated AtomArray suitable for model inference.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from rdkit import Chem

from protenix.data import ccd
from protenix.data.json_parser import (
    DesignAtomArrayBuilder,
    build_ligand,
)
from protenix.data.parser import AddAtomArrayAnnot, MMCIFParser

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MotifResidue:
    """Partial residue extracted from motif CIF file."""

    res_id: int
    res_name: str  # e.g., "HIS", "GLU", "THR"
    atom_names: tuple[str, ...]  # e.g., ("CG", "CD2", "ND1", "CE1", "NE2")
    coords: np.ndarray  # [N_atoms, 3]

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
        full_residue.coord = np.zeros_like(full_residue.coord)

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

        return full_residue


@dataclass(frozen=True)
class MotifLigand:
    """Ligand specification combining motif CIF fragment with full ligand description."""

    # From motif CIF file
    cif_res_name: str  # Residue name in the CIF (e.g., "ZN", "AL5")
    atom_names: tuple[str, ...]  # Atom names from CIF (e.g., ("O1A", "S1", "N21"))
    coords: np.ndarray  # [N_atoms, 3] coordinates from CIF
    elements: tuple[str, ...]  # Element types (e.g., ("O", "S", "N"))

    # From JSON specification
    full_ligand: str  # Full ligand: "CCD_ZN", "CCD_AL5", or SMILES string

    # Optional explicit atom mapping for SMILES
    # Maps CIF atom_name -> SMILES atom map number
    atom_map: Optional[dict[str, int]] = None

    # Metadata (optional, for debugging/logging)
    chain_id: Optional[str] = None
    res_id: Optional[int] = None

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
        entity_info = {"ligand": {"ligand": self.full_ligand, "count": 1}}

        ligand_info = build_ligand(entity_info)
        full_ligand = ligand_info["atom_array"]

        is_motif = np.zeros(len(full_ligand), dtype=bool)

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

        elif self.atom_map is not None:
            # SMILES with explicit atom_map

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

        else:
            raise ValueError(
                f"SMILES ligand '{self.full_ligand}' requires atom_map for reliable "
                f"coordinate matching. Use SMILES atom map notation [atom:N] and "
                f"provide atom_map dict mapping CIF atom names to map numbers."
            )

        full_ligand.set_annotation("is_motif", is_motif)

        return full_ligand


def parse_motif_cif(
    motif_cif_path: Path,
) -> tuple[list[MotifResidue], list[MotifLigand]]:
    """
    Parse motif CIF file to extract partial residues and ligand fragments.

    The motif CIF contains:
    - Protein residues (hetero=False): partial sidechains with coordinates
    - Ligand entities (hetero=True): fragments like ZN, AL5

    Args:
        motif_cif_path: Path to the motif CIF file

    Returns:
        tuple containing:
        - list of MotifResidue objects (protein sidechains)
        - list of MotifLigand objects (ligand fragments)
    """
    parser = MMCIFParser(motif_cif_path)
    motif_array = parser.get_structure()

    residues = []
    ligands = []

    # Group atoms by residue
    res_starts = struc.get_residue_starts(motif_array, add_exclusive_stop=True)

    for start, stop in zip(res_starts[:-1], res_starts[1:]):
        res_slice = motif_array[start:stop]

        if res_slice.hetero[0]:
            # Ligand entity
            # TODO: do res_name CCD code and smiles string stuff,
            # just need to add an extension to CCD codes.
            # to do this properly, might just need to get the whole json as input
            # to check if it is a CCD code or a smiles.
            ligands.append(
                MotifLigand(
                    chain_id=str(res_slice.chain_id[0]),
                    res_id=int(res_slice.res_id[0]),
                    res_name=str(res_slice.res_name[0]),
                    atom_names=res_slice.atom_name.tolist(),
                    coords=res_slice.coord.copy(),
                    elements=res_slice.element.tolist(),
                )
            )
        else:
            # Protein residue
            residues.append(
                MotifResidue(
                    res_id=int(res_slice.res_id[0]),
                    res_name=str(res_slice.res_name[0]),
                    atom_names=res_slice.atom_name.tolist(),
                    coords=res_slice.coord.copy(),
                )
            )

    logger.info(
        f"Parsed motif CIF: {len(residues)} residues, {len(ligands)} ligand fragments"
    )
    return residues, ligands


def expand_motif_residue(
    motif_res: MotifResidue,
) -> AtomArray:
    """
    Expand partial sidechain to full residue from CCD.

    1. Get full residue from CCD (e.g., complete HIS with all atoms)
    2. Match motif atoms by atom_name
    3. Transfer coordinates for matching atoms
    4. Return full residue with is_motif and is_motif_related annotations

    Args:
        motif_res: MotifResidue with partial atom coordinates

    Returns:
        AtomArray: Full residue with coordinates transferred where available,
            with is_motif and is_motif_related annotations added
    """
    full_residue = ccd.get_component_atom_array(
        motif_res.res_name,
        keep_leaving_atoms=False,
        keep_hydrogens=False,
    )

    if full_residue is None:
        raise ValueError(f"Could not find CCD component for {motif_res.res_name}")

    is_motif = np.zeros(len(full_residue), dtype=bool)

    # Set coords to zero as a safety measure
    full_residue.coord = np.zeros_like(full_residue.coord)

    # Transfer coordinates for matching atoms
    for i, atom_name in enumerate(motif_res.atom_names):
        match_idx = np.where(full_residue.atom_name == atom_name)[0]
        if len(match_idx) > 0:
            idx = match_idx[0]
            full_residue.coord[idx] = motif_res.coords[i]
            is_motif[idx] = True
        else:
            logger.warning(
                f"Motif atom {atom_name} not found in CCD component {motif_res.res_name}"
            )

    # Set residue ID
    full_residue.res_id[:] = motif_res.res_id

    # Add is_motif annotation
    full_residue.set_annotation("is_motif", is_motif)

    # Add is_motif_related annotation (all atoms in this residue are related)
    is_motif_related = np.ones(len(full_residue), dtype=bool)
    full_residue.set_annotation("is_motif_related", is_motif_related)

    return full_residue


def expand_ligand(motif_ligand: MotifLigand) -> AtomArray:
    """
    Expand motif ligand using the CCD code or the SMILES id.

    Args:
        motif_ligand: MotifLigand with partial coordinates

    Returns:
       AtomArray of the full ligand with the motif atom coordinates
       filled in.
    """

    pass


def match_ccd_ligand_coords(
    full_ligand: AtomArray,
    motif_ligand: MotifLigand,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match motif ligand atoms to full CCD ligand by atom_name.

    Args:
        full_ligand: Complete ligand AtomArray from CCD
        motif_ligand: MotifLigand with partial coordinates

    Returns:
        tuple containing:
        - np.ndarray: Updated coordinates with motif coords transferred
        - np.ndarray: Boolean mask indicating which atoms are from motif
    """
    is_motif = np.zeros(len(full_ligand), dtype=bool)
    coords = full_ligand.coord.copy()

    for i, atom_name in enumerate(motif_ligand.atom_names):
        match_idx = np.where(full_ligand.atom_name == atom_name)[0]
        if len(match_idx) > 0:
            idx = match_idx[0]
            coords[idx] = motif_ligand.coords[i]
            is_motif[idx] = True
        else:
            logger.warning(
                f"Motif atom {atom_name} not found in CCD ligand {motif_ligand.res_name}"
            )

    return coords, is_motif


def _create_mol_from_motif_fragment(motif_ligand: MotifLigand) -> Chem.Mol:
    """
    Create RDKit Mol from motif ligand fragment for substructure search.

    Uses element types and infers bonds from inter-atomic distances.

    Args:
        motif_ligand: MotifLigand with atom info and coordinates

    Returns:
        RDKit Mol object representing the fragment
    """
    mol = Chem.RWMol()

    # Add atoms (fix element capitalization for RDKit: ZN -> Zn)
    for element in motif_ligand.elements:
        # RDKit expects proper element capitalization
        element_fixed = element.capitalize()
        atom = Chem.Atom(element_fixed)
        mol.AddAtom(atom)

    # Infer bonds from distances
    coords = motif_ligand.coords
    n_atoms = len(coords)

    # Bond length thresholds by element pairs (approximate)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            # Use typical bond length threshold
            if dist < 2.0:  # Generous threshold for single/aromatic bonds
                mol.AddBond(i, j, Chem.BondType.SINGLE)

    return mol.GetMol()


def _get_heavy_atom_indices(mol: Chem.Mol) -> list[int]:
    """Get indices of non-hydrogen atoms in RDKit mol."""
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]


def match_smiles_ligand_coords(
    full_ligand: AtomArray,
    motif_ligand: MotifLigand,
    smiles: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match motif ligand atoms to SMILES ligand using RDKit substructure matching.

    The challenge: SMILES atom names (C1, C2...) don't match CIF atom names.
    Solution: Build mol from motif fragment, use GetSubstructMatches().

    Args:
        full_ligand: Complete ligand AtomArray from SMILES
        motif_ligand: MotifLigand with partial coordinates
        smiles: Original SMILES string

    Returns:
        tuple containing:
        - np.ndarray: Updated coordinates with motif coords transferred
        - np.ndarray: Boolean mask indicating which atoms are from motif
    """
    # Create RDKit mol from motif fragment
    motif_mol = _create_mol_from_motif_fragment(motif_ligand)

    # Get mol from SMILES
    full_mol = Chem.MolFromSmiles(smiles)
    if full_mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    full_mol = Chem.AddHs(full_mol)

    # Find substructure match
    matches = full_mol.GetSubstructMatches(motif_mol, useChirality=False)

    if len(matches) == 0:
        raise ValueError(
            f"Could not find motif fragment as substructure of SMILES ligand. "
            f"Motif atoms: {motif_ligand.atom_names}, SMILES: {smiles}"
        )

    # Use first match
    match = matches[0]  # Tuple: motif_idx -> full_mol_idx

    logger.info(f"Found {len(matches)} substructure match(es), using first")

    # Build mapping from full_mol heavy atom indices to atom_array indices
    heavy_atom_indices = _get_heavy_atom_indices(full_mol)
    mol_idx_to_array_idx = {
        mol_idx: array_idx for array_idx, mol_idx in enumerate(heavy_atom_indices)
    }

    # Transfer coordinates
    is_motif = np.zeros(len(full_ligand), dtype=bool)
    coords = full_ligand.coord.copy()

    for motif_atom_idx, full_mol_atom_idx in enumerate(match):
        # Check if this is a heavy atom (in our atom_array)
        if full_mol_atom_idx in mol_idx_to_array_idx:
            array_idx = mol_idx_to_array_idx[full_mol_atom_idx]
            coords[array_idx] = motif_ligand.coords[motif_atom_idx]
            is_motif[array_idx] = True

    return coords, is_motif


def add_is_motif_related(atom_array: AtomArray) -> AtomArray:
    """
    Set is_motif_related annotation.

    An atom is motif_related if ANY atom in its ref_space_uid group has is_motif=1.
    This follows the pattern from protenix/data/motif.py.

    Args:
        atom_array: AtomArray with is_motif and ref_space_uid annotations

    Returns:
        AtomArray with is_motif_related annotation added
    """
    is_motif_related = np.zeros(len(atom_array), dtype=int)

    # Get unique ref_space_uids that contain motif atoms
    motif_ref_uids = np.unique(atom_array[atom_array.is_motif > 0].ref_space_uid)

    # Mark all atoms in these residues/ligands
    is_motif_related[np.isin(atom_array.ref_space_uid, motif_ref_uids)] = 1

    atom_array.set_annotation("is_motif_related", is_motif_related)
    return atom_array


class MotifScaffoldParser:
    """
    Parser for motif scaffolding inference input.

    Given a JSON specification with:
    - Protein length
    - Motif CIF file (partial sidechains + ligand fragments with coordinates)
    - Ligand specification (CCD code or SMILES)

    Generates an annotated AtomArray suitable for model inference.
    """

    def __init__(
        self,
        json_input: Union[str, Path, dict],
        base_path: Optional[Path] = None,
    ):
        """
        Initialize parser with JSON input.

        Args:
            json_input: Path to JSON file, JSON string, or dict
            base_path: Base path for resolving relative file paths in JSON.
                       If None, uses the directory containing the JSON file.
        """
        if isinstance(json_input, dict):
            self.config = json_input
            self.base_path = base_path or Path.cwd()
        elif isinstance(json_input, (str, Path)):
            json_path = Path(json_input)
            with open(json_path) as f:
                data = json.load(f)
                # Handle both single config and list of configs
                self.config = data[0] if isinstance(data, list) else data
            self.base_path = base_path or json_path.parent
        else:
            raise TypeError(f"Invalid json_input type: {type(json_input)}")

        self._parse_config()

    def _parse_config(self) -> None:
        """Parse and validate the JSON configuration."""
        # Extract protein length from design
        design = self.config.get("design", [])
        if not design:
            raise ValueError("Missing 'design' section in config")

        protein_config = design[0].get("proteinChain", {})
        self.protein_length = protein_config.get("length")
        if self.protein_length is None:
            raise ValueError("Missing 'length' in proteinChain config")

        # Extract motif CIF path
        motif = self.config.get("motif", [])
        if not motif:
            raise ValueError("Missing 'motif' section in config")

        motif_file = motif[0].get("ligand", {}).get("file")
        if motif_file is None:
            raise ValueError("Missing 'file' in motif config")
        self.motif_cif_path = self.base_path / motif_file

        # Extract ligand specification(s)
        ligand_configs = self.config.get("ligand", [])
        self.ligand_specs = []
        for lig_config in ligand_configs:
            if "ligand" in lig_config:
                self.ligand_specs.append(
                    {
                        "type": "ligand",
                        "spec": lig_config["ligand"]["ligand"],
                        "count": lig_config["ligand"].get("count", 1),
                    }
                )
            elif "ion" in lig_config:
                self.ligand_specs.append(
                    {
                        "type": "ion",
                        "spec": lig_config["ion"]["ion"],
                        "count": lig_config["ion"].get("count", 1),
                    }
                )

    def _build_design_protein(self) -> AtomArray:
        """Generate poly-glycine backbone for the design protein."""
        builder = DesignAtomArrayBuilder(
            length=self.protein_length,
            mol_type="protein",
            chain_id="A",
            entity_id="1",
        )
        protein = builder.build_all_annotations()

        # Add is_motif annotation (all zeros for design backbone)
        protein.add_annotation("is_motif", dtype=bool)
        protein.is_motif[:] = False
        # TODO: add is motif related false as well

        return protein

    def _build_ligand(self, ligand_spec: dict) -> tuple[AtomArray, str]:
        """
        Build ligand from specification.

        Returns:
            tuple of (AtomArray, original_spec_string)
        """
        spec = ligand_spec["spec"]

        if ligand_spec["type"] == "ion":
            entity_info = {"ion": {"ion": spec, "count": 1}}
        else:
            entity_info = {"ligand": {"ligand": spec, "count": 1}}

        ligand_info = build_ligand(entity_info)
        ligand_array = ligand_info["atom_array"]

        # Add is_motif annotation (will be updated later)
        ligand_array.add_annotation("is_motif", dtype=int)
        ligand_array.is_motif[:] = 0

        return ligand_array, spec

    def _is_ccd_ligand(self, spec: str) -> bool:
        """Check if ligand spec is a CCD code."""
        return spec.startswith("CCD_") or len(spec) <= 3  # Ion codes are short

    def _match_ligand_to_motif(
        self,
        ligand_array: AtomArray,
        motif_ligands: list[MotifLigand],
        ligand_spec: str,
    ) -> AtomArray:
        """
        Match ligand atoms to motif fragments and transfer coordinates.

        Args:
            ligand_array: Full ligand AtomArray
            motif_ligands: List of MotifLigand fragments from CIF
            ligand_spec: Original ligand specification string

        Returns:
            Updated AtomArray with motif coordinates and is_motif annotation
        """
        # Find matching motif ligand by res_name
        # For CCD ligands, match by the CCD code
        if self._is_ccd_ligand(ligand_spec):
            ccd_code = (
                ligand_spec[4:] if ligand_spec.startswith("CCD_") else ligand_spec
            )
            # May have multiple CCD codes separated by underscore
            ccd_codes = ccd_code.split("_")

            for motif_lig in motif_ligands:
                if motif_lig.res_name in ccd_codes:
                    coords, is_motif = match_ccd_ligand_coords(ligand_array, motif_lig)
                    ligand_array.coord[:] = coords
                    ligand_array.is_motif[:] = is_motif.astype(int)
                    logger.info(
                        f"Matched {is_motif.sum()} atoms for ligand {motif_lig.res_name}"
                    )
        else:
            # SMILES ligand - use substructure matching
            # Only try to match multi-atom organic fragments, not metal ions
            organic_fragments = [
                lig
                for lig in motif_ligands
                if len(lig.atom_names) > 1  # Skip single-atom ions
            ]

            for motif_lig in organic_fragments:
                try:
                    coords, is_motif = match_smiles_ligand_coords(
                        ligand_array, motif_lig, ligand_spec
                    )
                    ligand_array.coord[:] = coords
                    ligand_array.is_motif[:] = is_motif.astype(int)
                    logger.info(
                        f"Matched {is_motif.sum()} atoms for SMILES ligand via substructure "
                        f"(fragment: {motif_lig.res_name})"
                    )
                    break
                except ValueError as e:
                    logger.warning(
                        f"Substructure match failed for {motif_lig.res_name}: {e}"
                    )

        return ligand_array

    def _assemble_atom_array(
        self,
        design_protein: AtomArray,
        motif_residues: list[tuple[AtomArray, np.ndarray]],
        ligands: list[AtomArray],
    ) -> AtomArray:
        """
        Assemble final AtomArray from components.

        Components:
        - Chain A: Design protein (poly-glycine backbone)
        - Chains B, C, ...: Motif residues (complete sidechains)
        - Remaining chains: Ligands

        Args:
            design_protein: Poly-glycine backbone
            motif_residues: List of (residue_array, is_motif_mask) tuples
            ligands: List of ligand AtomArrays

        Returns:
            Combined AtomArray with all components
        """
        # Start with design protein (Chain A)
        result = design_protein.copy()

        # Chain ID counter (B, C, D, ...)
        chain_idx = 1
        entity_idx = 2

        # Add motif residues as separate chains
        for residue_array, is_motif_mask in motif_residues:
            chain_id = chr(ord("A") + chain_idx)

            # Set chain and entity annotations
            residue_array = residue_array.copy()
            residue_array.chain_id[:] = chain_id

            if not hasattr(residue_array, "label_asym_id"):
                residue_array.add_annotation("label_asym_id", dtype=str)
            residue_array.label_asym_id[:] = chain_id

            if not hasattr(residue_array, "label_entity_id"):
                residue_array.add_annotation("label_entity_id", dtype=str)
            residue_array.label_entity_id[:] = str(entity_idx)

            if not hasattr(residue_array, "is_motif"):
                residue_array.add_annotation("is_motif", dtype=int)
            residue_array.is_motif[:] = is_motif_mask.astype(int)

            # Ensure all necessary annotations exist
            residue_array = self._ensure_annotations(residue_array)

            result = result + residue_array
            chain_idx += 1
            entity_idx += 1

        # Add ligands
        for ligand_array in ligands:
            chain_id = chr(ord("A") + chain_idx)

            ligand_array = ligand_array.copy()
            ligand_array.chain_id[:] = chain_id

            if not hasattr(ligand_array, "label_asym_id"):
                ligand_array.add_annotation("label_asym_id", dtype=str)
            ligand_array.label_asym_id[:] = chain_id

            if not hasattr(ligand_array, "label_entity_id"):
                ligand_array.add_annotation("label_entity_id", dtype=str)
            ligand_array.label_entity_id[:] = str(entity_idx)

            # Ensure all necessary annotations exist
            ligand_array = self._ensure_annotations(ligand_array)

            result = result + ligand_array
            chain_idx += 1
            entity_idx += 1

        return result

    def _ensure_annotations(self, atom_array: AtomArray) -> AtomArray:
        """Ensure all required annotations exist on atom array."""
        # List of annotations that might be missing
        annotations = {
            "ins_code": ("", str),
            "label_alt_id": (".", str),
            "occupancy": (1.0, np.float32),
            "b_factor": (0.0, np.float32),
            "charge": (0, np.int8),
            "auth_asym_id": (
                atom_array.chain_id[0] if len(atom_array) > 0 else "A",
                str,
            ),
            "auth_seq_id": (None, str),  # Will be set from res_id
            "label_seq_id": (None, str),  # Will be set from res_id
            "is_resolved": (True, bool),
        }

        for annot_name, (default_val, dtype) in annotations.items():
            if not hasattr(atom_array, annot_name):
                atom_array.add_annotation(annot_name, dtype=dtype)
                if annot_name in ("auth_seq_id", "label_seq_id"):
                    atom_array.set_annotation(
                        annot_name,
                        [str(rid) for rid in atom_array.res_id],
                    )
                elif annot_name == "auth_asym_id":
                    atom_array.set_annotation(annot_name, atom_array.chain_id.copy())
                else:
                    atom_array.set_annotation(
                        annot_name,
                        np.full(len(atom_array), default_val, dtype=dtype),
                    )

        return atom_array

    def _add_all_annotations(self, atom_array: AtomArray) -> AtomArray:
        """
        Add all required annotations for inference.

        Uses existing AddAtomArrayAnnot methods from parser.py.
        """
        # Build entity_poly_type mapping
        # Entity 1 is always the design protein
        entity_poly_type = {"1": "polypeptide(L)"}

        # Add mol_type annotations
        atom_array = AddAtomArrayAnnot.add_token_mol_type(atom_array, entity_poly_type)
        atom_array = AddAtomArrayAnnot.add_centre_atom_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_atom_mol_type_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_plddt_m_rep_atom_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_cano_seq_resname(atom_array)
        atom_array = AddAtomArrayAnnot.add_tokatom_idx(atom_array)
        atom_array = AddAtomArrayAnnot.add_modified_res_mask(atom_array)

        # Add chain/entity ID annotations
        atom_array = AddAtomArrayAnnot.unique_chain_and_add_ids(atom_array)
        atom_array = AddAtomArrayAnnot.find_equiv_mol_and_assign_ids(
            atom_array, entity_poly_type
        )

        # Add ref_space_uid for motif_related calculation
        atom_array = AddAtomArrayAnnot.add_ref_space_uid(atom_array)

        # Add reference features
        atom_array = AddAtomArrayAnnot.add_ref_info_and_res_perm(atom_array)

        # Add is_motif_related using ref_space_uid
        atom_array = add_is_motif_related(atom_array)

        return atom_array

    def parse(self) -> AtomArray:
        """
        Main entry point. Parse JSON input and generate annotated AtomArray.

        Returns:
            AtomArray with all annotations required for model inference,
            including is_motif and is_motif_related flags.
        """
        logger.info(
            f"Parsing motif scaffold config: {self.config.get('name', 'unnamed')}"
        )

        # 1. Build design protein (poly-glycine backbone)
        design_protein = self._build_design_protein()
        logger.info(f"Built design protein with {self.protein_length} residues")

        # 2. Parse motif CIF
        motif_residues_raw, motif_ligands = parse_motif_cif(self.motif_cif_path)

        # 3. Expand motif residues to full sidechains
        motif_residues = []
        for motif_res in motif_residues_raw:
            full_res, is_motif = expand_motif_residue(motif_res)
            motif_residues.append((full_res, is_motif))
            logger.info(
                f"Expanded {motif_res.res_name}{motif_res.res_id}: "
                f"{is_motif.sum()}/{len(full_res)} atoms from motif"
            )

        # 4. Build ligands and match to motif
        ligands = []
        for ligand_spec in self.ligand_specs:
            ligand_array, spec_str = self._build_ligand(ligand_spec)
            ligand_array = self._match_ligand_to_motif(
                ligand_array, motif_ligands, spec_str
            )
            ligands.append(ligand_array)

        # 5. Assemble all components
        atom_array = self._assemble_atom_array(design_protein, motif_residues, ligands)
        logger.info(f"Assembled atom array with {len(atom_array)} atoms")

        # 6. Add all annotations
        atom_array = self._add_all_annotations(atom_array)

        # Log summary
        n_motif = atom_array.is_motif.sum()
        n_motif_related = atom_array.is_motif_related.sum()
        logger.info(
            f"Final atom array: {len(atom_array)} atoms, "
            f"{n_motif} motif atoms, {n_motif_related} motif-related atoms"
        )

        return atom_array
