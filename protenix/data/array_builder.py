# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from protenix.common.constants import BACKBONE_DEFS
from protenix.data import ccd
from protenix.data.json_parser_new import (
    Backbone,
    CCDLigand,
    MotifInput,
    SequenceDesignInput,
    SMILESLigand,
)
from protenix.data.json_parser_utils import (
    DNA_1to3,
    PROTEIN_1to3,
    RNA_1to3,
    add_bonds_to_terminal_residues,
    add_reference_features,
    remove_leaving_atoms,
    remove_non_std_ccd_leaving_atoms,
)
from protenix.data.parser import MMCIFParser
from protenix.data.rdkit import AtomInfo, smiles_to_atom_info

logger = logging.getLogger(__name__)


def motif_input_to_atom_array(motif_input: MotifInput) -> AtomArray:
    """
    Process motif scaffolding input to build AtomArray.
    """
    assert len(motif_input.design) == 1, "Only one design entity is supported"

    # Design
    design_atom_array = build_backbone(motif_input.design[0])

    design_copy = copy.deepcopy(design_atom_array)
    design_chain_id = ["0"] * len(design_copy)
    design_copy.set_annotation("label_asym_id", design_chain_id)
    design_copy.set_annotation("auth_asym_id", design_chain_id)
    design_copy.set_annotation("chain_id", design_chain_id)
    design_copy.set_annotation("label_seq_id", design_copy.res_id)
    design_copy.set_annotation("copy_id", ["1"] * len(design_copy))
    design_copy.set_annotation("label_entity_id", ["1"] * len(design_copy))
    atom_array = design_copy

    # Motif
    motif_atom_array = Motif.from_input(motif_input).to_atom_array()
    motif_copy = copy.deepcopy(motif_atom_array)

    # TODO: temporary chain annotations before the correct IDs are inferred
    # in the `add_atom_array_attributes` function of SampleDictToFeatures
    motif_chain_id = ["1"] * len(motif_copy)
    motif_copy.set_annotation("label_asym_id", motif_chain_id)
    motif_copy.set_annotation("auth_asym_id", motif_chain_id)
    motif_copy.set_annotation("chain_id", motif_chain_id)
    motif_copy.set_annotation("label_seq_id", motif_copy.res_id)
    motif_copy.set_annotation("copy_id", ["1"] * len(motif_copy))
    motif_copy.set_annotation("label_entity_id", ["2"] * len(motif_copy))
    motif_copy.set_annotation("is_target", [False] * len(motif_copy))
    motif_copy.set_annotation("is_hotspot", [False] * len(motif_copy))
    motif_copy.set_annotation("is_resolved", [True] * len(motif_copy))
    atom_array += motif_copy

    return atom_array


def sequence_input_to_atom_array(sequence_input: SequenceDesignInput) -> AtomArray:
    """Handles atom array generation for tasks with a sequence input and backbone.
    This is required for packing design tasks."""
    return build_sequence_design_structure(sequence_input)


def backbone_input_to_atom_array(single_job_dict: dict) -> AtomArray:
    """Handles atom array generation for tasks with a backbone input.
    This is required for sequence design tasks."""
    raise NotImplementedError("Backbone generation is not supported yet.")


def unconditional_input_to_atom_arrays(single_job_dict: dict) -> dict:
    """Handles atom array generation for unconditional generation task."""
    raise NotImplementedError("Unconditional generation is not supported yet.")


def process_input_to_atom_array(
    input: MotifInput | SequenceDesignInput,
) -> AtomArray:
    """Dispatch to appropriate task-specific processor based on input structure."""
    if isinstance(input, MotifInput):
        return motif_input_to_atom_array(input)
    elif isinstance(input, SequenceDesignInput):
        return sequence_input_to_atom_array(input)
    else:
        raise ValueError(f"Unknown input type: {type(input)}")


def _build_polymer_atom_array(ccd_seqs: list[str]) -> AtomArray:
    """
    Build polymer atom_array from ccd codes, but not remove leaving atoms

    Args:
        ccd_seqs: a list of ccd code in sequence, ["MET", "ALA"] or ["DA", "DT"]

    Returns:
        AtomArray: Biotite AtomArray of chain
    """
    chain = struc.AtomArray(0)
    for res_id, res_name in enumerate(ccd_seqs):
        # Keep all leaving atoms, will remove leaving atoms later
        residue = ccd.get_component_atom_array(
            res_name, keep_leaving_atoms=True, keep_hydrogens=False
        )
        residue.res_id[:] = res_id + 1
        chain += residue
    res_starts = struc.get_residue_starts(chain, add_exclusive_stop=True)
    polymer_bonds = ccd._connect_inter_residue(chain, res_starts)

    if chain.bonds is None:
        chain.bonds = polymer_bonds
    else:
        chain.bonds = chain.bonds.merge(polymer_bonds)

    chain = add_bonds_to_terminal_residues(chain)

    bond_count = {}
    for i, j, t in polymer_bonds._bonds:
        bond_count[i] = bond_count.get(i, 0) + 1
        bond_count[j] = bond_count.get(j, 0) + 1

    chain = remove_leaving_atoms(chain, bond_count)

    chain = remove_non_std_ccd_leaving_atoms(chain)

    return chain


def build_backbone(backbone: Backbone) -> AtomArray:
    """
    Build a backbone from a Backbone pydantic object.
    """
    sequence = [PROTEIN_1to3["G"] for _ in range(backbone.length)]
    atom_array = _build_polymer_atom_array(sequence)

    atom_array = add_reference_features(atom_array)

    atom_array.set_annotation("is_motif_related", [False] * len(atom_array))
    atom_array.set_annotation("is_motif", [False] * len(atom_array))
    atom_array.set_annotation("is_resolved", [True] * len(atom_array))
    atom_array.set_annotation("is_target", [False] * len(atom_array))
    atom_array.set_annotation("is_hotspot", [False] * len(atom_array))
    atom_array.hetero[:] = False

    return atom_array


def build_polymer(entity_info: dict) -> AtomArray:
    """
    Build a polymer from a polymer info dict
    example: {
        "name": "polymer",
        "sequence": "GPDSMEEVVVPEEPPKLVSALATYVQQERLCTMFLSIANKLLPLKP",
        "count": 1
        }

    Args:
        item (dict): polymer info dict

    Returns:
        AtomArray: Biotite AtomArray of chain
    """
    poly_type, info = list(entity_info.items())[0]
    if poly_type == "proteinChain":
        ccd_seqs = [PROTEIN_1to3[x] for x in info["sequence"]]
        if modifications := info.get("modifications"):
            for m in modifications:
                index = m["ptmPosition"] - 1
                mtype = m["ptmType"]
                if mtype.startswith("CCD_"):
                    ccd_seqs[index] = mtype[4:]
                else:
                    raise ValueError(f"unknown modification type: {mtype}")
        if glycans := info.get("glycans"):
            logging.warning(f"glycans not supported: {glycans}")
        chain_array = _build_polymer_atom_array(ccd_seqs)

    elif poly_type in ("dnaSequence", "rnaSequence"):
        map_1to3 = DNA_1to3 if poly_type == "dnaSequence" else RNA_1to3
        ccd_seqs = [map_1to3[x] for x in info["sequence"]]
        if modifications := info.get("modifications"):
            for m in modifications:
                index = m["basePosition"] - 1
                mtype = m["modificationType"]
                if mtype.startswith("CCD_"):
                    ccd_seqs[index] = mtype[4:]
                else:
                    raise ValueError(f"unknown modification type: {mtype}")
        chain_array = _build_polymer_atom_array(ccd_seqs)

    else:
        raise ValueError(
            "polymer type must be proteinChain, dnaSequence or rnaSequence"
        )

    chain_array = add_reference_features(chain_array)

    chain_array.set_annotation("is_motif_related", [False] * len(chain_array))
    chain_array.set_annotation("is_motif", [False] * len(chain_array))
    chain_array.set_annotation("is_resolved", [True] * len(chain_array))
    chain_array.set_annotation("is_target", [False] * len(chain_array))
    chain_array.set_annotation("is_hotspot", [False] * len(chain_array))

    return chain_array


def build_ligand(ligand: CCDLigand | SMILESLigand) -> AtomInfo:
    """Build a ligand AtomArray from a typed ligand specification.

    Args:
        ligand: CCDLigand or SMILESLigand from json_parser_new

    Returns:
        AtomInfo with atom_array and atom_map_to_atom_name (for SMILES)

    Raises:
        TypeError: If ligand is neither CCDLigand nor SMILESLigand
    """
    # NOTE: the res_ids assigned here will either need to be changed downstream
    # or inside this function.
    if isinstance(ligand, CCDLigand):
        atom_array = AtomArray(0)
        res_ids = []
        for idx, code in enumerate(ligand.codes):
            ccd_atom_array = ccd.get_component_atom_array(
                code, keep_leaving_atoms=True, keep_hydrogens=False
            )
            atom_array += ccd_atom_array
            res_id = idx + 1
            res_ids += [res_id] * len(ccd_atom_array)
        atom_array.res_id[:] = res_ids
        atom_array = add_reference_features(atom_array)
        return AtomInfo(atom_array=atom_array)

    elif isinstance(ligand, SMILESLigand):
        atom_info = smiles_to_atom_info(ligand.smiles)
        atom_info.atom_array.res_id[:] = 1
        atom_array = add_reference_features(atom_info.atom_array)
        return AtomInfo(
            atom_array=atom_array,
            atom_map_to_atom_name=atom_info.atom_map_to_atom_name,
        )

    else:
        raise TypeError(
            f"Ligand must be CCDLigand or SMILESLigand, got {type(ligand).__name__}"
        )


def build_motif_residue(motif_residue: MotifResidue) -> AtomArray:
    """Expand partial sidechain to full residue AtomArray from CCD.

    Takes a MotifResidue dataclass and:
    1. Gets full residue from CCD (e.g., complete HIS with all atoms)
    2. Matches motif atoms by atom_name
    3. Transfers coordinates for matching atoms
    4. Adds is_motif annotation (True for atoms with motif coordinates)

    Args:
        motif_residue: MotifResidue dataclass with res_id, res_name,
            atom_names, and coords

    Returns:
        AtomArray: Full residue with coordinates transferred where available,
            with is_motif and is_motif_related annotations
    """
    full_residue = ccd.get_component_atom_array(
        motif_residue.res_name,
        keep_leaving_atoms=False,
        keep_hydrogens=False,
    )

    full_residue = full_residue.copy()

    if full_residue is None:
        raise ValueError(f"Could not find CCD component for {motif_residue.res_name}")

    is_motif = np.zeros(len(full_residue), dtype=bool)

    for i, atom_name in enumerate(motif_residue.atom_names):
        match_idx = np.where(full_residue.atom_name == atom_name)[0]
        if len(match_idx) > 0:
            full_residue.coord[match_idx[0]] = motif_residue.coords[i]
            is_motif[match_idx[0]] = True
        else:
            logger.warning(
                f"Motif atom {atom_name} not found in CCD component "
                f"{motif_residue.res_name}"
            )

    # NOTE: we need to change the res_id assigned here downstream and have
    # motif res_id annotations to prevent information leakage and avoid train-test discrepancy.
    full_residue.res_id[:] = motif_residue.res_id
    full_residue.set_annotation("is_motif", is_motif)
    full_residue.set_annotation("is_motif_related", np.ones_like(is_motif, dtype=bool))

    full_residue = add_reference_features(full_residue)
    return full_residue


def build_motif_ligand(motif_ligand: MotifLigand) -> AtomArray:
    """Expand motif ligand fragment to full ligand AtomArray.

    Takes a MotifLigand dataclass and:
    1. Builds the full ligand from full_ligand spec (CCD or SMILES)
    2. Matches motif atoms to full ligand atoms
    3. Transfers coordinates for matching atoms
    4. Adds is_motif annotation

    For CCD ligands: matches by atom_name
    For SMILES ligands: uses atom_map for mapping

    Args:
        motif_ligand: MotifLigand dataclass with cif_res_name, atom_names,
            coords, elements, and full_ligand spec

    Returns:
        AtomArray: Full ligand with motif coordinates transferred,
            with is_motif and is_motif_related annotations
    """
    ligand_info = build_ligand(motif_ligand.full_ligand)
    full_ligand_array = ligand_info.atom_array

    is_motif = np.zeros(len(full_ligand_array), dtype=bool)

    # CCD: match by atom_name
    if isinstance(motif_ligand.full_ligand, CCDLigand):
        for i, atom_name in enumerate(motif_ligand.atom_names):
            match_idx = np.where(full_ligand_array.atom_name == atom_name)[0]
            if len(match_idx) == 0:
                raise ValueError(
                    f"Motif atom '{atom_name}' not found in CCD ligand "
                    f"'{motif_ligand.full_ligand.codes}'. Available atoms: "
                    f"{full_ligand_array.atom_name.tolist()}"
                )
            full_ligand_array.coord[match_idx[0]] = motif_ligand.coords[i]
            is_motif[match_idx[0]] = True

    # SMILES: use atom_map for coordinate transfer
    else:
        atom_map = motif_ligand.full_ligand.atom_map
        if atom_map is None:
            raise ValueError(
                f"SMILES ligand requires atom_map for coordinate transfer. "
                f"SMILES: {motif_ligand.full_ligand.smiles}"
            )

        map_num_to_atom_name = ligand_info.atom_map_to_atom_name

        for i, cif_atom_name in enumerate(motif_ligand.atom_names):
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
            full_ligand_array.coord[match_idx[0]] = motif_ligand.coords[i]
            is_motif[match_idx[0]] = True

    full_ligand_array.set_annotation("is_motif", is_motif)
    full_ligand_array.set_annotation(
        "is_motif_related", np.ones_like(is_motif, dtype=bool)
    )

    return full_ligand_array


def build_motif(motif: Motif) -> AtomArray:
    """Build a motif from a Motif object."""
    arrays = []

    for residue in motif.residues:
        arrays.append(residue.to_atom_array())

    for ligand in motif.ligands:
        arrays.append(ligand.to_atom_array())

    if not arrays:
        raise ValueError("Motif has no residues or ligands")

    combined = arrays[0]
    for arr in arrays[1:]:
        combined += arr

    return combined


def build_sequence_design_structure(
    input_data: SequenceDesignInput,
) -> AtomArray:
    """Build atom array from sequence design input.

    Takes a SequenceDesignInput and:
    1. Loads backbone CIF via MMCIFParser
    2. Filters protein residues to backbone atoms only (N, CA, C, O)
    3. Keeps ligand atoms (HETATM) as-is from CIF
    4. Adds is_fixed annotation for fixed residues
    5. Adds standard annotations (reference features)

    Args:
        input_data: SequenceDesignInput with backbone file and fixed residues

    Returns:
        AtomArray: Complete structure with is_fixed annotation marking
            fixed residue atoms

    Raises:
        FileNotFoundError: If backbone CIF file doesn't exist
    """
    cif_path = input_data.cif_file.path
    if not cif_path.exists():
        raise FileNotFoundError(f"Backbone CIF file not found: {cif_path}")

    parser = MMCIFParser(cif_path)
    atoms = parser.get_structure()

    # Group atoms by residue and process
    res_starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    processed_arrays: list[AtomArray] = []

    for start, stop in zip(res_starts[:-1], res_starts[1:]):
        res_atoms = atoms[start:stop]
        is_hetero = res_atoms.hetero[0]

        # For protein residues, keep only backbone atoms
        if not is_hetero:
            backbone_atoms = BACKBONE_DEFS["protein"]
            backbone_mask = np.isin(res_atoms.atom_name, backbone_atoms)
            res_atoms = res_atoms[backbone_mask]

        # Ligands (HETATM) are kept as-is from CIF
        processed_arrays.append(res_atoms.copy())

    # Combine all arrays
    if not processed_arrays:
        raise ValueError("No atoms found in backbone CIF")

    combined = processed_arrays[0]
    for arr in processed_arrays[1:]:
        combined += arr

    # Add is_fixed annotation for fixed residues
    is_fixed = np.zeros(len(combined), dtype=bool)
    fixed_residue_map = {
        (fr.chain_id, fr.res_id): fr.residue for fr in input_data.fixed_residues
    }

    for start, stop in zip(
        struc.get_residue_starts(combined, add_exclusive_stop=True)[:-1],
        struc.get_residue_starts(combined, add_exclusive_stop=True)[1:],
    ):
        chain_id = combined.chain_id[start]
        res_id = combined.res_id[start]
        key = (chain_id, res_id)
        if key in fixed_residue_map:
            is_fixed[start:stop] = True

    combined.set_annotation("is_fixed", is_fixed)

    # Add reference features
    combined = add_reference_features(combined)

    return combined


#####
# Motif parsing
#####


@dataclass(frozen=True)
class MotifResidue:
    """Partial residue extracted from motif CIF file."""

    res_id: int
    res_name: str  # e.g., "HIS", "GLU", "THR"
    atom_names: tuple[str, ...]  # e.g., ("CG", "CD2", "ND1", "CE1", "NE2")
    coords: np.ndarray  # (N_atoms, 3)

    def to_atom_array(self) -> AtomArray:
        """Expand partial sidechain to full residue AtomArray from CCD.

        Returns:
            AtomArray: Full residue with coordinates transferred where available,
                with is_motif and is_motif_related annotations
        """
        return build_motif_residue(self)


@dataclass(frozen=True)
class MotifLigand:
    """Ligand specification combining motif CIF fragment with full ligand description."""

    cif_res_name: str  # Residue / CCD name in the CIF (e.g., "ZN", "AL5", "LIG")
    atom_names: tuple[str, ...]  # Atom names from CIF (e.g., ("O1A", "S1", "N21"))
    coords: np.ndarray  # (n_atoms, 3) coordinates from CIF
    elements: tuple[str, ...]  # Element types (e.g., ("O", "S", "N"))

    full_ligand: CCDLigand | SMILESLigand

    def to_atom_array(self) -> AtomArray:
        """Expand motif ligand to full ligand AtomArray.

        Returns:
            AtomArray: Full ligand with motif coordinates transferred,
                with is_motif and is_motif_related annotations
        """
        return build_motif_ligand(self)


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

    def to_atom_array(self) -> AtomArray:
        """Generate combined AtomArray from all residues and ligands.

        Returns:
            AtomArray with is_motif and is_motif_related annotations
        """
        return build_motif(self)
