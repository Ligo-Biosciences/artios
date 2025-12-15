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

import logging
import random
import warnings

import biotite.sequence.io as seqio
import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
from biotite.structure import AtomArray
from typing_extensions import assert_never

from protenix.common.constants import BACKBONE_DEFS
from protenix.common.residue_names import PROTEIN_COMMON_ONE_TO_THREE as PROTEIN_1to3
from protenix.data import ccd
from protenix.data.json_parser import (
    Backbone,
    CCDLigand,
    MotifInput,
    PackingInput,
    SequenceDesignInput,
    SMILESLigand,
)
from protenix.data.motif_parser import Motif, MotifLigand, MotifResidue
from protenix.data.parser import MMCIFParser
from protenix.data.rdkit import AtomInfo, smiles_to_atom_info

logger = logging.getLogger(__name__)


def set_chain_and_seq_ids(
    atom_array: AtomArray,
    chain_id: int,
) -> AtomArray:
    """Assign IDs."""
    chain_ids = [str(chain_id)] * len(atom_array)
    atom_array.set_annotation("label_asym_id", chain_ids)
    atom_array.set_annotation("auth_asym_id", chain_ids)
    atom_array.set_annotation("chain_id", chain_ids)
    atom_array.set_annotation("label_seq_id", atom_array.res_id)
    atom_array.set_annotation("copy_id", ["1"] * len(atom_array))
    atom_array.set_annotation("label_entity_id", [str(chain_id + 1)] * len(atom_array))
    return atom_array


def motif_input_to_atom_array(motif_input: MotifInput) -> AtomArray:
    """Process motif scaffolding input and builds AtomArray."""
    assert len(motif_input.design) == 1, "Only one design entity is supported"

    # Build protein array
    sequence = [PROTEIN_1to3["G"] for _ in range(motif_input.design[0].length)]
    design_atom_array = _build_polymer_atom_array(sequence)
    design_atom_array = add_reference_features(design_atom_array)
    design_atom_array = set_chain_and_seq_ids(design_atom_array, chain_id=0)

    design_atom_array.set_annotation(
        "is_motif_related", [False] * len(design_atom_array)
    )
    design_atom_array.set_annotation("is_motif", [False] * len(design_atom_array))

    # Build motif
    motif_atom_array = build_motif(Motif.from_input(motif_input))
    motif_atom_array = set_chain_and_seq_ids(motif_atom_array, chain_id=1)

    return design_atom_array + motif_atom_array


def sequence_input_to_atom_array(sequence_input: SequenceDesignInput) -> AtomArray:
    """Process sequence design input and builds AtomArray."""
    # Load stitched backbone from cif
    designed_array = strucio.load_structure(sequence_input.cif_file.path)

    arrays = []
    # Build protein array
    res_ids = np.unique(
        designed_array[designed_array.chain_id == "A"].res_id
    )  # TODO: chain ID always 0 in future
    len_sequence = len(res_ids)

    res_id_to_idx = {res_id: idx for idx, res_id in enumerate(res_ids)}

    sequence = [PROTEIN_1to3["G"] for _ in range(len_sequence)]

    for res in sequence_input.fixed_residues:
        residue = res.residue
        if len(residue) == 1:
            residue = PROTEIN_1to3[residue]

        idx = res_id_to_idx[res.res_id]
        sequence[idx] = residue

    protein_array = _build_polymer_atom_array(sequence)
    protein_array = add_reference_features(protein_array)
    protein_array = set_chain_and_seq_ids(protein_array, chain_id=0)

    arrays.append(protein_array)

    # Build ligand arrays
    for i, ligand in enumerate(sequence_input.ligands):
        ligand_array = build_ligand(ligand).atom_array
        ligand_array = set_chain_and_seq_ids(ligand_array, chain_id=i + 1)
        arrays.append(ligand_array)

    combined = arrays[0]
    for arr in arrays[1:]:
        combined += arr

    # Add the coordinates to combined array
    assert len(combined) == len(designed_array)
    combined.coord = designed_array.coord

    combined.set_annotation("is_motif_related", [False] * len(combined))
    combined.set_annotation("is_motif", [False] * len(combined))

    return combined


def packing_input_to_atom_array(packing_input: PackingInput) -> AtomArray:
    """Process packing input and builds AtomArray."""
    # Load stitched backbone from cif
    designed_array = strucio.load_structure(packing_input.cif_file.path)

    arrays = []
    # Build protein array
    fasta_dict = seqio.load_sequences(packing_input.fasta_file.path)
    sequence = list(fasta_dict.values())[0]
    ccd_seqs = [PROTEIN_1to3[aa] for aa in sequence]

    protein_array = _build_polymer_atom_array(ccd_seqs)
    protein_array = add_reference_features(protein_array)
    protein_array = set_chain_and_seq_ids(protein_array, chain_id=0)
    arrays.append(protein_array)

    # Build ligand arrays
    for i, ligand in enumerate(packing_input.ligands):
        ligand_array = build_ligand(ligand).atom_array
        ligand_array = set_chain_and_seq_ids(ligand_array, chain_id=i + 1)
        arrays.append(ligand_array)

    combined = arrays[0]
    for arr in arrays[1:]:
        combined += arr

    # Get backbone atom masks
    backbone_atoms = BACKBONE_DEFS["protein"]
    backbone_design_mask = (designed_array.hetero) | (
        (~designed_array.hetero) & (np.isin(designed_array.atom_name, backbone_atoms))
    )
    backbone_combined_mask = (combined.hetero) | (
        (~combined.hetero) & (np.isin(combined.atom_name, backbone_atoms))
    )

    # Add the coordinates to combined array
    assert sum(backbone_combined_mask.astype(int)) == sum(
        backbone_design_mask.astype(int)
    )
    combined.coord[backbone_combined_mask] = designed_array.coord[backbone_design_mask]

    combined.set_annotation("is_motif_related", [False] * len(combined))
    combined.set_annotation("is_motif", [False] * len(combined))

    return combined


def process_input_to_atom_array(
    input: MotifInput | SequenceDesignInput | PackingInput,
) -> AtomArray:
    """Dispatch to appropriate task-specific processor based on input structure."""
    match input:
        case MotifInput():
            return motif_input_to_atom_array(input)
        case SequenceDesignInput():
            return sequence_input_to_atom_array(input)
        case PackingInput():
            return packing_input_to_atom_array(input)
        case _:
            assert_never(input)


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

    return atom_array


def build_ligand(ligand: CCDLigand | SMILESLigand) -> AtomInfo:
    """Build a ligand AtomArray from a typed ligand specification.

    Args:
        ligand: CCDLigand or SMILESLigand from json_parser.py

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
        arrays.append(build_motif_residue(residue))

    for ligand in motif.ligands:
        arrays.append(build_motif_ligand(ligand))

    if not arrays:
        raise ValueError("Motif has no residues or ligands")

    combined = arrays[0]
    for arr in arrays[1:]:
        combined += arr

    return combined


def build_structure_from_backbone(
    input_data: SequenceDesignInput | PackingInput,
) -> AtomArray:
    """Build atom array from sequence design input.

    Takes a SequenceDesignInput or PackingInput and:
    1. Loads backbone CIF via MMCIFParser
    2. Filters protein residues to backbone atoms only (N, CA, C, O)
    3. Keeps ligand atoms (HETATM) as-is from CIF
    4. Adds is_fixed annotation for fixed residues
    5. Adds standard annotations (reference features)

    Args:
        input_data: SequenceDesignInput or PackingInput with backbone file and fixed residues

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
    fixed_residues = getattr(input_data, "fixed_residues", [])
    fixed_residue_map = {(fr.chain_id, fr.res_id): fr.residue for fr in fixed_residues}

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


def add_reference_features(atom_array: AtomArray) -> AtomArray:
    """
    Add reference features of each residue to atom_array

    Args:
        atom_array (AtomArray): biotite AtomArray

    Returns:
        AtomArray: biotite AtomArray with reference features
        - ref_pos: reference conformer atom positions
        - ref_charge (n): reference conformer atom charges
        - ref_mask: reference conformer atom masks
    """
    atom_count = len(atom_array)
    ref_pos = np.zeros((atom_count, 3), dtype=np.float32)
    ref_charge = np.zeros(atom_count, dtype=int)
    ref_mask = np.zeros(atom_count, dtype=int)

    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for start, stop in zip(starts[:-1], starts[1:]):
        res_name = atom_array.res_name[start]
        if res_name == "UNL":
            # UNL is smiles ligand, copy info from atom_array
            ref_pos[start:stop] = atom_array.coord[start:stop]
            ref_charge[start:stop] = atom_array.charge[start:stop]
            ref_mask[start:stop] = 1
            continue

        ref_info = ccd.get_ccd_ref_info(res_name)
        if ref_info:
            atom_sub_idx = [
                *map(ref_info.atom_map.get, atom_array.atom_name[start:stop])
            ]
            ref_pos[start:stop] = ref_info.coord[atom_sub_idx]
            ref_charge[start:stop] = ref_info.charge[atom_sub_idx]
            ref_mask[start:stop] = ref_info.mask[atom_sub_idx]
        else:
            logging.warning(f"no reference info for {res_name}")

    atom_array.set_annotation("ref_pos", ref_pos)
    atom_array.set_annotation("ref_charge", ref_charge)
    atom_array.set_annotation("ref_mask", ref_mask)
    return atom_array


def remove_non_std_ccd_leaving_atoms(atom_array: AtomArray) -> AtomArray:
    """
    Check polymer connections and remove non-standard leaving atoms

    Args:
        atom_array (AtomArray): biotite AtomArray

    Returns:
        AtomArray: biotite AtomArray with leaving atoms removed.
    """
    connected = np.zeros(atom_array.res_id[-1], dtype=bool)
    for i, j, t in atom_array.bonds._bonds:
        if abs(atom_array.res_id[i] - atom_array.res_id[j]) == 1:
            connected[atom_array.res_id[[i, j]].min()] = True

    leaving_atoms = np.zeros(len(atom_array), dtype=bool)
    for res_id, conn in enumerate(connected):
        if res_id == 0 or conn:
            continue

        # Res_id start from 1
        res_name_i = atom_array.res_name[atom_array.res_id == res_id][0]
        res_name_j = atom_array.res_name[atom_array.res_id == res_id + 1][0]
        warnings.warn(
            f"No C-N or O3'-P bond between residue {res_name_i}({res_id}) and residue {res_name_j}({res_id+1}). \n"
            f"all leaving atoms will be removed for both residues."
        )
        for idx, res_name in zip([res_id, res_id + 1], [res_name_i, res_name_j]):
            staying_atoms = ccd.get_component_atom_array(
                res_name, keep_leaving_atoms=False, keep_hydrogens=False
            ).atom_name
            if idx == 1 and ccd.get_mol_type(res_name) in ("dna", "rna"):
                staying_atoms = np.append(staying_atoms, ["OP3"])
            if idx == atom_array.res_id[-1] and ccd.get_mol_type(res_name) == "protein":
                staying_atoms = np.append(staying_atoms, ["OXT"])
            leaving_atoms |= (atom_array.res_id == idx) & (
                ~np.isin(atom_array.atom_name, staying_atoms)
            )
    return atom_array[~leaving_atoms]


def find_range_by_index(starts: np.ndarray, atom_index: int) -> tuple[int, int]:
    """
    Find the residue range of an atom index

    Args:
        starts (np.ndarray): Residue starts or Chain starts with exclusive stop.
        atom_index (int): Atom index.

    Returns:
        tuple[int, int]: range (start, stop).
    """
    for start, stop in zip(starts[:-1], starts[1:]):
        if start <= atom_index < stop:
            return start, stop
    raise ValueError(f"atom_index {atom_index} not found in starts {starts}")


def remove_leaving_atoms(atom_array: AtomArray, bond_count: dict) -> AtomArray:
    """
    Remove leaving atoms based on ccd info

    Args:
        atom_array (AtomArray): Biotite Atom array.
        bond_count (dict): atom index -> Bond count.

    Returns:
        AtomArray: Biotite Atom array with leaving atoms removed.
    """
    remove_indices = []
    res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for centre_idx, b_count in bond_count.items():
        res_name = atom_array.res_name[centre_idx]
        centre_name = atom_array.atom_name[centre_idx]

        comp = ccd.get_component_atom_array(
            res_name, keep_leaving_atoms=True, keep_hydrogens=False
        )
        if comp is None:
            continue

        leaving_groups = comp.central_to_leaving_groups.get(centre_name)
        if leaving_groups is None:
            continue

        if b_count > len(leaving_groups):
            warnings.warn(
                f"centre atom {centre_name=} {res_name=} {centre_idx=} has {b_count} inter residue bonds, greater than number of leaving groups:{leaving_groups}, remove all leaving atoms.\n"
                f"atom info: {atom_array[centre_idx]=}"
            )
            remove_groups = leaving_groups
        else:
            remove_groups = random.sample(leaving_groups, b_count)

        start, stop = find_range_by_index(res_starts, centre_idx)

        # Find leaving atom indices
        for group in remove_groups:
            for atom_name in group:
                leaving_idx = np.where(atom_array.atom_name[start:stop] == atom_name)[0]
                if len(leaving_idx) == 0:
                    logging.info(f"{atom_name=} not found in residue {res_name}, ")
                    continue

                remove_indices.append(leaving_idx[0] + start)

    if not remove_indices:
        return atom_array

    keep_mask = np.ones(len(atom_array), dtype=bool)
    keep_mask[remove_indices] = False
    return atom_array[keep_mask]


def add_bonds_to_terminal_residues(atom_array: AtomArray) -> AtomArray:
    """
    Add bonds to terminal residues (eg: ACE, NME)

    Args:
        atom_array (AtomArray): Biotite AtomArray

    Returns:
        AtomArray: Biotite AtomArray with non-standard polymer bonds
    """

    if atom_array.res_name[0] == "ACE":
        term_res_idx = atom_array.res_id[0]
        next_res_idx = term_res_idx + 1
        term_atom_idx = np.where(
            (atom_array.res_id == term_res_idx) & (atom_array.atom_name == "C")
        )[0]
        next_atom_idx = np.where(
            (atom_array.res_id == next_res_idx) & (atom_array.atom_name == "N")
        )[0]
        if len(term_atom_idx) > 0 and len(next_atom_idx) > 0:
            atom_array.bonds.add_bond(term_atom_idx[0], next_atom_idx[0], 1)

    if atom_array.res_name[-1] == "NME":
        term_res_idx = atom_array.res_id[-1]
        prev_res_idx = term_res_idx - 1
        term_atom_idx = np.where(
            (atom_array.res_id == term_res_idx) & (atom_array.atom_name == "N")
        )[0]
        prev_atom_idx = np.where(
            (atom_array.res_id == prev_res_idx) & (atom_array.atom_name == "C")
        )[0]
        if len(prev_atom_idx) > 0 and len(term_atom_idx) > 0:
            atom_array.bonds.add_bond(prev_atom_idx[0], term_atom_idx[0], 1)

    return atom_array
