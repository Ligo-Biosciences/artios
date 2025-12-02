"""
Utilities for building atom arrays from CCD codes and other sources.
"""

from __future__ import annotations

import logging
import random
import warnings
from typing import TYPE_CHECKING

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from protenix.common.constants import BACKBONE_DEFS
from protenix.common.residue_names import PROTEIN_ONE_TO_THREE_WITH_UNKNOWN
from protenix.data import ccd
from protenix.data.json_parser_new import CCDLigand, SequenceDesignInput, SMILESLigand
from protenix.data.parser import MMCIFParser
from protenix.data.rdkit import AtomInfo, smiles_to_atom_info

if TYPE_CHECKING:
    from protenix.data.motif_parser import MotifLigand, MotifResidue

logger = logging.getLogger(__name__)


def _remove_non_std_ccd_leaving_atoms(atom_array: AtomArray) -> AtomArray:
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
            if (
                idx == 1
                and ccd.get_chain_type_from_ccd_code(res_name).is_nucleic_acid()
            ):
                staying_atoms = np.append(staying_atoms, ["OP3"])
            if (
                idx == atom_array.res_id[-1]
                and ccd.get_chain_type_from_ccd_code(res_name).is_protein()
            ):
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


def _add_bonds_to_terminal_residues(atom_array: AtomArray) -> AtomArray:
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


def _build_polymer_atom_array(ccd_seqs: list[str]) -> tuple[AtomArray, struc.BondList]:
    """
    Build polymer atom_array from ccd codes, but not remove leaving atoms

    Args:
        ccd_seqs: a list of ccd code in sequence, ["MET", "ALA"] or ["DA", "DT"]

    Returns:
        AtomArray: Biotite AtomArray of chain
        BondList: Biotite BondList of polymer bonds (C-N or O3'-P)
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

    chain = _add_bonds_to_terminal_residues(chain)

    bond_count = {}
    for i, j, t in polymer_bonds._bonds:
        bond_count[i] = bond_count.get(i, 0) + 1
        bond_count[j] = bond_count.get(j, 0) + 1

    chain = remove_leaving_atoms(chain, bond_count)

    chain = _remove_non_std_ccd_leaving_atoms(chain)

    return chain


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
                *map(ref_info["atom_map"].get, atom_array.atom_name[start:stop])
            ]
            ref_pos[start:stop] = ref_info["coord"][atom_sub_idx]
            ref_charge[start:stop] = ref_info["charge"][atom_sub_idx]
            ref_mask[start:stop] = ref_info["mask"][atom_sub_idx]
        else:
            logging.warning(f"no reference info for {res_name}")

    atom_array.set_annotation("ref_pos", ref_pos)
    atom_array.set_annotation("ref_charge", ref_charge)
    atom_array.set_annotation("ref_mask", ref_mask)
    return atom_array


def build_polymer(sequence: str) -> AtomArray:
    """Build a protein polymer AtomArray from a sequence string.

    Args:
        sequence: One-letter amino acid sequence (e.g., "MVLSPADKTN...")

    Returns:
        AtomArray with all atoms, bonds, and reference features
    """
    ccd_seqs = [PROTEIN_ONE_TO_THREE_WITH_UNKNOWN[aa] for aa in sequence]
    chain_array = _build_polymer_atom_array(ccd_seqs)
    chain_array = add_reference_features(chain_array)
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


def build_ligand_with_coords(
    cif_ligand: AtomArray,
    full_ligand: CCDLigand | SMILESLigand,
) -> AtomArray:
    """Build full ligand with coordinates transferred from CIF fragment.

    Takes coordinates from a CIF ligand AtomArray and transfers them to
    the expanded full ligand structure. This is useful for sequence design
    where the backbone CIF contains ligand coordinates.

    For CCD ligands: matches by atom_name
    For SMILES ligands: uses atom_map for mapping

    Args:
        cif_ligand: AtomArray from the CIF file containing ligand coordinates
        full_ligand: CCDLigand or SMILESLigand specifying the full ligand

    Returns:
        AtomArray: Full ligand with coordinates transferred from CIF,
            with is_motif annotation (True for atoms with transferred coords)
    """
    ligand_info = build_ligand(full_ligand)
    full_ligand_array = ligand_info.atom_array

    is_motif = np.zeros(len(full_ligand_array), dtype=bool)

    # CCD: match by atom_name
    if isinstance(full_ligand, CCDLigand):
        for i, atom_name in enumerate(cif_ligand.atom_name):
            match_idx = np.where(full_ligand_array.atom_name == atom_name)[0]
            if len(match_idx) == 0:
                logger.warning(
                    f"CIF atom '{atom_name}' not found in CCD ligand "
                    f"'{full_ligand.codes}'. Skipping coordinate transfer."
                )
                continue
            full_ligand_array.coord[match_idx[0]] = cif_ligand.coord[i]
            is_motif[match_idx[0]] = True

    # SMILES: use atom_map for coordinate transfer
    else:
        atom_map = full_ligand.atom_map
        if atom_map is None:
            raise ValueError(
                f"SMILES ligand requires atom_map for coordinate transfer. "
                f"SMILES: {full_ligand.smiles}"
            )

        map_num_to_atom_name = ligand_info.atom_map_to_atom_name

        for i, cif_atom_name in enumerate(cif_ligand.atom_name):
            if cif_atom_name not in atom_map:
                logger.warning(
                    f"CIF atom '{cif_atom_name}' not found in atom_map. "
                    f"Skipping coordinate transfer."
                )
                continue

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
            full_ligand_array.coord[match_idx[0]] = cif_ligand.coord[i]
            is_motif[match_idx[0]] = True

    full_ligand_array.set_annotation("is_motif", is_motif)

    return full_ligand_array


def build_sequence_design_structure(
    input_data: SequenceDesignInput,
) -> AtomArray:
    """Build atom array from sequence design input.

    Takes a SequenceDesignInput and:
    1. Loads backbone CIF via MMCIFParser
    2. For each ligand mapping: expands CIF ligand to full_ligand, transfers coords
    3. Adds is_fixed annotation for fixed residues
    4. Adds standard annotations (reference features)

    Args:
        input_data: SequenceDesignInput with backbone file, ligand mappings,
            and fixed residues

    Returns:
        AtomArray: Complete structure with is_fixed annotation marking
            fixed residue atoms

    Raises:
        FileNotFoundError: If backbone CIF file doesn't exist
        ValueError: If a ligand mapping doesn't match any residue in the CIF
    """
    backbone_path = input_data.backbone.file
    if not backbone_path.exists():
        raise FileNotFoundError(f"Backbone CIF file not found: {backbone_path}")

    parser = MMCIFParser(backbone_path)
    atoms = parser.get_structure()

    # Build ligand mapping: cif_res_name -> full_ligand
    ligand_mappings = {
        mapping.cif_res_name: mapping.full_ligand
        for mapping in input_data.backbone.ligands
    }

    # Group atoms by residue and process
    res_starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    processed_arrays: list[AtomArray] = []
    ligands_found: set[str] = set()

    for start, stop in zip(res_starts[:-1], res_starts[1:]):
        res_atoms = atoms[start:stop]
        res_name = res_atoms.res_name[0]
        is_hetero = res_atoms.hetero[0]

        # Ligand that needs expansion
        if is_hetero and res_name in ligand_mappings:
            full_ligand = ligand_mappings[res_name]
            expanded_ligand = build_ligand_with_coords(res_atoms, full_ligand)
            # Preserve res_id from CIF
            expanded_ligand.res_id[:] = res_atoms.res_id[0]
            expanded_ligand.chain_id[:] = res_atoms.chain_id[0]
            processed_arrays.append(expanded_ligand)
            ligands_found.add(res_name)
        else:
            # For protein residues, keep only backbone atoms
            if not is_hetero:
                backbone_atoms = BACKBONE_DEFS["protein"]
                backbone_mask = np.isin(res_atoms.atom_name, backbone_atoms)
                res_atoms = res_atoms[backbone_mask]
            processed_arrays.append(res_atoms.copy())

    # Warn about unmapped ligands
    unmapped = set(ligand_mappings.keys()) - ligands_found
    if unmapped:
        logger.warning(
            f"Ligand mappings not found in CIF: {unmapped}. " f"These will be ignored."
        )

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
