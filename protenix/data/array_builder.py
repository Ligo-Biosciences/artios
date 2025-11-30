"""
Utilities for building atom arrays from CCD codes and other things.


_build_polymer_atom_array

build_polymer
PolymerInfo ->

build_ligand
LigandInfo -> AtomInfo

build_design_polymer?

build_target_polymer?

"""

import logging
import random
import warnings
from dataclasses import dataclass

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from protenix.data import ccd
from protenix.data.rdkit import lig_file_to_atom_info, smiles_to_atom_info

DNA_1to3 = {
    "A": "DA",
    "G": "DG",
    "C": "DC",
    "T": "DT",
    "X": "DN",
    "I": "DI",  # eg: pdb 114d
    "N": "DN",  # eg: pdb 7r6t-3DR
    "U": "DU",  # eg: pdb 7sd8
}
RNA_1to3 = {
    "A": "A",
    "G": "G",
    "C": "C",
    "U": "U",
    "X": "N",
    "I": "I",  # eg: pdb 7wv5
    "N": "N",
}

PROTEIN_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}


@dataclass(frozen=True)
class PolymerInfo:
    name: str
    sequence: str  # "GPDSMEEVVVPEE..."
    count: int = 1


# Build polymer
@dataclass(frozen=True)
class LigandInfo:
    name: str
    # type - ligand, ion
    full_ligand: str  # "CCD_ZN", "CCD_AL5", or SMILES string


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
    # TODO: move somewhere else, e.g. ccd.py
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


def build_polymer(entity_info: dict) -> dict:
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
        dict: {"atom_array": biotite_AtomArray_object}
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
    return {"atom_array": chain_array}


def build_ligand(entity_info: dict) -> dict:
    """
    Build a ligand from a ligand entity info dict
    example1: {
        "ligand": {
          "ligand": "CCD_ATP",
          "count": 1
        }
      },
    example2:{
        "ligand": {
          "ligand": "CCC=O",  # smiles
          "count": 1
        }
      },
    example3:{
        "ion": {
          "ion": "NA",
          "count": 3
        }
      },

    Args:
        entity_info (dict): ligand entity info

    Returns:
        dict: info of atoms
        example: {
            "atom_array": biotite_AtomArray_object,
            "index_to_atom_name": {1: "C2"}, # only for smiles
            }
    """
    if info := entity_info.get("ion"):
        ccd_code = [info["ion"]]
    elif info := entity_info.get("ligand"):
        ligand_str = info["ligand"]
        if ligand_str.startswith("CCD_"):
            ccd_code = ligand_str[4:].split("_")
        else:
            ccd_code = None

    atom_info = {}
    if ccd_code is not None:
        atom_array = AtomArray(0)
        res_ids = []
        for idx, code in enumerate(ccd_code):
            ccd_atom_array = ccd.get_component_atom_array(
                code, keep_leaving_atoms=True, keep_hydrogens=False
            )
            atom_array += ccd_atom_array
            res_id = idx + 1
            res_ids += [res_id] * len(ccd_atom_array)
        atom_info["atom_array"] = atom_array
        atom_info["atom_array"].res_id[:] = res_ids
    else:
        if info["ligand"].startswith("FILE_"):
            lig_file_path = ligand_str[5:]
            atom_info = lig_file_to_atom_info(lig_file_path)
        else:
            atom_info = smiles_to_atom_info(ligand_str)
        atom_info["atom_array"].res_id[:] = 1
    atom_info["atom_array"] = add_reference_features(atom_info["atom_array"])
    return atom_info
