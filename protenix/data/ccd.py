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

import functools
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
from atomworks.enums import ChainType
from atomworks.io.tools.rdkit import ccd_code_to_rdkit
from atomworks.io.utils.ccd import (
    atom_array_from_ccd_code,
    chem_comp_to_one_letter,
    get_available_ccd_codes,
    get_chain_type_from_ccd_code,
)
from biotite.structure import AtomArray
from rdkit import Chem

from configs.configs_data import data_configs
from protenix.data.substructure_perms import get_substructure_perms

logger = logging.getLogger(__name__)

COMPONENTS_FILE = Path(data_configs["ccd_components_file"])
RKDIT_MOL_PKL = Path(data_configs["ccd_components_rdkit_mol_file"])


@functools.lru_cache
def biotite_load_ccd_cif() -> pdbx.CIFFile:
    """biotite load CCD components file

    Returns:
        pdbx.CIFFile: ccd components file
    """
    return pdbx.CIFFile.read(COMPONENTS_FILE)


def _map_central_to_leaving_groups(
    component: AtomArray,
) -> Optional[dict[str, list[list[str]]]]:
    """
    Map each central atom (bonded atom) index to leaving atom groups in component (atom_array).

    Returns:
        dict[str, list[list[str]]]: central atom name to leaving atom groups (atom names).
    """
    comp = component.copy()
    # Eg: ions
    if comp.bonds is None:
        return {}
    central_to_leaving_groups = defaultdict(list)

    # ... iterate through the central atoms
    for central_idx in np.flatnonzero(~comp.is_leaving_atom):
        bonds, _ = comp.bonds.get_bonds(central_idx)

        # ... iterate through leaving groups defined by the bond graph
        for leaving_idx in bonds:
            if comp.is_leaving_atom[leaving_idx]:
                comp.bonds.remove_bond(central_idx, leaving_idx)
                connected_group = struc.find_connected(comp.bonds, leaving_idx)
                if not np.all(comp.is_leaving_atom[connected_group]):
                    return None
                central_to_leaving_groups[comp.atom_name[central_idx]].append(
                    comp.atom_name[connected_group].tolist()
                )
    return central_to_leaving_groups


@functools.lru_cache
def get_component_atom_array(
    ccd_code: str, keep_leaving_atoms: bool = False, keep_hydrogens=False
) -> AtomArray:
    """get component atom array

    Args:
        ccd_code (str): ccd code
        keep_leaving_atoms (bool, optional): keep leaving atoms. Defaults to False.
        keep_hydrogens (bool, optional): keep hydrogens. Defaults to False.

    Returns:
        AtomArray: Biotite AtomArray of CCD component
            with additional attribute: is_leaving_atom (bool)
    """
    comp = atom_array_from_ccd_code(ccd_code)

    if not keep_leaving_atoms:
        comp = comp[~comp.is_leaving_atom]
    if not keep_hydrogens:
        # EG: ND4
        comp = comp[~np.isin(comp.element, ["H", "D"])]

    # Map central atom index to leaving group in atom array.
    comp.central_to_leaving_groups = _map_central_to_leaving_groups(comp)
    if comp.central_to_leaving_groups is None:
        logger.warning(
            f"Warning: ccd {ccd_code} has leaving atom group bond to more than",
            "one central atom, central_to_leaving_groups is None.",
        )
    return comp


@functools.lru_cache(maxsize=None)
def get_one_letter_code(ccd_code: str) -> Union[str, None]:
    """get one_letter_code from CCD components file.

    normal return is one letter: ALA --> A, DT --> T
    unknown protein: X
    unknown DNA or RNA: N
    other unknown: None
    some ccd_code will return more than one letter:
    eg: XXY --> THG

    Args:
        ccd_code (str): _description_

    Returns:
        str: one letter code
    """
    # ccd_cif = biotite_load_ccd_cif()
    # if ccd_code not in ccd_cif:
    #     return None
    # one = ccd_cif[ccd_code]["chem_comp"]["one_letter_code"].as_item()
    # if one == "?":
    #     return None
    # else:
    #     return one
    return chem_comp_to_one_letter.get(ccd_code, None)


@functools.lru_cache(maxsize=None)
def get_mol_type_str(ccd_code: str) -> str:
    """Get mol_type as string from CCD components file.

    Args:
        ccd_code (str): ccd code

    Returns:
        str: mol_type, one of "protein", "rna", "dna", "ligand"
    """
    chain_type: ChainType = get_chain_type_from_ccd_code(ccd_code)

    if chain_type.is_protein():
        return "protein"
    if chain_type in (ChainType.DNA, ChainType.DNA_RNA_HYBRID):
        "dna"
    if chain_type == ChainType.RNA:
        "rna"
    return "ligand"


@functools.lru_cache(maxsize=None)
def get_smiles_str(ccd_code: str) -> str:
    """Get CACTVS canonical SMILES string from CCD components file.

    Returns only the CACTVS canonical SMILES representation. Returns None if either
    the CCD code is not found or no CACTVS canonical SMILES is available.

    Args:
        ccd_code (str): The CCD component code

    Returns:
        str: The CACTVS canonical SMILES string, or None if not found
    """
    ccd_cif = biotite_load_ccd_cif()
    if ccd_cif is None:
        raise ValueError(
            "Failed to load CCD components file. biotite_load_ccd_cif() returned None."
        )
    if ccd_code not in ccd_cif:
        return None

    descriptors = ccd_cif[ccd_code]["pdbx_chem_comp_descriptor"]
    mask = (descriptors["type"].as_array() == "SMILES_CANONICAL") & (
        descriptors["program"].as_array() == "CACTVS"
    )
    matching_smiles = descriptors["descriptor"].as_array()[mask]

    return matching_smiles[0] if len(matching_smiles) > 0 else None


def get_all_ccd_code() -> list:
    """get all ccd code from components file"""
    # ccd_cif = biotite_load_ccd_cif()
    # return list(ccd_cif.keys())
    return list(get_available_ccd_codes())


_ccd_rdkit_mols: dict[str, Chem.Mol] = {}


def get_component_rdkit_mol(ccd_code: str) -> Union[Chem.Mol, None]:
    """get rdkit mol by PDBeCCDUtils
    https://github.com/PDBeurope/ccdutils

    preprocessing all ccd components in _components_file at first time run.

    Args:
        ccd_code (str): ccd code

    Returns
        rdkit.Chem.Mol: rdkit mol with ref coord
    # TODO: move to rdkit.py
    """
    # global _ccd_rdkit_mols
    # # _ccd_rdkit_mols is not empty
    # if _ccd_rdkit_mols:
    #     return _ccd_rdkit_mols.get(ccd_code, None)

    # rdkit_mol_pkl = RKDIT_MOL_PKL
    # if rdkit_mol_pkl.exists():
    #     with open(rdkit_mol_pkl, "rb") as f:
    #         _ccd_rdkit_mols = pickle.load(f)
    #     return _ccd_rdkit_mols.get(ccd_code, None)
    # else:
    #     raise FileNotFoundError(
    #         f"CCD components file {rdkit_mol_pkl} not found, please download it to your DATA_ROOT_DIR before running."
    #         "See https://github.com/bytedance/Protenix"
    #     )
    return ccd_code_to_rdkit(ccd_code, ccd_mirror_path=None)


@functools.lru_cache
def get_ccd_ref_info(ccd_code: str, return_perm: bool = True) -> dict[str, Any]:
    """
    Ref: AlphaFold3 SI Chapter 2.8
    Reference features. Features derived from a residue, nucleotide or ligand's reference conformer.
    Given an input CCD code or SMILES string, the conformer is typically generated
    with RDKit v.2023_03_3 [25] using ETKDGv3 [26]. On error, we fall back to using the CCD ideal coordinates,
    or finally the representative coordinates
    if they are from before our training date cut-off (2021-09-30 unless otherwise stated).
    At the end, any atom coordinates still missing are set to zeros.

    Get reference atom mapping and coordinates.

    Args:
        name (str): CCD name
        return_perm (bool): return atom permutations.

    Returns:
        Dict:
            ccd: ccd code
            atom_map: atom name to atom index
            coord: atom coordinates
            charge: atom formal charge
            perm: atom permutation
    """
    # TODO: ccd mirror path
    mol = ccd_code_to_rdkit(
        ccd_code, ccd_mirror_path=None
    )  # get_component_rdkit_mol(ccd_code)
    if mol is None:
        return {}
    if mol.GetNumAtoms() == 0:  # eg: "UNL"
        logger.warning(
            f"Warning: mol {ccd_code} from get_component_rdkit_mol() has no atoms,"
            "get_ccd_ref_info() return empty dict"
        )
        return {}
    conf = mol.GetConformer(mol.ref_conf_id)
    coord = conf.GetPositions()
    charge = np.array([atom.GetFormalCharge() for atom in mol.GetAtoms()])

    results = {
        "ccd": ccd_code,  # str
        "atom_map": mol.atom_map,  # dict[str,int]: atom name to atom index
        "coord": coord,  # np.ndarray[float]: atom coordinates, shape:(n_atom,3)
        "mask": mol.ref_mask,  # np.ndarray[bool]: atom mask, shape:(n_atom,)
        "charge": charge,  # np.ndarray[int]: atom formal charge, shape:(n_atom,)
    }

    if return_perm:
        try:
            Chem.SanitizeMol(mol)
            perm = get_substructure_perms(mol, MaxMatches=1000)

        except Exception:
            # Sanitize failed, permutation is unavailable
            perm = np.array(
                [
                    [
                        i
                        for i, atom in enumerate(mol.GetAtoms())
                        if atom.GetAtomicNum() != 1
                    ]
                ]
            )
        # np.ndarray[int]: atom permutation, shape:(n_atom_wo_h, n_perm)
        results["perm"] = perm.T

    return results


# Modified from biotite to use consistent ccd components file
def _connect_inter_residue(
    atoms: AtomArray, residue_starts: np.ndarray
) -> struc.BondList:
    """
    Create a :class:`BondList` containing the bonds between adjacent
    amino acid or nucleotide residues.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The structure to create the :class:`BondList` for.
    residue_starts : ndarray, dtype=int
        Return value of
        ``get_residue_starts(atoms, add_exclusive_stop=True)``.

    Returns
    -------
    BondList
        A bond list containing all inter residue bonds.
    """

    bonds = []

    atom_names = atoms.atom_name
    res_names = atoms.res_name
    res_ids = atoms.res_id
    chain_ids = atoms.chain_id

    # Iterate over all starts excluding:
    #   - the last residue and
    #   - exclusive end index of 'atoms'
    for i in range(len(residue_starts) - 2):
        curr_start_i = residue_starts[i]
        next_start_i = residue_starts[i + 1]
        after_next_start_i = residue_starts[i + 2]

        # Check if the current and next residue is in the same chain
        if chain_ids[next_start_i] != chain_ids[curr_start_i]:
            continue
        # Check if the current and next residue
        # have consecutive residue IDs
        # (Same residue ID is also possible if insertion code is used)
        if res_ids[next_start_i] - res_ids[curr_start_i] > 1:
            continue

        # Get link type for this residue from RCSB components.cif
        # curr_link = get_mol_type(res_names[curr_start_i])
        # next_link = get_mol_type(res_names[next_start_i])

        curr_link: ChainType = get_chain_type_from_ccd_code(res_names[curr_start_i])
        next_link: ChainType = get_chain_type_from_ccd_code(res_names[next_start_i])

        if curr_link.is_protein() and next_link.is_protein():
            curr_connect_atom_name = "C"
            next_connect_atom_name = "N"
        elif curr_link.is_nucleic_acid() and next_link.is_nucleic_acid():
            curr_connect_atom_name = "O3'"
            next_connect_atom_name = "P"
        else:
            # Create no bond if the connection types of consecutive
            # residues are not compatible
            continue

        # Index in atom array for atom name in current residue
        # Addition of 'curr_start_i' is necessary, as only a slice of
        # 'atom_names' is taken, beginning at 'curr_start_i'
        curr_connect_indices = np.where(
            atom_names[curr_start_i:next_start_i] == curr_connect_atom_name
        )[0]
        curr_connect_indices += curr_start_i

        # Index in atom array for atom name in next residue
        next_connect_indices = np.where(
            atom_names[next_start_i:after_next_start_i] == next_connect_atom_name
        )[0]
        next_connect_indices += next_start_i

        if len(curr_connect_indices) == 0 or len(next_connect_indices) == 0:
            # The connector atoms are not found in the adjacent residues
            # -> skip this bond
            continue

        bonds.append(
            (curr_connect_indices[0], next_connect_indices[0], struc.BondType.SINGLE)
        )

    return struc.BondList(atoms.array_length(), np.array(bonds, dtype=np.uint32))


def add_inter_residue_bonds(
    atom_array: AtomArray,
    exclude_struct_conn_pairs: bool = False,
    remove_far_inter_chain_pairs: bool = False,
) -> AtomArray:
    """
    add polymer bonds (C-N or O3'-P) between adjacent residues based on auth_seq_id.

    exclude_struct_conn_pairs: if True, do not add bond between adjacent residues already has non-standard polymer bonds
                  on atom C or N or O3' or P.

    remove_far_inter_chain_pairs: if True, remove inter chain (based on label_asym_id) bonds that are far away from each other.

    returns:
        AtomArray: Biotite AtomArray merged inter residue bonds into atom_array.bonds
    """
    res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    inter_bonds = _connect_inter_residue(atom_array, res_starts)

    if atom_array.bonds is None:
        atom_array.bonds = inter_bonds
        return atom_array

    select_mask = np.ones(len(inter_bonds._bonds), dtype=bool)
    if exclude_struct_conn_pairs:
        for b_idx, (atom_i, atom_j, b_type) in enumerate(inter_bonds._bonds):
            atom_k = atom_i if atom_array.atom_name[atom_i] in ("N", "O3'") else atom_j
            bonds, types = atom_array.bonds.get_bonds(atom_k)
            if len(bonds) == 0:
                continue
            for b in bonds:
                if (
                    # adjacent residues
                    abs((res_starts <= b).sum() - (res_starts <= atom_k).sum()) == 1
                    and atom_array.chain_id[b] == atom_array.chain_id[atom_k]
                    and atom_array.atom_name[b] not in ("C", "P")
                ):
                    select_mask[b_idx] = False
                    break

    if remove_far_inter_chain_pairs:
        if not hasattr(atom_array, "label_asym_id"):
            logging.warning(
                "label_asym_id not found, far inter chain bonds will not be removed"
            )
        for b_idx, (atom_i, atom_j, b_type) in enumerate(inter_bonds._bonds):
            if atom_array.label_asym_id[atom_i] != atom_array.label_asym_id[atom_j]:
                coord_i = atom_array.coord[atom_i]
                coord_j = atom_array.coord[atom_j]
                if np.linalg.norm(coord_i - coord_j) > 2.5:
                    select_mask[b_idx] = False

    # filter out removed_inter_bonds from atom_array.bonds
    remove_bonds = inter_bonds._bonds[~select_mask]
    remove_mask = np.isin(atom_array.bonds._bonds[:, 0], remove_bonds[:, 0]) & np.isin(
        atom_array.bonds._bonds[:, 1], remove_bonds[:, 1]
    )
    atom_array.bonds._bonds = atom_array.bonds._bonds[~remove_mask]

    # merged normal inter_bonds into atom_array.bonds
    inter_bonds._bonds = inter_bonds._bonds[select_mask]
    atom_array.bonds = atom_array.bonds.merge(inter_bonds)
    return atom_array


def res_names_to_sequence(res_names: list[str]) -> str:
    """convert res_names to sequences {chain_id: canonical_sequence} based on CCD

    Return
        str: canonical_sequence
    """
    seq = ""
    for res_name in res_names:
        one = get_one_letter_code(res_name)
        one = "X" if one is None else one
        one = "X" if len(one) > 1 else one
        seq += one
    return seq


__all__ = ["get_chain_type_from_ccd_code"]
