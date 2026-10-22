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
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import biotite
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
from biotite.structure import AtomArray
from rdkit import Chem

from artios.common.chain_type import ChainType
from artios.common.residue_names import STANDARD_POLYMER_TYPES_SET
from artios.configs.configs_data import data_configs
from artios.data.chem.perms import get_substructure_perms

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
    ccd_cif = biotite_load_ccd_cif()
    if ccd_code not in ccd_cif:
        logger.warning(f"Warning: get_component_atom_array() can not parse {ccd_code}")
        return None
    try:
        comp = pdbx.get_component(ccd_cif, data_block=ccd_code, use_ideal_coord=True)
    except biotite.InvalidFileError as e:
        logger.warning(
            f"Warning: get_component_atom_array() can not parse {ccd_code} for {e}"
        )
        return None

    # biotite's get_component sets hetero=True on all atoms. Standard polymer
    # residues need hetero=False so PDB writers emit ATOM (not HETATM) records,
    # which is required for PyMOL/other viewers to render backbone connectivity.
    if ccd_code in STANDARD_POLYMER_TYPES_SET:
        comp.hetero[:] = False

    # Add leaving atom annotation (not provided by biotite's get_component)
    atom_category = ccd_cif[ccd_code]["chem_comp_atom"]
    leaving_atom_flag = atom_category["pdbx_leaving_atom_flag"].as_array()
    comp.set_annotation("is_leaving_atom", leaving_atom_flag == "Y")

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
    ccd_cif = biotite_load_ccd_cif()
    if ccd_code not in ccd_cif:
        return None
    one = ccd_cif[ccd_code]["chem_comp"]["one_letter_code"].as_item()
    if one == "?":
        return None
    return one


@functools.lru_cache(maxsize=None)
def get_mol_type(ccd_code: str) -> str:
    """Get mol_type from CCD components file.

    Based on _chem_comp.type:
    http://mmcif.rcsb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_chem_comp.type.html

    Args:
        ccd_code (str): ccd code

    Returns:
        str: mol_type, one of "protein", "rna", "dna", "ligand"
    """
    ccd_cif = biotite_load_ccd_cif()
    if ccd_code not in ccd_cif:
        return "ligand"

    link_type = ccd_cif[ccd_code]["chem_comp"]["type"].as_item().upper()

    if "PEPTIDE" in link_type and link_type != "PEPTIDE-LIKE":
        return "protein"
    if "DNA" in link_type:
        return "dna"
    if "RNA" in link_type:
        return "rna"
    return "ligand"


@functools.lru_cache(maxsize=None)
def get_chain_type_from_ccd_code(ccd_code: str) -> ChainType:
    """Get the ChainType enum for a CCD code.

    Uses _chem_comp.type from the CCD components file.

    Args:
        ccd_code: CCD code (e.g., "ALA", "DA", "ATP")

    Returns:
        ChainType enum value
    """
    ccd_cif = biotite_load_ccd_cif()
    if ccd_code not in ccd_cif:
        return ChainType.NON_POLYMER

    link_type = ccd_cif[ccd_code]["chem_comp"]["type"].as_item().upper()

    if "PEPTIDE" in link_type and link_type != "PEPTIDE-LIKE":
        if "D-PEPTIDE" in link_type or "D-BETA" in link_type or "D-GAMMA" in link_type:
            return ChainType.POLYPEPTIDE_D
        return ChainType.POLYPEPTIDE_L
    if "DNA" in link_type:
        if "RNA" in link_type:
            return ChainType.DNA_RNA_HYBRID
        return ChainType.DNA
    if "RNA" in link_type:
        return ChainType.RNA
    if "SACCHARIDE" in link_type:
        return ChainType.BRANCHED
    return ChainType.NON_POLYMER


@functools.lru_cache(maxsize=None)
def get_smiles_str(ccd_code: str) -> str | None:
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


def get_all_ccd_code() -> list[str]:
    """get all ccd code from components file"""
    ccd_cif = biotite_load_ccd_cif()
    return list(ccd_cif.keys())


@functools.lru_cache(maxsize=None)
def is_valid_ccd_code(ccd_code: str) -> bool:
    """Check if a CCD code exists in the Chemical Component Dictionary.

    Args:
        ccd_code: CCD code to validate (e.g., "ZN", "ATP", "HIS")

    Returns:
        True if the code exists, False otherwise
    """
    ccd_cif = biotite_load_ccd_cif()
    return ccd_code in ccd_cif


_ccd_rdkit_mols: dict[str, Chem.Mol] = {}


def get_component_rdkit_mol(ccd_code: str) -> Union[Chem.Mol, None]:
    """Returns the pre-computed reference RDKit mol corresponding to the CCD.
    The reference conformer is computed with RDKit ETKDGv3 as stated in
    AlphaFold3's supplementary information.

    NOTE: it is unclear the performance difference of this vs doing something
    like ccd_code_to_rdkit, but we save that for ablations.

    Args:
        ccd_code (str): ccd code

    Returns
        rdkit.Chem.Mol: rdkit mol with ref coord
    """
    global _ccd_rdkit_mols
    if _ccd_rdkit_mols:
        return _ccd_rdkit_mols.get(ccd_code, None)

    rdkit_mol_pkl = RKDIT_MOL_PKL
    if rdkit_mol_pkl.exists():
        with open(rdkit_mol_pkl, "rb") as f:
            _ccd_rdkit_mols = pickle.load(f)
        return _ccd_rdkit_mols.get(ccd_code, None)
    else:
        raise FileNotFoundError(
            f"CCD components file {rdkit_mol_pkl} not found, please download it to your DATA_ROOT_DIR before running."
            "or generate it with scripts/gen_ccd_cache.py"
        )
    # return ccd_code_to_rdkit(ccd_code)


@dataclass(frozen=True)
class RefInfo:
    """Reference conformer information for a CCD component."""

    ccd: str
    atom_map: Dict[str, int]
    coord: np.ndarray  # (n_atom, 3)
    charge: np.ndarray  # (n_atom,)
    mask: np.ndarray  # (n_atom,)
    perm: np.ndarray  # (n_atom_wo_h, n_perm)


@functools.lru_cache
def get_ccd_ref_info(ccd_code: str, return_perm: bool = True) -> Optional[RefInfo]:
    """Get reference atom mapping and coordinates from CCD.

    Ref: AlphaFold3 SI Chapter 2.8
    Reference features. Features derived from a residue, nucleotide or ligand's reference conformer.
    Given an input CCD code or SMILES string, the conformer is typically generated
    with RDKit v.2023_03_3 [25] using ETKDGv3 [26]. On error, we fall back to using the CCD ideal coordinates,
    or finally the representative coordinates
    if they are from before our training date cut-off (2021-09-30 unless otherwise stated).
    At the end, any atom coordinates still missing are set to zeros.

    Args:
        ccd_code (str): CCD code
        return_perm (bool): Whether to return atom permutations. Defaults to True.

    Returns:
        Optional[RefInfo]: Reference information containing:
            - ccd: CCD code
            - atom_map: Atom name to atom index mapping
            - coord: Atom coordinates, shape [n_atom, 3]
            - charge: Atom formal charges, shape [n_atom]
            - mask: Atom mask, shape [n_atom]
            - perm: Atom permutations, shape [n_atom_wo_h, n_perm] (if return_perm=True)
        Returns None if molecule cannot be loaded or has no atoms.
    """
    mol = get_component_rdkit_mol(ccd_code)
    if mol is None:
        return None
    if mol.GetNumAtoms() == 0:  # eg: "UNL"
        logger.warning(
            f"Warning: mol {ccd_code} from get_component_rdkit_mol() has no atoms, "
            "get_ccd_ref_info() return None"
        )
        return None

    conf = mol.GetConformer(mol.ref_conf_id)
    coord = conf.GetPositions()
    charge = np.array([atom.GetFormalCharge() for atom in mol.GetAtoms()])

    perm = None
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
        # np.ndarray[int]: atom permutation, (n_atom_wo_h, n_perm)
        perm = perm.T

    return RefInfo(
        ccd=ccd_code,
        atom_map=mol.atom_map,
        coord=coord,
        mask=mol.ref_mask,
        charge=charge,
        perm=perm if perm is not None else np.array([]),
    )


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
    Add polymer bonds (C-N or O3'-P) between adjacent residues based on auth_seq_id.

    Args:
        atom_array: input atom array
        exclude_struct_conn_pairs: if True, do not add bond between adjacent residues already
            has non-standard polymer bonds on atom C or N or O3' or P.
        remove_far_inter_chain_pairs: if True, remove inter chain (based on label_asym_id)
            bonds that are far away from each other.

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
    """
    Convert res_names to sequences {chain_id: canonical_sequence}
    based on CCD.

    Args:
        res_names: residue names (HIS, ARG, ...)

    Returns:
        str: canonical_sequence
    """
    seq = ""
    for res_name in res_names:
        one = get_one_letter_code(res_name)
        one = "X" if one is None else one
        one = "X" if len(one) > 1 else one
        seq += one
    return seq
