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

import concurrent.futures
import copy
import logging
import random
import warnings
from collections import Counter
from typing import Any, Optional

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import BondType as RDKitBondType

from protenix.data import ccd
from protenix.data.parser import DistillationMMCIFParser, MMCIFParser

logger = logging.getLogger(__name__)


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


# RDKit bond type to biotite bond type enumerations
RDKIT_TO_BIOTITE_BOND_TYPE = {
    RDKitBondType.SINGLE: struc.BondType.SINGLE,
    RDKitBondType.DOUBLE: struc.BondType.DOUBLE,
    RDKitBondType.TRIPLE: struc.BondType.TRIPLE,
    RDKitBondType.QUADRUPLE: struc.BondType.QUADRUPLE,
    # RDKitBondType.AROMATIC: struc.BondType.AROMATIC,
}


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


def rdkit_mol_to_atom_array(mol: Chem.Mol, removeHs: bool = True) -> AtomArray:
    """
    Convert rdkit mol to biotite AtomArray

    Args:
        mol (Chem.Mol): rdkit mol
        removeHs (bool): whether to remove hydrogens in atom_array

    Returns:
        AtomArray: biotite AtomArray
    """
    atom_count = mol.GetNumAtoms()
    atom_array = AtomArray(atom_count)
    atom_array.hetero[:] = True
    atom_array.res_name[:] = "UNL"
    atom_array.add_annotation("charge", int)

    conf = mol.GetConformer()
    coord = conf.GetPositions()

    element_count = Counter()
    for i, atom in enumerate(mol.GetAtoms()):
        element = atom.GetSymbol().upper()
        element_count[element] += 1
        atom_name = f"{element}{element_count[element]}"

        atom.SetProp("name", atom_name)

        atom_array.atom_name[i] = atom_name
        atom_array.element[i] = element
        atom_array.charge[i] = atom.GetFormalCharge()
        atom_array.coord[i, :] = coord[i, :]

    bonds = []
    for bond in mol.GetBonds():
        bonds.append(
            [
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                RDKIT_TO_BIOTITE_BOND_TYPE.get(bond.GetBondType(), struc.BondType.ANY),
            ]
        )
    atom_array.bonds = struc.BondList(atom_count, np.array(bonds))
    if removeHs:
        atom_array = atom_array[atom_array.element != "H"]
    return atom_array


def rdkit_mol_to_atom_info(mol: Chem.Mol) -> dict[str, Any]:
    """
    Convert RDKit Mol to atom_info dict.

    Args:
        mol (Chem.Mol): rdkit mol

    Returns:
        dict: info of atoms
        example: {
            "atom_array": biotite_AtomArray_object,
            "atom_map_to_atom_name": {1: "C2"}, # only for smiles
            }
    """
    atom_info = {}
    atom_map_to_atom_name = {}
    atom_idx_to_atom_name = {}

    element_count = Counter()
    for atom in mol.GetAtoms():
        element = atom.GetSymbol().upper()
        element_count[element] += 1
        atom_name = f"{element}{element_count[element]}"
        atom.SetProp("name", atom_name)
        if atom.GetAtomMapNum() != 0:
            atom_map_to_atom_name[atom.GetAtomMapNum()] = atom_name
        atom_idx_to_atom_name[atom.GetIdx()] = atom_name

    if atom_map_to_atom_name:
        # Atom map for input SMILES
        atom_info["atom_map_to_atom_name"] = atom_map_to_atom_name
    else:
        # Atom index for input file
        atom_info["atom_map_to_atom_name"] = atom_idx_to_atom_name

    # Atom_array without hydrogens
    atom_info["atom_array"] = rdkit_mol_to_atom_array(mol, removeHs=True)
    return atom_info


def lig_file_to_atom_info(lig_file_path: str) -> dict[str, Any]:
    """
    Convert ligand file to biotite AtomArray.

    Args:
        lig_file_path (str): ligand file path with one of the following suffixes: [mol, mol2, sdf, pdb]

    Returns:
        dict: info of atoms
        example: {
            "atom_array": biotite_AtomArray_object,
            "atom_map_to_atom_name": {1: "C2"}, # only for smiles
            }
    """
    if lig_file_path.endswith(".mol"):
        mol = Chem.MolFromMolFile(lig_file_path)
    elif lig_file_path.endswith(".sdf"):
        suppl = Chem.SDMolSupplier(lig_file_path)
        mol = next(suppl)
    elif lig_file_path.endswith(".pdb"):
        mol = Chem.MolFromPDBFile(lig_file_path)
    elif lig_file_path.endswith(".mol2"):
        mol = Chem.MolFromMol2File(lig_file_path)
    else:
        raise ValueError(f"Invalid ligand file type: .{lig_file_path.split('.')[-1]}")
    assert (
        mol is not None
    ), f"Failed to retrieve molecule from file, invalid ligand file: {lig_file_path}. \
        Please provide a file with one of the following suffixes: [mol, mol2, sdf, pdb]."

    assert (
        mol.GetConformer().Is3D()
    ), f"3D conformer not found in ligand file: {lig_file_path}"
    atom_info = rdkit_mol_to_atom_info(mol)
    return atom_info


def smiles_to_atom_info(smiles: str) -> dict:
    """
    Convert smiles to atom_array, and atom_map_to_atom_name

    Args:
        smiles (str): smiles string, like "CCCC", or "[C:1]NC(=O)" (use num to label covalent bond atom.)

    Returns:
        dict: info of atoms
        example: {
            "atom_array": biotite_AtomArray_object,
            "atom_map_to_atom_name": {1: "C2"}, # only for smiles
            }
    """
    atom_info = {}
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(AllChem.EmbedMolecule, mol)

        try:
            ret_code = future.result(timeout=90)
        except concurrent.futures.TimeoutError as exc:
            raise TimeoutError(
                'Conformer generation timed out.  \
                    Please change the "ligand" input format to "CCD_" or "FILE_".'
            ) from exc

    if ret_code != 0:
        # retry with random coords
        ret_code = AllChem.EmbedMolecule(mol, useRandomCoords=True)

    assert ret_code == 0, f"Conformer generation failed for input SMILES: {smiles}"
    atom_info = rdkit_mol_to_atom_info(mol)
    return atom_info


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


def add_entity_atom_array(single_job_dict: dict) -> dict:
    """
    Add atom_array to each entity in single_job_dict

    Args:
        single_job_dict (dict): input job dict

    Returns:
        dict: deepcopy and updated job dict with atom_array
    """

    valid_keys = ["design", "target", "name"]
    small_mol_keys = ["ligand"]
    polymer_keys = ["proteinChain", "dnaSequence", "rnaSequence"]
    biomol_keys = small_mol_keys + polymer_keys
    smiles_ligand_count = 0

    single_job_dict = copy.deepcopy(single_job_dict)

    assert all(
        key in valid_keys for key in single_job_dict.keys()
    ), f"single_job_dict must only contain allowed keys: {valid_keys}"

    for entity in single_job_dict.get("design", []):
        key = list(entity.keys())[0]

        assert (
            key in biomol_keys
        ), f"design entity must only contain allowed keys: {biomol_keys}"
        assert (
            len(list(entity.keys())) == 1
        ), f"design entity must only contain one key got {list(entity.keys())}"

        if key in polymer_keys:
            atom_info = build_design_polymer(entity)
        elif key in small_mol_keys:
            atom_info = build_ligand(entity)
            atom_info["atom_array"].is_target[:] = False
            if not entity["ligand"].startswith("CCD_"):
                smiles_ligand_count += 1
                assert smiles_ligand_count <= 99, "too many smiles ligands"
                # use lower case res_name (l01, l02, ..., l99) to avoid conflict with CCD code
                entity["atom_array"].res_name[:] = f"l{smiles_ligand_count:02d}"
        else:
            raise ValueError(f"invalid design entity key: {key}")

        entity[key].update(atom_info)

    for entity in single_job_dict.get("target", []):
        key = list(entity.keys())[0]

        assert (
            len(list(entity.keys())) == 1
        ), f"target entity must only contain one key got {list(entity.keys())}"

        if key in polymer_keys:
            atom_info = build_target_polymer(entity)
        elif key in small_mol_keys:
            atom_info = build_ligand(entity)
            atom_info["atom_array"].is_target[:] = True

            if not entity["ligand"].startswith("CCD_"):
                smiles_ligand_count += 1
                assert smiles_ligand_count <= 99, "too many smiles ligands"
                # use lower case res_name (l01, l02, ..., l99) to avoid conflict with CCD code
                entity["atom_array"].res_name[:] = f"l{smiles_ligand_count:02d}"
        else:
            raise ValueError(f"invalid design entity key: {key}")

        entity[key].update(atom_info)

    return single_job_dict


def build_design_polymer(entity: dict) -> dict:
    # TODO: tidy up please
    poly_to_mol = {
        "proteinChain": "protein",
        "dnaSequence": "dna",
        "rnaSequence": "rna",
    }
    poly_type, info = list(entity.items())[0]
    assert poly_type in poly_to_mol, f"invalid design entity key: {poly_type}"

    atom_array = create_design_atom_array(
        info["length"], mol_type=poly_to_mol[poly_type]
    )
    return {"atom_array": atom_array}


def build_target_polymer(entity: dict) -> dict:
    poly_type, info = list(entity.items())[0]

    pid = info["pdb_id"]
    if pid.startswith("AF-"):
        parser = DistillationMMCIFParser(mmcif_file=info["file"])
        atom_array = parser.get_structure_dict()["atom_array"]
    else:
        parser = MMCIFParser(mmcif_file=info["file"])
        atom_array = parser.get_bioassembly()["atom_array"]

    assert (
        np.unique(atom_array.chain_id).shape[0] == 1
    ), "Expected single chain structure, but found multiple chains"

    atom_array.add_annotation("is_target", dtype=np.int32)
    atom_array.is_target[:] = True

    atom_array.add_annotation("is_hotspot", dtype=np.int32)
    atom_array.is_hotspot[:] = False
    atom_array.is_hotspot[info["hotspots"]] = True

    # TODO: is_target_token is only given as true when the centre atoms are
    # resolved. Probably need to rethink how we're doing this.
    atom_array.add_annotation("is_resolved", dtype=np.int32)
    atom_array.is_resolved[:] = True

    return {"atom_array": atom_array}


# TODO: find better location
class DesignAtomArrayBuilder:
    """
    Build AtomArray with all annotations required by MMCIFParser pipeline.
    """

    def __init__(
        self, length: int, mol_type: str, chain_id: str = "A", entity_id: str = "1"
    ):
        """
        Initialise builder with basic parameters.

        Args:
            length (int): Number of atoms/residues
            mol_type (str): Molecule type ("protein", "ligand", "dna", "rna")
            chain_id (str): Chain identifier
            entity_id (str): Entity identifier
        """
        self.length = length
        self.mol_type = mol_type
        self.chain_id = chain_id
        self.entity_id = entity_id
        self.atom_array = AtomArray(length)

        if mol_type == "protein":
            self.atom_name = "CA"
            self.element = "C"
            self.res_name = "TBD"  # to be designed :)
            self.canonical_res_name = (
                "GLY"  # NOTE: has to be legit resname otherwise featurizer breaks
            )
        elif mol_type == "rna":
            self.atom_name = "C1'"
            self.element = "C"
            self.res_name = "TBD"
            self.canonical_res_name = "A"
        elif mol_type == "dna":
            self.atom_name = "C1'"
            self.element = "C"
            self.res_name = "TBD"
            self.canonical_res_name = "A"
        elif mol_type == "ligand":
            raise NotImplementedError("Ligand design not implemented")
        else:
            raise ValueError(f"Invalid molecule type: {mol_type}")

    def build_basic_properties(
        self,
        coordinates: Optional[np.ndarray] = None,
    ):
        """Build basic atomic properties."""

        # Coordinates
        if coordinates is None:
            coordinates = np.zeros((self.length, 3))
        self.atom_array.coord = coordinates

        # Basic properties
        self.atom_array.atom_name = [self.atom_name] * self.length
        self.atom_array.element = [self.element] * self.length
        self.atom_array.hetero = [False] * self.length
        self.atom_array.res_name = [self.res_name] * self.length
        self.atom_array.res_id = list(range(1, self.length + 1))
        self.atom_array.chain_id = [self.chain_id] * self.length

    def build_insertion_codes(self):
        """Build insertion code annotations."""
        self.atom_array.add_annotation("ins_code", dtype=str)
        self.atom_array.ins_code = [""] * self.length

    def build_altloc_and_occupancy(self):
        """Build alternate location and occupancy annotations."""
        self.atom_array.add_annotation("label_alt_id", dtype=str)
        self.atom_array.label_alt_id = ["."] * self.length

        self.atom_array.add_annotation("occupancy", dtype=np.float32)
        self.atom_array.occupancy = [1.0] * self.length

    def build_sequence_annotations(self):
        """Build sequence-related annotations."""
        self.atom_array.add_annotation("auth_seq_id", dtype=str)
        self.atom_array.auth_seq_id = [str(i) for i in range(1, self.length + 1)]

        self.atom_array.add_annotation("label_seq_id", dtype=str)
        self.atom_array.label_seq_id = [str(i) for i in range(1, self.length + 1)]

    def build_b_factor_and_charge(self, b_factor: float = 100.0):
        """Build B-factor and charge annotations."""
        self.atom_array.add_annotation("b_factor", dtype=np.float32)
        self.atom_array.b_factor = [b_factor] * self.length

        self.atom_array.add_annotation("charge", dtype=np.int8)
        self.atom_array.charge = [0] * self.length

    def build_chain_and_entity_annotations(self):
        """Build chain and entity-related annotations."""
        self.atom_array.add_annotation("auth_asym_id", dtype=str)
        self.atom_array.auth_asym_id = [self.chain_id] * self.length

        self.atom_array.add_annotation("label_asym_id", dtype=str)
        self.atom_array.label_asym_id = [self.chain_id] * self.length

        self.atom_array.add_annotation("label_entity_id", dtype=str)
        self.atom_array.label_entity_id = [self.entity_id] * self.length

    def build_resolution_mask(self, is_resolved: bool = False):
        """Build resolution mask annotation."""
        self.atom_array.add_annotation("is_resolved", dtype=bool)
        self.atom_array.is_resolved = [is_resolved] * self.length

    def build_molecule_type_annotations(self):
        """Build molecule type annotations."""
        self.atom_array.add_annotation("mol_type", dtype=str)
        self.atom_array.mol_type = [self.mol_type] * self.length

        self.atom_array.add_annotation("chain_mol_type", dtype=str)
        self.atom_array.chain_mol_type = [self.mol_type] * self.length

    def build_molecule_flags(self, mol_type: str = "protein"):
        """Build binary molecule type flags."""
        is_ligand = 1 if mol_type == "ligand" else 0
        is_dna = 1 if mol_type == "dna" else 0
        is_rna = 1 if mol_type == "rna" else 0
        is_protein = 1 if mol_type == "protein" else 0

        for flag_name, flag_value in [
            ("is_ligand", is_ligand),
            ("is_dna", is_dna),
            ("is_rna", is_rna),
            ("is_protein", is_protein),
        ]:
            self.atom_array.add_annotation(flag_name, dtype=np.int8)
            setattr(self.atom_array, flag_name, [flag_value] * self.length)

    def build_center_atom_mask(self):
        """Build center atom mask (all 1s for CA-only)."""
        self.atom_array.add_annotation("centre_atom_mask", dtype=np.int8)
        self.atom_array.centre_atom_mask = [1] * self.length

    def build_representation_masks(self):
        """Build representation atom masks."""
        # Distogram representation mask (all 1s for CA atoms)
        self.atom_array.add_annotation("distogram_rep_atom_mask", dtype=np.int8)
        self.atom_array.distogram_rep_atom_mask = [1] * self.length

        # PLDDT representation mask (0s for ligands, 1s for proteins)
        self.atom_array.add_annotation("plddt_m_rep_atom_mask", dtype=np.int8)
        # Check if this is protein (assuming ligand for now based on your output)
        plddt_value = 0  # Change to 1 for proteins
        self.atom_array.plddt_m_rep_atom_mask = [plddt_value] * self.length

    def build_canonical_sequence(self):
        """Build canonical sequence residue names."""
        self.atom_array.add_annotation("cano_seq_resname", dtype=str)
        self.atom_array.cano_seq_resname = [self.canonical_res_name] * self.length

    def build_token_annotations(self):
        """Build token-related annotations."""
        self.atom_array.add_annotation("tokatom_idx", dtype=np.int32)
        self.atom_array.tokatom_idx = [0] * self.length

        self.atom_array.add_annotation("modified_res_mask", dtype=np.int8)
        self.atom_array.modified_res_mask = [0] * self.length

    def build_id_annotations(self):
        """Build integer ID annotations."""
        # All zeros for single chain/entity
        # TODO: do sym_ids start from 0 or 1?
        for id_name in ["asym_id_int", "entity_id_int", "sym_id_int"]:
            self.atom_array.add_annotation(id_name, dtype=np.int32)
            setattr(self.atom_array, id_name, [0] * self.length)

    def build_molecule_id_annotations(self):
        """Build molecule ID annotations."""
        # mol_id: sequential for each atom (0, 1, 2, ...)
        self.atom_array.add_annotation("mol_id", dtype=np.int32)
        self.atom_array.mol_id = list(range(self.length))

        # entity_mol_id: first atom is 0, rest are 1
        self.atom_array.add_annotation("entity_mol_id", dtype=np.int32)
        entity_mol_ids = [0] + [1] * (self.length - 1)
        self.atom_array.entity_mol_id = entity_mol_ids

        # mol_atom_index: all zeros (first atom in each molecule)
        self.atom_array.add_annotation("mol_atom_index", dtype=np.int32)
        self.atom_array.mol_atom_index = [0] * self.length

    def build_reference_annotations(self):
        """Build reference space annotations."""
        # ref_space_uid: sequential (0, 1, 2, ...)
        self.atom_array.add_annotation("ref_space_uid", dtype=np.int32)
        self.atom_array.ref_space_uid = list(range(self.length))

        # ref_pos: all zeros
        self.atom_array.add_annotation("ref_pos", dtype=np.float32)
        self.atom_array.ref_pos = np.zeros((self.length, 3))

        # ref_charge: all zeros
        self.atom_array.add_annotation("ref_charge", dtype=np.int8)
        self.atom_array.ref_charge = [0] * self.length

        # ref_mask: all zeros
        self.atom_array.add_annotation("ref_mask", dtype=np.int8)
        self.atom_array.ref_mask = [0] * self.length

        # res_perm: all "0"
        self.atom_array.add_annotation("res_perm", dtype=str)
        self.atom_array.res_perm = ["0"] * self.length

    def build_target_flags(self):
        """Build target flags."""
        self.atom_array.add_annotation("is_target", dtype=np.int8)
        self.atom_array.is_target = [0] * self.length

    def build_bonds(self):
        """
        Build bonds for CA-only chain - consecutive residues are bonded.
        Creates a BondList compatible with biotite's adjacency_matrix() method.
        """
        if self.length <= 1:
            # No bonds possible with 0 or 1 atoms
            self.atom_array.bonds = struc.BondList(
                self.length, np.empty((0, 3), dtype=int)
            )
            return

        # Create bonds between consecutive CA atoms (i.e., consecutive residues)
        bonds = []
        for i in range(self.length - 1):
            # Bond between atom i and atom i+1
            # Format: [atom_i, atom_j, bond_type]
            bonds.append([i, i + 1, struc.BondType.SINGLE])

        bonds_array = np.array(bonds, dtype=int)

        # Create BondList object that biotite expects
        self.atom_array.bonds = struc.BondList(self.length, bonds_array)

    def build_all_annotations(
        self,
        coordinates: Optional[np.ndarray] = None,
    ) -> AtomArray:
        """
        Build all annotations in one call.

        Args:
            coordinates: Atom coordinates [length, 3]
            mol_type: Molecule type ("protein", "ligand", "dna", "rna")
            res_name: Residue name for canonical sequence

        Returns:
            Complete AtomArray with all annotations
        """
        self.build_basic_properties(coordinates=coordinates)
        self.build_insertion_codes()
        self.build_altloc_and_occupancy()
        self.build_sequence_annotations()
        self.build_b_factor_and_charge()
        self.build_chain_and_entity_annotations()
        self.build_resolution_mask()
        self.build_molecule_type_annotations()
        self.build_molecule_flags()
        self.build_center_atom_mask()
        self.build_representation_masks()
        self.build_canonical_sequence()
        self.build_token_annotations()
        self.build_id_annotations()
        self.build_molecule_id_annotations()
        self.build_reference_annotations()
        self.build_target_flags()
        self.build_bonds()

        return self.atom_array


def create_design_atom_array(length: int, mol_type: str) -> AtomArray:
    """Create a complete AtomArray suitable for design task with all annotations.

    Args:
        length (int): Number of atoms/residues
        mol_type (str): Molecule type ("protein", "ligand", "dna", "rna")

    Returns:
        AtomArray: Complete AtomArray with all annotations
    """
    builder = DesignAtomArrayBuilder(length=length, mol_type=mol_type)
    return builder.build_all_annotations()
