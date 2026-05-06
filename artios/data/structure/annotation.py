"""Legacy v1 annotation helpers retained for the cif.py pre-expand
chain and ``array_builder.add_atom_array_attributes``.

Everything here is either: (a) still called directly by cif.py /
array_builder for the pre-annotation atom_array prep that runs before
expand_assembly + filters, or (b) a legacy helper kept for back-compat
until its callers are migrated onto ``annotation_v2``.

Migration progress (cif.py refactor):
- pocket / fragment / unresolve_low_plddt → moved to annotation_v2
  (step 9).
- populate_ccd_reference / remove_bonds_between_polymer_chains → live
  in annotation_v2 (step 2).
- pre-expand chain and unique_chain_and_add_ids /
  find_equiv_mol_and_assign_ids / add_mol_id / add_ref_space_uid:
  **still here**; migration blocked on making a non-reordering v2 path
  for the pre-expand stage (expand_assembly is sensitive to atom order).
"""

import logging
from collections import Counter
from typing import Optional

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray, get_chain_starts
from biotite.structure.molecules import get_molecule_indices

from artios.common.constants import (
    DNA_STD_RESIDUES,
    PROT_STD_RESIDUES_ONE_TO_THREE,
    RES_ATOMS_DICT,
    RNA_STD_RESIDUES,
    STD_RESIDUES,
)
from artios.data.chem import ccd
from artios.data.structure.annotation_v2 import remove_bonds_between_polymer_chains
from artios.data.utils import get_starts_by

logger = logging.getLogger(__name__)


def add_token_mol_type(atom_array: AtomArray, sequences: dict[str, str]) -> AtomArray:
    """
    Add molecule types in atom_arry.mol_type based on ccd pdbx_type.

    Args:
        atom_array (AtomArray): Biotite AtomArray object.
        sequences (dict[str, str]): A dict of label_entity_id --> canonical_sequence

    Return
        AtomArray: add atom_arry.mol_type = "protein" | "rna" | "dna" | "ligand"
    """
    mol_types = np.zeros(len(atom_array), dtype="U7")
    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for start, stop in zip(starts[:-1], starts[1:]):
        entity_id = atom_array.label_entity_id[start]
        if entity_id not in sequences:
            # non-poly is ligand
            mol_types[start:stop] = "ligand"
            continue
        res_name = atom_array.res_name[start]

        mol_types[start:stop] = ccd.get_mol_type(res_name)

    atom_array.set_annotation("mol_type", mol_types)
    return atom_array


def add_atom_mol_type_mask(atom_array: AtomArray) -> AtomArray:
    """
    Mask indicates is_protein / rna / dna / ligand.
    It is atom-level which is different with paper (token-level).
    The type of each atom is determined based on the most frequently
    occurring type in the chain to which it belongs.

    Args:
        atom_array (AtomArray): Biotite AtomArray object

    Returns:
        AtomArray: Biotite AtomArray object with
                   "is_ligand", "is_dna", "is_rna", "is_protein" annotation added.
    """
    # it should be called after mmcif_parser.add_token_mol_type
    chain_starts = struc.get_chain_starts(atom_array, add_exclusive_stop=True)
    chain_mol_type = []
    for start, end in zip(chain_starts[:-1], chain_starts[1:]):
        mol_types = atom_array.mol_type[start:end]
        mol_type_count = Counter(mol_types)
        most_freq_mol_type = max(mol_type_count, key=mol_type_count.get)
        chain_mol_type.extend([most_freq_mol_type] * (end - start))
    atom_array.set_annotation("chain_mol_type", chain_mol_type)

    for type_str in ["ligand", "dna", "rna", "protein"]:
        mask = (atom_array.chain_mol_type == type_str).astype(int)
        atom_array.set_annotation(f"is_{type_str}", mask)
    return atom_array


def add_modified_res_mask(atom_array: AtomArray) -> AtomArray:
    """
    Ref: AlphaFold3 SI Chapter 5.9.3

    Determine if an atom belongs to a modified residue,
    which is used to calculate the Modified Residue Scores in sample ranking:
    Modified residue scores are ranked according to the average pLDDT of the modified residue.

    Args:
        atom_array (AtomArray): Biotite AtomArray object

    Returns:
        AtomArray: Biotite AtomArray object with
                   "modified_res_mask" annotation added.
    """
    modified_res_mask = []
    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for start, stop in zip(starts[:-1], starts[1:]):
        res_name = atom_array.res_name[start]
        mol_type = atom_array.mol_type[start]
        res_atom_nums = stop - start
        if res_name not in STD_RESIDUES and mol_type != "ligand":
            modified_res_mask.extend([1] * res_atom_nums)
        else:
            modified_res_mask.extend([0] * res_atom_nums)
    atom_array.set_annotation("modified_res_mask", modified_res_mask)
    return atom_array


def add_centre_atom_mask(atom_array: AtomArray) -> AtomArray:
    """
    Ref: AlphaFold3 SI Chapter 2.6
        • A standard amino acid residue (Table 13) is represented as a single token.
        • A standard nucleotide residue (Table 13) is represented as a single token.
        • A modified amino acid or nucleotide residue is tokenized per-atom (i.e. N tokens for an N-atom residue)
        • All ligands are tokenized per-atom
    For each token we also designate a token centre atom, used in various places below:
        • Cα for standard amino acids
        • C1′ for standard nucleotides
        • For other cases take the first and only atom as they are tokenized per-atom.

    Args:
        atom_array (AtomArray): Biotite AtomArray object

    Returns:
        AtomArray: Biotite AtomArray object with "centre_atom_mask" annotation added.
    """
    res_name = list(STD_RESIDUES.keys())
    std_res = np.isin(atom_array.res_name, res_name) & (atom_array.mol_type != "ligand")
    prot_res = np.char.str_len(atom_array.res_name) == 3
    prot_centre_atom = prot_res & (atom_array.atom_name == "CA")
    nuc_centre_atom = (~prot_res) & (atom_array.atom_name == r"C1'")
    not_std_res = ~std_res
    centre_atom_mask = (std_res & (prot_centre_atom | nuc_centre_atom)) | not_std_res
    centre_atom_mask = centre_atom_mask.astype(int)
    atom_array.set_annotation("centre_atom_mask", centre_atom_mask)
    return atom_array


def add_plddt_m_rep_atom_mask(atom_array: AtomArray) -> AtomArray:
    """
    Ref: AlphaFold3 SI Chapter 4.3.1
    the representative atom for plddt loss
    • Atoms such that the distance in the ground truth between atom l and atom m is less than 15 Å
        if m is a protein atom or less than 30 Å if m is a nucleic acid atom.
    • Only atoms in polymer chains.
    • One atom per token - Cα for standard protein residues
        and C1′ for standard nucleic acid residues.

    Args:
        atom_array (AtomArray): Biotite AtomArray object

    Returns:
        AtomArray: Biotite AtomArray object with "plddt_m_rep_atom_mask" annotation added.
    """
    std_res = np.isin(atom_array.res_name, list(STD_RESIDUES.keys())) & (
        atom_array.mol_type != "ligand"
    )
    ca_or_c1 = (atom_array.atom_name == "CA") | (atom_array.atom_name == r"C1'")
    plddt_m_rep_atom_mask = (std_res & ca_or_c1).astype(int)
    atom_array.set_annotation("plddt_m_rep_atom_mask", plddt_m_rep_atom_mask)
    return atom_array


def add_ref_space_uid(atom_array: AtomArray) -> AtomArray:
    """
    Ref: AlphaFold3 SI Chapter 2.8 Table 5
    Numerical encoding of the chain id and residue index associated with this reference conformer.
    Each (chain id, residue index) tuple is assigned an integer on first appearance.

    Args:
        atom_array (AtomArray): Biotite AtomArray object

    Returns:
        AtomArray: Biotite AtomArray object with "ref_space_uid" annotation added.
    """
    # [N_atom, 2]
    chain_res_id = np.vstack((atom_array.asym_id_int, atom_array.res_id)).T
    unique_id = np.unique(chain_res_id, axis=0)

    mapping_dict = {}
    for idx, chain_res_id_pair in enumerate(unique_id):
        asym_id_int, res_id = chain_res_id_pair
        mapping_dict[(asym_id_int, res_id)] = idx

    ref_space_uid = [
        mapping_dict[(asym_id_int, res_id)] for asym_id_int, res_id in chain_res_id
    ]
    atom_array.set_annotation("ref_space_uid", ref_space_uid)
    return atom_array


def add_cano_seq_resname(atom_array: AtomArray) -> AtomArray:
    """
    Assign to each atom the three-letter residue name (resname)
    corresponding to its place in the canonical sequences.
    Non-standard residues are mapped to standard ones.
    Residues that cannot be mapped to standard residues and ligands are all labeled as "UNK".

    Note: Some CCD Codes in the canonical sequence are mapped to three letters. It is labeled as one "UNK".

    Args:
        atom_array (AtomArray): Biotite AtomArray object

    Returns:
        AtomArray: Biotite AtomArray object with "cano_seq_resname" annotation added.
    """
    cano_seq_resname = []
    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for start, stop in zip(starts[:-1], starts[1:]):
        res_atom_nums = stop - start
        mol_type = atom_array.mol_type[start]
        resname = atom_array.res_name[start]

        one_letter_code = ccd.get_one_letter_code(resname)
        if one_letter_code is None or len(one_letter_code) != 1:
            # Some non-standard residues cannot be mapped back to one standard residue.
            one_letter_code = "X" if mol_type == "protein" else "N"

        if mol_type == "protein":
            res_name_in_cano_seq = PROT_STD_RESIDUES_ONE_TO_THREE.get(
                one_letter_code, "UNK"
            )
        elif mol_type == "dna":
            res_name_in_cano_seq = "D" + one_letter_code
            if res_name_in_cano_seq not in DNA_STD_RESIDUES:
                res_name_in_cano_seq = "DN"
        elif mol_type == "rna":
            res_name_in_cano_seq = one_letter_code
            if res_name_in_cano_seq not in RNA_STD_RESIDUES:
                res_name_in_cano_seq = "N"
        else:
            # some molecules attached to a polymer like ATP-RNA. e.g.
            res_name_in_cano_seq = "UNK"

        cano_seq_resname.extend([res_name_in_cano_seq] * res_atom_nums)

    atom_array.set_annotation("cano_seq_resname", cano_seq_resname)
    return atom_array


def find_equiv_mol_and_assign_ids(
    atom_array: AtomArray,
    entity_poly_type: Optional[dict[str, str]] = None,
    check_final_equiv: bool = True,
) -> AtomArray:
    """
    Assign a unique integer to each molecule in the structure.
    All atoms connected by covalent bonds are considered as a molecule, with unique mol_id (int).
    different copies of same molecule will assign same entity_mol_id (int).
    for each mol, assign mol_atom_index starting from 0.

    Args:
        atom_array (AtomArray): Biotite AtomArray object
        entity_poly_type (Optional[dict[str, str]]): label_entity_id to entity.poly_type.
                          Defaults to None.
        check_final_equiv (bool, optional): check if the final mol_ids of same entity_mol_id are all equivalent.

    Returns:
        AtomArray: Biotite AtomArray object with new annotations
        - mol_id: atoms with covalent bonds connected, 0-based int
        - entity_mol_id: equivalent molecules will assign same entity_mol_id, 0-based int
        - mol_residue_index: mol_atom_index for each mol, 0-based int
    """
    # Re-assign mol_id to AtomArray after break asym bonds
    if entity_poly_type is None:
        mol_indices: list[np.ndarray] = get_molecule_indices(atom_array)
    else:
        bonds_filtered = remove_bonds_between_polymer_chains(
            atom_array, entity_poly_type
        )
        mol_indices: list[np.ndarray] = get_molecule_indices(bonds_filtered)

    # assign mol_id
    mol_ids = np.array([-1] * len(atom_array), dtype=np.int32)
    for mol_id, atom_indices in enumerate(mol_indices):
        mol_ids[atom_indices] = mol_id
    atom_array.set_annotation("mol_id", mol_ids)

    assert ~np.isin(-1, atom_array.mol_id), "Some mol_id is not assigned."
    assert len(np.unique(atom_array.mol_id)) == len(
        mol_indices
    ), "Some mol_id is duplicated."

    # assign entity_mol_id
    # --------------------
    # first atom of mol with infos in attrubites, eg: info.num_atoms, info.bonds, ...
    ref_mol_infos = []
    # perm for keep multiple chains in one mol are together and in same chain order
    new_atom_perm = []
    chain_starts = struc.get_chain_starts(atom_array, add_exclusive_stop=False)
    entity_mol_ids = np.zeros_like(mol_ids)
    for mol_id, atom_indices in enumerate(mol_indices):
        atom_indices = np.sort(atom_indices)
        # keep multiple chains-mol has same chain order in different copies
        chain_perm = np.argsort(atom_array.label_entity_id[atom_indices], kind="stable")
        atom_indices = atom_indices[chain_perm]
        # save indices for finally re-ordering atom_array
        new_atom_perm.extend(atom_indices)

        # check mol equal, keep chain order consistent with atom_indices
        mol_chain_mask = np.isin(atom_indices, chain_starts)
        entity_ids = atom_array.label_entity_id[atom_indices][mol_chain_mask].tolist()

        match_entity_mol_id = None
        for entity_mol_id, mol_info in enumerate(ref_mol_infos):
            # check mol equal
            # same entity_ids and same atom name will assign same entity_mol_id
            if entity_ids != mol_info.entity_ids:
                continue

            if len(atom_indices) != len(mol_info.atom_name):
                continue

            atom_name_not_equal = (
                atom_array.atom_name[atom_indices] != mol_info.atom_name
            )
            if np.any(atom_name_not_equal):
                diff_indices = np.where(atom_name_not_equal)[0]
                query_atom = atom_array[atom_indices[diff_indices[0]]]
                ref_atom = atom_array[mol_info.atom_indices[diff_indices[0]]]
                logger.warning(
                    f"Two mols have entity_ids and same number of atoms, but diff atom name:\n{query_atom=}\n{  ref_atom=}"
                )
                continue

            # pass all checks, it is a match
            match_entity_mol_id = entity_mol_id
            break

        if match_entity_mol_id is None:  # not found match mol
            # use first atom as a placeholder for mol info.
            mol_info = atom_array[atom_indices[0]]
            mol_info.atom_indices = atom_indices
            mol_info.entity_ids = entity_ids
            mol_info.atom_name = atom_array.atom_name[atom_indices]
            mol_info.entity_mol_id = len(ref_mol_infos)
            ref_mol_infos.append(mol_info)
            match_entity_mol_id = mol_info.entity_mol_id

        entity_mol_ids[atom_indices] = match_entity_mol_id

    atom_array.set_annotation("entity_mol_id", entity_mol_ids)

    # re-order atom_array to make atoms with same mol_id together.
    atom_array = atom_array[new_atom_perm]

    # assign mol_atom_index
    mol_starts = get_starts_by(atom_array, by_annot="mol_id", add_exclusive_stop=True)
    mol_atom_index = np.zeros_like(atom_array.mol_id, dtype=np.int32)
    for start, stop in zip(mol_starts[:-1], mol_starts[1:]):
        mol_atom_index[start:stop] = np.arange(stop - start)
    atom_array.set_annotation("mol_atom_index", mol_atom_index)

    # check mol equivalence again
    if check_final_equiv:
        num_mols = len(mol_starts) - 1
        for i in range(num_mols):
            for j in range(i + 1, num_mols):
                start_i, stop_i = mol_starts[i], mol_starts[i + 1]
                start_j, stop_j = mol_starts[j], mol_starts[j + 1]
                if (
                    atom_array.entity_mol_id[start_i]
                    != atom_array.entity_mol_id[start_j]
                ):
                    continue
                for key in ["res_name", "atom_name", "mol_atom_index"]:
                    # not check res_id for ligand may have different res_id
                    annot = getattr(atom_array, key)
                    assert np.all(
                        annot[start_i:stop_i] == annot[start_j:stop_j]
                    ), f"not equal {key} when find_equiv_mol_and_assign_ids()"

    return atom_array


def add_tokatom_idx(atom_array: AtomArray) -> AtomArray:
    """
    Add a tokatom_idx corresponding to the residue and atom name for each atom.
    For non-standard residues or ligands, the tokatom_idx should be set to 0.

    Parameters:
    atom_array (AtomArray): The AtomArray object to which the annotation will be added.

    Returns:
    AtomArray: The AtomArray object with the 'tokatom_idx' annotation added.
    """
    # pre-defined atom name order for tokatom_idx
    tokatom_idx_list = []
    for atom in atom_array:
        atom_name_position = RES_ATOMS_DICT.get(atom.res_name, None)
        if atom.mol_type == "ligand" or atom_name_position is None:
            tokatom_idx = 0
        else:
            tokatom_idx = atom_name_position[atom.atom_name]
        tokatom_idx_list.append(tokatom_idx)
    atom_array.set_annotation("tokatom_idx", tokatom_idx_list)
    return atom_array


def add_mol_id(atom_array: AtomArray) -> AtomArray:
    """
    Assign a unique integer to each molecule in the structure.

    Args:
        atom_array (AtomArray): Biotite AtomArray object
    Returns:
        AtomArray: Biotite AtomArray object with new annotations
        - mol_id: atoms with covalent bonds connected, 0-based int
    """
    mol_indices = get_molecule_indices(atom_array)

    # assign mol_id
    mol_ids = np.array([-1] * len(atom_array), dtype=np.int32)
    for mol_id, atom_indices in enumerate(mol_indices):
        mol_ids[atom_indices] = mol_id
    atom_array.set_annotation("mol_id", mol_ids)
    return atom_array


def unique_chain_and_add_ids(atom_array: AtomArray) -> AtomArray:
    """
    Unique chain ID and add asym_id, entity_id, sym_id.
    Adds a number to the chain ID to make chain IDs in the assembly unique.
    Example: [A, B, A, B, C] -> [A, B, A.1, B.1, C]

    Args:
        atom_array (AtomArray): Biotite AtomArray object.

    Returns:
        AtomArray: Biotite AtomArray object with new annotations:
            - asym_id_int: np.array(int)
            - entity_id_int: np.array(int)
            - sym_id_int: np.array(int)
    """
    chain_ids = np.zeros(len(atom_array), dtype="<U8")
    chain_starts = get_chain_starts(atom_array, add_exclusive_stop=True)

    chain_counter = Counter()
    for start, stop in zip(chain_starts[:-1], chain_starts[1:]):
        ori_chain_id = atom_array.chain_id[start]
        cnt = chain_counter[ori_chain_id]
        if cnt == 0:
            new_chain_id = ori_chain_id
        else:
            new_chain_id = f"{ori_chain_id}.{chain_counter[ori_chain_id]}"

        chain_ids[start:stop] = new_chain_id
        chain_counter[ori_chain_id] += 1

    assert "" not in chain_ids
    # reset chain id
    atom_array.del_annotation("chain_id")
    atom_array.set_annotation("chain_id", chain_ids)

    entity_id_uniq = np.sort(np.unique(atom_array.label_entity_id))
    entity_id_dict = {e: i for i, e in enumerate(entity_id_uniq)}
    asym_ids = np.zeros(len(atom_array), dtype=int)
    entity_ids = np.zeros(len(atom_array), dtype=int)
    sym_ids = np.zeros(len(atom_array), dtype=int)
    counter = Counter()
    start_indices = struc.get_chain_starts(atom_array, add_exclusive_stop=True)
    for i in range(len(start_indices) - 1):
        start_i = start_indices[i]
        stop_i = start_indices[i + 1]
        asym_ids[start_i:stop_i] = i

        entity_id = atom_array.label_entity_id[start_i]
        entity_ids[start_i:stop_i] = entity_id_dict[entity_id]

        sym_ids[start_i:stop_i] = counter[entity_id]
        counter[entity_id] += 1

    atom_array.set_annotation("asym_id_int", asym_ids)
    atom_array.set_annotation("entity_id_int", entity_ids)
    atom_array.set_annotation("sym_id_int", sym_ids)
    return atom_array
