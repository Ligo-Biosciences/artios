"""RDKit utilities."""

from collections import Counter
from dataclasses import dataclass
from typing import Dict

from atomworks.io.tools.rdkit import (
    atom_array_from_rdkit,
)
from atomworks.io.tools.rdkit import (
    smiles_to_rdkit as smiles_to_rdkit_mol,
)
from biotite.structure import AtomArray
from rdkit import Chem


@dataclass(frozen=True)
class AtomInfo:
    atom_array: AtomArray
    atom_map_to_atom_name: Dict[int, str]  # TODO: pydantic?


def rdkit_mol_to_atom_info(mol: Chem.Mol, remove_hydrogens: bool = True) -> AtomInfo:
    """
    Convert RDKit Mol to AtomInfo.

    Args:
        mol (Chem.Mol): rdkit mol

    Returns:
        AtomInfo object
    """

    atom_map_to_atom_name = {}
    atom_idx_to_atom_name = {}

    # ... create atom map (SMILES atom map notation) to atom name
    # Count elements to generate unique atom names (e.g., C1, C2, C3)
    element_count = Counter()

    for atom in mol.GetAtoms():
        # Generate atom name from element symbol and count
        element = atom.GetSymbol().upper()
        element_count[element] += 1
        atom_name = f"{element}{element_count[element]}"

        # Store atom name as RDKit property
        atom.SetProp("name", atom_name)

        # Map atom map number to atom name (for SMILES with atom mapping)
        if atom.GetAtomMapNum() != 0:
            atom_map_to_atom_name[atom.GetAtomMapNum()] = atom_name
        atom_idx_to_atom_name[atom.GetIdx()] = atom_name

    # ... create atom array from RDKit mol
    atom_array = atom_array_from_rdkit(mol, remove_hydrogens=remove_hydrogens)
    return AtomInfo(atom_array=atom_array, atom_map_to_atom_name=atom_map_to_atom_name)


def smiles_to_atom_info(smiles: str, sanitize: bool = True) -> AtomInfo:
    """
    Convert smiles to AtomInfo.

    Args:
        smiles (str): smiles string, like "CCCC", or "[C:1]NC(=O)" (use num to label covalent bond atom.)
        sanitize (bool): whether to sanitize the molecule.

    Returns:
        AtomInfo object
    """
    mol = smiles_to_rdkit_mol(
        smiles=smiles, generate_conformers=True, sanitize=sanitize
    )
    return rdkit_mol_to_atom_info(mol)


def lig_file_to_atom_info(lig_file_path: str) -> AtomInfo:
    """
    Convert ligand file to biotite AtomArray.

    Args:
        lig_file_path (str): ligand file path with one of the following suffixes: [mol, mol2, sdf, pdb]

    Returns:
        AtomInfo
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


__all__ = ["smiles_to_rdkit_mol"]
