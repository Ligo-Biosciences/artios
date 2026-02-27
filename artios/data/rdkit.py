"""RDKit utilities."""

import copy
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional

import biotite.structure as struc
import numpy as np
from atomworks.io.tools.rdkit import (
    CONVERTIBLE_RDKIT_BOND_TYPES,
    RDKIT_BOND_TYPE_TO_BIOTITE,
    RDKIT_HYBRIDIZATION_TO_INT,
    not_isin,
)
from atomworks.io.tools.rdkit import (
    smiles_to_rdkit as smiles_to_rdkit_mol,
)
from biotite.structure import AtomArray
from rdkit import Chem
from rdkit.Chem import AdjustQueryParameters, AdjustQueryProperties

logger = logging.getLogger(__name__)

UNKNOWN_LIGAND = "UNL"
HYDROGEN_LIKE_SYMBOLS = ["H", "D", "T"]


@dataclass(frozen=True)
class AtomInfo:
    atom_array: AtomArray
    atom_map_to_atom_name: Dict[int, str] = field(default_factory=dict)


def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid.

    Args:
        smiles: SMILES string to validate

    Returns:
        True if valid, False otherwise
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def parse_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Parse a SMILES string to RDKit Mol.

    Args:
        smiles: SMILES string to parse

    Returns:
        RDKit Mol if valid, None otherwise
    """
    return Chem.MolFromSmiles(smiles)


def atom_array_from_rdkit(
    mol: Chem.Mol,
    *,
    set_coord_if_available: bool = True,
    conformer_id: int | None = None,
    remove_hydrogens: bool = True,
    remove_inferred_atoms: bool = False,
) -> AtomArray:
    """Convert an RDKit molecule to a Biotite AtomArray object.

    This is a local copy of atomworks.io.tools.rdkit.atom_array_from_rdkit
    with ONE CHANGE: atom naming uses 1-based indexing (C1, C2, O1) instead
    of 0-based (C0, C1, O0) to match our atom_map_to_atom_name convention.

    Args:
        mol: The RDKit molecule to convert.
        set_coord_if_available: Whether to set the coordinates from the RDKit
            molecule if a conformer is available.
        conformer_id: The conformer ID to use for coordinates. If None, the first
          conformer is used.
        remove_hydrogens: Whether to remove any explicit hydrogen atoms.
        remove_inferred_atoms: Whether to remove any atoms that do not carry the
            `rdkit_atom_id` annotation.

    Returns:
        An AtomArray containing the atoms and bonds from the input Mol object.
    """
    mol = copy.deepcopy(mol)

    n_atoms = mol.GetNumAtoms()

    # Get coordinates (if available)
    coords = np.full((mol.GetNumAtoms(), 3), np.nan)
    n_conformers = mol.GetNumConformers()
    if set_coord_if_available and n_conformers > 0:
        conformer_id = conformer_id if conformer_id is not None else -1
        if conformer_id >= n_conformers:
            raise ValueError(
                f"Conformer ID {conformer_id} out of range for molecule "
                f"with {n_conformers} conformers"
            )
        coords = mol.GetConformer(conformer_id).GetPositions()

    # Set atoms
    atoms = []
    element_counter = Counter()
    for idx, rdatom in enumerate(mol.GetAtoms()):
        # CHANGE: Use 1-based indexing for atom names (C1, C2, O1)
        # Original atomworks uses 0-based (C0, C1, O0)
        element_counter[rdatom.GetAtomicNum()] += 1
        element_occurence = element_counter[rdatom.GetAtomicNum()]

        atoms.append(
            struc.Atom(
                element=rdatom.GetSymbol(),
                atomic_number=rdatom.GetAtomicNum(),
                coord=coords[idx],
                charge=rdatom.GetFormalCharge(),
                hyb=RDKIT_HYBRIDIZATION_TO_INT[rdatom.GetHybridization()],
                nhyd=rdatom.GetTotalNumHs(),
                hvydeg=rdatom.GetDegree() - rdatom.GetTotalNumHs(),
                rdkit_atom_id=(
                    rdatom.GetIntProp("rdkit_atom_id")
                    if rdatom.HasProp("rdkit_atom_id")
                    else -1
                ),
                hetero=True,
                atom_name=f"{rdatom.GetSymbol().upper()}{element_occurence}",
                res_name=UNKNOWN_LIGAND,
                chiral_tag=int(rdatom.GetChiralTag()),
                is_aromatic=rdatom.GetIsAromatic(),
            )
        )
    atom_array = struc.array(atoms)

    # Set bonds
    Chem.Kekulize(mol)
    bond_list = []
    for bond in mol.GetBonds():
        rdkit_bond_type = bond.GetBondType()

        if rdkit_bond_type not in CONVERTIBLE_RDKIT_BOND_TYPES:
            logger.warning(
                f"Skipping {rdkit_bond_type=}. Not in convertible bond types."
            )
            continue

        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        is_bond_aromatic = bond.GetIsAromatic()
        bond_type = RDKIT_BOND_TYPE_TO_BIOTITE[(bond.GetBondType(), is_bond_aromatic)]
        bond_list.append((begin_atom_idx, end_atom_idx, bond_type))
    atom_array.bonds = struc.bonds.BondList(n_atoms, np.array(bond_list))

    # Set extra annotations
    annotations = mol._annotations if hasattr(mol, "_annotations") else {}
    if len(annotations) > 0:
        _rdkit_id_to_annotation_idx = {
            rdkit_atom_id: idx
            for idx, rdkit_atom_id in enumerate(annotations["rdkit_atom_id"])
        }

        array_idx_to_annotation_idx = []
        for idx, atom in enumerate(atoms):
            if atom.rdkit_atom_id in _rdkit_id_to_annotation_idx:
                array_idx_to_annotation_idx.append(
                    (idx, _rdkit_id_to_annotation_idx[atom.rdkit_atom_id])
                )
        array_idx_to_annotation_idx = np.array(array_idx_to_annotation_idx)

        if len(array_idx_to_annotation_idx) == 0:
            logger.warning(
                "No rdkit atoms match any annotation. You may want to check that you "
                "are using the @preserve_annotations decorator correctly. And set the "
                "rdkit pickle options to preserve properties. Returning."
            )
            return atom_array

        for key, val in annotations.items():
            if key in ["coord", "charge"]:
                logger.warning(
                    f"Found built-in annotation: {key} as custom annotation. "
                    "Using re-calculated values from RDKit instead."
                )
                continue

            if np.issubdtype(val.dtype, np.integer):
                if np.issubdtype(val.dtype, np.unsignedinteger):
                    default = np.iinfo(val.dtype).max
                else:
                    default = -1
            elif np.issubdtype(val.dtype, np.floating):
                default = np.nan
            elif np.issubdtype(val.dtype, np.str_ or str):
                default = ""
            elif np.issubdtype(val.dtype, bool):
                default = False
            else:
                logger.warning(
                    f"Unsupported annotation dtype: {val.dtype} for annotation: "
                    f"{key}. Skipping."
                )
                continue

            vals = np.full(len(atom_array), default, dtype=val.dtype)
            vals[array_idx_to_annotation_idx[:, 0]] = annotations[key][
                array_idx_to_annotation_idx[:, 1]
            ]
            atom_array.set_annotation(key, vals)

    if remove_hydrogens:
        atom_array = atom_array[
            not_isin(atom_array.element, np.array(HYDROGEN_LIKE_SYMBOLS))
        ]

    if remove_inferred_atoms:
        atom_array = atom_array[atom_array.rdkit_atom_id != -1]

    return atom_array


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

    # ... create atom array from RDKit mol (using our 1-based naming version)
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


def atoms_to_rdkit_mol(
    elements: tuple[str, ...],
    coords: np.ndarray,
) -> Chem.Mol:
    """Create RDKit mol from elements and coordinates, inferring bonds.

    Only infers connectivity (which atoms are bonded), not bond orders.
    This is sufficient for substructure matching and avoids failures on
    molecular fragments that can't satisfy valence rules.

    Args:
        elements: Tuple of element symbols (e.g., ("C", "N", "O"))
        coords: (n_atoms, 3) array of coordinates

    Returns:
        RDKit Mol with inferred connectivity (all single bonds)
    """
    from rdkit.Chem import rdDetermineBonds

    # Create editable mol and add atoms
    mol = Chem.RWMol()
    for element in elements:
        atom = Chem.Atom(element)
        mol.AddAtom(atom)

    # Add conformer with coordinates
    conf = Chem.Conformer(len(elements))
    for i, coord in enumerate(coords):
        conf.SetAtomPosition(i, coord.tolist())
    mol.AddConformer(conf)

    # Infer connectivity only (not bond orders) - sufficient for substructure matching
    # Bond order inference fails on molecular fragments that can't satisfy valence rules
    rdDetermineBonds.DetermineConnectivity(mol)

    return mol.GetMol()


def infer_atom_map_from_substructure(
    motif_elements: tuple[str, ...],
    motif_coords: np.ndarray,
    motif_atom_names: tuple[str, ...],
    full_smiles: str,
) -> dict[str, str]:
    """Infer atom name mapping by RDKit's substructure matching of the
    motif against the full ligand.
    Uses RDKit substructure matching to find which atoms in the full ligand
    correspond to the motif atoms, then returns a direct mapping from
    CIF atom names to the generated atom names in the full ligand.

    Args:
        motif_elements: Element symbols from motif CIF
        motif_coords: Coordinates from motif CIF
        motif_atom_names: Atom names from motif CIF (e.g., "C1", "N3")
        full_smiles: SMILES string of full ligand

    Returns:
        Dict mapping CIF atom names to full ligand atom names (e.g., {"C1": "C3", "N3": "N1"})

    Raises:
        ValueError: If substructure match fails
    """
    motif_mol = atoms_to_rdkit_mol(motif_elements, motif_coords)

    # Convert to query mol with flexible bond matching (ignore bond orders)
    # This is necessary because atoms_to_rdkit_mol only infers connectivity (single bonds)
    # while the full mol from SMILES has proper bond orders (aromatic/double)
    params = AdjustQueryParameters()
    params.makeBondsGeneric = True  # Match any bond type
    query_mol = AdjustQueryProperties(motif_mol, params)

    # Build RDKit mol from SMILES
    full_mol = Chem.MolFromSmiles(full_smiles)
    if full_mol is None:
        raise ValueError(f"Failed to parse SMILES: {full_smiles}")

    # Substructure match: returns tuple of full_mol atom indices
    # that correspond to motif_mol atoms 0, 1, 2, ...
    match = full_mol.GetSubstructMatch(query_mol)

    if not match:
        raise ValueError(
            f"Substructure match failed. Motif ({len(motif_elements)} atoms) "
            f"not found in full ligand SMILES. This can happen with symmetric "
            f"molecules or if the motif contains atoms not in the full ligand."
        )

    # Build index -> atom_name mapping for full mol
    element_count: Counter = Counter()
    idx_to_atom_name = {}
    for atom in full_mol.GetAtoms():
        element = atom.GetSymbol().upper()
        element_count[element] += 1
        idx_to_atom_name[atom.GetIdx()] = f"{element}{element_count[element]}"

    # Build direct mapping: CIF atom name -> full ligand atom name
    cif_to_full_atom_name = {}
    for motif_idx, full_idx in enumerate(match):
        cif_atom_name = motif_atom_names[motif_idx]
        cif_to_full_atom_name[cif_atom_name] = idx_to_atom_name[full_idx]

    return cif_to_full_atom_name


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


__all__ = ["smiles_to_rdkit_mol", "atom_array_from_rdkit"]
