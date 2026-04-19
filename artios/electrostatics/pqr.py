"""PQR file generation for Poisson-Boltzmann calculations.

Converts biotite AtomArrays into PQR files (PDB + charge + radius per atom).

Two strategies:
- Protein: PDB2PQR with PARSE force field for partial charges and radii.
- Ligand (CCD): MMFF94 partial charges via RDKit + PARSE radii by element.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray

from artios.common.elements import get_parse_radius

logger = logging.getLogger(__name__)


def write_pqr(
    structure: AtomArray,
    charges: np.ndarray,
    radii: np.ndarray,
    output_path: Path,
) -> None:
    """Write a PQR file from an AtomArray with charges and radii.

    Args:
        structure: AtomArray with coordinates.
        charges: Partial charges per atom, shape (n_atoms,).
        radii: VdW radii per atom, shape (n_atoms,).
        output_path: Path to write PQR file.
    """
    lines = []
    for i in range(len(structure)):
        # PQR format: 1-char chain ID; truncate if longer (e.g. mmCIF label_asym_id)
        chain = structure.chain_id[i][:1] if structure.chain_id[i] else " "
        line = (
            f"ATOM  {i + 1:5d} {structure.atom_name[i]:<4s} "
            f"{structure.res_name[i]:<3s} "
            f"{chain:1s}{structure.res_id[i]:4d}    "
            f"{structure.coord[i, 0]:8.3f}"
            f"{structure.coord[i, 1]:8.3f}"
            f"{structure.coord[i, 2]:8.3f} "
            f"{charges[i]:7.4f} {radii[i]:6.4f}"
        )
        lines.append(line)

    lines.append("END")
    output_path.write_text("\n".join(lines) + "\n")


def generate_protein_pqr(
    protein: AtomArray,
    output_path: Path,
) -> Path:
    """Generate PQR for protein using PDB2PQR with PARSE force field.

    Args:
        protein: Protein-only AtomArray.
        output_path: Where to write the PQR file.

    Returns:
        Path to the generated PQR file.
    """
    import biotite.structure.io.pdb as pdb_io

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp_pdb = Path(tmp.name)

    try:
        # ... PDB format requires single-char chain IDs; remap if needed
        protein = protein.copy()
        unique_chains = np.unique(protein.chain_id)
        if any(len(c) > 1 for c in unique_chains):
            chain_map = {c: chr(65 + i) for i, c in enumerate(unique_chains)}
            protein.chain_id = np.array([chain_map[c] for c in protein.chain_id])

        pdb_file = pdb_io.PDBFile()
        pdb_io.set_structure(pdb_file, protein)
        pdb_file.write(tmp_pdb)

        pdb2pqr_bin = shutil.which("pdb2pqr30") or shutil.which("pdb2pqr")
        assert pdb2pqr_bin is not None, (
            "pdb2pqr30/pdb2pqr not found on PATH. " "Install with: pip install pdb2pqr"
        )

        subprocess.run(
            [
                pdb2pqr_bin,
                "--ff=PARSE",
                "--noopt",
                "--nodebump",
                "--keep-chain",
                str(tmp_pdb),
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        return output_path

    finally:
        tmp_pdb.unlink(missing_ok=True)
        output_path.with_suffix(".log").unlink(missing_ok=True)


def compute_mmff94_charges(ligand: AtomArray) -> np.ndarray:
    """Compute MMFF94 partial charges for a ligand AtomArray.

    For each CCD residue: load the RDKit mol from cache, add hydrogens
    for correct MMFF94 atom typing, extract partial charges, and map
    back to AtomArray heavy-atom ordering via mol.atom_map.

    Args:
        ligand: Ligand AtomArray (may contain multiple CCD residues).

    Returns:
        Array of partial charges, shape (n_atoms,).
    """
    from rdkit import Chem
    from rdkit.Chem import rdForceFieldHelpers

    from artios.data.chem.ccd import get_component_rdkit_mol

    charges = np.zeros(len(ligand), dtype=np.float64)

    for res_name in np.unique(ligand.res_name):
        # ... load the pre-computed RDKit mol for this CCD code
        mol = get_component_rdkit_mol(res_name)
        assert mol is not None, (
            f"No RDKit mol for CCD code '{res_name}'. "
            "Ensure the CCD components cache is available."
        )

        # ... add hydrogens (needed for correct MMFF94 atom typing)
        # Heavy atom indices are preserved — H atoms are appended at the end.
        mol_h = Chem.AddHs(mol)

        # ... compute MMFF94 properties (includes partial charges per atom)
        mmff_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol_h)
        assert (
            mmff_props is not None
        ), f"MMFF94 parameterization failed for '{res_name}'."

        # ... extract charges for heavy atoms using mol.atom_map
        # mol.atom_map: Dict[atom_name, heavy_atom_idx_in_original_mol]
        # These indices are valid in mol_h since AddHs preserves heavy atom order.
        atom_map: dict[str, int] = mol.atom_map
        rdkit_charges = {
            name: mmff_props.GetMMFFPartialCharge(idx) for name, idx in atom_map.items()
        }

        # ... assign charges to the corresponding AtomArray atoms
        res_mask = ligand.res_name == res_name
        for i in np.where(res_mask)[0]:
            atom_name = ligand.atom_name[i]
            assert atom_name in rdkit_charges, (
                f"Atom '{atom_name}' in '{res_name}' not in mol.atom_map. "
                f"Available: {sorted(rdkit_charges.keys())}"
            )
            charges[i] = rdkit_charges[atom_name]

    return charges


def generate_ligand_pqr(
    ligand: AtomArray,
    output_path: Path,
) -> Path:
    """Generate PQR for a CCD ligand using MMFF94 charges and PARSE radii.

    Args:
        ligand: Ligand-only AtomArray.
        output_path: Where to write the PQR file.

    Returns:
        Path to the generated PQR file.
    """
    charges = compute_mmff94_charges(ligand)
    radii = np.array([get_parse_radius(e) for e in ligand.element])
    write_pqr(ligand, charges, radii, output_path)
    return output_path
