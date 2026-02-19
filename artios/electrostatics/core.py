"""Poisson-Boltzmann electrostatic complementarity (EC) pipeline.

Computes how well a protein's electrostatic potential complements the
ligand's charge distribution at their interface. A negative EC score
indicates good complementarity — the physical basis for molecular
recognition and enzyme catalysis.

Pipeline:
1. Generate molecular surface points at the binding interface
2. Generate PQR files (charge + radius assignment)
3. Run APBS twice (protein-only, ligand-only) on a focused grid
4. Sample both potential maps at interface surface points
5. Pearson correlation → EC score
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

from artios.electrostatics.apbs import (
    ECConfig,
    focused_grid_dims,
    generate_apbs_input,
    read_dx_potential,
    run_apbs,
    sample_potential_at_points,
)
from artios.electrostatics.elements import get_parse_radius
from artios.electrostatics.pqr import (
    generate_ligand_pqr,
    generate_protein_pqr,
    write_pqr,
)
from artios.electrostatics.surface import interface_surface_points

INTERFACE_CUTOFF: float = 5.0  # Angstroms


@dataclass
class ECResult:
    """Result of an electrostatic complementarity calculation."""

    ec_score: float  # Pearson correlation (negative = complementary)
    n_interface_points: int
    protein_potentials: np.ndarray  # phi_protein at interface points
    ligand_potentials: np.ndarray  # phi_ligand at interface points
    surface_points: np.ndarray  # (M, 3) interface surface point coordinates


def split_complex(
    structure: AtomArray,
    ligand_chain_id: str,
) -> tuple[AtomArray, AtomArray]:
    """Split a complex into protein and ligand by chain ID.

    Args:
        structure: Full complex AtomArray.
        ligand_chain_id: Chain ID of the ligand.

    Returns:
        (protein, ligand) AtomArrays.
    """
    ligand_mask = structure.chain_id == ligand_chain_id
    assert ligand_mask.any(), (
        f"No atoms found with chain_id='{ligand_chain_id}'. "
        f"Available chains: {np.unique(structure.chain_id)}"
    )
    return structure[~ligand_mask], structure[ligand_mask]


def interface_atoms(
    protein: AtomArray,
    ligand: AtomArray,
    cutoff: float = INTERFACE_CUTOFF,
) -> np.ndarray:
    """Find protein atom indices within cutoff of any ligand atom.

    Args:
        protein: Protein AtomArray.
        ligand: Ligand AtomArray.
        cutoff: Distance cutoff in Angstroms.

    Returns:
        Boolean mask over protein atoms (True = interface).
    """
    tree = cKDTree(ligand.coord)
    distances, _ = tree.query(protein.coord)
    return distances <= cutoff


def _run_apbs_for_pqr(
    pqr_path: Path,
    prefix: str,
    center: np.ndarray,
    dime: np.ndarray,
    fine_len: np.ndarray,
    coarse_len: np.ndarray,
    config: ECConfig,
    work_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run APBS for a single PQR file and return the potential grid in memory.

    Returns:
        (potential, origin, delta) numpy arrays.
    """
    input_content = generate_apbs_input(
        pqr_path,
        prefix,
        center,
        dime,
        fine_len,
        coarse_len,
        config,
    )
    input_path = work_dir / f"{prefix}.in"
    input_path.write_text(input_content)

    dx_path = run_apbs(input_path, work_dir, dx_prefix=prefix)
    return read_dx_potential(dx_path)


def compute_ec(
    protein: AtomArray,
    ligand: AtomArray,
    *,
    ligand_charges: np.ndarray | None = None,
    config: ECConfig | None = None,
    interface_cutoff: float = INTERFACE_CUTOFF,
) -> ECResult:
    """Compute electrostatic complementarity between a protein and ligand.

    Args:
        protein: Protein AtomArray.
        ligand: Ligand AtomArray.
        ligand_charges: Custom partial charges for ligand atoms. If None,
            uses MMFF94 charges from CCD component lookup.
        config: EC configuration. Uses defaults if None.
        interface_cutoff: Distance cutoff for grid sizing (Angstroms).

    Returns:
        ECResult with EC score, potentials, and surface point coordinates.
    """
    if config is None:
        config = ECConfig()

    # ... generate surface points at the protein-ligand contact
    points = interface_surface_points(
        protein,
        ligand,
        probe_radius=config.probe_radius,
    )
    n_points = len(points)
    assert n_points >= 3, f"Only {n_points} interface surface points found (need >= 3)."

    # ... compute focused grid: fine grid on binding site, coarse covers full protein
    iface_mask = interface_atoms(protein, ligand, interface_cutoff)
    binding_site_coords = np.concatenate(
        [
            ligand.coord,
            protein.coord[iface_mask],
        ]
    )
    all_coords = np.concatenate([protein.coord, ligand.coord])
    center, dime, fine_len, coarse_len = focused_grid_dims(
        all_coords,
        binding_site_coords,
        config,
    )

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)

        # ... protein PQR (always PDB2PQR + PARSE)
        protein_pqr = generate_protein_pqr(protein, work_dir / "protein.pqr")

        # ... ligand PQR: custom charges or MMFF94 from CCD
        if ligand_charges is not None:
            assert len(ligand_charges) == len(ligand), (
                f"ligand_charges length ({len(ligand_charges)}) != "
                f"ligand atoms ({len(ligand)})"
            )
            ligand_radii = np.array([get_parse_radius(e) for e in ligand.element])
            ligand_pqr = work_dir / "ligand.pqr"
            write_pqr(ligand, ligand_charges, ligand_radii, ligand_pqr)
        else:
            ligand_pqr = generate_ligand_pqr(ligand, work_dir / "ligand.pqr")

        # ... solve PB for protein alone
        prot_potential, prot_origin, prot_delta = _run_apbs_for_pqr(
            protein_pqr,
            "protein",
            center,
            dime,
            fine_len,
            coarse_len,
            config,
            work_dir,
        )

        # ... solve PB for ligand alone
        lig_potential, lig_origin, lig_delta = _run_apbs_for_pqr(
            ligand_pqr,
            "ligand",
            center,
            dime,
            fine_len,
            coarse_len,
            config,
            work_dir,
        )

    # ... sample both potentials at interface surface points
    phi_protein = sample_potential_at_points(
        prot_potential,
        prot_origin,
        prot_delta,
        points,
    )
    phi_ligand = sample_potential_at_points(
        lig_potential,
        lig_origin,
        lig_delta,
        points,
    )

    # ... Pearson correlation: negative = complementary
    ec_score, _ = pearsonr(phi_protein, phi_ligand)

    return ECResult(
        ec_score=float(ec_score),
        n_interface_points=n_points,
        protein_potentials=phi_protein,
        ligand_potentials=phi_ligand,
        surface_points=points,
    )
