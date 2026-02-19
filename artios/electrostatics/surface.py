"""Molecular surface point generation for electrostatic complementarity.

Generates Shrake-Rupley style surface points on the protein's solvent-accessible
surface (SAS) at the protein-ligand interface. These points replace raw atom
coordinates as sampling locations for EC calculation, giving spatially uniform
coverage of the binding interface.
"""

from __future__ import annotations

import numpy as np
from biotite.structure import AtomArray
from scipy.spatial import cKDTree

from artios.electrostatics.elements import get_parse_radius


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n approximately evenly-spaced points on a unit sphere.

    Uses the Fibonacci lattice (golden spiral) method, which gives near-optimal
    point distributions without iterative optimization.

    Args:
        n: Number of points to generate.

    Returns:
        Unit sphere points, shape (n, 3).
    """
    golden_ratio = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    theta = 2 * np.pi * i / golden_ratio
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    return np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
        axis=-1,
    )


def interface_surface_points(
    protein: AtomArray,
    ligand: AtomArray,
    probe_radius: float = 1.4,
    n_points_per_atom: int = 100,
    contact_cutoff: float = 1.4,
) -> np.ndarray:
    """Generate solvent-accessible surface points on the protein at the ligand contact.

    Uses the Shrake-Rupley approach: for each protein atom near the ligand,
    distribute test points on its SAS sphere (VdW + probe radius), then keep
    only those that are solvent-exposed and in contact with the ligand's VdW
    surface.

    A surface point is "in contact" if its distance to the nearest ligand
    VdW surface is <= contact_cutoff (default: one probe radius). This gives
    just the contacting shell, not a generous 5A envelope.

    Args:
        protein: Protein AtomArray.
        ligand: Ligand AtomArray.
        probe_radius: Water probe radius in Angstroms.
        n_points_per_atom: Number of Fibonacci sphere points per atom.
        contact_cutoff: Max distance from the ligand's VdW surface for a
            surface point to be kept, in Angstroms.

    Returns:
        Interface surface points, shape (M, 3). Empty (0, 3) if none found.
    """
    # ... VdW radii for all protein and ligand atoms
    protein_radii = np.array([get_parse_radius(e) for e in protein.element])
    ligand_radii = np.array([get_parse_radius(e) for e in ligand.element])
    sas_radii = protein_radii + probe_radius

    # ... pre-filter: only protein atoms close enough to contribute contact points
    # Surface point distance to ligand VdW surface ≈ D - sas_radius - ligand_vdw.
    # For it to be <= contact_cutoff: D <= sas_radius + max_ligand_vdw + contact_cutoff
    max_sas = sas_radii.max()
    max_lig_vdw = ligand_radii.max()
    ligand_tree = cKDTree(ligand.coord)
    dists_to_ligand, _ = ligand_tree.query(protein.coord)
    nearby_mask = dists_to_ligand <= max_sas + max_lig_vdw + contact_cutoff
    nearby_indices = np.where(nearby_mask)[0]

    if len(nearby_indices) == 0:
        return np.empty((0, 3))

    nearby_coords = protein.coord[nearby_indices]
    nearby_sas_radii = sas_radii[nearby_indices]

    # ... build KDTree of ALL protein atoms for burial checks
    protein_tree = cKDTree(protein.coord)

    # ... unit sphere template
    unit_sphere = _fibonacci_sphere(n_points_per_atom)  # (n_points, 3)

    all_points = []
    for idx_local, idx_global in enumerate(nearby_indices):
        center = nearby_coords[idx_local]
        r = nearby_sas_radii[idx_local]
        points = center + r * unit_sphere  # (n_points, 3)

        # ... check burial: remove points inside any other protein atom's SAS sphere
        search_radius = r + max_sas
        neighbor_lists = protein_tree.query_ball_point(center, search_radius)

        exposed = np.ones(len(points), dtype=bool)
        for neighbor_idx in neighbor_lists:
            if neighbor_idx == idx_global:
                continue
            neighbor_center = protein.coord[neighbor_idx]
            neighbor_sas = sas_radii[neighbor_idx]
            dist_sq = np.sum((points - neighbor_center) ** 2, axis=1)
            exposed &= dist_sq > neighbor_sas**2

        points = points[exposed]

        if len(points) == 0:
            continue

        # ... keep only points in contact with ligand VdW surface
        # Distance to ligand surface = distance to ligand atom - ligand VdW radius
        dists_to_lig_atoms, nearest_lig_idx = ligand_tree.query(points)
        dist_to_lig_surface = dists_to_lig_atoms - ligand_radii[nearest_lig_idx]
        contact_mask = dist_to_lig_surface <= contact_cutoff
        points = points[contact_mask]

        if len(points) > 0:
            all_points.append(points)

    if not all_points:
        return np.empty((0, 3))

    return np.concatenate(all_points)
