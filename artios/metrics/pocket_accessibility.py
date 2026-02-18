"""Pocket solvent accessibility metric for protein-ligand complexes.

Determines whether a ligand-binding pocket is connected to bulk solvent
(accessible pocket / real active site) or is an enclosed cavity (folding
artifact). Works by voxelizing the protein around the ligand, computing a
distance transform, and checking connectivity at various probe radii.

The bottleneck radius reports the largest sphere that can pass from the
pocket to the protein exterior — a proxy for how wide the pocket entrance is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from biotite.structure import AtomArray
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label
from scipy.spatial import KDTree

from artios.metrics.shape_complementarity import _get_vdw_radii

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PocketAccessibilityResult:
    """Result of a pocket solvent accessibility calculation.

    Attributes:
        pocket_is_accessible: Whether the pocket is connected to bulk
            solvent at water-probe radius (1.4 A).
        bottleneck_radius: Largest probe radius (A) that maintains
            connectivity from pocket to exterior. 0.0 if fully enclosed.
    """

    pocket_is_accessible: bool
    bottleneck_radius: float


def _rasterize_atoms(
    coords: NDArray[np.floating],
    radii: NDArray[np.floating],
    grid_origin: NDArray[np.floating],
    grid_shape: tuple[int, int, int],
    voxel_size: float,
) -> NDArray[np.bool_]:
    """Rasterize atoms as filled spheres into a boolean voxel grid.

    For each atom, marks all voxels whose centers fall within the atom's
    van der Waals radius as True.

    Args:
        coords: Atom positions, shape ``(N, 3)``.
        radii: Per-atom vdW radii, shape ``(N,)``.
        grid_origin: World-space coordinate of voxel (0,0,0).
        grid_shape: Grid dimensions ``(nx, ny, nz)``.
        voxel_size: Edge length of each cubic voxel in Angstroms.

    Returns:
        Boolean grid of shape ``grid_shape``.
    """
    grid = np.zeros(grid_shape, dtype=np.bool_)
    nx, ny, nz = grid_shape

    for i in range(len(coords)):
        # Atom center in grid (fractional) coordinates
        gc = (coords[i] - grid_origin) / voxel_size

        r_vox = radii[i] / voxel_size
        half = int(np.ceil(r_vox))

        # Bounding box in grid indices
        lo = np.maximum(np.floor(gc).astype(int) - half, 0)
        hi = np.minimum(np.floor(gc).astype(int) + half + 1, [nx, ny, nz])

        # Build local index arrays
        ix = np.arange(lo[0], hi[0])
        iy = np.arange(lo[1], hi[1])
        iz = np.arange(lo[2], hi[2])
        if len(ix) == 0 or len(iy) == 0 or len(iz) == 0:
            continue

        gx, gy, gz = np.meshgrid(ix, iy, iz, indexing="ij")
        # Distance from each voxel center to atom center (in voxel units)
        dist_sq = (gx - gc[0]) ** 2 + (gy - gc[1]) ** 2 + (gz - gc[2]) ** 2
        mask = dist_sq <= r_vox**2
        grid[gx[mask], gy[mask], gz[mask]] = True

    return grid


def _pocket_connects_to_exterior(
    dist_map: NDArray[np.floating],
    min_radius: float,
    seed_indices: list[tuple[int, int, int]],
    struct: NDArray,
) -> bool:
    """Check if the pocket is connected to the grid exterior at a given probe radius.

    Args:
        dist_map: Distance transform (in Angstroms) from each free voxel
            to the nearest protein surface.
        min_radius: Minimum clearance radius required for passage.
        seed_indices: Grid indices ``(i, j, k)`` of pocket seed voxels.
            Connectivity is checked for each seed; returns True if ANY
            seed connects to exterior.
        struct: Connectivity structure for ``scipy.ndimage.label``.

    Returns:
        True if any pocket seed is in the same connected component as
        any voxel on a grid boundary face.
    """
    passable = dist_map >= min_radius
    labeled, n_components = label(passable, structure=struct)

    if n_components == 0:
        return False

    # Collect labels touching any of the 6 boundary faces
    boundary_labels: set[int] = set()
    for axis in range(3):
        for side in [0, -1]:
            slicer = [slice(None)] * 3
            slicer[axis] = side
            face = labeled[tuple(slicer)]
            boundary_labels.update(face[face > 0].tolist())

    if not boundary_labels:
        return False

    # Check if ANY seed voxel connects to a boundary component
    for seed_idx in seed_indices:
        pocket_label = labeled[seed_idx]
        if pocket_label != 0 and pocket_label in boundary_labels:
            return True

    return False


def compute_pocket_accessibility(
    atoms: AtomArray,
    protein_chain_id: str,
    ligand_chain_id: str,
    voxel_size: float = 0.5,
    grid_margin: float = 12.0,
    max_probe_radius: float = 10.0,
    probe_precision: float = 0.1,
) -> PocketAccessibilityResult:
    """Compute pocket solvent accessibility for a protein-ligand complex.

    Determines whether the ligand-binding pocket is connected to bulk
    solvent and measures the bottleneck radius of the entrance channel.

    Args:
        atoms: Full structure as a Biotite AtomArray.
        protein_chain_id: Chain ID of the protein.
        ligand_chain_id: Chain ID of the ligand.
        voxel_size: Voxel edge length in Angstroms.
        grid_margin: Padding around the ligand bounding box in Angstroms.
        max_probe_radius: Upper bound for bottleneck binary search.
        probe_precision: Binary search stops when interval < this value.

    Returns:
        A :class:`PocketAccessibilityResult` with accessibility flag
        and bottleneck radius.
    """
    # --- 1. Filter to heavy atoms, split by chain ---
    heavy_mask = atoms.element != "H"
    heavy_atoms = atoms[heavy_mask]

    protein_mask = heavy_atoms.chain_id == protein_chain_id
    ligand_mask = heavy_atoms.chain_id == ligand_chain_id

    protein_atoms = heavy_atoms[protein_mask]
    ligand_atoms = heavy_atoms[ligand_mask]

    # Edge case: too few ligand atoms
    if len(ligand_atoms) < 3:
        logger.warning(
            "Ligand %s has fewer than 3 heavy atoms (%d), returning inaccessible.",
            ligand_chain_id,
            len(ligand_atoms),
        )
        return PocketAccessibilityResult(
            pocket_is_accessible=False, bottleneck_radius=0.0
        )

    ligand_coords = ligand_atoms.coord
    protein_coords = protein_atoms.coord

    # Edge case: no protein atoms
    if len(protein_atoms) == 0:
        logger.warning(
            "No protein atoms for chain %s, returning fully accessible.",
            protein_chain_id,
        )
        return PocketAccessibilityResult(
            pocket_is_accessible=True, bottleneck_radius=max_probe_radius
        )

    # --- 2. Build 3D grid around ligand with margin ---
    ligand_min = ligand_coords.min(axis=0)
    ligand_max = ligand_coords.max(axis=0)
    grid_origin = ligand_min - grid_margin
    grid_end = ligand_max + grid_margin

    grid_shape = tuple(
        int(np.ceil((grid_end[i] - grid_origin[i]) / voxel_size)) + 1 for i in range(3)
    )

    # --- 3. Select protein atoms within reach of the grid ---
    # Include all protein atoms whose centers are within grid bounds + max vdW radius
    protein_radii = _get_vdw_radii(protein_atoms.element)
    max_vdw = protein_radii.max()

    # Use KDTree for efficient spatial filtering
    grid_center = (grid_origin + grid_end) / 2.0
    grid_half_diag = np.linalg.norm(grid_end - grid_origin) / 2.0 + max_vdw
    protein_tree = KDTree(protein_coords)
    nearby_idxs = protein_tree.query_ball_point(grid_center, grid_half_diag)

    if len(nearby_idxs) == 0:
        logger.warning("No protein atoms near the grid, returning fully accessible.")
        return PocketAccessibilityResult(
            pocket_is_accessible=True, bottleneck_radius=max_probe_radius
        )

    nearby_coords = protein_coords[nearby_idxs]
    nearby_radii = protein_radii[nearby_idxs]

    # --- 4. Rasterize protein atoms into grid ---
    protein_grid = _rasterize_atoms(
        nearby_coords, nearby_radii, grid_origin, grid_shape, voxel_size
    )

    # --- 5. Compute distance transform ---
    # dist_map[i,j,k] = distance (in Angstroms) from voxel to nearest protein surface
    dist_map = distance_transform_edt(~protein_grid) * voxel_size

    # --- 6. Find pocket seed voxels (all ligand atom positions + centroid) ---
    seed_indices: list[tuple[int, int, int]] = []

    # Use every ligand atom position as a potential seed
    all_seed_coords = np.vstack(
        [ligand_coords, ligand_coords.mean(axis=0, keepdims=True)]
    )
    for coord in all_seed_coords:
        idx = tuple(
            max(
                0,
                min(
                    int(round((coord[i] - grid_origin[i]) / voxel_size)),
                    grid_shape[i] - 1,
                ),
            )
            for i in range(3)
        )
        if not protein_grid[idx]:
            seed_indices.append(idx)

    # If all seeds landed inside protein, find the nearest free voxel to centroid
    if not seed_indices:
        free_voxels = np.argwhere(~protein_grid)
        if len(free_voxels) == 0:
            return PocketAccessibilityResult(
                pocket_is_accessible=False, bottleneck_radius=0.0
            )
        centroid_grid = np.array(
            [
                (ligand_coords.mean(axis=0)[i] - grid_origin[i]) / voxel_size
                for i in range(3)
            ]
        )
        dists_sq = np.sum((free_voxels - centroid_grid) ** 2, axis=1)
        nearest = free_voxels[np.argmin(dists_sq)]
        seed_indices.append(tuple(nearest))

    # Deduplicate seeds
    seed_indices = list(set(seed_indices))

    # --- 7. Connectivity structure (18-connectivity: face + edge neighbors) ---
    struct = generate_binary_structure(3, 2)

    # --- 8. Check water accessibility (probe = 1.4 A) ---
    water_probe = 1.4
    pocket_is_accessible = _pocket_connects_to_exterior(
        dist_map, water_probe, seed_indices, struct
    )

    # --- 9. Binary search for bottleneck radius ---
    if not pocket_is_accessible:
        bottleneck_radius = 0.0
    else:
        lo, hi = water_probe, max_probe_radius

        # First check if max probe radius still connects
        if _pocket_connects_to_exterior(dist_map, hi, seed_indices, struct):
            bottleneck_radius = hi
        else:
            while (hi - lo) >= probe_precision:
                mid = (lo + hi) / 2.0
                if _pocket_connects_to_exterior(dist_map, mid, seed_indices, struct):
                    lo = mid
                else:
                    hi = mid
            bottleneck_radius = lo

    return PocketAccessibilityResult(
        pocket_is_accessible=pocket_is_accessible,
        bottleneck_radius=round(bottleneck_radius, 2),
    )
