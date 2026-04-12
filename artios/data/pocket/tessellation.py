"""Vectorized alpha-sphere computation from Delaunay tessellation.

Drop-in replacement for pocketeer.core.compute_alpha_spheres with
bulk numpy operations instead of a per-simplex Python loop.
"""

import numpy as np
import numpy.typing as npt
from pocketeer.core.types import AlphaSphere
from scipy.spatial import Delaunay, cKDTree


def compute_alpha_spheres(
    coords: npt.NDArray[np.float64],
    *,
    r_min: float = 3.0,
    r_max: float = 6.0,
    min_atoms: int = 10,
) -> list[AlphaSphere]:
    """Compute alpha-spheres from Delaunay tessellation (vectorized).

    Equivalent to pocketeer.core.compute_alpha_spheres but ~10x faster
    by batch-computing circumspheres and emptiness checks.

    Args:
        coords: (N, 3) atom coordinates
        r_min: Minimum alpha-sphere radius (A)
        r_max: Maximum alpha-sphere radius (A)
        min_atoms: Minimum atoms required for tessellation

    Returns:
        List of AlphaSphere objects
    """
    if len(coords) < min_atoms:
        return []

    tri = Delaunay(coords)
    simplices = tri.simplices  # (M, 4) indices into coords

    # --- Batch circumsphere computation ---
    # For each tetrahedron, solve for circumsphere center and radius.
    # Place first vertex at origin: p1, p2, p3 = tet[1:] - tet[0]
    tet_coords = coords[simplices]  # (M, 4, 3)
    p0 = tet_coords[:, 0]  # (M, 3)
    d = tet_coords[:, 1:] - p0[:, None, :]  # (M, 3, 3) — rows are p1-p0, p2-p0, p3-p0

    # A @ center_rel = b  where A = 2*d, b = [|p1|^2, |p2|^2, |p3|^2]
    A = 2.0 * d  # (M, 3, 3)
    b = np.sum(d * d, axis=2)  # (M, 3)

    # Batch solve — filter out singular matrices first
    det = np.linalg.det(A)
    valid_mask = np.abs(det) > 1e-10
    n_valid = valid_mask.sum()

    centers = np.full((len(simplices), 3), np.nan)
    radii = np.full(len(simplices), 0.0)

    if n_valid > 0:
        # solve expects (M, 3, 3) @ (M, 3) but b must be (M, 3, 1) for batch
        centers_rel = np.linalg.solve(A[valid_mask], b[valid_mask, :, None]).squeeze(-1)
        centers[valid_mask] = centers_rel + p0[valid_mask]
        radii[valid_mask] = np.linalg.norm(centers_rel, axis=1)

    # --- Radius filter ---
    radius_mask = valid_mask & (radii >= r_min) & (radii <= r_max)
    candidate_idx = np.where(radius_mask)[0]

    if len(candidate_idx) == 0:
        return []

    # --- Batch emptiness check ---
    # A sphere is "empty" if no atom (other than its 4 defining vertices)
    # lies strictly inside it.
    candidate_centers = centers[candidate_idx]
    candidate_radii = radii[candidate_idx]
    candidate_simplices = simplices[candidate_idx]

    tree = cKDTree(coords)
    # Query all atoms within each sphere's radius (batched)
    neighbors = tree.query_ball_point(candidate_centers, candidate_radii)

    spheres = []
    for i, (nbrs, simplex, center, radius) in enumerate(
        zip(neighbors, candidate_simplices, candidate_centers, candidate_radii)
    ):
        defining = set(simplex.tolist())
        empty = True
        for idx in nbrs:
            if idx in defining:
                continue
            dist = np.linalg.norm(coords[idx] - center)
            if dist < radius - 1e-6:
                empty = False
                break
        if empty:
            spheres.append(
                AlphaSphere(
                    sphere_id=candidate_idx[i],
                    center=center,
                    radius=float(radius),
                    is_buried=True,
                    atom_indices=simplex.tolist(),
                )
            )

    return spheres
