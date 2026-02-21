"""Shape complementarity (Sc) metric for protein-ligand interfaces.

Implements a Lawrence & Colman Sc-inspired score using Shrake-Rupley
surface generation (numpy/scipy only). Measures how well a ligand's
surface meshes with the protein pocket's surface via paired normal
dot products.

Reference:
    Lawrence, M. C., & Colman, P. M. (1993).
    Shape complementarity at protein/protein interfaces.
    J. Mol. Biol., 234(4), 946–950.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from biotite.structure import AtomArray, sasa
from biotite.structure.info import vdw_radius_single
from numpy.typing import NDArray
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BurialWeightedScResult:
    """Result of a burial-weighted shape complementarity calculation.

    Tailored for enzyme-substrate evaluation: the ligand should be
    partially exposed (substrate entry/exit), but the buried portion
    should have high shape complementarity with the pocket.

    Attributes:
        score: Burial-weighted Sc (ligand→protein direction).
        exposure_fraction: Fraction of ligand surface exposed to solvent
            (0 = fully buried, 1 = fully exposed).
        n_ligand_surface_points: Total ligand surface points generated.
        mean_burial_weight: Average burial weight across ligand surface
            points.
    """

    score: float
    exposure_fraction: float
    n_ligand_surface_points: int
    mean_burial_weight: float


@dataclass(frozen=True)
class ShapeComplementarityResult:
    """Result of a shape complementarity calculation.

    Attributes:
        score: Symmetrized Sc score in [0, 1] (or NaN on degenerate input).
        score_protein_to_ligand: Directional Sc from protein to ligand.
        score_ligand_to_protein: Directional Sc from ligand to protein.
        n_interface_points_protein: Number of protein surface points at interface.
        n_interface_points_ligand: Number of ligand surface points at interface.
    """

    score: float
    score_protein_to_ligand: float
    score_ligand_to_protein: float
    n_interface_points_protein: int
    n_interface_points_ligand: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fibonacci_sphere_points(n: int) -> NDArray[np.floating]:
    """Generate *n* approximately uniform points on the unit sphere.

    Uses the Fibonacci lattice (golden-angle spiral).

    Returns:
        Array of shape ``(n, 3)``.
    """
    indices = np.arange(n, dtype=np.float64)
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle

    # y goes from ~+1 to ~-1
    y = 1.0 - (2.0 * indices / (n - 1)) if n > 1 else np.array([0.0])
    radius = np.sqrt(1.0 - y * y)
    theta = phi * indices

    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return np.column_stack([x, y, z])


def _get_vdw_radii(elements: NDArray) -> NDArray[np.floating]:
    """Look up van der Waals radii for an array of element symbols.

    Falls back to 1.7 A for unknown elements.
    """
    fallback = 1.7
    radii = np.empty(len(elements), dtype=np.float64)
    for i, elem in enumerate(elements):
        r = vdw_radius_single(elem)
        radii[i] = r if r is not None else fallback
    return radii


def _generate_surface(
    coords: NDArray[np.floating],
    elements: NDArray,
    n_points: int,
    probe_radius: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.intp]]:
    """Shrake-Rupley surface point generation.

    For each atom, places *n_points* on its solvent-accessible sphere
    (vdW radius + probe radius) via a Fibonacci lattice, then culls
    points that are buried inside any neighboring atom's sphere.

    Args:
        coords: Atom positions, shape ``(N, 3)``.
        elements: Element symbols, length ``N``.
        n_points: Number of sample points per atom.
        probe_radius: Probe (solvent) radius in Angstroms.

    Returns:
        ``(surface_points, surface_normals, atom_indices)`` where
        points and normals have shape ``(M, 3)`` and atom_indices
        has shape ``(M,)`` mapping each surface point to its
        generating atom index.
    """
    n_atoms = len(coords)
    radii = _get_vdw_radii(elements) + probe_radius  # (N,)

    # Pre-compute the unit sphere template once
    template = _fibonacci_sphere_points(n_points)  # (n_points, 3)

    # Build KDTree of atom centres for fast neighbor lookup
    max_radius = radii.max()
    tree = KDTree(coords)

    all_points: list[NDArray] = []
    all_normals: list[NDArray] = []
    all_atom_indices: list[NDArray] = []

    for i in range(n_atoms):
        # Points on this atom's sphere
        pts = coords[i] + radii[i] * template  # (n_points, 3)

        # Find neighbor atoms that could possibly bury these points.
        # A point at distance d from atom j is buried if d < radii[j].
        # Maximum possible burial distance is max_radius.
        neighbor_idxs = tree.query_ball_point(coords[i], radii[i] + max_radius)

        # Check burial against each neighbor (skip self)
        buried = np.zeros(n_points, dtype=bool)
        for j in neighbor_idxs:
            if j == i:
                continue
            dists_sq = np.sum((pts - coords[j]) ** 2, axis=1)
            buried |= dists_sq < radii[j] ** 2

        exposed = ~buried
        if exposed.any():
            all_points.append(pts[exposed])
            all_normals.append(template[exposed])  # unit normals
            all_atom_indices.append(np.full(exposed.sum(), i, dtype=np.intp))

    if not all_points:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0, dtype=np.intp)

    return (
        np.concatenate(all_points),
        np.concatenate(all_normals),
        np.concatenate(all_atom_indices),
    )


def _select_interface_points(
    points: NDArray[np.floating],
    normals: NDArray[np.floating],
    other_atom_coords: NDArray[np.floating],
    cutoff: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Keep only surface points within *cutoff* of any atom on the other molecule."""
    if len(points) == 0 or len(other_atom_coords) == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    other_tree = KDTree(other_atom_coords)
    dists, _ = other_tree.query(points)
    mask = dists <= cutoff
    return points[mask], normals[mask]


def _compute_directional_sc(
    pts_a: NDArray[np.floating],
    norms_a: NDArray[np.floating],
    pts_b: NDArray[np.floating],
    norms_b: NDArray[np.floating],
    sigma: float,
) -> float:
    """Compute one-directional shape complementarity score.

    For each point on surface A, find the nearest point on surface B,
    then compute ``s_i = -(n_A . n_B) * exp(-d^2 / (2 * sigma^2))``.

    Complementary normals point in opposite directions, so their dot
    product is ~ -1, and after negation the contribution is ~ +1.
    """
    if len(pts_a) == 0 or len(pts_b) == 0:
        return 0.0

    tree_b = KDTree(pts_b)
    dists, indices = tree_b.query(pts_a)

    matched_normals_b = norms_b[indices]
    dots = np.sum(norms_a * matched_normals_b, axis=1)
    weights = np.exp(-(dists**2) / (2.0 * sigma**2))
    scores = -dots * weights
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_shape_complementarity(
    atoms: AtomArray,
    protein_chain_id: str,
    ligand_chain_id: str,
    n_points_per_atom: int = 128,
    probe_radius: float = 1.4,
    residue_selection_cutoff: float = 5.0,
    interface_cutoff: float = 5.0,
    sigma: float = 2.5,
) -> ShapeComplementarityResult:
    """Compute the shape complementarity score for a protein-ligand interface.

    Uses a Shrake-Rupley surface approximation and Lawrence & Colman-style
    normal-based scoring.

    Args:
        atoms: Full structure as a Biotite AtomArray.
        protein_chain_id: Chain ID of the protein.
        ligand_chain_id: Chain ID of the ligand.
        n_points_per_atom: Fibonacci lattice points per atom for surface
            generation.
        probe_radius: Solvent probe radius in Angstroms.
        residue_selection_cutoff: Distance (A) to select protein residues
            near the ligand.
        interface_cutoff: Distance (A) to select surface points at the
            interface.
        sigma: Gaussian decay width (A) for distance weighting.

    Returns:
        A :class:`ShapeComplementarityResult` with the symmetrized score
        and per-direction scores.
    """
    nan_result = ShapeComplementarityResult(
        score=float("nan"),
        score_protein_to_ligand=float("nan"),
        score_ligand_to_protein=float("nan"),
        n_interface_points_protein=0,
        n_interface_points_ligand=0,
    )

    # --- 1. Filter to heavy atoms ----------------------------------------
    heavy_mask = atoms.element != "H"
    heavy_atoms = atoms[heavy_mask]

    # --- 2. Split by chain ------------------------------------------------
    protein_mask = heavy_atoms.chain_id == protein_chain_id
    ligand_mask = heavy_atoms.chain_id == ligand_chain_id

    protein_atoms = heavy_atoms[protein_mask]
    ligand_atoms = heavy_atoms[ligand_mask]

    if len(ligand_atoms) < 3 or len(protein_atoms) == 0:
        logger.warning(
            f"Not enough atoms found for protein {protein_chain_id} ({len(protein_atoms)}) and ligand {ligand_chain_id} ({len(ligand_atoms)})"
        )
        return nan_result

    protein_coords = protein_atoms.coord
    ligand_coords = ligand_atoms.coord

    # --- 3. Select nearby protein residues --------------------------------
    ligand_tree = KDTree(ligand_coords)
    dists, _ = ligand_tree.query(protein_coords)
    near_mask = dists <= residue_selection_cutoff

    # Expand to full residues: any atom close → keep entire residue
    near_res_ids = set(
        zip(protein_atoms.chain_id[near_mask], protein_atoms.res_id[near_mask])
    )
    if not near_res_ids:
        logger.warning(
            f"No nearby residues found for protein {protein_chain_id} ({len(protein_atoms)}) and ligand {ligand_chain_id} ({len(ligand_atoms)})"
        )
        return nan_result

    residue_mask = np.array(
        [
            (cid, rid) in near_res_ids
            for cid, rid in zip(protein_atoms.chain_id, protein_atoms.res_id)
        ]
    )
    pocket_atoms = protein_atoms[residue_mask]
    pocket_coords = pocket_atoms.coord

    # --- 4. Generate surfaces ---------------------------------------------
    pocket_pts, pocket_norms, _ = _generate_surface(
        pocket_coords, pocket_atoms.element, n_points_per_atom, probe_radius
    )
    ligand_pts, ligand_norms, _ = _generate_surface(
        ligand_coords, ligand_atoms.element, n_points_per_atom, probe_radius
    )

    if len(pocket_pts) == 0 or len(ligand_pts) == 0:
        logger.warning(
            f"No surface points found for protein {protein_chain_id} and ligand {ligand_chain_id}"
        )
        return nan_result

    # --- 5. Select interface points ---------------------------------------
    iface_pocket_pts, iface_pocket_norms = _select_interface_points(
        pocket_pts, pocket_norms, ligand_coords, interface_cutoff
    )
    iface_ligand_pts, iface_ligand_norms = _select_interface_points(
        ligand_pts, ligand_norms, pocket_coords, interface_cutoff
    )

    n_iface_prot = len(iface_pocket_pts)
    n_iface_lig = len(iface_ligand_pts)

    if n_iface_prot == 0 or n_iface_lig == 0:
        logger.warning(
            f"No interface points found for protein {protein_chain_id} and ligand {ligand_chain_id}"
        )
        return ShapeComplementarityResult(
            score=0.0,
            score_protein_to_ligand=0.0,
            score_ligand_to_protein=0.0,
            n_interface_points_protein=n_iface_prot,
            n_interface_points_ligand=n_iface_lig,
        )

    # --- 6. Compute directional Sc ----------------------------------------
    sc_p2l = _compute_directional_sc(
        iface_pocket_pts,
        iface_pocket_norms,
        iface_ligand_pts,
        iface_ligand_norms,
        sigma,
    )
    sc_l2p = _compute_directional_sc(
        iface_ligand_pts,
        iface_ligand_norms,
        iface_pocket_pts,
        iface_pocket_norms,
        sigma,
    )

    # --- 7. Symmetrize ----------------------------------------------------
    sc = (sc_p2l + sc_l2p) / 2.0

    return ShapeComplementarityResult(
        score=sc,
        score_protein_to_ligand=sc_p2l,
        score_ligand_to_protein=sc_l2p,
        n_interface_points_protein=n_iface_prot,
        n_interface_points_ligand=n_iface_lig,
    )


def compute_burial_weighted_sc(
    atoms: AtomArray,
    protein_chain_id: str,
    ligand_chain_id: str,
    n_points_per_atom: int = 128,
    probe_radius: float = 1.4,
    residue_selection_cutoff: float = 10.0,
    sigma: float = 2.5,
) -> BurialWeightedScResult:
    """Compute burial-weighted shape complementarity for enzyme-substrate evaluation.

    Instead of averaging Sc equally over all ligand surface points, each
    point's contribution is weighted by the SASA-derived burial fraction
    of its generating atom.  This rewards structures where the buried
    portion of the ligand fits the pocket well, while naturally
    down-weighting exposed (solvent-facing) regions.

    Only the ligand→protein direction is computed — the protein pocket's
    complementarity to the ligand is less meaningful when the ligand is
    partially exposed by design.

    Args:
        atoms: Full structure as a Biotite AtomArray.
        protein_chain_id: Chain ID of the protein.
        ligand_chain_id: Chain ID of the ligand.
        n_points_per_atom: Fibonacci lattice points per atom for surface
            generation.
        probe_radius: Solvent probe radius in Angstroms.
        residue_selection_cutoff: Distance (A) to select protein residues
            near the ligand.
        sigma: Gaussian decay width (A) for distance weighting.

    Returns:
        A :class:`BurialWeightedScResult` with the weighted score and
        burial statistics.
    """
    nan_result = BurialWeightedScResult(
        score=float("nan"),
        exposure_fraction=float("nan"),
        n_ligand_surface_points=0,
        mean_burial_weight=float("nan"),
    )

    # --- 1. Filter to heavy atoms ----------------------------------------
    heavy_mask = atoms.element != "H"
    heavy_atoms = atoms[heavy_mask]

    # --- 2. Split by chain ------------------------------------------------
    protein_mask = heavy_atoms.chain_id == protein_chain_id
    ligand_mask = heavy_atoms.chain_id == ligand_chain_id

    protein_atoms = heavy_atoms[protein_mask]
    ligand_atoms = heavy_atoms[ligand_mask]

    if len(ligand_atoms) < 3 or len(protein_atoms) == 0:
        logger.warning(
            "Not enough atoms: protein %s (%d), ligand %s (%d)",
            protein_chain_id,
            len(protein_atoms),
            ligand_chain_id,
            len(ligand_atoms),
        )
        return nan_result

    protein_coords = protein_atoms.coord
    ligand_coords = ligand_atoms.coord

    # --- 3. Select nearby protein residues (pocket) -----------------------
    ligand_tree = KDTree(ligand_coords)
    dists, _ = ligand_tree.query(protein_coords)
    near_mask = dists <= residue_selection_cutoff

    near_res_ids = set(
        zip(protein_atoms.chain_id[near_mask], protein_atoms.res_id[near_mask])
    )
    if not near_res_ids:
        logger.warning(
            "No nearby residues for protein %s and ligand %s",
            protein_chain_id,
            ligand_chain_id,
        )
        return nan_result

    residue_mask = np.array(
        [
            (cid, rid) in near_res_ids
            for cid, rid in zip(protein_atoms.chain_id, protein_atoms.res_id)
        ]
    )
    pocket_atoms = protein_atoms[residue_mask]
    pocket_coords = pocket_atoms.coord

    # --- 4. Compute per-atom burial weights via SASA ----------------------
    sasa_alone = sasa(ligand_atoms, vdw_radii="Single")  # (N_lig,)
    sasa_complex = sasa(heavy_atoms, atom_filter=ligand_mask, vdw_radii="Single")
    sasa_complex_lig = sasa_complex[ligand_mask]  # (N_lig,)

    burial_weights = np.ones(len(ligand_atoms), dtype=np.float64)
    nonzero = sasa_alone > 0
    burial_weights[nonzero] = np.clip(
        1.0 - sasa_complex_lig[nonzero] / sasa_alone[nonzero], 0.0, 1.0
    )
    # Atoms with sasa_alone == 0 (fully internal) keep weight 1.0

    # --- 5. Exposure fraction ---------------------------------------------
    sasa_alone_total = sasa_alone.sum()
    if sasa_alone_total > 0:
        exposure_fraction = float(sasa_complex_lig.sum() / sasa_alone_total)
    else:
        exposure_fraction = 0.0

    # --- 6. Generate surfaces ---------------------------------------------
    ligand_pts, ligand_norms, ligand_atom_idx = _generate_surface(
        ligand_coords, ligand_atoms.element, n_points_per_atom, probe_radius
    )
    pocket_pts, pocket_norms, _ = _generate_surface(
        pocket_coords, pocket_atoms.element, n_points_per_atom, probe_radius
    )

    n_lig_pts = len(ligand_pts)
    if n_lig_pts == 0 or len(pocket_pts) == 0:
        logger.warning(
            "No surface points for protein %s and ligand %s",
            protein_chain_id,
            ligand_chain_id,
        )
        return nan_result

    # --- 7. Map burial weights to surface points --------------------------
    point_weights = burial_weights[ligand_atom_idx]  # (M,)
    weight_sum = point_weights.sum()

    if weight_sum == 0:
        # Fully exposed ligand — no buried face to evaluate
        return BurialWeightedScResult(
            score=float("nan"),
            exposure_fraction=exposure_fraction,
            n_ligand_surface_points=n_lig_pts,
            mean_burial_weight=0.0,
        )

    # --- 8. Compute burial-weighted directional Sc (ligand→protein) -------
    pocket_tree = KDTree(pocket_pts)
    dists, indices = pocket_tree.query(ligand_pts)

    matched_pocket_norms = pocket_norms[indices]
    dots = np.sum(ligand_norms * matched_pocket_norms, axis=1)
    decay = np.exp(-(dists**2) / (2.0 * sigma**2))
    per_point_sc = -dots * decay  # (M,)

    weighted_sc = float(np.sum(per_point_sc * point_weights) / weight_sum)

    return BurialWeightedScResult(
        score=weighted_sc,
        exposure_fraction=exposure_fraction,
        n_ligand_surface_points=n_lig_pts,
        mean_burial_weight=float(np.mean(point_weights)),
    )
