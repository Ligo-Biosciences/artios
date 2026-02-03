"""Aromatic ring interactions: pi-stacking and cation-pi.

Based on Molstar's algorithm for cation-pi detection.
Reference: https://github.com/molstar/molstar
"""

from dataclasses import dataclass
from enum import IntEnum
from math import radians

import networkx as nx
import numpy as np
from biotite.structure import AtomArray, BondType, connect_via_distances
from biotite.structure.rings import find_aromatic_rings
from numpy.typing import NDArray

from protenix.common.elements import AROMATIC_ELEMENTS
from protenix.contacts.geometry import (
    minimum_angle,
    point_plane_offset,
    triangle_normal,
)
from protenix.contacts.valence import ValenceModel, compute_valence_model


class StackingType(IntEnum):
    """Type of pi-stacking interaction."""

    PARALLEL = 0
    PERPENDICULAR = 1


@dataclass
class PiStackingInteraction:
    """A detected pi-stacking interaction."""

    ring1_indices: NDArray[np.intp]
    ring2_indices: NDArray[np.intp]
    stacking_type: StackingType


@dataclass
class CationPiInteraction:
    """A detected cation-pi interaction."""

    ring_indices: NDArray[np.intp]
    cation_indices: NDArray[np.intp]  # Indices of atoms in the cation group
    offset: float  # Perpendicular offset from ring center


def find_all_rings(
    atoms: AtomArray,
    infer_bonds: bool = False,
) -> list[NDArray[np.intp]]:
    """Find all rings using geometry instead of aromatic bond flags.

    Args:
        atoms: AtomArray with coordinates and bonds set.
        infer_bonds: If True, infer bonds via distances (adds missing bonds
            even if some bonds already exist, e.g., ligands without bonds).

    Returns:
        List of atom index arrays for each ring found.
    """
    if infer_bonds:
        atoms.bonds = connect_via_distances(atoms)
    elif atoms.bonds is None:
        return []

    bond_array = atoms.bonds.as_array()

    # Build graph from all covalent bonds (exclude coordination bonds)
    covalent_mask = bond_array[:, 2] != BondType.COORDINATION
    covalent_bonds = bond_array[covalent_mask, :2]

    if len(covalent_bonds) == 0:
        return []

    graph = nx.Graph()
    graph.add_edges_from(covalent_bonds.tolist())

    # Find minimal cycles
    rings = nx.cycle_basis(graph)
    return [np.array(ring, dtype=np.intp) for ring in rings]


def find_planar_rings(
    atoms: AtomArray,
    min_size: int = 5,
    max_size: int = 7,
    planarity_threshold: float = 0.05,
    infer_bonds: bool = False,
) -> list[NDArray[np.intp]]:
    """Find planar rings using geometry instead of aromatic bond flags.

    Fallback for structures where aromatic bonds are not annotated
    (e.g., from structure prediction tools). Uses SVD to check planarity.

    Args:
        atoms: AtomArray with coordinates and bonds set.
        min_size: Minimum ring size to consider (default 5 for pyrrole).
        max_size: Maximum ring size to consider (default 7 for tropylium).
        planarity_threshold: Maximum deviation from planarity (Å). Rings with
            smallest singular value below this are considered planar.
        infer_bonds: If True and bonds are missing, infer them via distances.

    Returns:
        List of atom index arrays for each planar ring found.
    """
    all_rings = find_all_rings(atoms, infer_bonds=infer_bonds)
    if len(all_rings) == 0:
        return []

    elements = atoms.element
    coords = atoms.coord

    planar_rings: list[NDArray[np.intp]] = []

    for ring in all_rings:
        # Size filter
        if not (min_size <= len(ring) <= max_size):
            continue

        # Element filter: all atoms must be aromatic-compatible
        ring_elements = elements[ring]
        if not all(elem in AROMATIC_ELEMENTS for elem in ring_elements):
            continue

        # Planarity check using SVD
        ring_coords = coords[ring]
        centroid = ring_coords.mean(axis=0)
        centered = ring_coords - centroid

        # SVD: smallest singular value indicates out-of-plane deviation
        _, singular_values, _ = np.linalg.svd(centered)
        min_singular = singular_values[-1]

        if min_singular < planarity_threshold:
            planar_rings.append(ring)

    return planar_rings


def _find_all_aromatic_rings(
    atoms: AtomArray,
    infer_bonds: bool = False,
) -> list[NDArray[np.intp]]:
    """Find aromatic rings using both bond flags and planarity detection.

    Combines biotite's bond-flag-based detection with planarity-based
    detection to handle structures with mixed annotation (e.g., protein
    with proper aromatic flags + ligand without).

    Args:
        atoms: AtomArray with coordinates and bonds set.
        infer_bonds: If True and bonds are missing, infer them via distances.

    Returns:
        List of ring atom index arrays (deduplicated).
    """
    # Get rings from both methods
    if infer_bonds:
        flagged_rings = []
    else:
        flagged_rings = find_aromatic_rings(atoms)

    planar_rings = find_planar_rings(atoms, infer_bonds=infer_bonds)

    if len(flagged_rings) == 0:
        return planar_rings
    if len(planar_rings) == 0:
        return flagged_rings

    # Deduplicate by converting to frozensets
    seen: set[frozenset[int]] = set()
    combined: list[NDArray[np.intp]] = []

    for ring in flagged_rings:
        key = frozenset(ring.tolist())
        if key not in seen:
            seen.add(key)
            combined.append(ring)

    for ring in planar_rings:
        key = frozenset(ring.tolist())
        if key not in seen:
            seen.add(key)
            combined.append(ring)

    return combined


def detect_pi_stacking(
    atoms: AtomArray,
    *,
    centroid_cutoff: float = 5.5,
    offset_max: float = 2.0,
    plane_angle_tol: float = 30.0,
    infer_bonds: bool = False,
) -> list[PiStackingInteraction]:
    """Detect pi-stacking interactions using Molstar's algorithm.

    Uses offset-based criterion instead of shift-angle criterion.
    Takes the minimum offset from both ring perspectives.

    Args:
        atoms: AtomArray with coordinates and bonds set.
        centroid_cutoff: Maximum distance between ring centroids (Å).
        offset_max: Maximum perpendicular offset between ring centers (Å).
        plane_angle_tol: Tolerance for parallel/perpendicular plane angles (degrees).
        infer_bonds: If True and bonds are missing, infer them via distances.

    Returns:
        List of detected PiStackingInteraction objects.
    """
    plane_angle_tol_rad = radians(plane_angle_tol)

    rings = _find_all_aromatic_rings(atoms, infer_bonds=infer_bonds)
    if len(rings) == 0:
        return []

    coords = atoms.coord
    ring_centroids = np.array([coords[ring].mean(axis=0) for ring in rings])
    ring_normals = np.array([triangle_normal(coords[ring]) for ring in rings])

    centroid_cutoff_sq = centroid_cutoff**2
    interactions: list[PiStackingInteraction] = []

    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            # Distance check
            diff = ring_centroids[j] - ring_centroids[i]
            dist_sq = float(np.dot(diff, diff))
            if dist_sq > centroid_cutoff_sq:
                continue

            # Offset check (Molstar criterion) - take minimum from both perspectives
            offset_i = point_plane_offset(
                ring_centroids[j], ring_centroids[i], ring_normals[i]
            )
            offset_j = point_plane_offset(
                ring_centroids[i], ring_centroids[j], ring_normals[j]
            )
            offset = min(offset_i, offset_j)

            if offset > offset_max:
                continue

            # Plane angle check (parallel or perpendicular)
            plane_angle = float(minimum_angle(ring_normals[i], ring_normals[j]))
            is_parallel = plane_angle < plane_angle_tol_rad
            is_perpendicular = abs(plane_angle - np.pi / 2) < plane_angle_tol_rad

            if not (is_parallel or is_perpendicular):
                continue

            interactions.append(
                PiStackingInteraction(
                    ring1_indices=rings[i],
                    ring2_indices=rings[j],
                    stacking_type=StackingType.PARALLEL
                    if is_parallel
                    else StackingType.PERPENDICULAR,
                )
            )

    return interactions


# Residues with cationic nitrogens at physiological pH
CATIONIC_RESIDUES: frozenset[str] = frozenset({"ARG", "LYS", "HIS"})

# Backbone atom names to exclude from cation detection
BACKBONE_ATOMS: frozenset[str] = frozenset({"N", "CA", "C", "O", "H", "HA"})


def _identify_cation_groups(
    atoms: AtomArray,
    valence_model: ValenceModel,
) -> list[NDArray[np.intp]]:
    """Identify cation groups using valence model and residue patterns.

    Groups cationic atoms by residue (like Molstar). For example, ARG's
    guanidinium (NE, NH1, NH2) becomes one group with centroid used for
    distance calculations.

    Combines two approaches:
    1. Valence model charges > 0 (catches ligands, unusual chemistry)
    2. Non-backbone nitrogens in ARG/LYS/HIS (physiological protonation)

    Args:
        atoms: AtomArray with element, res_name, atom_name annotations.
        valence_model: Pre-computed valence model.

    Returns:
        List of cation groups, each an array of atom indices.
    """
    # Collect all cation atoms
    cation_atoms: set[int] = set()

    # Method 1: Valence model charges
    cation_atoms.update(np.where(valence_model.charge > 0)[0])

    # Method 2: Residue-based (ARG/LYS/HIS non-backbone nitrogens)
    for i in range(len(atoms)):
        if (
            atoms.res_name[i] in CATIONIC_RESIDUES
            and atoms.element[i] == "N"
            and atoms.atom_name[i] not in BACKBONE_ATOMS
        ):
            cation_atoms.add(i)

    if not cation_atoms:
        return []

    # Group by residue (chain_id, res_id)
    groups: dict[tuple[str, int], list[int]] = {}
    for idx in cation_atoms:
        key = (atoms.chain_id[idx], int(atoms.res_id[idx]))
        if key not in groups:
            groups[key] = []
        groups[key].append(idx)

    return [np.array(sorted(indices), dtype=np.intp) for indices in groups.values()]


def detect_cation_pi(
    atoms: AtomArray,
    *,
    distance_max: float = 6.0,
    offset_max: float = 2.0,
    infer_bonds: bool = False,
    valence_model: ValenceModel | None = None,
) -> list[CationPiInteraction]:
    """Detect cation-pi interactions using Molstar's algorithm.

    Uses perpendicular offset check instead of angle check:
    - Distance between ring centroid and cation group centroid ≤ distance_max
    - Perpendicular offset (lateral displacement) ≤ offset_max

    Cationic atoms are grouped by residue (e.g., ARG's NE/NH1/NH2 form one
    group), and the group centroid is used for distance calculations.

    Args:
        atoms: AtomArray with coordinates and bonds set.
        distance_max: Maximum distance between ring centroid and cation.
        offset_max: Maximum perpendicular offset from ring center.
        infer_bonds: If True and bonds are missing, infer them via distances.
        valence_model: Pre-computed valence model. If None, will be computed.

    Returns:
        List of detected CationPiInteraction objects.
    """
    if valence_model is None:
        valence_model = compute_valence_model(atoms)

    # Find aromatic rings (using both bond-flag and planarity detection)
    rings = _find_all_aromatic_rings(atoms, infer_bonds=infer_bonds)
    if len(rings) == 0:
        return []

    # Identify cation groups (grouped by residue)
    cation_groups = _identify_cation_groups(atoms, valence_model)
    if len(cation_groups) == 0:
        return []

    # Precompute ring centroids and normals
    coords = atoms.coord
    ring_centroids = np.array([coords[ring].mean(axis=0) for ring in rings])
    ring_normals = np.array([triangle_normal(coords[ring]) for ring in rings])

    # Precompute cation group centroids
    cation_centroids = np.array([coords[group].mean(axis=0) for group in cation_groups])

    distance_max_sq = distance_max**2
    interactions: list[CationPiInteraction] = []

    for ring, ring_centroid, normal in zip(rings, ring_centroids, ring_normals):
        ring_set = set(ring.tolist())

        for cation_group, cation_centroid in zip(cation_groups, cation_centroids):
            # Skip if any cation atom is part of the ring (e.g., HIS ND1/NE2)
            if any(idx in ring_set for idx in cation_group):
                continue

            # Distance check (using group centroid)
            dist_sq = float(np.sum((ring_centroid - cation_centroid) ** 2))
            if dist_sq > distance_max_sq:
                continue

            # Offset check (Molstar's criterion)
            offset = point_plane_offset(cation_centroid, ring_centroid, normal)
            if offset > offset_max:
                continue

            interactions.append(
                CationPiInteraction(
                    ring_indices=ring,
                    cation_indices=cation_group,
                    offset=offset,
                )
            )

    return interactions
