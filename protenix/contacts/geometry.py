"""Atom geometry types and associated ideal angles.

Based on VSEPR theory and Mol* implementation.
Reference: https://github.com/molstar/molstar
"""

from enum import IntEnum
from math import radians

import numpy as np
from biotite.structure import AtomArray, BondType
from numpy.typing import NDArray

from protenix.common.elements import H


class AtomGeometry(IntEnum):
    """Atom geometry based on coordination number (VSEPR theory).

    Numbering mostly inline with coordination number, except SquarePlanar.
    """

    Spherical = 0  # Isolated atom (e.g., noble gas, ion)
    Terminal = 1  # Single bond (e.g., H in H2O)
    Linear = 2  # Two bonds at 180° (e.g., CO2 carbon)
    Trigonal = 3  # Three bonds at 120° (e.g., BF3 boron, sp2 carbon)
    Tetrahedral = 4  # Four bonds at 109.47° (e.g., CH4 carbon, sp3)
    TrigonalBipyramidal = 5  # Five bonds (e.g., PCl5 phosphorus)
    Octahedral = 6  # Six bonds at 90° (e.g., SF6 sulfur)
    SquarePlanar = 7  # Four bonds in plane at 90° (e.g., XeF4)
    Unknown = 8


# Ideal bond angles for each geometry (in radians)
GEOMETRY_IDEAL_ANGLES: dict[AtomGeometry, float] = {
    AtomGeometry.Linear: radians(180.0),
    AtomGeometry.Trigonal: radians(120.0),
    AtomGeometry.Tetrahedral: radians(109.4721),
    AtomGeometry.Octahedral: radians(90.0),
}

DEFAULT_IDEAL_ANGLE: float = radians(120.0)

COORDINATION_TO_GEOMETRY: dict[int, AtomGeometry] = {
    0: AtomGeometry.Spherical,
    1: AtomGeometry.Terminal,
    2: AtomGeometry.Linear,
    3: AtomGeometry.Trigonal,
    4: AtomGeometry.Tetrahedral,
}


def get_ideal_angle(geometry: AtomGeometry) -> float:
    """Get the ideal bond angle for a given atom geometry."""
    return GEOMETRY_IDEAL_ANGLES.get(geometry, DEFAULT_IDEAL_ANGLE)


def assign_geometry(total_coordination: int) -> AtomGeometry:
    """Assign atom geometry based on total coordination number."""
    return COORDINATION_TO_GEOMETRY.get(total_coordination, AtomGeometry.Unknown)


def _get_bonded_indices(atoms: AtomArray, atom_idx: int) -> list[int]:
    """Get indices of atoms covalently bonded to the given atom.

    Args:
        atoms: AtomArray with bonds set.
        atom_idx: Index of the atom to get neighbors for.

    Returns:
        List of bonded atom indices.
    """
    if atoms.bonds is None:
        return []

    bonded = atoms.bonds.get_bonds(atom_idx)
    if bonded is None or len(bonded[0]) == 0:
        return []

    bonded_indices, bond_types = bonded
    # Filter to covalent bonds only (exclude coordination bonds)
    return [
        int(idx)
        for idx, bt in zip(bonded_indices, bond_types)
        if BondType(bt) != BondType.COORDINATION
    ]


def _vector_angle(v1: NDArray[np.floating], v2: NDArray[np.floating]) -> float:
    """Calculate angle between two vectors in radians.

    Args:
        v1: First vector, shape (3,).
        v2: Second vector, shape (3,).

    Returns:
        Angle in radians [0, pi].
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def calc_angles(
    atoms: AtomArray,
    index_a: int,
    index_b: int,
    ignore_hydrogens: bool = True,
) -> tuple[list[float], list[float]]:
    """Calculate the angles X-A-B for all X bonded to atom A.

    For each atom X bonded to A, calculates the angle formed by X-A-B.
    Separates angles into heavy atoms and hydrogens.

    Args:
        atoms: AtomArray with coordinates and bonds.
        index_a: Index of atom A (the angle vertex).
        index_b: Index of atom B.
        ignore_hydrogens: If True, only return heavy atom angles.
            If False, also compute hydrogen angles separately.

    Returns:
        Tuple of (heavy_atom_angles, hydrogen_angles) in radians.
        If ignore_hydrogens=True, hydrogen_angles will be empty.
    """
    coords = atoms.coord
    elements = atoms.element

    pos_a = coords[index_a]
    pos_b = coords[index_b]
    dir_ab = pos_b - pos_a  # Direction from A to B

    angles_heavy: list[float] = []
    angles_h: list[float] = []

    for idx_x in _get_bonded_indices(atoms, index_a):
        pos_x = coords[idx_x]
        dir_ax = pos_x - pos_a  # Direction from A to X
        angle = _vector_angle(dir_ab, dir_ax)

        if elements[idx_x] != H:
            angles_heavy.append(angle)
        elif not ignore_hydrogens:
            angles_h.append(angle)

    return angles_heavy, angles_h


def calc_plane_angle(
    atoms: AtomArray,
    index_a: int,
    index_b: int,
) -> float | None:
    """Calculate the out-of-plane angle from atom B to the plane at atom A.

    Finds two heavy-atom neighbors of A to define a plane, then measures
    the angle from that plane to atom B. Used for checking trigonal
    geometry constraints.

    If A has only one heavy neighbor, walks to that neighbor to find a
    second atom to define the plane.

    Args:
        atoms: AtomArray with coordinates and bonds.
        index_a: Index of atom A (plane center).
        index_b: Index of atom B (out-of-plane atom).

    Returns:
        Out-of-plane angle in radians [0, pi/2], or None if plane
        cannot be defined (fewer than 2 neighbors found).
    """
    coords = atoms.coord
    elements = atoms.element

    pos_a = coords[index_a]
    pos_b = coords[index_b]
    dir_ab = pos_b - pos_a

    # Collect up to 2 heavy-atom neighbor directions
    neighbor_dirs: list[NDArray[np.floating]] = []
    first_neighbor_idx: int | None = None

    for idx_x in _get_bonded_indices(atoms, index_a):
        if len(neighbor_dirs) >= 2:
            break
        if elements[idx_x] == H:
            continue

        pos_x = coords[idx_x]
        neighbor_dirs.append(pos_x - pos_a)

        if first_neighbor_idx is None:
            first_neighbor_idx = idx_x

    # If only one neighbor found, walk to that neighbor's bonds
    if len(neighbor_dirs) == 1 and first_neighbor_idx is not None:
        for idx_x in _get_bonded_indices(atoms, first_neighbor_idx):
            if len(neighbor_dirs) >= 2:
                break
            if idx_x == index_a:
                continue
            if elements[idx_x] == H:
                continue

            pos_x = coords[idx_x]
            neighbor_dirs.append(pos_x - pos_a)

    if len(neighbor_dirs) < 2:
        return None

    # Compute plane normal via cross product
    normal = np.cross(neighbor_dirs[0], neighbor_dirs[1])
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-8:
        # Neighbors are collinear, no plane defined
        return None

    # Angle between normal and dir_ab
    angle_to_normal = _vector_angle(normal, dir_ab)

    # Out-of-plane angle is |90° - angle_to_normal|
    return float(abs(np.pi / 2 - angle_to_normal))


def triangle_normal(
    coords: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute unit normal vector from three or more points.

    Uses the first three points to define a plane via cross product.
    Useful for computing ring plane normals.

    Args:
        coords: Array of shape (N, 3) with N >= 3 points.

    Returns:
        Unit normal vector of shape (3,). Returns [0, 0, 1] if
        points are collinear.
    """
    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0])
    return normal / norm


def closest_hydrogen_index(
    atoms: AtomArray,
    donor_idx: int,
    acceptor_idx: int,
) -> int:
    """Find the hydrogen bonded to the donor that is closest to the acceptor.

    When a donor has multiple hydrogens attached, this finds which one
    is most likely participating in the hydrogen bond.

    Note: Only returns a hydrogen index if that hydrogen is closer to
    the acceptor than the donor heavy atom itself. This filters out
    geometrically unfavorable configurations.

    Args:
        atoms: AtomArray with coordinates and bonds.
        donor_idx: Index of the hydrogen bond donor atom.
        acceptor_idx: Index of the hydrogen bond acceptor atom.

    Returns:
        Index of the closest hydrogen if one is closer than the donor,
        otherwise returns donor_idx.
    """
    coords = atoms.coord
    elements = atoms.element

    pos_donor = coords[donor_idx]
    pos_acceptor = coords[acceptor_idx]

    # Initialize to donor-acceptor distance; H must be closer than this
    min_dist_sq = float(np.sum((pos_donor - pos_acceptor) ** 2))
    closest_idx = donor_idx

    for idx_h in _get_bonded_indices(atoms, donor_idx):
        if elements[idx_h] != H:
            continue

        pos_h = coords[idx_h]
        dist_sq = float(np.sum((pos_h - pos_acceptor) ** 2))

        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_idx = idx_h

    return closest_idx


def minimum_angle(
    v1: NDArray[np.floating],
    v2: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Get minimum angle between vectors (range [0, pi/2]).

    Useful for ring normal comparisons where direction is ambiguous.

    Args:
        v1: First vector(s), shape (..., 3).
        v2: Second vector(s), shape (..., 3).

    Returns:
        Angle(s) in radians, range [0, pi/2].
    """
    dot = np.einsum("...i,...i->...", v1, v2)
    return np.arccos(np.minimum(np.abs(dot), 1.0))


def point_plane_offset(
    point: NDArray[np.floating],
    plane_origin: NDArray[np.floating],
    plane_normal: NDArray[np.floating],
) -> float:
    """Compute lateral offset of a point from a plane origin.

    Projects the point onto the plane and measures the distance
    from the plane origin to that projection.

    Args:
        point: 3D position of the point.
        plane_origin: 3D position of the reference point on the plane.
        plane_normal: Unit normal vector of the plane.

    Returns:
        Distance from plane origin to the projected point.
    """
    v = point - plane_origin
    v_projected = v - np.dot(v, plane_normal) * plane_normal
    return float(np.linalg.norm(v_projected))
