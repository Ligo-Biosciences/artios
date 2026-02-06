"""Hydrogen bond detection.

Identifies atoms that can participate in hydrogen bonds based on
element type, charge, geometry, and local environment. Detects
hydrogen bonds using geometric criteria.

Based on Mol* implementation.
Reference: https://github.com/molstar/molstar

Algorithm references:
- HBPLUS: https://doi.org/10.1006/jmbi.1994.1334
- Sulfur H-bonds: https://doi.org/10.1002/prot.22327
"""

from dataclasses import dataclass
from math import radians

import numpy as np
from biotite.structure import AtomArray
from numpy.typing import NDArray

from protenix.common import residue_names as residues
from protenix.common.atom_types import PROTEIN_BACKBONE_ATOMS
from protenix.common.elements import N, O, S
from protenix.contacts.geometry import (
    AtomGeometry,
    calc_angles,
    calc_plane_angle,
    closest_hydrogen_index,
    get_ideal_angle,
)
from protenix.contacts.valence import ValenceModel, compute_valence_model

# Histidine ring nitrogen atom names
HIS_RING_NITROGENS: frozenset[str] = frozenset({"ND1", "NE2"})


def _is_histidine_ring_nitrogen(
    res_names: NDArray[np.str_],
    atom_names: NDArray[np.str_],
    elements: NDArray[np.str_],
    idx: int,
) -> bool:
    """Check if atom is a histidine ring nitrogen."""
    return (
        res_names[idx] == residues.HIS
        and elements[idx] == N
        and atom_names[idx] in HIS_RING_NITROGENS
    )


def detect_hydrogen_donors(
    atoms: AtomArray,
    valence_model: ValenceModel | None = None,
) -> NDArray[np.bool_]:
    """Detect potential hydrogen bond donors.

    A hydrogen donor is an atom that can donate a hydrogen in an H-bond.
    Criteria:
    - Histidine ring nitrogens (ND1, NE2) - always included due to
      ambiguous protonation
    - N, O, or S atoms with total_h > 0

    Args:
        atoms: AtomArray with bonds set.
        valence_model: Pre-computed valence model. If None, will be computed.

    Returns:
        Boolean mask of hydrogen donor atoms.
    """
    if valence_model is None:
        valence_model = compute_valence_model(atoms)

    n_atoms = len(atoms)
    is_donor = np.zeros(n_atoms, dtype=np.bool_)

    elements = atoms.element
    res_names = atoms.res_name
    atom_names = atoms.atom_name
    total_h = valence_model.total_h

    for i in range(n_atoms):
        # Histidine ring nitrogens are always potential donors
        if _is_histidine_ring_nitrogen(res_names, atom_names, elements, i):
            is_donor[i] = True
        # N, O, S with hydrogens attached
        elif elements[i] in (N, O, S) and total_h[i] > 0:
            is_donor[i] = True

    return is_donor


def detect_hydrogen_acceptors(
    atoms: AtomArray,
    valence_model: ValenceModel | None = None,
) -> NDArray[np.bool_]:
    """Detect potential hydrogen bond acceptors.

    A hydrogen acceptor is an atom that can accept a hydrogen in an H-bond.
    Criteria:
    - All oxygen atoms
    - Nitrogen atoms if:
      - Histidine ring nitrogen (ND1, NE2), OR
      - Neutral (charge < 1) with at least one lone pair available
    - Sulfur atoms in CYS or MET, or with formal charge -1

    Args:
        atoms: AtomArray with bonds set.
        valence_model: Pre-computed valence model. If None, will be computed.

    Returns:
        Boolean mask of hydrogen acceptor atoms.
    """
    if valence_model is None:
        valence_model = compute_valence_model(atoms)

    n_atoms = len(atoms)
    is_acceptor = np.zeros(n_atoms, dtype=np.bool_)

    elements = atoms.element
    res_names = atoms.res_name
    atom_names = atoms.atom_name
    charge = valence_model.charge
    implicit_h = valence_model.implicit_h
    geometry = valence_model.ideal_geometry

    # Get bond counts for lone pair calculation
    bonds = atoms.bonds
    bond_counts = np.zeros(n_atoms, dtype=np.int_)
    for i in range(n_atoms):
        bonded = bonds.get_bonds(i)
        if bonded is not None and len(bonded[0]) > 0:
            bond_counts[i] = len(bonded[0])

    for i in range(n_atoms):
        elem = elements[i]

        if elem == O:
            # All oxygens are acceptors
            is_acceptor[i] = True

        elif elem == N:
            # Histidine ring nitrogens
            if _is_histidine_ring_nitrogen(res_names, atom_names, elements, i):
                is_acceptor[i] = True
            # Neutral nitrogen with lone pair
            elif charge[i] < 1:
                total_bonds = bond_counts[i] + implicit_h[i]
                geom = AtomGeometry(geometry[i])

                # Check if there's a lone pair available based on geometry
                has_lone_pair = (
                    (geom == AtomGeometry.Tetrahedral and total_bonds < 4)
                    or (geom == AtomGeometry.Trigonal and total_bonds < 3)
                    or (geom == AtomGeometry.Linear and total_bonds < 2)
                )
                if has_lone_pair:
                    is_acceptor[i] = True

        elif elem == S:
            # Sulfur in CYS or MET
            if res_names[i] in (residues.CYS, residues.MET):
                is_acceptor[i] = True
            # Or negatively charged sulfur (thiolate)
            elif charge[i] == -1:
                is_acceptor[i] = True

    return is_acceptor


def detect_hbond_features(
    atoms: AtomArray,
    valence_model: ValenceModel | None = None,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Detect both hydrogen bond donors and acceptors.

    Args:
        atoms: AtomArray with bonds set.
        valence_model: Pre-computed valence model. If None, will be computed.

    Returns:
        Tuple of (donor_mask, acceptor_mask) boolean arrays.
    """
    if valence_model is None:
        valence_model = compute_valence_model(atoms)

    donors = detect_hydrogen_donors(atoms, valence_model)
    acceptors = detect_hydrogen_acceptors(atoms, valence_model)

    return donors, acceptors


def add_hbond_feature_annotations(
    atoms: AtomArray,
    valence_model: ValenceModel | None = None,
) -> AtomArray:
    """Detect H-bond features and add as annotations to the AtomArray.

    Adds the following annotations:
    - `hbond_donor`: Boolean, True if atom is a potential H-bond donor
    - `hbond_acceptor`: Boolean, True if atom is a potential H-bond acceptor

    Args:
        atoms: AtomArray with bonds set.
        valence_model: Pre-computed valence model. If None, will be computed.

    Returns:
        The same AtomArray with annotations added.
    """
    donors, acceptors = detect_hbond_features(atoms, valence_model)

    atoms.set_annotation("hbond_donor", donors)
    atoms.set_annotation("hbond_acceptor", acceptors)

    return atoms


# ---------------------------------------------------------------------------
# Hydrogen Bond Detection
# ---------------------------------------------------------------------------

# Default ideal angle when geometry is unknown (120° in radians)
DEFAULT_IDEAL_ANGLE: float = radians(120.0)


@dataclass
class _HBondParams:
    """Internal parameters for hydrogen bond detection."""

    distance_max: float
    sulfur_distance_max: float
    don_angle_dev_max: float
    acc_angle_dev_max: float
    don_out_of_plane_max: float
    acc_out_of_plane_max: float
    ignore_hydrogens: bool
    include_backbone: bool
    include_water: bool

    @property
    def distance_max_sq(self) -> float:
        return self.distance_max**2

    @property
    def sulfur_distance_max_sq(self) -> float:
        return self.sulfur_distance_max**2


@dataclass
class HydrogenBond:
    """A detected hydrogen bond.

    Attributes:
        donor_idx: Index of the donor heavy atom.
        acceptor_idx: Index of the acceptor atom.
        hydrogen_idx: Index of the hydrogen atom (may equal donor_idx if implicit).
        distance: Donor-acceptor distance in Ångströms.
    """

    donor_idx: int
    acceptor_idx: int
    hydrogen_idx: int
    distance: float


def _is_backbone_atom(atom_names: NDArray[np.str_], idx: int) -> bool:
    """Check if an atom is a protein backbone atom."""
    return atom_names[idx] in PROTEIN_BACKBONE_ATOMS


def _is_water(res_names: NDArray[np.str_], idx: int) -> bool:
    """Check if an atom belongs to a water molecule."""
    return res_names[idx] in ("HOH", "WAT", "H2O", "DOD", "SOL")


def _check_geometry(
    atoms: AtomArray,
    donor_idx: int,
    acceptor_idx: int,
    valence_model: ValenceModel,
    params: _HBondParams,
) -> bool:
    """Check if donor-acceptor pair satisfies geometric constraints.

    Args:
        atoms: AtomArray with coordinates and bonds.
        donor_idx: Index of the donor atom.
        acceptor_idx: Index of the acceptor atom.
        valence_model: Pre-computed valence model.
        params: H-bond detection parameters.

    Returns:
        True if geometry is valid for an H-bond, False otherwise.
    """
    atom_names = atoms.atom_name
    ideal_geometry = valence_model.ideal_geometry

    # 1. Backbone exclusion
    if not params.include_backbone:
        if _is_backbone_atom(atom_names, donor_idx) and _is_backbone_atom(
            atom_names, acceptor_idx
        ):
            return False

    # 2. Donor angle validation
    don_angles, don_h_angles = calc_angles(
        atoms, donor_idx, acceptor_idx, ignore_hydrogens=params.ignore_hydrogens
    )

    donor_geom = AtomGeometry(ideal_geometry[donor_idx])
    ideal_don_angle = get_ideal_angle(donor_geom)

    # Check heavy atom angles: ALL must be within tolerance (two-sided)
    for don_angle in don_angles:
        if abs(ideal_don_angle - don_angle) > params.don_angle_dev_max:
            return False

    # Check hydrogen angles: at least ONE must point toward acceptor
    if don_h_angles and not any(
        angle < params.don_angle_dev_max for angle in don_h_angles
    ):
        return False

    # 3. Donor out-of-plane check (trigonal geometry only)
    if donor_geom == AtomGeometry.Trigonal:
        out_of_plane = calc_plane_angle(atoms, donor_idx, acceptor_idx)
        if out_of_plane is not None and out_of_plane > params.don_out_of_plane_max:
            return False

    # 4. Find closest hydrogen for acceptor angle calculation
    hydrogen_idx = donor_idx
    if not params.ignore_hydrogens and don_h_angles:
        hydrogen_idx = closest_hydrogen_index(atoms, donor_idx, acceptor_idx)

    # 5. Acceptor angle validation
    acc_angles, acc_h_angles = calc_angles(
        atoms, acceptor_idx, hydrogen_idx, ignore_hydrogens=params.ignore_hydrogens
    )

    acceptor_geom = AtomGeometry(ideal_geometry[acceptor_idx])
    ideal_acc_angle = get_ideal_angle(acceptor_geom)

    # Check acceptor angles: ONE-SIDED (large angles OK, small angles bad)
    # Reject if (ideal - measured) > max, i.e., measured is too small
    for acc_angle in acc_angles:
        if ideal_acc_angle - acc_angle > params.acc_angle_dev_max:
            return False

    for acc_h_angle in acc_h_angles:
        if ideal_acc_angle - acc_h_angle > params.acc_angle_dev_max:
            return False

    # 6. Acceptor out-of-plane check (trigonal geometry only)
    if acceptor_geom == AtomGeometry.Trigonal:
        out_of_plane = calc_plane_angle(atoms, acceptor_idx, donor_idx)
        if out_of_plane is not None and out_of_plane > params.acc_out_of_plane_max:
            return False

    return True


def test_hydrogen_bond(
    atoms: AtomArray,
    donor_idx: int,
    acceptor_idx: int,
    valence_model: ValenceModel,
    params: _HBondParams,
) -> HydrogenBond | None:
    """Test if a donor-acceptor pair forms a valid hydrogen bond.

    Args:
        atoms: AtomArray with coordinates and bonds.
        donor_idx: Index of the potential donor atom.
        acceptor_idx: Index of the potential acceptor atom.
        valence_model: Pre-computed valence model.
        params: H-bond detection parameters.

    Returns:
        HydrogenBond if valid, None otherwise.
    """
    coords = atoms.coord
    elements = atoms.element
    res_names = atoms.res_name

    # Calculate distance
    pos_donor = coords[donor_idx]
    pos_acceptor = coords[acceptor_idx]
    dist_sq = float(np.sum((pos_donor - pos_acceptor) ** 2))

    # Distance check (use sulfur distance if either atom is S)
    if elements[donor_idx] == S or elements[acceptor_idx] == S:
        max_dist_sq = params.sulfur_distance_max_sq
    else:
        max_dist_sq = params.distance_max_sq

    if dist_sq > max_dist_sq:
        return None

    # Water exclusion
    if not params.include_water:
        if _is_water(res_names, donor_idx) and _is_water(res_names, acceptor_idx):
            return None

    # Geometry check
    if not _check_geometry(atoms, donor_idx, acceptor_idx, valence_model, params):
        return None

    # Find the hydrogen involved
    hydrogen_idx = donor_idx
    if not params.ignore_hydrogens:
        hydrogen_idx = closest_hydrogen_index(atoms, donor_idx, acceptor_idx)

    return HydrogenBond(
        donor_idx=donor_idx,
        acceptor_idx=acceptor_idx,
        hydrogen_idx=hydrogen_idx,
        distance=float(np.sqrt(dist_sq)),
    )


def detect_hydrogen_bonds(
    atoms: AtomArray,
    distance_max: float = 3.5,
    sulfur_distance_max: float = 4.1,
    don_angle_dev_max: float = 45.0,
    acc_angle_dev_max: float = 45.0,
    don_out_of_plane_max: float = 45.0,
    acc_out_of_plane_max: float = 90.0,
    ignore_hydrogens: bool = True,
    include_backbone: bool = False,
    include_water: bool = False,
    valence_model: ValenceModel | None = None,
) -> list[HydrogenBond]:
    """Detect all hydrogen bonds in a structure.

    Args:
        atoms: AtomArray with coordinates and bonds set.
        distance_max: Maximum donor-acceptor distance (Å).
        sulfur_distance_max: Maximum distance when sulfur is involved (Å).
        don_angle_dev_max: Maximum deviation from ideal donor angle (degrees).
        acc_angle_dev_max: Maximum deviation from ideal acceptor angle (degrees).
        don_out_of_plane_max: Maximum out-of-plane angle for trigonal donor (degrees).
        acc_out_of_plane_max: Maximum out-of-plane angle for trigonal acceptor (degrees).
        ignore_hydrogens: If True, use implicit hydrogen positions.
        include_backbone: If True, include backbone-to-backbone H-bonds.
        include_water: If True, include water-to-water H-bonds.
        valence_model: Pre-computed valence model. If None, will be computed.

    Returns:
        List of detected HydrogenBond objects.
    """
    params = _HBondParams(
        distance_max=distance_max,
        sulfur_distance_max=sulfur_distance_max,
        don_angle_dev_max=radians(don_angle_dev_max),
        acc_angle_dev_max=radians(acc_angle_dev_max),
        don_out_of_plane_max=radians(don_out_of_plane_max),
        acc_out_of_plane_max=radians(acc_out_of_plane_max),
        ignore_hydrogens=ignore_hydrogens,
        include_backbone=include_backbone,
        include_water=include_water,
    )

    if valence_model is None:
        valence_model = compute_valence_model(atoms)

    # Detect donors and acceptors
    donor_mask = detect_hydrogen_donors(atoms, valence_model)
    acceptor_mask = detect_hydrogen_acceptors(atoms, valence_model)

    donor_indices = np.where(donor_mask)[0]
    acceptor_indices = np.where(acceptor_mask)[0]

    # Use the larger of the two distance cutoffs for initial filtering
    max_dist = max(params.distance_max, params.sulfur_distance_max)
    max_dist_sq = max_dist * max_dist

    coords = atoms.coord
    chain_ids = atoms.chain_id
    res_ids = atoms.res_id
    hbonds: list[HydrogenBond] = []

    # Test all donor-acceptor pairs within distance cutoff
    for donor_idx in donor_indices:
        pos_donor = coords[donor_idx]

        for acceptor_idx in acceptor_indices:
            # Skip if same atom (can happen if atom is both donor and acceptor)
            if donor_idx == acceptor_idx:
                continue

            # Skip if same residue
            if (
                chain_ids[donor_idx] == chain_ids[acceptor_idx]
                and res_ids[donor_idx] == res_ids[acceptor_idx]
            ):
                continue

            # Quick distance pre-filter
            pos_acceptor = coords[acceptor_idx]
            dist_sq = float(np.sum((pos_donor - pos_acceptor) ** 2))
            if dist_sq > max_dist_sq:
                continue

            # Full H-bond test
            hbond = test_hydrogen_bond(
                atoms, donor_idx, acceptor_idx, valence_model, params
            )
            if hbond is not None:
                hbonds.append(hbond)

    return hbonds
