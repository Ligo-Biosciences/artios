"""Hydrogen bond donor and acceptor feature detection.

Identifies atoms that can participate in hydrogen bonds based on
element type, charge, geometry, and local environment.

Based on Mol* implementation.
Reference: https://github.com/molstar/molstar
"""

import numpy as np
from biotite.structure import AtomArray
from numpy.typing import NDArray

from protenix.common import residue_names as residues
from protenix.common.elements import N, O, S
from protenix.contacts.geometry import AtomGeometry
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
