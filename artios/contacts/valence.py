"""Valence model computation for molecular structures.

Computes charge, implicit hydrogens, total hydrogens, and ideal geometry
for each atom in a structure. Based on Mol* implementation.

Reference:
    - Mol*: https://github.com/molstar/molstar
    - OpenEye hydrogen count model:
      https://docs.eyesopen.com/toolkits/python/oechemtk/valence.html
"""

from dataclasses import dataclass

import numpy as np
from biotite.structure import AtomArray, BondList, BondType
from numpy.typing import NDArray

from artios.common.elements import (
    ALKALI_METALS,
    ALKALINE_EARTH_METALS,
    HALOGENS,
    METALS,
    C,
    H,
    N,
    O,
    P,
    S,
)
from artios.contacts.geometry import AtomGeometry, assign_geometry


@dataclass
class ValenceModel:
    """Valence model containing computed properties for each atom.

    Attributes:
        charge: Computed formal charge for each atom. Shape: (n_atoms,)
        implicit_h: Number of implicit (missing) hydrogens. Shape: (n_atoms,)
        total_h: Total hydrogens (implicit + explicit bonds). Shape: (n_atoms,)
        ideal_geometry: Ideal geometry enum value for each atom. Shape: (n_atoms,)
    """

    charge: NDArray[np.int8]
    implicit_h: NDArray[np.int8]
    total_h: NDArray[np.int8]
    ideal_geometry: NDArray[np.int8]

    def __len__(self) -> int:
        return len(self.charge)


def _bond_order(bond_type: BondType) -> int:
    """Get the bond order from a BondType."""
    match bond_type:
        case BondType.SINGLE | BondType.AROMATIC_SINGLE | BondType.AROMATIC:
            return 1
        case BondType.DOUBLE | BondType.AROMATIC_DOUBLE:
            return 2
        case BondType.TRIPLE | BondType.AROMATIC_TRIPLE:
            return 3
        case BondType.QUADRUPLE:
            return 4
        case BondType.COORDINATION:
            return 0
        case _:
            return 1


def _is_covalent_bond(bond_type: BondType) -> bool:
    """Check if a bond type is covalent (not coordination)."""
    return bond_type != BondType.COORDINATION


def _get_bond_count(bonds: BondList, atom_idx: int) -> int:
    """Count the number of covalent bonds to an atom.

    Note: This counts bonds, not bond order. A double bond counts as 1 bond.
    For sum of bond orders, use _get_explicit_valence.
    """
    bonded = bonds.get_bonds(atom_idx)
    if bonded is None or len(bonded) == 0:
        return 0
    _, bond_types = bonded
    return sum(1 for bt in bond_types if _is_covalent_bond(BondType(bt)))


def _get_explicit_valence(bonds: BondList, atom_idx: int) -> int:
    """Calculate explicit valence (sum of bond orders) for an atom."""
    bonded = bonds.get_bonds(atom_idx)
    if bonded is None or len(bonded) == 0:
        return 0
    _, bond_types = bonded
    return sum(
        _bond_order(BondType(bt))
        for bt in bond_types
        if _is_covalent_bond(BondType(bt))
    )


def _count_bonds_to_element(
    bonds: BondList,
    elements: NDArray[np.str_],
    atom_idx: int,
    target_element: str,
) -> int:
    """Count bonds from an atom to atoms of a specific element."""
    bonded = bonds.get_bonds(atom_idx)
    if bonded is None or len(bonded) == 0:
        return 0
    bonded_indices, bond_types = bonded
    count = 0
    for bidx, bt in zip(bonded_indices, bond_types):
        if _is_covalent_bond(BondType(bt)) and elements[bidx] == target_element:
            count += 1
    return count


def _is_conjugated(
    bonds: BondList,
    elements: NDArray[np.str_],
    atom_idx: int,
) -> bool:
    """Check if an atom is involved in a pi system (conjugated).

    An atom is conjugated if:
    - It forms a double/triple bond, OR
    - It's N or O adjacent to a double bond (except P=O or S=O systems)
    """
    element = elements[atom_idx]
    is_hetero = element in (N, O)

    # N,O with degree 4 cannot be conjugated
    if is_hetero and _get_bond_count(bonds, atom_idx) == 4:
        return False

    bonded = bonds.get_bonds(atom_idx)
    if bonded is None or len(bonded) == 0:
        return False

    bonded_indices, bond_types = bonded

    for bidx, bt in zip(bonded_indices, bond_types):
        bond_type = BondType(bt)
        if not _is_covalent_bond(bond_type):
            continue

        # Direct double/triple bond
        if _bond_order(bond_type) > 1:
            return True

        # For N/O: check if neighbor has a double bond
        if is_hetero:
            neighbor_element = elements[bidx]
            neighbor_bonds = bonds.get_bonds(bidx)
            if neighbor_bonds is None:
                continue
            nb_indices, nb_types = neighbor_bonds

            for nb_idx, nb_bt in zip(nb_indices, nb_types):
                if _bond_order(BondType(nb_bt)) > 1:
                    # Exclude P=O and S=O (keeps sulfonamide N as sp3)
                    if neighbor_element in (P, S) and elements[nb_idx] == O:
                        continue
                    return True

    return False


def _has_bonded_metal(
    bonds: BondList,
    elements: NDArray[np.str_],
    atom_idx: int,
) -> bool:
    """Check if an atom is bonded to a metal."""
    bonded = bonds.get_bonds(atom_idx)
    if bonded is None or len(bonded) == 0:
        return False
    bonded_indices, _ = bonded
    return any(elements[bidx] in METALS for bidx in bonded_indices)


def _calculate_h_and_charge(
    bonds: BondList,
    elements: NDArray[np.str_],
    formal_charges: NDArray[np.int_],
    atom_idx: int,
    assign_charge: bool,
    assign_h: bool,
) -> tuple[int, int, int, AtomGeometry]:
    """Calculate charge, implicit H count, total H count, and geometry for one atom."""
    element = elements[atom_idx]
    charge = int(formal_charges[atom_idx])
    degree = _get_bond_count(bonds, atom_idx)
    valence = _get_explicit_valence(bonds, atom_idx)
    hydrogen_count = _count_bonds_to_element(bonds, elements, atom_idx, H)

    conjugated = _is_conjugated(bonds, elements, atom_idx)
    multi_bond = valence - degree > 0

    implicit_h = 0
    geom = AtomGeometry.Unknown

    if element == H:
        if assign_charge:
            if degree == 0:
                charge = 1
                geom = AtomGeometry.Spherical
            elif degree == 1:
                charge = 0
                geom = AtomGeometry.Terminal
        else:
            geom = AtomGeometry.Terminal if degree >= 1 else AtomGeometry.Spherical

    elif element == C:
        if assign_charge:
            charge = 0  # Carbon is always neutral
        if assign_h:
            implicit_h = max(0, 4 - valence - abs(charge))
        geom = assign_geometry(degree + implicit_h + max(0, -charge))

    elif element == N:
        if assign_charge:
            if not assign_h:
                charge = valence - 3
            elif conjugated and valence < 4:
                if degree - hydrogen_count == 1 and valence - hydrogen_count == 2:
                    charge = 1
                else:
                    charge = 0
            else:
                if _has_bonded_metal(bonds, elements, atom_idx):
                    charge = 0
                elif _count_bonds_to_element(bonds, elements, atom_idx, S) > 0:
                    charge = 0
                else:
                    charge = 1 if valence >= 4 else 0

        if assign_h:
            implicit_h = max(0, 3 - valence + charge)

        if conjugated and not multi_bond:
            geom = assign_geometry(degree + implicit_h - charge)
        else:
            geom = assign_geometry(degree + implicit_h + 1 - charge)

    elif element == O:
        if assign_charge:
            if not assign_h:
                charge = valence - 2
            elif valence == 1:
                # Check for carboxylate-like pattern
                bonded = bonds.get_bonds(atom_idx)
                if bonded is not None and len(bonded[0]) > 0:
                    for bidx in bonded[0]:
                        neighbor_bonds = bonds.get_bonds(bidx)
                        if neighbor_bonds is None:
                            continue
                        for nb_idx, nb_bt in zip(neighbor_bonds[0], neighbor_bonds[1]):
                            if nb_idx != atom_idx:
                                if (
                                    elements[nb_idx] == O
                                    and _bond_order(BondType(nb_bt)) == 2
                                ):
                                    charge = -1
                                    break
                        if charge == -1:
                            break

        if assign_h:
            implicit_h = max(0, 2 - valence + charge)

        if conjugated and not multi_bond:
            geom = assign_geometry(degree + implicit_h - charge + 1)
        else:
            geom = assign_geometry(degree + implicit_h - charge + 2)

    elif element == S:
        if assign_charge:
            if not assign_h:
                if (
                    valence <= 3
                    and _count_bonds_to_element(bonds, elements, atom_idx, O) == 0
                ):
                    charge = valence - 2
                else:
                    charge = 0
        if assign_h:
            if valence < 2:
                implicit_h = max(0, 2 - valence + charge)

        if valence <= 3:
            geom = assign_geometry(degree + implicit_h - charge + 2)
        else:
            geom = AtomGeometry.Unknown

    elif element in HALOGENS:
        if assign_charge:
            charge = valence - 1
        geom = AtomGeometry.Terminal if degree >= 1 else AtomGeometry.Spherical

    elif element in ALKALI_METALS:
        if assign_charge:
            charge = 1 - valence
        geom = AtomGeometry.Spherical

    elif element in ALKALINE_EARTH_METALS:
        if assign_charge:
            charge = 2 - valence
        geom = AtomGeometry.Spherical

    total_h = implicit_h + hydrogen_count
    return charge, implicit_h, total_h, geom


def compute_valence_model(
    atoms: AtomArray,
    assign_charge: bool = True,
    assign_h: bool = True,
) -> ValenceModel:
    """Compute the valence model for a structure.

    Args:
        atoms: Biotite AtomArray with bonds. Must have `bonds` attribute set.
        assign_charge: If True, compute charges. If False, use formal charges
            from the structure (requires `charge` annotation to be present).
        assign_h: If True, compute implicit hydrogens for atoms without explicit H.

    Returns:
        ValenceModel with arrays for each property.

    Raises:
        ValueError: If the AtomArray has no bonds defined.
        ValueError: If assign_charge=False but no charge annotation is present.
    """
    if atoms.bonds is None:
        raise ValueError(
            "AtomArray must have bonds defined. Use biotite.structure.connect_via_residue_names() "
            "or similar to add bonds."
        )

    n_atoms = len(atoms)
    bonds = atoms.bonds
    elements = atoms.element

    # Get formal charges
    if not assign_charge:
        if not hasattr(atoms, "charge") or atoms.charge is None:
            raise ValueError(
                "assign_charge=False requires the AtomArray to have a 'charge' annotation. "
                "Either set assign_charge=True or add charges to the structure."
            )
        formal_charges = atoms.charge
    else:
        # When computing charges, use existing as starting point if available
        if hasattr(atoms, "charge") and atoms.charge is not None:
            formal_charges = atoms.charge
        else:
            formal_charges = np.zeros(n_atoms, dtype=np.int_)

    # Allocate output arrays
    charge = np.zeros(n_atoms, dtype=np.int8)
    implicit_h = np.zeros(n_atoms, dtype=np.int8)
    total_h = np.zeros(n_atoms, dtype=np.int8)
    ideal_geometry = np.zeros(n_atoms, dtype=np.int8)

    for i in range(n_atoms):
        c, ih, th, geom = _calculate_h_and_charge(
            bonds=bonds,
            elements=elements,
            formal_charges=formal_charges,
            atom_idx=i,
            assign_charge=assign_charge,
            assign_h=assign_h,
        )
        charge[i] = c
        implicit_h[i] = ih
        total_h[i] = th
        ideal_geometry[i] = geom.value

    return ValenceModel(
        charge=charge,
        implicit_h=implicit_h,
        total_h=total_h,
        ideal_geometry=ideal_geometry,
    )


def add_valence_annotations(
    atoms: AtomArray,
    assign_charge: bool = True,
    assign_h: bool = True,
) -> AtomArray:
    """Compute valence model and add as annotations to the AtomArray.

    Adds the following annotations:
    - `valence_charge`: Computed formal charge
    - `valence_implicit_h`: Number of implicit hydrogens
    - `valence_total_h`: Total hydrogens (implicit + explicit)
    - `valence_geometry`: Ideal geometry (as AtomGeometry enum value)

    Args:
        atoms: Biotite AtomArray with bonds.
        assign_charge: If True, compute charges.
        assign_h: If True, compute implicit hydrogens.

    Returns:
        The same AtomArray with annotations added.
    """
    model = compute_valence_model(atoms, assign_charge, assign_h)

    atoms.set_annotation("valence_charge", model.charge)
    atoms.set_annotation("valence_implicit_h", model.implicit_h)
    atoms.set_annotation("valence_total_h", model.total_h)
    atoms.set_annotation("valence_geometry", model.ideal_geometry)

    return atoms
