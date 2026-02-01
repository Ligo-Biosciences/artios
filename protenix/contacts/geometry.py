"""Atom geometry types and associated ideal angles.

Based on VSEPR theory and Mol* implementation.
Reference: https://github.com/molstar/molstar
"""

from enum import IntEnum
from math import radians


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
