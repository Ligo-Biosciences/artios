"""Electrostatic complementarity (EC) scoring for protein-ligand interfaces.

Public API:

    from artios.electrostatics import compute_ec, ECResult, ECConfig
    from artios.electrostatics import split_complex, interface_surface_points, interface_atoms

For low-level access (custom charge pipelines, direct APBS control):

    from artios.electrostatics.apbs import generate_apbs_input, run_apbs, ...
    from artios.electrostatics.pqr import write_pqr, generate_protein_pqr, ...
"""

from artios.electrostatics.apbs import ECConfig, focused_grid_dims
from artios.electrostatics.core import (
    ECResult,
    compute_ec,
    interface_atoms,
    split_complex,
)
from artios.electrostatics.surface import interface_surface_points

__all__ = [
    "compute_ec",
    "ECConfig",
    "ECResult",
    "split_complex",
    "interface_surface_points",
    "interface_atoms",
    "focused_grid_dims",
]
