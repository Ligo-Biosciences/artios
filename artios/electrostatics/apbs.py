"""APBS (Adaptive Poisson-Boltzmann Solver) wrapper.

Generates input files, runs APBS, reads .dx potential grids, and samples
potentials at arbitrary 3D points via trilinear interpolation.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass
class ECConfig:
    """Configuration for APBS Poisson-Boltzmann calculations."""

    fine_grid_spacing: float = 0.5  # Angstroms
    grid_padding: float = 20.0  # Angstroms beyond molecule extent
    protein_dielectric: float = 2.0
    solvent_dielectric: float = 78.54
    ionic_strength: float = 0.15  # Molar (150 mM, physiological)
    temperature: float = 298.15  # Kelvin
    probe_radius: float = 1.4  # Water probe, Angstroms


def _make_odd(n: int) -> int:
    """APBS requires odd grid dimensions."""
    return n if n % 2 == 1 else n + 1


def focused_grid_dims(
    all_coords: np.ndarray,
    binding_site_coords: np.ndarray,
    config: ECConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute focused grid dimensions for binding-site EC calculation.

    The fine grid covers only the binding site at high resolution (0.5 A),
    while the coarse grid covers the full protein for correct boundary
    conditions. Both grids share a center (the binding site centroid).

    Args:
        all_coords: All atom coordinates (full protein + ligand), shape (N, 3).
        binding_site_coords: Binding site coordinates (ligand + nearby protein
            atoms), shape (M, 3). Determines fine grid extent.
        config: APBS configuration.

    Returns:
        center: Grid center (binding site centroid), [3,].
        dime: Grid point counts [nx, ny, nz].
        fine_len: Fine grid physical length [lx, ly, lz].
        coarse_len: Coarse grid physical length [lx, ly, lz].
    """
    center = binding_site_coords.mean(axis=0)

    # ... fine grid: covers binding site at fine_grid_spacing
    fine_extent = np.ptp(binding_site_coords, axis=0) + 2 * config.grid_padding
    dime = np.array(
        [_make_odd(int(np.ceil(e / config.fine_grid_spacing))) for e in fine_extent]
    )
    fine_len = dime * config.fine_grid_spacing

    # ... coarse grid: must cover all atoms from the binding site center
    all_centered = all_coords - center
    half_extent = np.abs(all_centered).max(axis=0) + config.grid_padding
    coarse_len = 2 * half_extent
    # APBS requires cglen >= ~2x fglen for stable focusing
    coarse_len = np.maximum(coarse_len, 2 * fine_len)

    return center, dime, fine_len, coarse_len


def generate_apbs_input(
    pqr_path: Path,
    dx_prefix: str,
    center: np.ndarray,
    dime: np.ndarray,
    fine_len: np.ndarray,
    coarse_len: np.ndarray,
    config: ECConfig,
) -> str:
    """Generate APBS input file content.

    Uses mg-auto mode with sequential focusing: a coarse grid provides
    boundary conditions for a fine grid centered on the binding site.

    Args:
        pqr_path: Path to input PQR file.
        dx_prefix: Prefix for output .dx file (APBS appends suffix).
        center: Grid center (x, y, z).
        dime: Grid point counts (nx, ny, nz).
        fine_len: Fine grid physical length (lx, ly, lz) in Angstroms.
        coarse_len: Coarse grid physical length (lx, ly, lz) in Angstroms.
        config: APBS configuration.

    Returns:
        APBS input file content as string.
    """
    cx, cy, cz = center
    nx, ny, nz = dime

    return f"""\
read
    mol pqr {pqr_path}
end

elec name solvation
    mg-auto
    dime {nx} {ny} {nz}
    cglen {coarse_len[0]:.1f} {coarse_len[1]:.1f} {coarse_len[2]:.1f}
    fglen {fine_len[0]:.1f} {fine_len[1]:.1f} {fine_len[2]:.1f}
    cgcent {cx:.3f} {cy:.3f} {cz:.3f}
    fgcent {cx:.3f} {cy:.3f} {cz:.3f}
    mol 1
    lpbe
    bcfl sdh
    ion charge 1 conc {config.ionic_strength:.3f} radius 2.0
    ion charge -1 conc {config.ionic_strength:.3f} radius 1.8
    pdie {config.protein_dielectric:.1f}
    sdie {config.solvent_dielectric:.2f}
    srfm smol
    chgm spl2
    sdens 10.0
    srad {config.probe_radius:.1f}
    swin 0.3
    temp {config.temperature:.2f}
    write pot dx {dx_prefix}
end

quit
"""


def run_apbs(input_path: Path, work_dir: Path, dx_prefix: str) -> Path:
    """Run APBS and return path to the output .dx file.

    Args:
        input_path: Path to APBS .in file.
        work_dir: Working directory for APBS execution.
        dx_prefix: Prefix used in the APBS input's `write pot dx` directive.

    Returns:
        Path to the output .dx potential grid file.
    """
    subprocess.run(
        ["apbs", str(input_path)],
        capture_output=True,
        text=True,
        cwd=str(work_dir),
        check=True,
    )

    # ... APBS writes <prefix>.dx (or <prefix>-PE0.dx for parallel) in work_dir
    dx_files = list(work_dir.glob(f"{dx_prefix}*.dx"))
    assert len(dx_files) >= 1, (
        f"APBS produced no .dx files matching '{dx_prefix}*.dx' in {work_dir}. "
        f"Contents: {[p.name for p in work_dir.iterdir()]}"
    )
    return dx_files[0]


def read_dx_potential(
    dx_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read an APBS .dx potential grid into numpy arrays.

    Args:
        dx_path: Path to .dx file.

    Returns:
        potential: 3D array of potentials in kT/e units.
        origin: Grid origin [3,].
        delta: Grid spacing [3,] (diagonal of delta matrix).
    """
    from gridData import Grid

    grid = Grid(str(dx_path))
    origin = np.array(grid.origin)
    delta = np.asarray(grid.delta, dtype=np.float64)
    return grid.grid, origin, delta


def sample_potential_at_points(
    potential: np.ndarray,
    origin: np.ndarray,
    delta: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Sample potential values at arbitrary 3D points via trilinear interpolation.

    Args:
        potential: 3D array of potential values.
        origin: Grid origin [3,].
        delta: Grid spacing [3,].
        points: Coordinates to sample at, shape (N, 3).

    Returns:
        Interpolated potential values, shape (N,).
    """
    # ... convert world coordinates to fractional grid indices
    fractional = (points - origin) / delta  # (N, 3)

    # ... scipy expects (3, N)
    return map_coordinates(
        potential,
        fractional.T,
        order=1,
        mode="constant",
        cval=0.0,
    )
