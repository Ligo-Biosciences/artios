"""ANM-based conformational ensemble generation.

Builds an Anisotropic Network Model on Cα atoms, computes normal modes,
and samples conformations by displacing along low-frequency modes with
physically calibrated amplitudes.
"""

from typing import Optional

import biotite.structure as struc
import numpy as np
import torch
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform

# Spring constant calibrated as median across 55 X-ray structures
# by fitting ANM-predicted B-factors to experimental B-factors.
# Can be tuned down (e.g. 0.075) to produce larger ensembles (~0.8 Å mean RMSD).
GAMMA = 0.2388  # kcal/mol/Å²
KB_T_300K = 0.6  # kcal/mol


def build_anm_hessian(
    ca_coords: np.ndarray, cutoff: float = 15.0
) -> tuple[np.ndarray, np.ndarray]:
    """Build the ANM Hessian and return non-trivial eigenvectors/eigenvalues.

    Args:
        ca_coords: Cα coordinates, shape (n_ca, 3).
        cutoff: Distance cutoff in Å for spring connections.

    Returns:
        eigvecs: Eigenvectors of shape (n_ca * 3, n_modes).
        eigvals: Eigenvalues of shape (n_modes,).
        Both exclude the 6 trivial (zero-frequency) modes.
    """
    n = len(ca_coords)
    dists = squareform(pdist(ca_coords))
    contacts = (dists < cutoff) & (dists > 0)

    hessian = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(i + 1, n):
            if not contacts[i, j]:
                continue
            diff = ca_coords[i] - ca_coords[j]
            d2 = dists[i, j] ** 2
            block = -np.outer(diff, diff) / d2
            hessian[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = block
            hessian[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] = block
            hessian[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] -= block
            hessian[3 * j : 3 * j + 3, 3 * j : 3 * j + 3] -= block

    eigvals, eigvecs = eigh(hessian)

    # Skip first 6 trivial modes (translation + rotation)
    return eigvecs[:, 6:], eigvals[6:]


def generate_ensemble(
    atom_array: struc.AtomArray,
    n_conformations: int = 20,
    n_modes: int = 20,
    temperature: float = 300.0,
    gamma: float = GAMMA,
    cutoff: float = 15.0,
    seed: int = 42,
) -> struc.AtomArrayStack:
    """Generate an ANM conformational ensemble.

    Args:
        atom_array: Full structure (all atoms). Backbone and Cα are selected internally.
        n_conformations: Number of conformations to sample.
        n_modes: Number of lowest-frequency modes to use.
        temperature: Temperature in Kelvin.
        gamma: Spring constant in kcal/mol/Å². Default is the B-factor-calibrated value.
        seed: Random seed.

    Returns:
        AtomArrayStack of backbone conformations with shape (n_conformations, n_bb, 3).
    """
    rng = np.random.default_rng(seed)

    bb_mask = (
        (atom_array.atom_name == "N")
        | (atom_array.atom_name == "CA")
        | (atom_array.atom_name == "C")
        | (atom_array.atom_name == "O")
    ) & ~atom_array.hetero
    ca_mask = (atom_array.atom_name == "CA") & ~atom_array.hetero

    backbone = atom_array[bb_mask]
    ca_coords = atom_array.coord[ca_mask]

    eigvecs, eigvals = build_anm_hessian(ca_coords, cutoff=cutoff)
    eigvecs = eigvecs[:, :n_modes]
    eigvals = eigvals[:n_modes]

    # Physical mode variances: σ²_k = k_B T / (γ λ_k)
    kbt = KB_T_300K * (temperature / 300.0)
    mode_variances = kbt / (gamma * eigvals)

    # Sample mode coefficients
    coefficients = rng.normal(
        0, np.sqrt(mode_variances), size=(n_conformations, n_modes)
    )
    ca_displacements = (coefficients @ eigvecs.T).reshape(
        n_conformations, len(ca_coords), 3
    )

    # Map Cα displacements to backbone atoms via residue index
    ca_res_ids = atom_array.res_id[ca_mask]
    bb_res_ids = backbone.res_id
    ca_rid_to_idx = {rid: i for i, rid in enumerate(ca_res_ids)}
    bb_to_ca = np.array([ca_rid_to_idx[rid] for rid in bb_res_ids])

    arrays = []
    for i in range(n_conformations):
        frame = backbone.copy()
        frame.coord = backbone.coord + ca_displacements[i][bb_to_ca]
        arrays.append(frame)

    return struc.stack(arrays)


def sample_anm_displacements(
    ca_coords: torch.Tensor,
    n_conformations: int,
    n_modes: int = 20,
    gamma: float = GAMMA,
    cutoff: float = 15.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample ANM displacements for Cα atoms (PyTorch version).

    Builds the ANM Hessian, eigendecomposes it, and samples displacements
    along the lowest-frequency normal modes with physically calibrated amplitudes.

    Args:
        ca_coords: Cα coordinates, shape (n_ca, 3).
        n_conformations: Number of displacement sets to sample.
        n_modes: Number of lowest-frequency modes to use.
        gamma: Spring constant in kcal/mol/Å².
        cutoff: Distance cutoff in Å for spring connections.
        generator: Optional torch Generator for reproducibility.

    Returns:
        Displacements of shape (n_conformations, n_ca, 3).
    """
    n = ca_coords.shape[0]
    device = ca_coords.device
    input_dtype = ca_coords.dtype

    # Need at least 7 atoms for non-trivial modes (6 trivial = translation + rotation)
    if n <= 6 or gamma <= 0:
        return torch.zeros(n_conformations, n, 3, device=device, dtype=input_dtype)

    n_modes = min(n_modes, 3 * n - 6)

    # Work in float32 for numerical stability in eigendecomposition
    coords = ca_coords.float()

    # Pairwise distances and contact mask
    dists = torch.cdist(coords, coords)  # [n, n]
    contacts = (dists < cutoff) & (dists > 0)

    # Direction vectors and 3x3 off-diagonal blocks
    diffs = coords[:, None, :] - coords[None, :, :]  # [n, n, 3]
    dists_sq = dists.pow(2).clamp(min=1e-12)
    blocks = (
        -(diffs[..., :, None] * diffs[..., None, :]) / dists_sq[:, :, None, None]
    )  # [n, n, 3, 3]
    blocks = blocks * contacts[:, :, None, None]

    # Build Hessian: off-diagonal blocks
    hessian = blocks.permute(0, 2, 1, 3).reshape(3 * n, 3 * n)

    # Diagonal blocks: H_ii = -sum_{j != i} H_ij
    diag_blocks = -blocks.sum(dim=1)  # [n, 3, 3]
    idx = torch.arange(n, device=device)
    for a in range(3):
        for b in range(3):
            hessian[3 * idx + a, 3 * idx + b] = diag_blocks[:, a, b]

    # Eigendecomposition, skip 6 trivial modes
    eigvals, eigvecs = torch.linalg.eigh(hessian)
    eigvals = eigvals[6 : 6 + n_modes]
    eigvecs = eigvecs[:, 6 : 6 + n_modes]

    # Mode variances: σ²_k = k_B T / (γ λ_k)
    # Clamp eigenvalues to avoid numerical issues (near-zero or slightly negative
    # from floating-point noise in the eigendecomposition)
    eigvals = eigvals.clamp(min=1e-4)
    mode_variances = KB_T_300K / (gamma * eigvals)

    # Sample coefficients and compute displacements
    coeffs = (
        torch.randn(
            n_conformations,
            n_modes,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * mode_variances.sqrt()
    )
    displacements = (coeffs @ eigvecs.T).reshape(n_conformations, n, 3)

    return displacements.to(input_dtype)
