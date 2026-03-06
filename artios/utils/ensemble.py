"""ANM-based conformational ensemble generation.

Builds an Anisotropic Network Model on Cα atoms, computes normal modes,
and samples conformations by displacing along low-frequency modes with
physically calibrated amplitudes.
"""

import biotite.structure as struc
import numpy as np
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

    eigvecs, eigvals = build_anm_hessian(ca_coords)
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
