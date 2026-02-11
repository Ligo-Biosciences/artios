from collections import Counter

import biotite.structure as struc
import numpy as np


def compute_mean_sasa(atom_array: struc.AtomArray) -> float:
    """
    Calculate mean solvent accessible surface area (SASA) per residue.

    Args:
        atom_array: Biotite AtomArray structure

    Returns:
        Mean SASA across all residues
    """
    # Calculate atom-level SASA
    atom_sasa = struc.sasa(atom_array, vdw_radii="Single")
    return np.nanmean(atom_sasa)


def compute_radius_of_gyration(atom_array: struc.AtomArray) -> float:
    """
    Calculate radius of gyration (Rg).

    Args:
        atom_array: Biotite AtomArray structure

    Returns:
        Radius of gyration in Angstroms

    Notes:
        This is not normalized.
    """
    coords = atom_array.coord
    center = coords.mean(axis=0)
    distances_squared = np.sum((coords - center) ** 2, axis=1)
    rg = np.sqrt(np.mean(distances_squared))
    return float(rg)


def compute_plddt_statistics(atom_array: struc.AtomArray) -> dict[str, float]:
    """
    Calculate comprehensive pLDDT statistics.
    We access the pLDDT from the b_factor annotation.

    Args:
        atom_array: Biotite AtomArray structure

    Returns:
        Dictionary containing:
            - mean: Mean pLDDT score
            - median: Median pLDDT score
            - std: Standard deviation of pLDDT scores
            - top_80_mean: Mean of top 80% pLDDT scores (excludes bottom 20%)
            - p25: 25th percentile
            - p75: 75th percentile
    """
    plddt_scores = atom_array.b_factor

    # Basic statistics
    mean_plddt = float(np.mean(plddt_scores))
    median_plddt = float(np.median(plddt_scores))
    std_plddt = float(np.std(plddt_scores))

    # Top 80% mean (excludes bottom 20%)
    sorted_scores = np.sort(plddt_scores)
    top_80_idx = int(len(sorted_scores) * 0.2)  # Start from 20th percentile
    top_80_mean = float(np.mean(sorted_scores[top_80_idx:]))

    # Percentiles
    p25 = float(np.percentile(plddt_scores, 25))
    p75 = float(np.percentile(plddt_scores, 75))

    return {
        "mean": mean_plddt,
        "median": median_plddt,
        "std": std_plddt,
        "top_80_mean": top_80_mean,
        "p25": p25,
        "p75": p75,
    }


def compute_sse(atom_array: struc.AtomArray) -> float:
    """
    Calculates the secondary structure splits.
    """
    sse_array = struc.annotate_sse(atom_array)
    sse_counter = Counter(sse_array.tolist())

    total_size = sse_counter.total()
    return {
        "alpha": sse_counter.get("a", 0) / total_size,
        "beta": sse_counter.get("b", 0) / total_size,
        "coil": sse_counter.get("c", 0) / total_size,
    }
