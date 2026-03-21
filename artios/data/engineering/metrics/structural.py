from collections import Counter
from dataclasses import dataclass

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


@dataclass(frozen=True)
class FoldQuality:
    """Fold quality metrics for a protein structure or fragment.

    Attributes:
        lr_cd: Long-range CA contact density (CA pairs < 8A, |i-j| >= 12, per residue).
        dssp_coil: DSSP coil fraction (structural complexity — loops/turns).
        dssp_helix_align: Mean pairwise |cos(angle)| of nearby DSSP helices
            (high = repetitive parallel helices, low = diverse orientations).
        score: P(good) from the logistic model. Higher = more designable.
    """

    lr_cd: float
    dssp_coil: float
    dssp_helix_align: float
    score: float

    # Logistic regression trained on 3250 manually classified structures
    # (2585 good, 665 bad). Features: lr_cd, dssp_coil, dssp_helix_align.
    _MEANS = np.array([1.6047, 0.5176, 0.6085])
    _STDS = np.array([0.4555, 0.1382, 0.1360])
    _COEFS = np.array([0.8588, 0.8901, -0.3422])
    _INTERCEPT = 2.0351
    # Threshold retaining 90% of good structures, removing 67.7% of bad.
    DEFAULT_THRESHOLD = 0.6477

    @staticmethod
    def compute_score(lr_cd: float, dssp_coil: float, dssp_helix_align: float) -> float:
        """Compute P(good) from the 3-feature logistic model."""
        x = np.array([lr_cd, dssp_coil, dssp_helix_align])
        logit = float(
            (x - FoldQuality._MEANS) / FoldQuality._STDS @ FoldQuality._COEFS
            + FoldQuality._INTERCEPT
        )
        return 1.0 / (1.0 + np.exp(-logit))


def compute_fold_quality(atom_array: struc.AtomArray) -> FoldQuality:
    """Compute fold quality metrics for filtering repetitive/simplistic structures.

    Three metrics that together capture backbone designability:
        - lr_cd: long-range CA contact density (packing quality)
        - dssp_coil: DSSP coil fraction (structural complexity)
        - dssp_helix_align: mean pairwise alignment of nearby DSSP helices
          (high = repetitive parallel helices, low = diverse orientations)

    Args:
        atom_array: Biotite AtomArray structure.

    Returns:
        FoldQuality dataclass with metrics and combined score.
    """
    from scipy.spatial.distance import pdist, squareform

    ca_mask = atom_array.atom_name == "CA"
    ca_atoms = atom_array[ca_mask]
    coords = ca_atoms.coord
    n = len(coords)

    if n < 10:
        return FoldQuality(
            lr_cd=0.0,
            dssp_coil=1.0,
            dssp_helix_align=0.5,
            score=FoldQuality.compute_score(0.0, 1.0, 0.5),
        )

    # Long-range contact density
    dm = squareform(pdist(coords))
    seq_sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    lr_cd = float(((dm < 8.0) & (seq_sep >= 12)).sum() / 2 / n)

    # DSSP
    sse = struc.annotate_sse(atom_array)
    dssp_coil = float((sse == "c").sum() / len(sse))

    # Helix alignment
    is_helix = sse == "a"
    helix_runs: list[tuple[int, int]] = []
    start = None
    for i in range(len(sse)):
        if is_helix[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                helix_runs.append((start, i))
                start = None
    if start is not None:
        helix_runs.append((start, len(sse)))

    dssp_helix_align = 0.5
    if len(helix_runs) >= 3:
        ca_res_ids = ca_atoms.res_id
        sse_res_ids = atom_array.res_id

        dirs: list[np.ndarray] = []
        mids: list[np.ndarray] = []
        for s, e in helix_runs:
            if e - s < 5:
                continue
            helix_res = set(sse_res_ids[s:e].tolist())
            helix_ca_mask = np.isin(ca_res_ids, list(helix_res))
            helix_ca = coords[helix_ca_mask]
            if len(helix_ca) < 5:
                continue
            d = helix_ca[-1] - helix_ca[0]
            norm = np.linalg.norm(d)
            if norm > 1e-6:
                dirs.append(d / norm)
                mids.append(helix_ca.mean(axis=0))

        if len(dirs) >= 3:
            als: list[float] = []
            for i in range(len(dirs)):
                for j in range(i + 1, len(dirs)):
                    if np.linalg.norm(mids[i] - mids[j]) < 20.0:
                        als.append(float(abs(np.dot(dirs[i], dirs[j]))))
            if als:
                dssp_helix_align = float(np.mean(als))

    score = FoldQuality.compute_score(lr_cd, dssp_coil, dssp_helix_align)
    return FoldQuality(
        lr_cd=lr_cd,
        dssp_coil=dssp_coil,
        dssp_helix_align=dssp_helix_align,
        score=score,
    )
