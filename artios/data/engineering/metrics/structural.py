from collections import Counter
from dataclasses import dataclass

import biotite.structure as struc
import numpy as np
from scipy.spatial.distance import pdist, squareform

from artios.common.constants import HYDROPHOBIC_RESIDUES
from artios.data.engineering.metrics.dssp import compute_dssp


def _extract_helix_axes(
    sse: np.ndarray,
    atom_array: struc.AtomArray,
    min_helix_len: int = 5,
    max_midpoint_dist: float = 20.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract helix direction vectors and midpoints from SSE annotation.

    Returns (dirs, mids) where dirs[i] is the unit direction vector of helix i
    and mids[i] is its CA centroid. Only helices with >= min_helix_len residues
    and >= min_helix_len CA atoms are included.
    """
    ca_mask = atom_array.atom_name == "CA"
    ca_atoms = atom_array[ca_mask]
    coords = ca_atoms.coord

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

    ca_res_ids = ca_atoms.res_id
    sse_res_ids = atom_array.res_id

    dirs: list[np.ndarray] = []
    mids: list[np.ndarray] = []
    for s, e in helix_runs:
        if e - s < min_helix_len:
            continue
        helix_res = set(sse_res_ids[s:e].tolist())
        helix_ca_mask = np.isin(ca_res_ids, list(helix_res))
        helix_ca = coords[helix_ca_mask]
        if len(helix_ca) < min_helix_len:
            continue
        d = helix_ca[-1] - helix_ca[0]
        norm = np.linalg.norm(d)
        if norm > 1e-6:
            dirs.append(d / norm)
            mids.append(helix_ca.mean(axis=0))

    return dirs, mids


def _pairwise_helix_cosines(
    dirs: list[np.ndarray],
    mids: list[np.ndarray],
    max_midpoint_dist: float = 20.0,
) -> list[float]:
    """Compute |cos(angle)| for all helix pairs with midpoints within max_midpoint_dist."""
    als: list[float] = []
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            if np.linalg.norm(mids[i] - mids[j]) < max_midpoint_dist:
                als.append(float(abs(np.dot(dirs[i], dirs[j]))))
    return als


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


def compute_sse(atom_array: struc.AtomArray, sse: np.ndarray | None = None) -> float:
    """
    Calculates the secondary structure splits.
    """
    if sse is None:
        sse = compute_dssp(atom_array)
    sse_counter = Counter(sse.tolist())

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


def compute_fold_quality(
    atom_array: struc.AtomArray, sse: np.ndarray | None = None
) -> FoldQuality:
    """Compute fold quality metrics for filtering repetitive/simplistic structures.

    Three metrics that together capture backbone designability:
        - lr_cd: long-range CA contact density (packing quality)
        - dssp_coil: DSSP coil fraction (structural complexity)
        - dssp_helix_align: mean pairwise alignment of nearby DSSP helices
          (high = repetitive parallel helices, low = diverse orientations)

    Args:
        atom_array: Biotite AtomArray structure.
        sse: Pre-computed SSE array from compute_dssp(). Computed if None.

    Returns:
        FoldQuality dataclass with metrics and combined score.
    """
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
    if sse is None:
        sse = compute_dssp(atom_array)
    dssp_coil = float((sse == "c").sum() / len(sse))

    # Helix alignment
    dirs, mids = _extract_helix_axes(sse, atom_array)
    dssp_helix_align = 0.5
    if len(dirs) >= 3:
        als = _pairwise_helix_cosines(dirs, mids)
        if als:
            dssp_helix_align = float(np.mean(als))

    score = FoldQuality.compute_score(lr_cd, dssp_coil, dssp_helix_align)
    return FoldQuality(
        lr_cd=lr_cd,
        dssp_coil=dssp_coil,
        dssp_helix_align=dssp_helix_align,
        score=score,
    )


def compute_relative_contact_order(
    atom_array: struc.AtomArray, contact_cutoff: float = 8.0
) -> float:
    """Relative contact order: mean sequence separation of contacting CA pairs,
    normalized by chain length.

    CO = (1 / (L * N)) * sum(|i - j|) for all CA pairs within contact_cutoff
    where L = number of residues, N = number of contacting pairs.

    High values indicate non-local packing (globular enzymes).
    Low values indicate local contacts (membrane helical bundles).

    Returns 0.0 for chains with < 10 residues or no contacts.
    """
    ca_mask = atom_array.atom_name == "CA"
    ca_atoms = atom_array[ca_mask]
    n = len(ca_atoms)

    if n < 10:
        return 0.0

    dm = squareform(pdist(ca_atoms.coord))
    seq_sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])

    # Contacts: within cutoff, exclude self and trivial neighbors (|i-j| >= 3)
    contact_mask = (dm < contact_cutoff) & (seq_sep >= 3)
    # Upper triangle only to avoid double-counting
    contact_mask = np.triu(contact_mask, k=1)

    num_contacts = contact_mask.sum()
    if num_contacts == 0:
        return 0.0

    total_sep = seq_sep[contact_mask].sum()
    return float(total_sep / (n * num_contacts))


def compute_sse_segment_count(
    atom_array: struc.AtomArray, sse: np.ndarray | None = None
) -> dict[str, float]:
    """Count distinct SSE segments (runs of consecutive same-type residues).

    Returns raw counts and length-normalized counts for helix, sheet, coil.
    Enzymes with mixed topology have many short segments; simple repeats have few long ones.
    """
    if sse is None:
        sse = compute_dssp(atom_array)
    n = len(sse)

    if n == 0:
        return {
            "helix_segments": 0,
            "sheet_segments": 0,
            "coil_segments": 0,
            "helix_segments_per_100": 0.0,
            "sheet_segments_per_100": 0.0,
            "coil_segments_per_100": 0.0,
            "total_segments": 0,
            "total_segments_per_100": 0.0,
        }

    # Count transitions: a new segment starts when sse[i] != sse[i-1]
    counts = {"a": 0, "b": 0, "c": 0}
    counts[sse[0]] = 1
    for i in range(1, n):
        if sse[i] != sse[i - 1]:
            counts[sse[i]] += 1

    total = counts["a"] + counts["b"] + counts["c"]
    per_100 = 100.0 / n

    return {
        "helix_segments": counts["a"],
        "sheet_segments": counts["b"],
        "coil_segments": counts["c"],
        "helix_segments_per_100": counts["a"] * per_100,
        "sheet_segments_per_100": counts["b"] * per_100,
        "coil_segments_per_100": counts["c"] * per_100,
        "total_segments": total,
        "total_segments_per_100": total * per_100,
    }


def compute_sse_entropy(
    atom_array: struc.AtomArray, sse: np.ndarray | None = None
) -> float:
    """Shannon entropy over SSE composition {helix, sheet, coil}.

    0.0 for all-helix or all-coil structures.
    ~1.58 bits for perfectly uniform {1/3, 1/3, 1/3} mix.
    """
    if sse is None:
        sse = compute_dssp(atom_array)
    n = len(sse)

    if n == 0:
        return 0.0

    counter = Counter(sse.tolist())
    entropy = 0.0
    for count in counter.values():
        p = count / n
        if p > 0:
            entropy -= p * np.log2(p)

    return float(entropy)


def compute_longest_coil_run(
    atom_array: struc.AtomArray, sse: np.ndarray | None = None
) -> dict[str, float]:
    """Longest uninterrupted coil run, absolute and as fraction of chain length.

    Pathological predicted structures (compact but unfolded spaghetti) have
    enormous coil runs (0.30-0.45 of length). Real folded domains stay below ~0.20.
    """
    if sse is None:
        sse = compute_dssp(atom_array)
    n = len(sse)

    if n == 0:
        return {"longest_coil_run": 0, "longest_coil_run_fraction": 0.0}

    max_run = 0
    current_run = 0
    for s in sse:
        if s == "c":
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0

    return {
        "longest_coil_run": max_run,
        "longest_coil_run_fraction": max_run / n,
    }


def compute_repeat_score(
    atom_array: struc.AtomArray, min_period: int = 15
) -> dict[str, float]:
    """Detect structural repetitiveness via CA distance matrix autocorrelation.

    For each sequence separation k, computes mean CA-CA distance d(k).
    A repeat protein (beta-solenoid, beta-helix) oscillates with period P;
    a globular protein decays smoothly. The repeat score is the max
    autocorrelation of the detrended d(k) signal at lags >= min_period.

    Returns repeat_score (0-1, higher = more repetitive) and repeat_period.
    """
    ca = atom_array[atom_array.atom_name == "CA"]
    n = len(ca)

    if n < 2 * min_period:
        return {"repeat_score": 0.0, "repeat_period": 0}

    dm = squareform(pdist(ca.coord))

    # Mean CA-CA distance at each sequence separation k
    max_k = n // 2
    d_k = np.array([np.mean(np.diag(dm, k=k)) for k in range(1, max_k + 1)])

    # Detrend: subtract running average to remove the monotonic rise
    window = min(min_period, len(d_k) // 3)
    if window < 3:
        return {"repeat_score": 0.0, "repeat_period": 0}
    kernel = np.ones(window) / window
    smooth = np.convolve(d_k, kernel, mode="same")
    detrended = d_k - smooth

    # Autocorrelation of detrended signal
    detrended = detrended - detrended.mean()
    norm = np.sum(detrended**2)
    if norm < 1e-10:
        return {"repeat_score": 0.0, "repeat_period": 0}

    autocorr = np.correlate(detrended, detrended, mode="full")
    autocorr = autocorr[len(detrended) - 1 :]  # positive lags only
    autocorr = autocorr / norm

    # Find max autocorrelation at lag >= min_period
    if len(autocorr) <= min_period:
        return {"repeat_score": 0.0, "repeat_period": 0}

    search = autocorr[min_period:]
    best_idx = np.argmax(search)

    return {
        "repeat_score": float(search[best_idx]),
        "repeat_period": int(best_idx + min_period),
    }


def compute_hydrophobic_sasa_fraction(atom_array: struc.AtomArray) -> dict[str, float]:
    """Fraction of solvent-accessible surface area that is hydrophobic.

    A stable monomer buries its hydrophobics in the core (~40-45% hydrophobic SASA).
    Interface-dependent structures expose large hydrophobic patches (>50%)
    where the binding partner would sit.

    Also returns the buried hydrophobic fraction: proportion of hydrophobic
    residues with SASA < 10 A^2. Low values indicate exposed hydrophobics
    that need a partner.
    """
    atom_sasa = struc.sasa(atom_array, vdw_radii="Single", point_number=20)

    # Per-residue SASA via np.add.reduceat (no Python loop)
    res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    sasa_safe = np.nan_to_num(atom_sasa, nan=0.0)
    res_sasa = np.add.reduceat(sasa_safe, res_starts[:-1])

    # Hydrophobic mask (one per residue)
    res_names = atom_array.res_name[res_starts[:-1]]
    is_hydrophobic = np.isin(res_names, list(HYDROPHOBIC_RESIDUES))

    total_sasa = float(res_sasa.sum())
    hydrophobic_sasa = float(res_sasa[is_hydrophobic].sum())
    n_hydrophobic = int(is_hydrophobic.sum())
    n_hydrophobic_buried = int(np.sum(is_hydrophobic & (res_sasa < 10.0)))

    hydrophobic_fraction = hydrophobic_sasa / total_sasa if total_sasa > 0 else 0.0
    buried_fraction = n_hydrophobic_buried / n_hydrophobic if n_hydrophobic > 0 else 0.0

    return {
        "hydrophobic_sasa_fraction": float(hydrophobic_fraction),
        "buried_hydrophobic_fraction": float(buried_fraction),
    }


def compute_sphericity(atom_array: struc.AtomArray) -> float:
    """Sphericity from CA coordinate inertia tensor eigenvalues.

    Sphericity = (lambda_min / lambda_max)^(1/2), where lambda are
    eigenvalues of the gyration tensor. 1.0 = perfect sphere,
    low values = elongated or flat (common in interface-dependent structures).
    """
    ca = atom_array[atom_array.atom_name == "CA"]
    coords = ca.coord
    n = len(coords)

    if n < 4:
        return 0.0

    center = coords.mean(axis=0)
    centered = coords - center

    # Gyration tensor (mass-weighted inertia, unit masses)
    gyration = centered.T @ centered / n
    eigenvalues = np.linalg.eigvalsh(gyration)
    eigenvalues = np.sort(eigenvalues)

    if eigenvalues[2] < 1e-10:
        return 0.0

    return float(np.sqrt(eigenvalues[0] / eigenvalues[2]))


def compute_local_packing_variance(
    atom_array: struc.AtomArray, radius: float = 10.0
) -> dict[str, float]:
    """Variance of per-residue CA contact counts within a sphere.

    Uniform packing (stable monomer) → low variance.
    Asymmetric packing (exposed interface on one face) → high variance.
    The coefficient of variation (CV = std/mean) normalizes for protein size.
    """
    ca = atom_array[atom_array.atom_name == "CA"]
    coords = ca.coord
    n = len(coords)

    if n < 10:
        return {"packing_cv": 0.0, "packing_mean": 0.0, "packing_std": 0.0}

    from scipy.spatial import KDTree

    tree = KDTree(coords)
    counts = np.array([len(tree.query_ball_point(c, r=radius)) for c in coords])

    mean_count = float(np.mean(counts))
    std_count = float(np.std(counts))
    cv = std_count / mean_count if mean_count > 0 else 0.0

    return {
        "packing_cv": float(cv),
        "packing_mean": mean_count,
        "packing_std": std_count,
    }
