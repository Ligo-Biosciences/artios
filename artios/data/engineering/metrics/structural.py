from collections import Counter

import biotite.structure as struc
import numpy as np
from scipy.spatial.distance import pdist, squareform

from artios.common.constants import HYDROPHOBIC_RESIDUES
from artios.utils.dssp import compute_dssp


def compute_corrected_contact_order(
    atom_array: struc.AtomArray, contact_cutoff: float = 8.0
) -> float:
    """Length-corrected contact order: ACO / sqrt(L).

    Absolute contact order (mean sequence separation of contacting CA pairs)
    scales empirically as L^0.49. Dividing by sqrt(L) removes this dependence,
    unlike the standard relative contact order which overcorrects by dividing
    by L.

    Returns 0.0 for chains with < 10 residues or no contacts.
    """
    ca_mask = atom_array.atom_name == "CA"
    ca_atoms = atom_array[ca_mask]
    n = len(ca_atoms)

    if n < 10:
        return 0.0

    dm = squareform(pdist(ca_atoms.coord))
    seq_sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])

    contact_mask = (dm < contact_cutoff) & (seq_sep >= 3)
    contact_mask = np.triu(contact_mask, k=1)

    num_contacts = contact_mask.sum()
    if num_contacts == 0:
        return 0.0

    aco = float(seq_sep[contact_mask].sum() / num_contacts)
    return aco / np.sqrt(n)


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


def compute_packing_efficiency(
    atom_array: struc.AtomArray,
) -> dict[str, float]:
    """Packing efficiency from convex hull of CA coordinates.

    Packing efficiency = N_residues / hull_volume. Lower values indicate
    more internal void space (cavities/clefts), characteristic of enzymes.

    The corrected version (PE * L^0.3) removes length dependence. The raw
    exponent is empirically ~L^-0.36 on predicted structures and ~L^-0.27
    on enzymes; L^0.3 minimizes worst-case correlation across both sets.
    """
    from scipy.spatial import ConvexHull

    ca = atom_array[atom_array.atom_name == "CA"]
    coords = ca.coord
    n = len(coords)

    if n < 4:
        return {"packing_efficiency": 0.0, "corrected_packing_efficiency": 0.0}

    hull = ConvexHull(coords)
    pe = n / hull.volume
    cpe = pe * (n**0.3)

    return {
        "packing_efficiency": float(pe),
        "corrected_packing_efficiency": float(cpe),
    }
