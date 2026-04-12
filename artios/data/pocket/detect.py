"""Two-stage pocket detection: per-sphere RF + per-pocket GBM.

Usage:
    from artios.data.pocket import find_pockets

    pockets = find_pockets(protein_atoms)                    # balanced (default)
    pockets = find_pockets(protein_atoms, mode="high_recall") # more pockets, some FPs
    pockets = find_pockets(protein_atoms, mode="high_precision") # fewer but confident
"""

from pathlib import Path
from typing import Literal

import biotite.structure as struc
import numpy as np
from pocketeer.core import Pocket, cluster_spheres, create_pocket
from pocketeer.utils.constants import MIN_ATOMS_FOR_TESSELLATION
from scipy.spatial import KDTree

from artios.common.atom_types import (
    AROMATIC_SIDECHAIN_ATOMS,
    HBOND_ACCEPTOR_ATOMS,
    HBOND_DONOR_ATOMS,
)
from artios.common.residue_names import (
    CHARGED_RESIDUES,
    HYDROPHILIC_RESIDUES,
    HYDROPHOBIC_RESIDUES,
)
from artios.data.pocket.legacy import _compute_binding_features
from artios.data.pocket.tessellation import compute_alpha_spheres

# Thresholds calibrated on 78 held-out enzyme structures (holo4k test set
# metrics in parentheses for reference).
#
#                         enzyme eval                    holo4k test
# Mode              Prec   Recall  F1              Prec   Recall  F1
# high_recall       ~49%   ~86%    0.63            ~58%   ~89%    0.70
# balanced          ~54%   ~82%    0.65            ~64%   ~84%    0.73
# high_precision    ~73%   ~63%    0.68            ~75%   ~67%    0.70
SCORE_THRESHOLDS: dict[str, float] = {
    "high_recall": 0.10,
    "balanced": 0.15,
    "high_precision": 0.40,
}

_MODEL_DIR = Path(__file__).parent
_SPHERE_RF = None
_POCKET_RF = None
_CHEM_RADIUS = 8.0


def _load_models():
    global _SPHERE_RF, _POCKET_RF
    if _SPHERE_RF is None:
        import joblib

        _SPHERE_RF = joblib.load(_MODEL_DIR / "sphere_rf.joblib")
        _POCKET_RF = joblib.load(_MODEL_DIR / "pocket_rf.joblib")
    return _SPHERE_RF, _POCKET_RF


def _threshold_sparse(dist_csr, r):
    """Binary neighbor matrix from a sparse distance matrix at threshold r."""
    m = dist_csr.copy()
    m.data = (m.data <= r).astype(np.float64)
    m.eliminate_zeros()
    return m


def _compute_sphere_features(
    sphere_centers,
    sphere_radii,
    sphere_is_buried,
    sphere_atom_indices,
    protein_coord,
    protein_element,
    protein_atom_name,
    protein_res_name,
    atom_sasa,
):
    protein_tree = KDTree(protein_coord)
    sphere_tree = KDTree(sphere_centers)

    props = {
        "hydrophobic": np.isin(protein_res_name, list(HYDROPHOBIC_RESIDUES)).astype(
            np.float64
        ),
        "hydrophilic": np.isin(protein_res_name, list(HYDROPHILIC_RESIDUES)).astype(
            np.float64
        ),
        "charged": np.isin(protein_res_name, list(CHARGED_RESIDUES)).astype(np.float64),
        "aromatic": np.isin(protein_atom_name, list(AROMATIC_SIDECHAIN_ATOMS)).astype(
            np.float64
        ),
        "hbond_donor": np.isin(protein_atom_name, list(HBOND_DONOR_ATOMS)).astype(
            np.float64
        ),
        "hbond_acceptor": np.isin(protein_atom_name, list(HBOND_ACCEPTOR_ATOMS)).astype(
            np.float64
        ),
        "polar": (
            (protein_element == "N")
            | (protein_element == "O")
            | (protein_element == "S")
        ).astype(np.float64),
        "C": (protein_element == "C").astype(np.float64),
        "N": (protein_element == "N").astype(np.float64),
        "O": (protein_element == "O").astype(np.float64),
        "S": (protein_element == "S").astype(np.float64),
    }

    # One distance query each, threshold for sub-radii
    prot_dist = sphere_tree.sparse_distance_matrix(
        protein_tree, 12.0, output_type="coo_matrix"
    ).tocsr()
    nbr_6, nbr_8, nbr_10, nbr_12 = (
        _threshold_sparse(prot_dist, r) for r in (6, 8, 10, 12)
    )

    sph_dist = sphere_tree.sparse_distance_matrix(
        sphere_tree, 5.0, output_type="coo_matrix"
    ).tocsr()
    sph_3 = _threshold_sparse(sph_dist, 3.0)
    sph_5 = sph_dist.copy()
    sph_5.data = np.ones_like(sph_5.data)

    counts_8 = np.array(nbr_8.sum(axis=1)).ravel()
    safe_8 = np.maximum(counts_8, 1.0)

    def frac(key):
        return np.array(nbr_8 @ props[key]).ravel() / safe_8

    n_buried_of_4 = np.sum(atom_sasa[sphere_atom_indices] <= 10.0, axis=1).astype(
        np.float64
    )
    sph_counts_3 = np.array(sph_3.sum(axis=1)).ravel() - 1
    sph_counts_5 = np.array(sph_5.sum(axis=1)).ravel() - 1
    mean_nbr_r = (np.array(sph_5 @ sphere_radii).ravel() - sphere_radii) / np.maximum(
        sph_counts_5, 1.0
    )

    return np.column_stack(
        [
            sphere_radii,
            sphere_is_buried.astype(np.float64),
            n_buried_of_4,
            np.array(nbr_6.sum(axis=1)).ravel(),
            counts_8,
            np.array(nbr_10.sum(axis=1)).ravel(),
            np.array(nbr_12.sum(axis=1)).ravel(),
            frac("hydrophobic"),
            frac("hydrophilic"),
            frac("aromatic"),
            frac("charged"),
            frac("hbond_donor"),
            frac("hbond_acceptor"),
            frac("polar"),
            frac("C"),
            frac("N"),
            frac("O"),
            frac("S"),
            counts_8 / ((4 / 3) * np.pi * _CHEM_RADIUS**3),
            sph_counts_3,
            sph_counts_5,
            mean_nbr_r,
        ]
    )


def find_pockets(
    atomarray: struc.AtomArray,
    *,
    mode: Literal["high_recall", "balanced", "high_precision"] = "balanced",
    plddt_cutoff: float | None = None,
    min_atoms_in_radius_15: int = 425,
    min_atoms_in_radius_20: int = 900,
) -> list[Pocket]:
    """Detect ligand binding pockets in a protein structure.

    Uses a two-stage classifier trained on 3,161 holo crystal structures:
      Stage 1: Per-sphere random forest (geometric + chemical features)
      Stage 2: Per-pocket gradient boosting (enclosure, aggregated sphere
               features, stage 1 score statistics)

    Args:
        atomarray: Biotite AtomArray. Hetero atoms, hydrogens, and water
            are stripped automatically.
        mode: Operating point.
            "high_recall"    — ~86% recall, ~49% precision (more pockets)
            "balanced"       — ~82% recall, ~54% precision (default)
            "high_precision" — ~63% recall, ~73% precision (fewer, confident)
        plddt_cutoff: If set, discard spheres whose defining atoms have
            pLDDT (b_factor) below this value. Useful for AlphaFold models.
        min_atoms_in_radius_15: Minimum protein atoms within 15A of
            pocket centroid. Filters sparse regions. Default: 425
        min_atoms_in_radius_20: Same at 20A. Default: 900

    Returns:
        List of Pocket objects, sorted by score (highest first).
        Each pocket has a ``binding_score`` attribute (0-1).
    """
    threshold = SCORE_THRESHOLDS[mode]
    sphere_rf, pocket_rf = _load_models()

    # Clean input
    atomarray = atomarray[atomarray.element != "H"]
    atomarray = atomarray[atomarray.res_name != "HOH"]
    atomarray = atomarray[~atomarray.hetero]

    if len(atomarray) < MIN_ATOMS_FOR_TESSELLATION:
        return []

    # Alpha spheres
    spheres = compute_alpha_spheres(atomarray.coord, r_min=3.0, r_max=6.0)
    if len(spheres) < 35:
        return []

    # SASA + polarity
    atom_sasa = struc.sasa(atomarray, probe_radius=5.0, point_number=200)
    for s in spheres:
        s.is_buried = bool(np.all(atom_sasa[s.atom_indices] <= 10.0))

    # Optional pLDDT filter
    if plddt_cutoff is not None:
        spheres = [
            s
            for s in spheres
            if np.all(atomarray.b_factor[s.atom_indices] >= plddt_cutoff)
        ]
        if len(spheres) < 35:
            return []

    # Stage 1: sphere features + RF predictions
    centers = np.array([s.center for s in spheres])
    radii = np.array([s.radius for s in spheres])
    buried = np.array([s.is_buried for s in spheres])
    aidx = np.array([s.atom_indices for s in spheres])

    sf = _compute_sphere_features(
        centers,
        radii,
        buried,
        aidx,
        atomarray.coord,
        np.array(atomarray.element),
        np.array(atomarray.atom_name),
        np.array(atomarray.res_name),
        atom_sasa,
    )
    sphere_probs = sphere_rf.predict_proba(sf)[:, 1]

    # Cluster buried spheres into pockets
    buried_spheres = [s for s in spheres if s.is_buried]
    clusters = cluster_spheres(buried_spheres, merge_distance=1.75, min_spheres=35)
    if not clusters:
        return []

    # Stage 2: pocket scoring
    sphere_id_map = {id(s): i for i, s in enumerate(spheres)}
    protein_tree = KDTree(atomarray.coord)

    pockets = []
    for cluster in clusters:
        pocket = create_pocket(0, cluster)

        if (
            len(protein_tree.query_ball_point(pocket.centroid, r=15.0))
            < min_atoms_in_radius_15
        ):
            continue
        if (
            len(protein_tree.query_ball_point(pocket.centroid, r=20.0))
            < min_atoms_in_radius_20
        ):
            continue

        try:
            binding_feats = _compute_binding_features(pocket, atomarray, protein_tree)
        except Exception:
            continue

        cidx = [sphere_id_map[id(s)] for s in cluster if id(s) in sphere_id_map]
        if not cidx:
            continue

        csf = sf[cidx]
        cprobs = sphere_probs[cidx]

        pocket_feat = np.concatenate(
            [
                binding_feats,
                csf.mean(axis=0),
                csf.std(axis=0),
                [
                    np.mean(cprobs),
                    np.median(cprobs),
                    np.percentile(cprobs, 90),
                    np.std(cprobs),
                    np.mean(cprobs >= 0.3),
                    np.mean(cprobs >= 0.5),
                    np.mean(cprobs >= 0.7),
                ],
            ]
        )

        pocket.binding_score = float(
            pocket_rf.predict_proba(pocket_feat.reshape(1, -1))[0, 1]
        )
        if pocket.binding_score >= threshold:
            pockets.append(pocket)

    pockets.sort(key=lambda p: p.binding_score, reverse=True)
    return pockets
