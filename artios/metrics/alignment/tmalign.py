"""Public API for TM-align.

Implements the orchestration logic from TMalign's `TMalign_main`: run each
initial-alignment strategy, refine candidates with `dp_iter`, do a final
`detailed_search_standard` pass, then report TM-scores under three length
normalizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from artios.metrics.alignment.dp import dp_iter
from artios.metrics.alignment.initials import (
    initial_fragment_gapless,
    initial_gapless,
    initial_local_superposition,
    initial_ss,
    initial_ss_plus_local,
)
from artios.metrics.alignment.kernels import (
    apply_transform,
    assign_secondary_structure,
    d0_of,
    d0_search_clamped,
    get_score_fast,
    score_d8,
    tm_kernel,
    tmscore8_search,
)


@dataclass(frozen=True)
class TMScoreResult:
    """Result of a single TM-align computation.

    Attributes:
        tm_norm_a: TM-score normalized by `len(ca_a)`.
        tm_norm_b: TM-score normalized by `len(ca_b)`.
        tm_norm_avg: TM-score normalized by `(len(ca_a) + len(ca_b)) // 2`.
        aligned_length: Number of residue pairs in the final alignment.
        alignment_a: Indices into `ca_a` (None unless `return_alignment=True`).
        alignment_b: Same length as `alignment_a`, indices into `ca_b`.
        rotation: Rotation matrix (3, 3) (None unless `return_transform=True`).
        translation: Translation vector (3,).
    """

    tm_norm_a: float
    tm_norm_b: float
    tm_norm_avg: float
    aligned_length: int
    alignment_a: NDArray[np.int_] | None
    alignment_b: NDArray[np.int_] | None
    rotation: NDArray[np.floating] | None
    translation: NDArray[np.floating] | None


def _finalize_scores(
    ca_a: NDArray[np.floating],
    ca_b: NDArray[np.floating],
    idx_a: NDArray[np.int_],
    idx_b: NDArray[np.int_],
    r: NDArray[np.floating],
    t: NDArray[np.floating],
) -> tuple[float, float, float]:
    """Re-score aligned pairs three times under d0(L_a), d0(L_b), d0(L_avg)."""
    la, lb = ca_a.shape[0], ca_b.shape[0]
    if idx_a.size == 0:
        return 0.0, 0.0, 0.0

    rotated = apply_transform(ca_a[idx_a], r, t)
    d_sq = np.sum((rotated - ca_b[idx_b]) ** 2, axis=-1)

    def score_for(lnorm: int) -> float:
        d0 = d0_of(lnorm, "final")
        cutoff = score_d8(lnorm)
        mask = d_sq <= cutoff * cutoff
        return float(tm_kernel(d_sq[mask], d0).sum() / lnorm)

    return score_for(la), score_for(lb), score_for((la + lb) // 2)


def _tmalign_main(
    ca_a: NDArray[np.floating],
    ca_b: NDArray[np.floating],
    fast: bool,
    ss_a: NDArray[np.int8] | None = None,
    ss_b: NDArray[np.int8] | None = None,
    tm_cutoff: float | None = None,
) -> tuple[
    NDArray[np.int_], NDArray[np.int_], NDArray[np.floating], NDArray[np.floating]
]:
    """Orchestrate the TMalign pipeline on a single pair.

    `ss_a`/`ss_b` let callers pass precomputed secondary structure; used by
    `tm_scores_pairwise` to amortise SS computation across a batch.

    `tm_cutoff` — if the best candidate's raw score after ranking (pre-refine)
    gives an estimated TM-score < tm_cutoff × Lnorm_search, skip `dp_iter` and
    return the ranking's rotation directly. Intended for k-NN filtering:
    pairs known to be non-neighbours short-circuit the expensive steps.
    """
    la, lb = ca_a.shape[0], ca_b.shape[0]
    lnorm_search = min(la, lb)
    d0 = d0_of(lnorm_search, "search")
    d0_search = d0_search_clamped(d0)

    if ss_a is None:
        ss_a = assign_secondary_structure(ca_a)
    if ss_b is None:
        ss_b = assign_secondary_structure(ca_b)

    candidates: list[tuple[NDArray[np.int_], NDArray[np.int_]]] = []

    # Strategy 1 — gapless threading.
    gapless_stride = 5 if fast else 1
    candidates.append(initial_gapless(ca_a, ca_b, d0, d0_search, stride=gapless_stride))

    # Strategy 2 — secondary-structure DP.
    candidates.append(initial_ss(ss_a, ss_b))

    # Strategy 3 — local superposition. Both modes run it — dropping it
    # entirely costs >0.1 TM-score on topology-diverse pairs. Fast mode just
    # strides 5× coarser over fragment start positions.
    candidates.append(
        initial_local_superposition(
            ca_a, ca_b, d0, d0_search, stride_scale=5 if fast else 1
        )
    )

    # Strategy 4 — local superposition + SS (accurate only — the first local
    # superposition already covers most of the benefit).
    if not fast:
        candidates.append(
            initial_ss_plus_local(ca_a, ca_b, ss_a, ss_b, d0, d0_search, stride_scale=1)
        )

    # Strategy 5 — fragment gapless threading.
    fgt_stride = 3 if fast else 1
    candidates.append(
        initial_fragment_gapless(ca_a, ca_b, d0, d0_search, stride=fgt_stride)
    )

    if fast:
        dp_iter_max = 3
        gap_opens = (-0.6, 0.0)
        polish_step = 40
    else:
        dp_iter_max = 15
        gap_opens = (-0.6, 0.0)
        polish_step = 8

    d8_cutoff = score_d8(lnorm_search)

    # Early-termination for k-NN filtering. Cheap pre-refinement score via
    # get_score_fast on each candidate; if the best is below `tm_cutoff`, we
    # return that rotation without running dp_iter or tmscore8_search.
    # get_score_fast's score is typically lower than the final TM-score
    # (which refinement improves), so pairs below cutoff here are very likely
    # below cutoff after refinement.
    if tm_cutoff is not None:
        best_cand_score = -np.inf
        best_cand_rt: (
            tuple[
                NDArray[np.int_],
                NDArray[np.int_],
                NDArray[np.floating],
                NDArray[np.floating],
            ]
            | None
        ) = None
        for idx_a, idx_b in candidates:
            if idx_a.size < 3:
                continue
            s, r, t = get_score_fast(ca_a, ca_b, idx_a, idx_b, d0, d0_search)
            if s > best_cand_score:
                best_cand_score = s
                best_cand_rt = (idx_a, idx_b, r, t)
        if best_cand_rt is not None and best_cand_score / lnorm_search < tm_cutoff:
            return best_cand_rt

    # Refine every candidate with dp_iter (cheap — uses get_score_fast inside)
    # then polish the winner once with tmscore8_search. Iterating every
    # candidate is what catches hard topology-swap pairs; restricting to top-k
    # measurably hurt accuracy.
    best_score = -np.inf
    best_r = np.eye(3, dtype=np.float64)
    best_t = np.zeros(3, dtype=np.float64)
    best_idx_a = np.empty(0, dtype=np.int_)
    best_idx_b = np.empty(0, dtype=np.int_)

    for idx_a, idx_b in candidates:
        if idx_a.size < 3:
            continue
        score, r, t, ia, ib = dp_iter(
            ca_a,
            ca_b,
            idx_a,
            idx_b,
            d0=d0,
            d0_search=d0_search,
            gap_opens=gap_opens,
            max_iters=dp_iter_max,
        )
        if score > best_score:
            best_score = score
            best_r, best_t = r, t
            best_idx_a, best_idx_b = ia, ib

    if best_idx_a.size >= 4:
        polished_score, polished_r, polished_t = tmscore8_search(
            ca_a,
            ca_b,
            best_idx_a,
            best_idx_b,
            d0=d0,
            d0_search=d0_search,
            score_d8_cutoff=d8_cutoff,
            simplify_step=polish_step,
        )
        if polished_score > best_score:
            best_r, best_t = polished_r, polished_t

    return best_idx_a, best_idx_b, best_r, best_t


def tm_score(
    ca_a: NDArray[np.floating],
    ca_b: NDArray[np.floating],
    *,
    fast: bool = True,
    return_transform: bool = False,
    return_alignment: bool = False,
    ss_a: NDArray[np.int8] | None = None,
    ss_b: NDArray[np.int8] | None = None,
    tm_cutoff: float | None = None,
) -> TMScoreResult:
    """Compute TM-score between two Cα-only structures.

    Args:
        ca_a, ca_b: Cα coordinates of the two structures, shape (N, 3).
        fast: Fast mode ≈ parity with `TMalign -fast`; off runs the full search.
        return_transform: If True, include `rotation` and `translation`.
        return_alignment: If True, include `alignment_a`/`alignment_b`.
        ss_a, ss_b: Precomputed secondary structure arrays (avoid recompute).
        tm_cutoff: If set, pairs whose pre-refinement TM-score is below the
            cutoff skip `dp_iter` + `tmscore8_search` and return an approximate
            score. Intended for k-NN filtering — the exact rank of
            "not a neighbour" pairs doesn't matter. Typical: 0.3 for TM-based
            structural similarity thresholds.

    Returns:
        `TMScoreResult` with TM-scores normalized by L_a, L_b, and L_avg.
    """
    ca_a = np.ascontiguousarray(ca_a, dtype=np.float64)
    ca_b = np.ascontiguousarray(ca_b, dtype=np.float64)
    idx_a, idx_b, r, t = _tmalign_main(
        ca_a, ca_b, fast=fast, ss_a=ss_a, ss_b=ss_b, tm_cutoff=tm_cutoff
    )
    tm_a, tm_b, tm_avg = _finalize_scores(ca_a, ca_b, idx_a, idx_b, r, t)
    return TMScoreResult(
        tm_norm_a=tm_a,
        tm_norm_b=tm_b,
        tm_norm_avg=tm_avg,
        aligned_length=int(idx_a.size),
        alignment_a=idx_a if return_alignment else None,
        alignment_b=idx_b if return_alignment else None,
        rotation=r if return_transform else None,
        translation=t if return_transform else None,
    )


def tm_scores_pairwise(
    query: NDArray[np.floating],
    database: list[NDArray[np.floating]],
    *,
    fast: bool = True,
    normalization: Literal["query", "ref", "avg"] = "query",
    tm_cutoff: float | None = None,
) -> NDArray[np.floating]:
    """Compute TM-scores between one query and a list of reference structures.

    Sequential by design. Secondary structure is cached (query once, each
    reference once). If you want process-level parallelism, wrap this call in
    your own pool — an inner worker pool would oversubscribe CPUs when the
    caller is already parallelising.

    Args:
        query: Query Cα coordinates, shape (N_q, 3).
        database: List of reference Cα coordinate arrays (variable length).
        fast: Passed through to `tm_score`.
        normalization: "query" → tm_norm_a, "ref" → tm_norm_b, "avg" → tm_norm_avg.

    Returns:
        1D array of TM-scores, shape `(len(database),)`.
    """
    query = np.ascontiguousarray(query, dtype=np.float64)
    db_arrays = [np.ascontiguousarray(r, dtype=np.float64) for r in database]

    ss_query = assign_secondary_structure(query)
    ss_db = [assign_secondary_structure(r) for r in db_arrays]

    scores = np.empty(len(db_arrays), dtype=np.float64)
    for i, ref in enumerate(db_arrays):
        r = tm_score(
            query, ref, fast=fast, ss_a=ss_query, ss_b=ss_db[i], tm_cutoff=tm_cutoff
        )
        if normalization == "query":
            scores[i] = r.tm_norm_a
        elif normalization == "ref":
            scores[i] = r.tm_norm_b
        else:
            scores[i] = r.tm_norm_avg
    return scores
