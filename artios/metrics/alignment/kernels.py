"""Numerical kernels for TM-align: Kabsch, d0/score_d8 helpers, secondary
structure, TM-score distance kernel, and the 3-round refinement."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numba import njit
from numpy.typing import NDArray

# Secondary-structure one-hot encoding used throughout.
SS_COIL = 0
SS_HELIX = 1
SS_STRAND = 2
SS_TURN = 3


@njit(cache=True, fastmath=True)
def _kabsch_impl(
    p: NDArray[np.float64], q: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Kabsch core — JIT'd so the per-call Python overhead disappears. Skips
    `np.linalg.det` in favour of the explicit 3x3 determinant (8x faster
    for this size)."""
    n = p.shape[0]
    # Weighted centroids.
    pc = np.zeros(3)
    qc = np.zeros(3)
    for i in range(n):
        for k in range(3):
            pc[k] += p[i, k]
            qc[k] += q[i, k]
    inv_n = 1.0 / n
    for k in range(3):
        pc[k] *= inv_n
        qc[k] *= inv_n

    # Covariance matrix H = P_c^T Q_c.
    h = np.zeros((3, 3))
    for i in range(n):
        dp0 = p[i, 0] - pc[0]
        dp1 = p[i, 1] - pc[1]
        dp2 = p[i, 2] - pc[2]
        dq0 = q[i, 0] - qc[0]
        dq1 = q[i, 1] - qc[1]
        dq2 = q[i, 2] - qc[2]
        h[0, 0] += dp0 * dq0
        h[0, 1] += dp0 * dq1
        h[0, 2] += dp0 * dq2
        h[1, 0] += dp1 * dq0
        h[1, 1] += dp1 * dq1
        h[1, 2] += dp1 * dq2
        h[2, 0] += dp2 * dq0
        h[2, 1] += dp2 * dq1
        h[2, 2] += dp2 * dq2

    u, _s, vt = np.linalg.svd(h)
    vt_ut = vt.T @ u.T
    # Explicit 3x3 determinant.
    det = (
        vt_ut[0, 0] * (vt_ut[1, 1] * vt_ut[2, 2] - vt_ut[1, 2] * vt_ut[2, 1])
        - vt_ut[0, 1] * (vt_ut[1, 0] * vt_ut[2, 2] - vt_ut[1, 2] * vt_ut[2, 0])
        + vt_ut[0, 2] * (vt_ut[1, 0] * vt_ut[2, 1] - vt_ut[1, 1] * vt_ut[2, 0])
    )
    if det < 0.0:
        d_sign = -1.0
    else:
        d_sign = 1.0
    # R = V · diag(1,1,d) · Uᵀ
    r = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            r[i, j] = (
                vt[0, i] * u[j, 0] + vt[1, i] * u[j, 1] + d_sign * vt[2, i] * u[j, 2]
            )
    t = np.empty(3)
    for k in range(3):
        t[k] = qc[k] - (r[k, 0] * pc[0] + r[k, 1] * pc[1] + r[k, 2] * pc[2])
    return r, t


def kabsch(
    p: NDArray[np.floating],
    q: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Rigid superposition of p onto q. Returns (R, t) so that p @ R.T + t ≈ q.

    If fewer than 3 pairs are supplied, the identity rotation and a
    difference-of-centroids translation are returned.
    """
    n = p.shape[0]
    if n < 3:
        R = np.eye(3, dtype=np.float64)
        t = q.mean(axis=0) - p.mean(axis=0) if n > 0 else np.zeros(3, dtype=np.float64)
        return R, t
    return _kabsch_impl(
        np.ascontiguousarray(p, dtype=np.float64),
        np.ascontiguousarray(q, dtype=np.float64),
    )


def apply_transform(
    x: NDArray[np.floating],
    r: NDArray[np.floating],
    t: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Apply `x @ R.T + t`."""
    return x @ r.T + t


def tm_kernel(d_sq: NDArray[np.floating], d0: float) -> NDArray[np.floating]:
    """TM-score kernel: `1 / (1 + d² / d0²)` applied element-wise."""
    return 1.0 / (1.0 + d_sq / (d0 * d0))


def d0_of(lnorm: int, phase: Literal["search", "final"]) -> float:
    """TMalign d0 formula. `phase` picks between parameter_set4search (used for
    ranking candidates) and parameter_set4final (used for output scores)."""
    if phase == "search":
        if lnorm <= 19:
            d0 = 0.168
        else:
            d0 = 1.24 * (lnorm - 15) ** (1.0 / 3.0) - 1.8
        return float(d0 + 0.8)
    if lnorm <= 21:
        return 0.5
    d0 = 1.24 * (lnorm - 15) ** (1.0 / 3.0) - 1.8
    return float(max(d0, 0.5))


def d0_search_clamped(d0: float) -> float:
    """Clamp d0 into [4.5, 8.0] as TMalign does for the search threshold."""
    if d0 < 4.5:
        return 4.5
    if d0 > 8.0:
        return 8.0
    return d0


def score_d8(lnorm: int) -> float:
    """Distance cutoff for final-score aggregation: 1.5 · L^0.3 + 3.5."""
    return float(1.5 * lnorm**0.3 + 3.5)


def assign_secondary_structure(coords: NDArray[np.floating]) -> NDArray[np.int8]:
    """Assign secondary structure from Cα geometry (TMalign `make_sec` rule).

    Uses CA-CA distances at offsets 2/3/4 against helix and strand templates,
    plus turn detection and a 3-pass smoothing. Returns an (N,) int8 array with
    values drawn from {SS_COIL, SS_HELIX, SS_STRAND, SS_TURN}.
    """
    n = coords.shape[0]
    ss = np.full(n, SS_COIL, dtype=np.int8)
    if n < 5:
        return ss

    def dist(i: NDArray[np.int_], j: NDArray[np.int_]) -> NDArray[np.floating]:
        return np.linalg.norm(coords[i] - coords[j], axis=-1)

    idx = np.arange(2, n - 2)
    d13 = dist(idx - 2, idx)
    d14 = dist(idx - 2, idx + 1)
    d15 = dist(idx - 2, idx + 2)
    d24 = dist(idx - 1, idx + 1)
    d25 = dist(idx - 1, idx + 2)
    d35 = dist(idx, idx + 2)

    delta_h = 2.1
    helix = (
        (np.abs(d13 - 5.45) < delta_h)
        & (np.abs(d14 - 5.18) < delta_h)
        & (np.abs(d15 - 6.37) < delta_h)
        & (np.abs(d24 - 5.45) < delta_h)
        & (np.abs(d25 - 5.18) < delta_h)
        & (np.abs(d35 - 5.45) < delta_h)
    )
    delta_e = 1.42
    strand = (
        (np.abs(d13 - 6.1) < delta_e)
        & (np.abs(d14 - 10.4) < delta_e)
        & (np.abs(d15 - 13.0) < delta_e)
        & (np.abs(d24 - 6.1) < delta_e)
        & (np.abs(d25 - 10.4) < delta_e)
        & (np.abs(d35 - 6.1) < delta_e)
    )
    turn = d15 < 8.0

    ss[idx[helix]] = SS_HELIX
    ss[idx[strand]] = SS_STRAND
    turn_only = turn & ~helix & ~strand
    ss[idx[turn_only]] = SS_TURN

    # Smoothing passes (TMalign lines ~2392–2434).
    out = ss.copy()
    # Fill one-residue gaps flanked by matching SS (A-X-A -> A-A-A) for H/E.
    for tag in (SS_HELIX, SS_STRAND):
        for i in range(2, n - 2):
            if out[i] != tag and out[i - 1] == tag and out[i + 1] == tag:
                out[i] = tag
        for i in range(1, n - 2):
            # doublet bridge: X-A-X-A-X -> X-A-A-A-X
            if out[i] == tag and out[i + 2] == tag and out[i + 1] != tag:
                out[i + 1] = tag
    # Remove isolated singletons.
    for i in range(1, n - 1):
        if out[i] in (SS_HELIX, SS_STRAND):
            if out[i - 1] != out[i] and out[i + 1] != out[i]:
                out[i] = SS_COIL
    return out


def _sq_dists(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
) -> NDArray[np.floating]:
    return np.sum((x - y) ** 2, axis=-1)


def get_score_fast(
    ca_a: NDArray[np.floating],
    ca_b: NDArray[np.floating],
    idx_a: NDArray[np.int_],
    idx_b: NDArray[np.int_],
    d0: float,
    d0_search: float,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    """Three-round Kabsch refinement used to rank initial alignments.

    Rounds (matching TMalign `get_score_fast`):
    1. Kabsch over all aligned pairs; score with tm_kernel.
    2. Filter pairs d ≤ d0_search; Kabsch on survivors; score.
    3. Filter pairs d ≤ d0_search + 1; Kabsch on survivors; score.
    If a filter leaves fewer than 3 pairs, the threshold is relaxed by +0.5 Å
    until at least 3 pairs survive (bounded by the aligned-pair count).

    Returns `(best_score, R, t)` where R and t are the transformation that
    produced `best_score`. `best_score` is a *sum* over aligned pairs — callers
    divide by Lnorm to convert to TM-score.
    """
    p = ca_a[idx_a]
    q = ca_b[idx_b]
    n_ali = p.shape[0]
    if n_ali == 0:
        return 0.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    best_score = -1.0
    best_r = np.eye(3, dtype=np.float64)
    best_t = np.zeros(3, dtype=np.float64)

    thresholds = (None, d0_search, d0_search + 1.0)
    current_p, current_q = p, q
    for cutoff in thresholds:
        if cutoff is not None:
            # Filter against the current transform.
            r, t = kabsch(current_p, current_q)
            rotated = apply_transform(p, r, t)
            d_sq = _sq_dists(rotated, q)
            mask = d_sq <= cutoff * cutoff
            relaxed = cutoff
            while mask.sum() < 3 and relaxed + 0.5 < 100.0 and n_ali > 3:
                relaxed += 0.5
                mask = d_sq <= relaxed * relaxed
            if mask.sum() < 3:
                continue
            current_p = p[mask]
            current_q = q[mask]
        r, t = kabsch(current_p, current_q)
        rotated = apply_transform(p, r, t)
        d_sq = _sq_dists(rotated, q)
        score = float(tm_kernel(d_sq, d0).sum())
        if score > best_score:
            best_score = score
            best_r = r
            best_t = t

    return best_score, best_r, best_t


def tm_score_given_rt(
    ca_a: NDArray[np.floating],
    ca_b: NDArray[np.floating],
    idx_a: NDArray[np.int_],
    idx_b: NDArray[np.int_],
    r: NDArray[np.floating],
    t: NDArray[np.floating],
    d0: float,
    cutoff: float | None = None,
) -> float:
    """Score aligned pairs under a given (R, t). Sums `tm_kernel` over pairs
    whose transformed distance is ≤ `cutoff` (if set). Returns the raw sum;
    the caller divides by the appropriate Lnorm."""
    rotated = apply_transform(ca_a[idx_a], r, t)
    d_sq = _sq_dists(rotated, ca_b[idx_b])
    scores = tm_kernel(d_sq, d0)
    if cutoff is not None:
        scores = np.where(d_sq <= cutoff * cutoff, scores, 0.0)
    return float(scores.sum())


def _refine_from_subset(
    aligned_p: NDArray[np.floating],
    aligned_q: NDArray[np.floating],
    start: int,
    length: int,
    d0: float,
    d0_search: float,
    score_cutoff: float,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    """Seed a Kabsch from a sub-fragment of the aligned set, then run the
    same 3-round filter-and-refit as `get_score_fast` but scoped to the
    *current aligned pairs only*. Returns the best (score_sum, R, t)."""
    seed_p = aligned_p[start : start + length]
    seed_q = aligned_q[start : start + length]
    r, t = kabsch(seed_p, seed_q)

    rotated = aligned_p @ r.T + t
    d_sq = _sq_dists(rotated, aligned_q)
    mask = d_sq <= score_cutoff * score_cutoff
    if mask.sum() >= 3:
        r, t = kabsch(aligned_p[mask], aligned_q[mask])
        rotated = aligned_p @ r.T + t
        d_sq = _sq_dists(rotated, aligned_q)

    # Two more filter-refit passes, matching TMalign's TMscore8_search.
    best_score = float(tm_kernel(d_sq, d0).sum())
    best_r = r
    best_t = t
    for cutoff in (d0_search, d0_search + 1.0):
        mask = d_sq <= cutoff * cutoff
        if mask.sum() < 3:
            continue
        r, t = kabsch(aligned_p[mask], aligned_q[mask])
        rotated = aligned_p @ r.T + t
        d_sq = _sq_dists(rotated, aligned_q)
        score = float(tm_kernel(d_sq, d0).sum())
        if score > best_score:
            best_score = score
            best_r = r
            best_t = t
    return best_score, best_r, best_t


def tmscore8_search(
    ca_a: NDArray[np.floating],
    ca_b: NDArray[np.floating],
    idx_a: NDArray[np.int_],
    idx_b: NDArray[np.int_],
    d0: float,
    d0_search: float,
    score_d8_cutoff: float,
    simplify_step: int = 1,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    """Fragment-based rotation polish (TMalign's `TMscore8_search`).

    Given a current alignment (idx_a, idx_b), this tries a family of seed
    fragments — lengths {n_ali, n_ali/2, n_ali/4, …, ≥4} at starting positions
    stepping by `simplify_step` — and keeps the rotation that maximizes the
    TM-kernel sum (with `score_d8_cutoff` as the distance cutoff for
    counting pairs). Returns the best `(score_sum, R, t)`.

    This is the step that gets us from "approximately correct" rotations to
    TMalign-exact rotations; `get_score_fast` alone leaves ~0.01–0.05 on the
    table for harder pairs.
    """
    aligned_p = ca_a[idx_a]
    aligned_q = ca_b[idx_b]
    n_ali = aligned_p.shape[0]
    if n_ali < 4:
        # Fall back to the simple 3-round refinement.
        return get_score_fast(ca_a, ca_b, idx_a, idx_b, d0, d0_search)

    # Fragment lengths: n_ali/2, n_ali/4, … stop at 6 (TMalign's floor).
    # Skipping the full-alignment length matches TMalign's TMscore8_search —
    # that case is already handled by the caller via `get_score_fast`.
    lengths: list[int] = []
    l = n_ali // 2
    while l >= 6:
        lengths.append(l)
        l //= 2

    best_score = -np.inf
    best_r = np.eye(3, dtype=np.float64)
    best_t = np.zeros(3, dtype=np.float64)

    for length in lengths:
        step = max(1, simplify_step)
        for start in range(0, n_ali - length + 1, step):
            score, r, t = _refine_from_subset(
                aligned_p,
                aligned_q,
                start,
                length,
                d0,
                d0_search,
                score_d8_cutoff,
            )
            if score > best_score:
                best_score = score
                best_r = r
                best_t = t

    return best_score, best_r, best_t
