"""Needleman–Wunsch dynamic programming for TM-align.

Two backends share a traceback:
- `nw_dp_jit` — Numba-JIT'd double-loop DP (primary, hot path).
- `nw_dp_naive` — pure-Python reference, permanently used to fuzz `nw_dp_jit`
  for correctness.

Scoring matrix entries are match scores (typically TM-score values in [0, 1]).
Gap-open is applied to transitions from aligned to unaligned, matching
TMalign's asymmetric affine-ish gap.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray


def _traceback(
    path: NDArray[np.int8],
    la: int,
    lb: int,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Follow `path` back from (la, lb) to (0, 0). `path` encodes:
    0 = diagonal (match), 1 = up (gap in b), 2 = left (gap in a)."""
    idx_a: list[int] = []
    idx_b: list[int] = []
    i, j = la, lb
    while i > 0 and j > 0:
        step = path[i, j]
        if step == 0:
            idx_a.append(i - 1)
            idx_b.append(j - 1)
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    idx_a.reverse()
    idx_b.reverse()
    return np.asarray(idx_a, dtype=np.int_), np.asarray(idx_b, dtype=np.int_)


def nw_dp_naive(
    score_matrix: NDArray[np.floating],
    gap_open: float,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Pure-Python Needleman–Wunsch reference. TMalign's recurrence applies the
    gap-open penalty only when transitioning *out of* an aligned cell, so we
    track an aligned/unaligned flag per cell (mirrors the `path` matrix)."""
    la, lb = score_matrix.shape
    val = np.full((la + 1, lb + 1), -np.inf, dtype=np.float64)
    path = np.zeros((la + 1, lb + 1), dtype=np.int8)
    aligned = np.zeros((la + 1, lb + 1), dtype=bool)
    val[0, 0] = 0.0
    for i in range(1, la + 1):
        val[i, 0] = 0.0
        path[i, 0] = 1
    for j in range(1, lb + 1):
        val[0, j] = 0.0
        path[0, j] = 2

    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            d = val[i - 1, j - 1] + float(score_matrix[i - 1, j - 1])
            h = val[i - 1, j] + (gap_open if aligned[i - 1, j] else 0.0)
            v = val[i, j - 1] + (gap_open if aligned[i, j - 1] else 0.0)
            if d >= h and d >= v:
                val[i, j] = d
                path[i, j] = 0
                aligned[i, j] = True
            elif h >= v:
                val[i, j] = h
                path[i, j] = 1
                aligned[i, j] = False
            else:
                val[i, j] = v
                path[i, j] = 2
                aligned[i, j] = False
    return _traceback(path, la, lb)


@njit(cache=True, fastmath=True)
def _nw_dp_core(
    score_matrix: NDArray[np.float64], gap_open: float
) -> tuple[NDArray[np.float64], NDArray[np.int8], NDArray[np.bool_]]:
    """JIT'd double-loop DP. Returns (val, path, aligned) tables that the
    shared traceback walks."""
    la, lb = score_matrix.shape
    val = np.zeros((la + 1, lb + 1), dtype=np.float64)
    path = np.zeros((la + 1, lb + 1), dtype=np.int8)
    aligned = np.zeros((la + 1, lb + 1), dtype=np.bool_)
    for i in range(1, la + 1):
        path[i, 0] = 1
    for j in range(1, lb + 1):
        path[0, j] = 2

    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            d = val[i - 1, j - 1] + score_matrix[i - 1, j - 1]
            h = val[i - 1, j]
            if aligned[i - 1, j]:
                h += gap_open
            v = val[i, j - 1]
            if aligned[i, j - 1]:
                v += gap_open
            if d >= h and d >= v:
                val[i, j] = d
                path[i, j] = 0
                aligned[i, j] = True
            elif h >= v:
                val[i, j] = h
                path[i, j] = 1
            else:
                val[i, j] = v
                path[i, j] = 2
    return val, path, aligned


def nw_dp_jit(
    score_matrix: NDArray[np.floating],
    gap_open: float,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Numba-JIT'd Needleman-Wunsch. Primary DP backend — ~10× faster than
    the NumPy anti-diagonal version on L≈300 protein pairs."""
    sm = np.ascontiguousarray(score_matrix, dtype=np.float64)
    _val, path, _aligned = _nw_dp_core(sm, float(gap_open))
    la, lb = sm.shape
    return _traceback(path, la, lb)


_fast_nw_dp = nw_dp_jit


def build_score_matrix(
    ca_a: NDArray[np.floating],
    ca_b: NDArray[np.floating],
    r: NDArray[np.floating],
    t: NDArray[np.floating],
    d0: float,
) -> NDArray[np.floating]:
    """Score matrix for TMalign's DP: tm_kernel of transformed distances.

    Computes the pairwise squared distance via the |a|² − 2 a·b + |b|²
    identity instead of a (La, Lb, 3) broadcast, which cuts this step by ~3×
    and avoids allocating a huge intermediate tensor on long proteins."""
    rotated = ca_a @ r.T + t
    a_sq = np.einsum("ij,ij->i", rotated, rotated)  # (La,)
    b_sq = np.einsum("ij,ij->i", ca_b, ca_b)  # (Lb,)
    d_sq = a_sq[:, None] + b_sq[None, :] - 2.0 * (rotated @ ca_b.T)
    np.maximum(d_sq, 0.0, out=d_sq)  # Numerical floor — |a-b|² ≥ 0.
    return 1.0 / (1.0 + d_sq / (d0 * d0))


def dp_iter(
    ca_a: NDArray[np.floating],
    ca_b: NDArray[np.floating],
    idx_a: NDArray[np.int_],
    idx_b: NDArray[np.int_],
    d0: float,
    d0_search: float,
    gap_opens: tuple[float, ...],
    max_iters: int,
) -> tuple[
    float,
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.int_],
    NDArray[np.int_],
]:
    """Iterative DP ↔ `get_score_fast` refinement.

    Alternates between (a) DP alignment with the current rotation, and
    (b) `get_score_fast` 3-round Kabsch refinement for a new rotation. Tracks
    the best (score, R, t, alignment) across iterations and gap-opens.

    This is the *inner* loop — cheap and well-behaved. The orchestrator runs
    `tmscore8_search` once on the winner as a final polish; that's where the
    fragment-based rotation search happens.

    Convergence: |Δ score| < 1e-6 or max_iters reached.
    """
    from artios.metrics.alignment.kernels import get_score_fast

    best_score = -np.inf
    best_r = np.eye(3, dtype=np.float64)
    best_t = np.zeros(3, dtype=np.float64)
    best_idx_a = idx_a
    best_idx_b = idx_b

    cur_idx_a = idx_a
    cur_idx_b = idx_b
    for gap_open in gap_opens:
        score_prev = -np.inf
        for _ in range(max_iters):
            score, r, t = get_score_fast(
                ca_a, ca_b, cur_idx_a, cur_idx_b, d0, d0_search
            )
            if score > best_score:
                best_score = score
                best_r = r
                best_t = t
                best_idx_a = cur_idx_a
                best_idx_b = cur_idx_b
            if abs(score - score_prev) < 1e-6:
                break
            score_prev = score
            sm = build_score_matrix(ca_a, ca_b, r, t, d0)
            cur_idx_a, cur_idx_b = _fast_nw_dp(sm, gap_open)
            if cur_idx_a.size == 0:
                break
        cur_idx_a = best_idx_a
        cur_idx_b = best_idx_b

    return best_score, best_r, best_t, best_idx_a, best_idx_b
