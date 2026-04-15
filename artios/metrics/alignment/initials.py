"""Initial alignment strategies for TM-align.

Each strategy produces a candidate pair of aligned-index arrays `(idx_a, idx_b)`
that the orchestrator feeds into `get_score_fast` (and optionally `dp_iter`) to
decide which strategy won.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from artios.metrics.alignment.dp import _fast_nw_dp, build_score_matrix
from artios.metrics.alignment.kernels import (
    SS_HELIX,
    SS_STRAND,
    get_score_fast,
    kabsch,
)


def _gapless_pair(
    offset: int, la: int, lb: int
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Alignment `ca_a[i] ↔ ca_b[i + offset]` for every i where both indices
    are in-range."""
    if offset >= 0:
        a_start, b_start = 0, offset
    else:
        a_start, b_start = -offset, 0
    length = min(la - a_start, lb - b_start)
    if length <= 0:
        return np.empty(0, dtype=np.int_), np.empty(0, dtype=np.int_)
    idx_a = np.arange(a_start, a_start + length, dtype=np.int_)
    idx_b = np.arange(b_start, b_start + length, dtype=np.int_)
    return idx_a, idx_b


def initial_gapless(
    ca_a: NDArray[np.floating],
    ca_b: NDArray[np.floating],
    d0: float,
    d0_search: float,
    stride: int,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Slide the shorter chain over the longer chain at the given stride and
    return the offset-aligned pair with the highest `get_score_fast` score.
    Corresponds to TMalign's `get_initial`."""
    la, lb = ca_a.shape[0], ca_b.shape[0]
    min_ali = max(5, min(la, lb) // 2)
    best_score = -np.inf
    best = (
        np.arange(min(la, lb), dtype=np.int_),
        np.arange(min(la, lb), dtype=np.int_),
    )

    offsets = list(range(-(lb - min_ali), la - min_ali + 1, stride))
    for offset in offsets:
        idx_a, idx_b = _gapless_pair(offset, la, lb)
        if idx_a.size < min_ali:
            continue
        score, _r, _t = get_score_fast(ca_a, ca_b, idx_a, idx_b, d0, d0_search)
        if score > best_score:
            best_score = score
            best = (idx_a, idx_b)
    return best


def initial_ss(
    ss_a: NDArray[np.int8],
    ss_b: NDArray[np.int8],
    gap_open: float = -1.0,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """NW-DP over a secondary-structure match score matrix (+1 for matching
    SS, 0 otherwise). Corresponds to TMalign's `get_initial_ss`."""
    match = (ss_a[:, None] == ss_b[None, :]).astype(np.float64)
    return _fast_nw_dp(match, gap_open)


def initial_fragment_gapless(
    ca_a: NDArray[np.floating],
    ca_b: NDArray[np.floating],
    d0: float,
    d0_search: float,
    stride: int,
    dcu0: float = 4.25,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Find the longest contiguous Cα–Cα < dcu0 fragment in the *shorter*
    chain and slide it rigidly over the other. A coarse substitute for
    TMalign's `get_initial_fgt` — we pick one fragment from the shorter
    chain rather than enumerating fragments in both.
    """
    la, lb = ca_a.shape[0], ca_b.shape[0]
    if la <= lb:
        short, long_ = ca_a, ca_b
        short_is_a = True
    else:
        short, long_ = ca_b, ca_a
        short_is_a = False

    if short.shape[0] < 4:
        return initial_gapless(ca_a, ca_b, d0, d0_search, stride=max(stride, 1))

    consec = np.linalg.norm(np.diff(short, axis=0), axis=-1)
    continuous = consec < dcu0
    best_len = 0
    best_start = 0
    run_start = 0
    run_len = 0
    for i, ok in enumerate(continuous):
        if ok:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len + 1 > best_len:
                best_len = run_len + 1
                best_start = run_start
        else:
            run_len = 0
    if best_len < 4:
        return initial_gapless(ca_a, ca_b, d0, d0_search, stride=max(stride, 1))

    best_score = -np.inf
    best_idx = (np.empty(0, dtype=np.int_), np.empty(0, dtype=np.int_))
    for offset in range(0, long_.shape[0] - best_len + 1, stride):
        idx_short = np.arange(best_start, best_start + best_len, dtype=np.int_)
        idx_long = np.arange(offset, offset + best_len, dtype=np.int_)
        if short_is_a:
            idx_a, idx_b = idx_short, idx_long
        else:
            idx_a, idx_b = idx_long, idx_short
        score, _r, _t = get_score_fast(ca_a, ca_b, idx_a, idx_b, d0, d0_search)
        if score > best_score:
            best_score = score
            best_idx = (idx_a, idx_b)
    return best_idx


def _fragment_kabsch_align(
    ca_a: NDArray[np.floating],
    ca_b: NDArray[np.floating],
    a_start: int,
    b_start: int,
    length: int,
    d0: float,
    gap_open: float,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Kabsch-align a fragment pair, then run one NW-DP on the resulting full
    score matrix to get an alignment."""
    p = ca_a[a_start : a_start + length]
    q = ca_b[b_start : b_start + length]
    r, t = kabsch(p, q)
    sm = build_score_matrix(ca_a, ca_b, r, t, d0)
    return _fast_nw_dp(sm, gap_open)


def initial_local_superposition(
    ca_a: NDArray[np.floating],
    ca_b: NDArray[np.floating],
    d0: float,
    d0_search: float,
    stride_scale: int = 1,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Enumerate fragment pairs across both chains, Kabsch-align each pair,
    run one NW-DP against the resulting score matrix, and keep the best.
    Corresponds to TMalign's `get_initial5`. The O(L³)-ish cost dominates
    accurate mode; `stride_scale` multiplies the outer stride to dial it down
    (TMalign's `-fast` uses 5; we'd only call this in accurate mode with 1)."""
    la, lb = ca_a.shape[0], ca_b.shape[0]
    min_len = min(la, lb)
    fragment_sizes = [20, 100]
    fragment_sizes = [min(f, min_len // 2) for f in fragment_sizes if f <= min_len]
    if not fragment_sizes:
        fragment_sizes = [max(4, min_len // 2)]

    # Stride table from TMalign, coarsened by `stride_scale`.
    if min_len <= 50:
        base_stride = 15
    elif min_len <= 100:
        base_stride = 25
    elif min_len <= 200:
        base_stride = 35
    else:
        base_stride = 45
    stride_outer = max(1, base_stride * stride_scale)
    gap_open = 0.0

    best_score = -np.inf
    best_idx = (np.empty(0, dtype=np.int_), np.empty(0, dtype=np.int_))

    for flen in fragment_sizes:
        for a_start in range(0, la - flen + 1, stride_outer):
            for b_start in range(0, lb - flen + 1, stride_outer):
                idx_a, idx_b = _fragment_kabsch_align(
                    ca_a, ca_b, a_start, b_start, flen, d0, gap_open
                )
                if idx_a.size == 0:
                    continue
                score, _r, _t = get_score_fast(ca_a, ca_b, idx_a, idx_b, d0, d0_search)
                if score > best_score:
                    best_score = score
                    best_idx = (idx_a, idx_b)
    return best_idx


def initial_ss_plus_local(
    ca_a: NDArray[np.floating],
    ca_b: NDArray[np.floating],
    ss_a: NDArray[np.int8],
    ss_b: NDArray[np.int8],
    d0: float,
    d0_search: float,
    stride_scale: int = 1,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Hybrid of local superposition and SS matching. For each fragment pair,
    build a score matrix that combines TM-kernel distance scoring with a
    half-point SS-match bonus, then NW-DP. Corresponds to TMalign's
    `get_initial_ssplus`."""
    la, lb = ca_a.shape[0], ca_b.shape[0]
    min_len = min(la, lb)
    fragment_sizes = [20, 100]
    fragment_sizes = [min(f, min_len // 2) for f in fragment_sizes if f <= min_len]
    if not fragment_sizes:
        fragment_sizes = [max(4, min_len // 2)]
    if min_len <= 50:
        base_stride = 15
    elif min_len <= 100:
        base_stride = 25
    elif min_len <= 200:
        base_stride = 35
    else:
        base_stride = 45
    stride_outer = max(1, base_stride * stride_scale)

    ss_match = (ss_a[:, None] == ss_b[None, :]).astype(np.float64) * 0.5
    structured = ((ss_a[:, None] == SS_HELIX) | (ss_a[:, None] == SS_STRAND)).astype(
        np.float64
    )
    ss_bonus = ss_match * structured

    gap_open = -0.6
    best_score = -np.inf
    best_idx = (np.empty(0, dtype=np.int_), np.empty(0, dtype=np.int_))

    for flen in fragment_sizes:
        for a_start in range(0, la - flen + 1, stride_outer):
            for b_start in range(0, lb - flen + 1, stride_outer):
                p = ca_a[a_start : a_start + flen]
                q = ca_b[b_start : b_start + flen]
                r, t = kabsch(p, q)
                sm = build_score_matrix(ca_a, ca_b, r, t, d0) + ss_bonus
                idx_a, idx_b = _fast_nw_dp(sm, gap_open)
                if idx_a.size == 0:
                    continue
                score, _r, _t = get_score_fast(ca_a, ca_b, idx_a, idx_b, d0, d0_search)
                if score > best_score:
                    best_score = score
                    best_idx = (idx_a, idx_b)
    return best_idx
