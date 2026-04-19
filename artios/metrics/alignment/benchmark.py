"""Benchmark `tm_score` against the reference TMalign binary.

Reads pairs from the fixture files, runs both implementations, reports
per-length-bucket wall-clock and accuracy deltas.

    python -m artios.metrics.alignment.benchmark --help

Filled out in Phase 4; Phase 1 ships a skeleton that only reports timings for
`tm_score` itself so we can track regressions early.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from artios.metrics.alignment import load_ca, tm_score


def _run(fixture_path: Path, mode: str) -> None:
    fixture = json.loads(fixture_path.read_text())
    pairs = fixture["pairs"]
    sizes: list[int] = []
    times: list[float] = []
    deltas: list[float] = []

    for pair in pairs:
        ca_a = load_ca(pair["path_a"])
        ca_b = load_ca(pair["path_b"])
        ref_tm_a = float(pair["tm_norm_a"])
        ref_tm_b = float(pair["tm_norm_b"])
        ref_tm_avg = float(pair["tm_norm_avg"])

        if mode in ("fast", "both"):
            t0 = time.perf_counter()
            r = tm_score(ca_a, ca_b, fast=True)
            elapsed = time.perf_counter() - t0
            sizes.append(max(ca_a.shape[0], ca_b.shape[0]))
            times.append(elapsed)
            deltas.append(
                max(
                    abs(r.tm_norm_a - ref_tm_a),
                    abs(r.tm_norm_b - ref_tm_b),
                    abs(r.tm_norm_avg - ref_tm_avg),
                )
            )
        if mode in ("accurate", "both"):
            t0 = time.perf_counter()
            r = tm_score(ca_a, ca_b, fast=False)
            elapsed = time.perf_counter() - t0
            sizes.append(max(ca_a.shape[0], ca_b.shape[0]))
            times.append(elapsed)
            deltas.append(
                max(
                    abs(r.tm_norm_a - ref_tm_a),
                    abs(r.tm_norm_b - ref_tm_b),
                    abs(r.tm_norm_avg - ref_tm_avg),
                )
            )

    print(f"[{mode}] pairs={len(pairs)}")
    print(f"  mean time   = {np.mean(times) * 1000:.1f} ms")
    print(f"  median time = {np.median(times) * 1000:.1f} ms")
    print(f"  mean |Δ|    = {np.mean(deltas):.4f}")
    print(f"  max  |Δ|    = {np.max(deltas):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        choices=["small", "comprehensive"],
        default="small",
        help="Which fixture to run against.",
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "accurate", "both"],
        default="both",
    )
    args = parser.parse_args()

    fixture_dir = Path(__file__).resolve().parents[3] / "tests" / "fixtures"
    fname = {
        "small": "tmalign_small.json",
        "comprehensive": "tmalign_comprehensive.json",
    }[args.set]
    _run(fixture_dir / fname, args.mode)


if __name__ == "__main__":
    main()
