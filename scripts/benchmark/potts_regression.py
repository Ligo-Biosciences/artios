"""Bit-exact regression check for ArtiosPotts inference.

``capture`` runs a fixed set of inputs, seeds and sampling configurations
through ``InferenceRunner.predict`` and saves every output tensor. ``compare``
reruns the same cases with the current code and requires each tensor to be
identical. Run ``capture`` before a change to model code and ``compare`` after.

    uv run python scripts/benchmark/potts_regression.py capture --out ref.pt
    uv run python scripts/benchmark/potts_regression.py compare --ref ref.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from potts_benchmark import build_configs, load_context  # noqa: E402

from artios.potts import batch_contexts, load_target, parse_selection  # noqa: E402
from artios.runner.inference import InferenceRunner  # noqa: E402
from artios.utils.inference import ModelType  # noqa: E402
from artios.utils.seed import seed_everything  # noqa: E402

FIELDS = ("h", "J", "edge_index", "edge_mask", "sampled_sequence", "energy")
SAMPLING = "--artios_mpnn.potts.sampling"
CONFIGS = {
    "default": "",
    "cfg": f"{SAMPLING}.cfg_scale 2.0",
    "penalties": (
        f"{SAMPLING}.aromatic_penalty_weight 1.0 {SAMPLING}.charged_penalty_weight 1.0 "
        f"{SAMPLING}.composition_penalty_weight 1.0 {SAMPLING}.net_charge_penalty_weight 1.0 "
        f"{SAMPLING}.complexity_nat_prior_penalty_weight 0.5"
    ),
    "dlmc": f"{SAMPLING}.proposal dlmc",
}


def cases(args):
    """(name, config key, context builder, seeds)."""
    bench = args.bench_dir
    example = Path("examples/sequence_design/backbone.npz")
    natives = [bench / f"{i}.npz" for i in ("7bmf", "7eut", "7exh")]
    out = [("example_batched", "default", ("npz", example), (101, 202))]
    out += [(f"native_{p.stem}", "default", ("npz", p), (101,)) for p in natives]
    if args.cif is not None:
        out.append(("loader_fixed_chain", "default", ("cif", args.cif), (7,)))
    for key in ("cfg", "penalties", "dlmc"):
        out.append((f"{key}_7bmf", key, ("npz", natives[0]), (101,)))
        out.append((f"{key}_example", key, ("npz", example), (101,)))
    return out


def build_context(runner, configs, source):
    kind, path = source
    if kind == "npz":
        return load_context(runner, configs, path)
    target = load_target(path)
    chains = target.protein_chains
    design = chains[-1]
    first = next(r for r in target.residues if r is not None and r.chain == design)
    context = target.build_context(
        design_chains=[design],
        fixed=parse_selection(f"{design}{first.number}-{first.number + 4}"),
    )
    return batch_contexts([context]).to(runner.device)


def run(args) -> dict:
    results = {}
    runners: dict[str, tuple] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, key, source, seeds in cases(args):
            if key not in runners:
                configs = build_configs(Path(tmp) / key, CONFIGS[key])
                runners[key] = (InferenceRunner(configs, ModelType.POTTS), configs)
            runner, configs = runners[key]
            context = build_context(runner, configs, source)
            for seed in seeds:
                seed_everything(seed=seed, deterministic=False)
                output = runner.predict(context)
                results[f"{name}/seed{seed}"] = {
                    f: getattr(output, f).detach().cpu() for f in FIELDS
                }
                logging.info("ran %s seed %d", name, seed)
    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("cmd", choices=["capture", "compare"])
    p.add_argument("--out", type=Path, help="capture: where to save reference outputs")
    p.add_argument("--ref", type=Path, help="compare: reference from capture")
    p.add_argument(
        "--bench-dir",
        type=Path,
        default=Path.home() / "bench/recentPDB_OOD_curated_npzs_v2",
    )
    p.add_argument(
        "--cif",
        type=Path,
        default=None,
        help="multi-chain mmCIF/PDB for the fixed-chain case",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger(__name__).setLevel(logging.INFO)

    results = run(args)
    if args.cmd == "capture":
        torch.save(results, args.out)
        print(f"captured {len(results)} cases -> {args.out}")
        return

    ref = torch.load(args.ref)
    failures = []
    if set(ref) != set(results):
        failures.append(f"case sets differ: {sorted(set(ref) ^ set(results))}")
    for case in sorted(set(ref) & set(results)):
        for f in FIELDS:
            a, b = ref[case][f], results[case][f]
            if a.shape != b.shape or not torch.equal(a, b):
                diff = (
                    (a.float() - b.float()).abs().max().item()
                    if a.shape == b.shape
                    else "shape"
                )
                failures.append(f"{case}.{f}: max abs diff {diff}")
    for line in failures:
        print("MISMATCH", line)
    print(f"{len(ref)} cases, {len(failures)} mismatches")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
