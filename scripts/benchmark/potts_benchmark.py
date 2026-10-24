"""Benchmark ArtiosPotts on natural backbones through the production inference path.

Each NPZ (a depth-1 ``AtomArrayStack`` from natural-backbone curation, carrying
the native residue names) is run through ``get_inference_dataloader`` and
``InferenceRunner.predict`` exactly as ``python -m artios.runner.inference``
would. Per structure and seed it records native sequence recovery (over all
designed positions and over the active site the model itself uses), the Potts
energy of the sample and the runtime.

    uv run python scripts/benchmark/potts_benchmark.py native \\
        --npz-dir ~/bench/recentPDB_OOD_curated_npzs_v2 --out ~/runs/baseline

``colouring`` measures how often parallel Gibbs updates touch neighbouring
residues when several backbones share a batch (the colouring is computed from
the first backbone's graph only), and compares batched against one-at-a-time
sampling energies.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from artios.common.constants import STD_RESIDUES_AND_GAP
from artios.common.residue_names import PROTEIN_COMMON_THREE_TO_ONE
from artios.config import parse_configs
from artios.configs.configs_base import configs as configs_base
from artios.configs.configs_data import data_configs
from artios.configs.configs_inference import inference_configs
from artios.data.infer_data_pipeline import get_inference_dataloader
from artios.model.generator import _color_graph
from artios.model.utils import build_backbone_ligand_proximity_mask
from artios.potts import batch_contexts, load_target
from artios.runner.inference import InferenceRunner
from artios.utils.inference import ModelType
from artios.utils.seed import seed_everything

logger = logging.getLogger(__name__)

IDX_TO_RESTYPE = {v: k for k, v in STD_RESIDUES_AND_GAP.items()}


def build_configs(dump_dir: Path, extra_args: str = ""):
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    return parse_configs(
        configs=configs,
        arg_str=f"--model_type potts --dump_dir {dump_dir} {extra_args}",
        fill_required_with_null=True,
    )


def load_context(runner: InferenceRunner, configs, npz_path: Path, index_map=None):
    """Featurize ``npz_path`` with the same dataloader the inference CLI uses."""
    spec = {"npz_file": str(npz_path)}
    if index_map is not None:
        spec["backbone_index_map"] = index_map
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"name": npz_path.stem, "task": "sequence_design", "spec": spec}, f)
    configs.input_json_path = f.name
    (context,) = list(get_inference_dataloader(configs, use_distributed_sampler=False))
    Path(f.name).unlink()
    return context.to(runner.device)


def list_npzs(npz_dir: Path) -> list[Path]:
    """Benchmark entries: the curation manifest if present, else every NPZ."""
    manifest = npz_dir / "manifest.parquet"
    if manifest.exists():
        import polars as pl

        ids = pl.read_parquet(manifest)["pdb_id"].to_list()
        return [npz_dir / f"{i}.npz" for i in sorted(ids)]
    return sorted(p for p in npz_dir.glob("*.npz") if not p.name.startswith("."))


def to_sequence(onehot: torch.Tensor, mask: torch.Tensor) -> str:
    idx = onehot.argmax(-1)[mask].tolist()
    return "".join(
        PROTEIN_COMMON_THREE_TO_ONE.get(IDX_TO_RESTYPE.get(i), "X") for i in idx
    )


def native_metrics(runner, context, output) -> list[dict]:
    model = runner.model
    design = (
        context.mask.token_mask
        & context.entity.is_protein_token
        & ~context.motif.is_motif_related_token
    )
    active = (
        build_backbone_ligand_proximity_mask(
            backbone_coordinate=context.coordinate.backbone_coordinate,
            centre_coordinate=context.coordinate.centre_coordinate,
            is_ligand_token=context.entity.is_ligand_token,
            token_mask=context.mask.token_mask,
            backbone_mask=context.mask.backbone_mask,
            distance_cutoff=model.active_site_distance_cutoff,
        )
        & design
    )
    native = context.token.restype.argmax(-1)
    sampled = output.sampled_sequence.argmax(-1)
    correct = sampled == native
    rows = []
    for b in range(sampled.shape[0]):
        d, a = design[b], active[b]
        rows.append(
            {
                "n_design": int(d.sum()),
                "n_active": int(a.sum()),
                "recovery": float(correct[b][d].float().mean()),
                "recovery_active": (
                    float(correct[b][a].float().mean()) if a.any() else None
                ),
                "energy": float(output.energy[b][d].sum()),
                "native_sequence": to_sequence(context.token.restype[b], d),
                "designed_sequence": to_sequence(output.sampled_sequence[b], d),
            }
        )
    return rows


def run_native(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    configs = build_configs(args.out / "dump", args.extra_args)
    runner = InferenceRunner(configs, ModelType.POTTS)
    if args.targets is not None:
        entries = [json.loads(line) for line in args.targets.read_text().splitlines()]
        entries = [(e["pdb_id"], e) for e in entries][: args.limit]
    else:
        entries = [(p.stem, p) for p in list_npzs(args.npz_dir)][: args.limit]
    results_path = args.out / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["pdb_id"], r["seed"]))
    with results_path.open("a") as out:
        for i, (pdb_id, source) in enumerate(entries):
            for seed in args.seeds:
                if (pdb_id, seed) in done:
                    continue
                try:
                    if isinstance(source, dict):
                        target = load_target(
                            source["path"], chains=source.get("chains")
                        )
                        context = batch_contexts([target.build_context()]).to(
                            runner.device
                        )
                    else:
                        context = load_context(runner, configs, source)
                    seed_everything(seed=seed, deterministic=False)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    t0 = time.perf_counter()
                    output = runner.predict(context)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    elapsed = time.perf_counter() - t0
                    (row,) = native_metrics(runner, context, output)
                    row.update(pdb_id=pdb_id, seed=seed, seconds=elapsed)
                except Exception as e:  # keep going; record the failure
                    row = {"pdb_id": pdb_id, "seed": seed, "error": repr(e)}
                out.write(json.dumps(row) + "\n")
                out.flush()
                logger.info(
                    "[%d/%d] %s seed=%d %s",
                    i + 1,
                    len(entries),
                    pdb_id,
                    seed,
                    {
                        k: row.get(k)
                        for k in ("recovery", "recovery_active", "seconds", "error")
                    },
                )
    summarize(results_path)


def summarize(results_path: Path) -> None:
    rows = [json.loads(line) for line in results_path.read_text().splitlines()]
    ok = [r for r in rows if "error" not in r]
    failed = sorted({r["pdb_id"] for r in rows if "error" in r})
    rec = np.array([r["recovery"] for r in ok])
    rec_a = np.array(
        [r["recovery_active"] for r in ok if r["recovery_active"] is not None]
    )
    secs = np.array([r["seconds"] for r in ok])
    summary = {
        "n_runs": len(ok),
        "n_structures": len({r["pdb_id"] for r in ok}),
        "failed": failed,
        "recovery_mean": float(rec.mean()) if len(rec) else None,
        "recovery_median": float(np.median(rec)) if len(rec) else None,
        "recovery_active_mean": float(rec_a.mean()) if len(rec_a) else None,
        "seconds_mean": float(secs.mean()) if len(secs) else None,
        "seconds_total": float(secs.sum()) if len(secs) else None,
    }
    (results_path.parent / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def colouring_conflicts(edge_index, edge_mask, color_masks) -> list[int]:
    """Per batch element, count graph edges whose endpoints share a colour."""
    colour = torch.full(edge_index.shape[:2], -1, device=edge_index.device)
    for c, m in enumerate(color_masks):
        colour[m] = c
    counts = []
    for b in range(edge_index.shape[0]):
        src = torch.arange(edge_index.shape[1], device=edge_index.device)
        src = src[:, None].expand_as(edge_index[b])
        dst = edge_index[b]
        valid = edge_mask[b] & (src != dst)
        same = colour[b][src] == colour[b][dst]
        counts.append(int((same & valid).sum()))
    return counts


def run_colouring(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    configs = build_configs(args.out / "dump", args.extra_args)
    runner = InferenceRunner(configs, ModelType.POTTS)

    context = load_context(runner, configs, args.npz)
    n = context.token.restype.shape[0]
    seed_everything(seed=args.seeds[0], deterministic=False)
    with torch.no_grad():
        output = runner.predict(context)
    colours = _color_graph(output.edge_index, output.edge_mask)
    conflicts = colouring_conflicts(output.edge_index, output.edge_mask, colours)
    n_edges = [int(m.sum()) for m in output.edge_mask]

    energies = {
        "batched": np.zeros((len(args.seeds), n)),
        "solo": np.zeros((len(args.seeds), n)),
    }
    for s, seed in enumerate(args.seeds):
        seed_everything(seed=seed, deterministic=False)
        out = runner.predict(context)
        energies["batched"][s] = [
            r["energy"] for r in native_metrics(runner, context, out)
        ]
        for b in range(n):
            solo_ctx = load_context(runner, configs, args.npz, index_map=[b])
            seed_everything(seed=seed, deterministic=False)
            out = runner.predict(solo_ctx)
            energies["solo"][s, b] = native_metrics(runner, solo_ctx, out)[0]["energy"]

    report = {
        "npz": str(args.npz),
        "seeds": args.seeds,
        "n_colours": len(colours),
        "same_colour_edges": conflicts,
        "n_edges": n_edges,
        "energy_batched_mean": energies["batched"].mean(0).tolist(),
        "energy_solo_mean": energies["solo"].mean(0).tolist(),
        "energy_batched": energies["batched"].tolist(),
        "energy_solo": energies["solo"].tolist(),
    }
    (args.out / "colouring.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in list(report)[:7]}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    native = sub.add_parser("native", help="native sequence recovery benchmark")
    source = native.add_mutually_exclusive_group(required=True)
    source.add_argument("--npz-dir", type=Path, help="curated NPZs (production path)")
    source.add_argument(
        "--targets",
        type=Path,
        help="JSONL of {pdb_id, path, chains} loaded with artios.potts.load_target",
    )
    native.add_argument("--limit", type=int, default=None)
    colouring = sub.add_parser("colouring", help="batched Gibbs colouring check")
    colouring.add_argument(
        "--npz", type=Path, default=Path("examples/sequence_design/backbone.npz")
    )
    for sp in (native, colouring):
        sp.add_argument("--out", type=Path, required=True)
        sp.add_argument("--seeds", type=int, nargs="+", default=[101])
        sp.add_argument(
            "--extra-args",
            default="",
            help="extra config overrides, e.g. '--load_potts_checkpoint_path x.pt'",
        )
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    {"native": run_native, "colouring": run_colouring}[args.cmd](args)


if __name__ == "__main__":
    main()
