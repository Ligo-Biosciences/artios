"""Artios: design ligand-binding proteins.

    artios scaffold examples/backbone/input.json -o out/backbone
    artios design out/backbone/odcase_backbone.npz -o out/designs -n 8
    artios design complex.pdb --design-chains B --fixed "B10-14 B42" -o out/binder
    artios score complex.pdb out/binder/designs.fasta -o out/scores
    artios params complex.pdb -o out/params

Structures are PDB, mmCIF or ArtiosBackbone NPZ files (one target per
backbone), or directories of them. Residues are selected in author numbering:
``A`` (chain), ``A42``, ``A52B`` (insertion code), ``A10-20``, ``1:42`` (chain
IDs that are not letters).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("artios.cli")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Progress and errors go to stderr; library logs and warnings only with -v.
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("artios: %(message)s"))
    logger.handlers[:] = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logging.captureWarnings(True)
    library_level = logging.INFO if args.verbose else logging.ERROR
    for name in ("artios", "py.warnings"):
        logging.getLogger(name).setLevel(library_level)
    try:
        return args.func(args)
    except ValueError as e:  # bad selections, sequences, settings
        if args.verbose:
            raise
        logger.error("%s", e)
        return 2


# --------------------------------------------------------------------- parser


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="artios",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(title="commands", required=True, metavar="COMMAND")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="output directory",
    )
    common.add_argument(
        "--checkpoint",
        type=Path,
        metavar="PATH",
        help="model weights (default: released weights)",
    )
    common.add_argument(
        "--device",
        help="cuda, cuda:N or cpu (default: cuda if available); Apple MPS is "
        "not supported",
    )
    common.add_argument("-v", "--verbose", action="store_true", help="log details")

    s = sub.add_parser(
        "scaffold",
        parents=[common],
        help="generate backbones around a motif (ArtiosBackbone)",
        description="Generate backbones around a motif with ArtiosBackbone. "
        "INPUT is a backbone input JSON (see docs/motif_input_format.md).",
    )
    s.add_argument("input", type=Path, metavar="INPUT")
    s.add_argument(
        "-n", "--num-samples", type=int, default=5, metavar="N", help="default: 5"
    )
    s.add_argument("--seed", type=int, default=101, help="default: 101")
    s.add_argument(
        "--format",
        choices=["npz", "cif"],
        default="npz",
        help="npz (one file, input to `artios design`) or one CIF per sample",
    )
    s.set_defaults(func=_scaffold)

    potts = argparse.ArgumentParser(add_help=False, parents=[common])
    potts.add_argument(
        "--chains",
        type=_comma_list,
        metavar="A,B",
        help="keep only these author chains of PDB/mmCIF inputs, e.g. A,B",
    )
    potts.add_argument("--seed", type=int, default=0, help="default: 0")
    potts.add_argument(
        "--set",
        dest="sampling",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="any artios_mpnn.potts.sampling option, e.g. cfg_scale=2.0 or "
        "energy_bias.W=-0.5 (repeatable)",
    )

    d = sub.add_parser(
        "design",
        parents=[potts],
        help="design sequences (ArtiosPotts)",
        description="Design sequences with ArtiosPotts. Writes designs.fasta "
        "(the designed chains, ':'-separated) and designs.jsonl (all chains, "
        "energies, designed positions). Rerunning into the same directory "
        "skips finished targets.",
    )
    d.add_argument("structures", type=Path, nargs="+", metavar="STRUCTURE")
    d.add_argument(
        "--design-chains",
        type=_comma_list,
        metavar="B,C",
        help="chains to design; others keep their sequence (default: all)",
    )
    d.add_argument(
        "--fixed",
        action="append",
        metavar="SEL",
        help='residues that keep their sequence, e.g. "B10-14 B42"',
    )
    d.add_argument(
        "--redesign",
        action="append",
        metavar="SEL",
        help="design only these residues (--fixed still applies)",
    )
    d.add_argument(
        "-n", "--num-sequences", type=int, default=8, metavar="N", help="default: 8"
    )
    d.add_argument("--batch-size", type=int, default=8, metavar="N", help="default: 8")
    d.add_argument(
        "--temperature",
        type=float,
        metavar="T",
        help="final sampling temperature (default: 0.1)",
    )
    d.add_argument("--allow-cys", action="store_true", help="allow cysteine")
    d.add_argument("--allow-met", action="store_true", help="allow methionine")
    d.add_argument(
        "--overwrite", action="store_true", help="discard existing results in OUT_DIR"
    )
    d.set_defaults(func=_design)

    sc = sub.add_parser(
        "score",
        parents=[potts],
        help="score sequences on a structure (ArtiosPotts)",
        description="Score sequences with ArtiosPotts: Potts energy and "
        "pseudo-log-likelihood. Records whose header has target=NAME (as "
        "written by `artios design`) are scored on that target only. Chains "
        "are ':'-separated, in the order of chains=A:B in the header, else "
        "the protein chains in file order; chains left out keep their "
        "sequence. Without SEQUENCES the input structure's sequence is scored.",
    )
    sc.add_argument("structure", type=Path, metavar="STRUCTURE")
    sc.add_argument("sequences", type=Path, nargs="?", metavar="SEQUENCES")
    sc.set_defaults(func=_score)

    pa = sub.add_parser(
        "params",
        parents=[potts],
        help="export Potts fields h, couplings J and the neighbour graph",
        description="Write {target}.npz with h [N, 32], J [N, K, 32, 32], "
        "edge_index [N, K] and edge_mask [N, K] over all N tokens, plus the "
        "input's token identities (native), per-token residue labels ('' for "
        "ligands and motif atoms) and the alphabet. Energies are sums over tokens of "
        "h[i, s_i] + 1/2 sum_k J[i, k, s_i, s_edge_index[i, k]], lower is better.",
    )
    pa.add_argument("structures", type=Path, nargs="+", metavar="STRUCTURE")
    pa.set_defaults(func=_params)
    return p


def _comma_list(text: str) -> list[str]:
    return [t for t in text.replace(" ", ",").split(",") if t]


# ------------------------------------------------------------------- commands


def _scaffold(args) -> int:
    import os

    # The backbone runner picks its own device; restrict it before torch loads.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.device and args.device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.removeprefix("cuda:")

    from artios.config import parse_configs
    from artios.configs.configs_base import configs as configs_base
    from artios.configs.configs_data import data_configs
    from artios.configs.configs_inference import inference_configs
    from artios.runner.inference import InferenceRunner, infer_predict
    from artios.utils.inference import ModelType

    if not args.verbose:  # artios.utils.logger resets the root level on import
        logging.getLogger().setLevel(logging.WARNING)
    arg_str = (
        f"--model_type backbone --input_json_path {_q(args.input)} "
        f"--dump_dir {_q(args.out_dir)} --sample_diffusion.N_sample {args.num_samples} "
        f"--seeds {args.seed} --output_format {args.format} --num_workers 0"
    )
    if args.checkpoint:
        arg_str += f" --load_backbone_checkpoint_path {_q(args.checkpoint)}"
    configs = parse_configs(
        configs={**configs_base, "data": data_configs, **inference_configs},
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    runner = InferenceRunner(configs, ModelType.BACKBONE)
    result = infer_predict(runner, configs)
    paths = result[0] if isinstance(result, tuple) else result
    # The runner pre-creates per-sample and error directories; drop empty ones.
    for d in [*runner.dump_dir_list, runner.error_dir]:
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
    if not paths:
        logger.error("backbone generation failed; rerun with -v for details")
        return 1
    for path in paths:
        print(path)
    return 0


def _design(args) -> int:
    from artios.potts import parse_selection

    fixed = parse_selection(args.fixed)
    redesign = parse_selection(args.redesign) if args.redesign else None
    settings = {
        k: _jsonable(v)
        for k, v in vars(args).items()
        if k not in {"func", "structures", "out_dir", "verbose", "overwrite", "device"}
    }
    out = args.out_dir
    jsonl, fasta, settings_file = (
        out / "designs.jsonl",
        out / "designs.fasta",
        out / "settings.json",
    )
    if not args.overwrite and _read_jsonl(jsonl) and settings_file.exists():
        if json.loads(settings_file.read_text()) != settings:
            logger.error(
                "%s holds results from different settings; use --overwrite or "
                "another --out-dir",
                out,
            )
            return 1
    # Check the options (by building the designer) before touching OUT_DIR.
    designer = _designer(args, _sampling(args))
    if args.overwrite:
        for f in (jsonl, fasta, settings_file):
            f.unlink(missing_ok=True)
    existing = _read_jsonl(jsonl)
    out.mkdir(parents=True, exist_ok=True)
    settings_file.write_text(json.dumps(settings, indent=2) + "\n")

    done = {r["target"] for r in existing}
    # FASTA is derived from the JSONL, which is written last for each target.
    with fasta.open("w") as f:
        for record in existing:
            f.write(_fasta_record(record))

    load_failures: list[Path] = []
    failures = 0
    for target in _targets(args.structures, args.chains, load_failures):
        if target.name in done:
            logger.info("%s: done, skipping", target.name)
            continue
        logger.info("%s: designing %d sequences", target.name, args.num_sequences)
        start = time.perf_counter()
        try:
            designs = designer.design(
                target,
                design_chains=args.design_chains,
                fixed=fixed,
                redesign=redesign,
                num_sequences=args.num_sequences,
                batch_size=args.batch_size,
                seed=args.seed,
            )
        except Exception as e:  # keep going over the other targets
            logger.error("%s: %s", target.name, e, exc_info=args.verbose)
            failures += 1
            continue
        records = [_design_record(d) for d in designs]
        with fasta.open("a") as f:
            f.writelines(_fasta_record(r) for r in records)
        with jsonl.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
        best = min(d.energy for d in designs)
        logger.info(
            "%s: %d designs, best energy %.2f (%.1f s)",
            target.name,
            len(designs),
            best,
            time.perf_counter() - start,
        )
    failures += len(load_failures)
    if fasta.stat().st_size:
        print(fasta)
    return 1 if failures else 0


def _score(args) -> int:
    failures: list[Path] = []
    targets = list(_targets([args.structure], args.chains, failures))
    if failures:
        return 1
    designer = _designer(args, _sampling(args))
    records = list(_read_fasta(args.sequences)) if args.sequences else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "scores.jsonl"
    rows = []
    with out.open("w") as f:
        for target in targets:
            if records is None:
                batch = [(f"{target.name}_native", {})]
            else:
                batch = [
                    (r.id, _chain_sequences(target, r))
                    for r in records
                    if r.fields.get("target", target.name) == target.name
                ]
            if not batch:
                continue
            scores = designer.score(target, [s for _, s in batch], seed=args.seed)
            for (rid, _), score in zip(batch, scores):
                record = {"id": rid, **asdict(score)}
                f.write(json.dumps(_jsonable(record)) + "\n")
                rows.append((rid, target.name, score))
    if records is not None and not rows:
        logger.error(
            "No sequence in %s matches a target in %s", args.sequences, args.structure
        )
        return 1
    print("id\ttarget\tenergy\tpseudo_log_likelihood")
    for rid, name, score in rows:
        print(f"{rid}\t{name}\t{score.energy:.3f}\t{score.pseudo_log_likelihood:.4f}")
    return 0


def _params(args) -> int:
    import numpy as np

    from artios.common.constants import STD_RESIDUES_AND_GAP

    designer = _designer(args, _sampling(args))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    alphabet = sorted(STD_RESIDUES_AND_GAP, key=STD_RESIDUES_AND_GAP.get)
    failures = []
    for target in _targets(args.structures, args.chains, failures):
        params, context = designer.potts_params(target, seed=args.seed)
        n = len(target.residues)
        path = args.out_dir / f"{target.name}.npz"
        edge_index = params.edge_index[0, :n].cpu().numpy()
        # The graph lists each token as its own neighbour; energies skip it.
        not_self = edge_index != np.arange(n)[:, None]
        np.savez_compressed(
            path,
            h=params.h[0, :n].float().cpu().numpy(),
            J=params.J[0, :n].float().cpu().numpy(),
            edge_index=edge_index,
            edge_mask=params.edge_mask[0, :n].cpu().numpy() & not_self,
            native=context.token.restype[0, :n].argmax(-1).cpu().numpy(),
            labels=np.array([r.label if r else "" for r in target.residues]),
            alphabet=np.array(alphabet),
        )
        print(path)
    return 1 if failures else 0


# -------------------------------------------------------------------- helpers


def _designer(args, sampling: dict):
    from artios.potts import PottsDesigner

    return PottsDesigner(
        checkpoint=args.checkpoint, device=args.device, sampling=sampling
    )


def _sampling(args) -> dict[str, str]:
    sampling = {}
    for item in args.sampling:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        sampling[key] = value
    if getattr(args, "temperature", None) is not None:
        sampling["temperature"] = args.temperature
    if getattr(args, "allow_cys", False):
        sampling["exclude_cys"] = False
    if getattr(args, "allow_met", False):
        sampling["exclude_met"] = False
    return sampling


def _targets(paths: list[Path], chains, failures: list) -> Iterator:
    """Targets of every structure file in ``paths`` (directories are expanded).

    Files that fail to load are logged and appended to ``failures``.
    """
    from artios.potts.structure import STRUCTURE_SUFFIXES, load_targets

    files = []
    for path in paths:
        if path.is_dir():
            files += sorted(
                f
                for f in path.iterdir()
                if f.name.lower().endswith(STRUCTURE_SUFFIXES)
                and not f.name.startswith(".")
            )
        elif path.exists():
            files.append(path)
        else:
            logger.error("%s: no such file or directory", path)
            failures.append(path)
    # Targets are named after the file stem, which keys outputs and resuming.
    stems: dict[str, list[str]] = {}
    for f in files:
        stems.setdefault(f.name.split(".")[0], []).append(str(f))
    clashes = [group for group in stems.values() if len(group) > 1]
    if clashes:
        raise ValueError(f"Inputs share a file name, rename them: {clashes}")
    for f in files:
        try:
            targets = load_targets(f, chains=chains)
        except Exception as e:
            logger.error("%s: %s", f, e)
            failures.append(f)
            continue
        yield from targets


def _design_record(d) -> dict:
    record = asdict(d)
    return {"id": f"{d.target}_{d.index}", **_jsonable(record)}


def _fasta_record(r: dict) -> str:
    chains = [c for c, m in r["designed"].items() if "1" in m]
    header = (
        f">{r['id']} target={r['target']} seed={r['seed']} "
        f"energy={r['energy']:.3f} chains={':'.join(chains)}"
    )
    if r["native_identity"] is not None:
        header += f" native_identity={r['native_identity']:.3f}"
    return header + "\n" + ":".join(r["sequences"][c] for c in chains) + "\n"


@dataclass(frozen=True)
class _FastaRecord:
    id: str
    fields: dict[str, str]
    sequence: str


def _read_fasta(path: Path) -> Iterator[_FastaRecord]:
    header, lines = None, []
    for line in [*path.read_text().splitlines(), ">"]:
        line = line.strip()
        if line.startswith(">"):
            if header is not None:
                rid, *rest = header.split()
                fields = dict(t.split("=", 1) for t in rest if "=" in t)
                yield _FastaRecord(rid, fields, "".join(lines))
            header, lines = line[1:], []
        elif line:
            lines.append(line)


def _chain_sequences(target, record: _FastaRecord) -> dict[str, str]:
    parts = record.sequence.split(":")
    if "chains" in record.fields:
        chains = record.fields["chains"].split(":")
    else:
        chains = target.protein_chains
    if len(parts) != len(chains):
        raise ValueError(
            f"{record.id} has {len(parts)} ':'-separated chains, expected "
            f"{len(chains)} ({':'.join(chains)}); set chains=... in the header"
        )
    return dict(zip(chains, parts))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _q(path) -> str:
    import shlex

    return shlex.quote(str(path))


if __name__ == "__main__":
    sys.exit(main())
