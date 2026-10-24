"""Compare ``artios.potts.load_target`` against curated natural-backbone NPZs.

For each curated NPZ (from ``evaluation/scripts/curate_natural_backbones.py``
in the research repo) this loads the deposited mmCIF with ``load_target``,
finds the protein chain containing the curated sequence, and checks that the
two paths give the same residues, the same backbone (RMSD after superposition)
and the same ligands. It writes ``targets.jsonl`` (pdb_id, mmCIF path, chain)
for ``potts_benchmark.py native --targets``.

    uv run python scripts/benchmark/loader_parity.py \\
        --npz-dir ~/bench/golden-monomers-curated --cif-dir ~/scratch/golden_cif \\
        --out ~/runs/parity_golden
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np

from artios.common.residue_names import PROTEIN_COMMON_THREE_TO_ONE
from artios.data.array_builder import sequence_design_input_to_structure
from artios.potts import load_target
from artios.utils.file_io import load_atomarray_stack

logger = logging.getLogger(__name__)


def _protein_ca(structure):
    """(one-letter sequence, CA coords, ligand res_names) of a structure."""
    atoms = structure.atoms
    centre = structure.centre_atom_indices
    is_prot = structure.type_masks.chain_mol_type[centre] == "protein"
    seq = "".join(
        PROTEIN_COMMON_THREE_TO_ONE.get(str(atoms.res_name[i]), "X")
        for i in centre[is_prot]
    )
    ca = []
    for i in centre[is_prot]:
        res = (atoms.chain_id == atoms.chain_id[i]) & (atoms.res_id == atoms.res_id[i])
        hit = np.flatnonzero(res & (atoms.atom_name == "CA"))
        ca.append(atoms.coord[hit[0]] if len(hit) else np.full(3, np.nan))
    ligands = sorted(str(atoms.res_name[i]) for i in centre[~is_prot])
    return seq, np.array(ca), ligands


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    ok = ~(np.isnan(a).any(1) | np.isnan(b).any(1))
    a, b = a[ok] - a[ok].mean(0), b[ok] - b[ok].mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((a @ rot - b) ** 2).sum(1).mean()))


def compare_one(npz: Path, cif_dir: Path) -> dict:
    pdb_id = npz.stem
    row: dict = {"pdb_id": pdb_id}
    try:
        stack = load_atomarray_stack(npz)
        curated, _ = sequence_design_input_to_structure(stack[0])
        c_seq, c_ca, c_lig = _protein_ca(curated)
        cif = next(cif_dir.glob(f"{pdb_id}.cif*"))
        full = load_target(cif)
        seqs = full.native_sequences()
        chain = next((c for c, s in seqs.items() if c_seq in s), None)
        if chain is None:
            row.update(status="no_matching_chain", curated_len=len(c_seq))
            return row
        target = load_target(cif, chains=[chain])
        t_seq, t_ca, t_lig = _protein_ca(target.structure)
        offset = t_seq.index(c_seq)
        window = t_ca[offset : offset + len(c_seq)]
        row.update(
            status="ok",
            path=str(cif),
            chain=chain,
            curated_len=len(c_seq),
            loader_len=len(t_seq),
            extra_residues=len(t_seq) - len(c_seq),
            ca_rmsd=_kabsch_rmsd(window, c_ca),
            curated_ligands=c_lig,
            loader_ligands=t_lig,
            same_ligands=c_lig == t_lig,
        )
    except Exception as e:
        row.update(status="error", error=repr(e))
    return row


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--npz-dir", type=Path, required=True)
    p.add_argument("--cif-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    import polars as pl

    ids = sorted(pl.read_parquet(args.npz_dir / "manifest.parquet")["pdb_id"].to_list())
    npzs = [args.npz_dir / f"{i}.npz" for i in ids][: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    with mp.Pool(args.workers) as pool:
        rows = pool.starmap(compare_one, [(n, args.cif_dir) for n in npzs])

    (args.out / "parity.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ok = [r for r in rows if r["status"] == "ok"]
    with (args.out / "targets.jsonl").open("w") as f:
        for r in ok:
            f.write(
                json.dumps(
                    {"pdb_id": r["pdb_id"], "path": r["path"], "chains": [r["chain"]]}
                )
                + "\n"
            )

    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    rmsd = np.array([r["ca_rmsd"] for r in ok])
    extra = np.array([r["extra_residues"] for r in ok])
    summary = {
        "n": len(rows),
        "status": statuses,
        "ca_rmsd_max": float(rmsd.max()) if len(rmsd) else None,
        "n_ca_rmsd_gt_0.01": int((rmsd > 0.01).sum()),
        "n_same_ligands": sum(r["same_ligands"] for r in ok),
        "n_extra_residues": int((extra > 0).sum()),
        "extra_residues_max": int(extra.max()) if len(extra) else None,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
