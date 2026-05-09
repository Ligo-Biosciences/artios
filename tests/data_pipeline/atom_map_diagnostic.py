"""Atom-map inference diagnostic visualiser.

For each ground-truth fixture in `test_atom_map.py`, render a single
multi-panel PNG showing:

  1. The hand-curated GROUND TRUTH labelling (where each CIF atom name
     lands in the SMILES per the JSON), scored under a representative
     diagnostic conformer.
  2. The TOP-K candidate substructure matches by Kabsch RMSD on that
     SAME conformer — so all RMSDs are directly comparable.
  3. Each panel is annotated with how often the function picked that
     match across N independent inference runs (each generating a
     fresh ETKDG conformer). This exposes the "function flips between
     near-tied candidates depending on conformer seed" failure mode.

Run from the repo root:

    python tests/data_pipeline/atom_map_diagnostic.py

Output: `atom_map_diagnostic/<case_id>.png`. Filename is deterministic
per case so the script overwrites on re-run.

Not a test (no `test_` prefix) — pytest won't collect it.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AdjustQueryParameters, AdjustQueryProperties, AllChem, Draw

from artios.data.chem.atom_map import (
    _kabsch_align,
    infer_atom_map_with_disambiguation,
)
from artios.data.chem.rdkit import atoms_to_rdkit_mol, smiles_to_rdkit_mol
from artios.data.parsing.mmcif import MmcifReader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path("atom_map_diagnostic")
TOP_K = 5
N_TRIALS = 10  # for measuring conformer-noise stability of the function

# Mirrors the parametrize list in test_atom_map.py.
CASES = [
    (
        "dtz_2_atom",
        "evaluation/inputs/artios/dtz/his_arg_4_atom_dtz.json",
        "evaluation/inputs/motifs/dtz/his_arg_4_atom_dtz.cif",
        "1",
        "DTZ",
    ),
    (
        "cutinase_5_atom",
        "evaluation/inputs/artios/cutinase/cutinase.json",
        "evaluation/inputs/motifs/cutinase/cutinase_motif.cif",
        "1",
        "UNK",
    ),
    (
        "cysteine_hydrolase_5_atom",
        "evaluation/inputs/artios/cysteine_hydrolase/cysteine_hydrolase_no_ser_trp_no_asn_n.json",
        "evaluation/inputs/motifs/cysteine-hydrolase/no_ser_trp_motif_clean_no_asn_n.cif",
        "1",
        "UNK",
    ),
]


def _strip_atom_map_numbers(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def _render_2d_image(
    plain_smiles: str,
    matched_indices: tuple[int, ...],
    motif_atom_names: tuple[str, ...],
    *,
    size: tuple[int, int] = (900, 700),
):
    """2D depiction with CIF atom names labelled on matched atoms,
    matched atoms highlighted. Returns a PIL Image."""
    mol = Chem.MolFromSmiles(plain_smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    AllChem.Compute2DCoords(mol)
    for motif_order, full_idx in enumerate(matched_indices):
        mol.GetAtomWithIdx(int(full_idx)).SetProp(
            "atomNote", motif_atom_names[motif_order]
        )
    return Draw.MolToImage(
        mol, size=size, highlightAtoms=[int(i) for i in matched_indices]
    )


def _diagnose(
    case_id: str, json_rel: str, cif_rel: str, chain_id: str, res_name: str
) -> None:
    print(f"\n=== {case_id} ===")
    spec = json.loads((PROJECT_ROOT / json_rel).read_text())
    full_ligand = spec["spec"]["motif"]["ligands"][0]["full_ligand"]
    gt_smiles = full_ligand["smiles"]
    gt_atom_map = full_ligand["atom_map"]
    plain_smiles = _strip_atom_map_numbers(gt_smiles)

    atom_array = MmcifReader(PROJECT_ROOT / cif_rel).atom_array()
    mask = (
        (atom_array.chain_id == chain_id)
        & (atom_array.res_name == res_name)
        & atom_array.hetero
    )
    ligand = atom_array[mask]
    elements = tuple(ligand.element)
    coords = np.asarray(ligand.coord, dtype=float)
    atom_names = tuple(str(n) for n in ligand.atom_name)
    print(f"  motif atoms: {list(zip(atom_names, [str(e) for e in elements]))}")

    # Bridge GT label numbers (defined in `gt_smiles`'s atom indexing)
    # to atom indices in the canonicalised plain SMILES — which is what
    # the function and substructure search use.
    full_mol = Chem.MolFromSmiles(plain_smiles)
    gt_mol_with_labels = Chem.MolFromSmiles(gt_smiles)
    gt_mol_stripped = Chem.MolFromSmiles(gt_smiles)
    for a in gt_mol_stripped.GetAtoms():
        a.SetAtomMapNum(0)
    labelled_to_plain = full_mol.GetSubstructMatch(gt_mol_stripped)
    if not labelled_to_plain:
        raise RuntimeError(
            f"Could not bridge labelled→plain SMILES atom indices for "
            f"{case_id}; canonical plain isn't isomorphic to labelled."
        )
    label_to_plain_idx = {
        atom.GetAtomMapNum(): labelled_to_plain[atom.GetIdx()]
        for atom in gt_mol_with_labels.GetAtoms()
        if atom.GetAtomMapNum() != 0
    }
    gt_match = tuple(label_to_plain_idx[gt_atom_map[name]] for name in atom_names)

    # Enumerate candidate matches under the same settings the inference
    # function uses (uniquify=False so symmetric mappings aren't
    # collapsed).
    motif_query = atoms_to_rdkit_mol(elements, coords)
    params = AdjustQueryParameters()
    params.makeBondsGeneric = True
    query_mol = AdjustQueryProperties(motif_query, params)
    matches = full_mol.GetSubstructMatches(query_mol, uniquify=False)

    # Score every candidate under ONE diagnostic conformer so all RMSDs
    # in the rendered figure are directly comparable.
    diag_mol_3d = smiles_to_rdkit_mol(plain_smiles, generate_conformers=True)
    diag_conf = diag_mol_3d.GetConformer().GetPositions()
    scored = []
    for match in matches:
        rmsd, max_disp = _kabsch_align(coords, diag_conf[list(match)])
        scored.append((match, float(rmsd), float(max_disp)))
    scored.sort(key=lambda x: x[1])

    # Track the function's pick across N independent runs (each with its
    # own random conformer) to expose conformer-noise.
    fn_picks = Counter()
    for _ in range(N_TRIALS):
        r = infer_atom_map_with_disambiguation(
            motif_elements=elements,
            motif_coords=coords,
            motif_atom_names=atom_names,
            full_smiles=plain_smiles,
        )
        fn_picks[r.matched_indices] += 1

    gt_rank = next((i + 1 for i, (m, _, _) in enumerate(scored) if m == gt_match), None)
    gt_rmsd = next((r for m, r, _ in scored if m == gt_match), None)

    print(f"  total candidates: {len(scored)}")
    print(
        f"  GT match : indices={gt_match} "
        f"diag_rmsd={gt_rmsd:.3f} rank={gt_rank}/{len(scored)} "
        f"trials_won={fn_picks.get(gt_match, 0)}/{N_TRIALS}"
    )
    print(f"  function pick distribution across {N_TRIALS} trials:")
    for indices, count in fn_picks.most_common():
        diag_rmsd_for_pick = next((r for m, r, _ in scored if m == indices), None)
        print(
            f"    {indices}: {count}/{N_TRIALS}  "
            f"(diag_rmsd={diag_rmsd_for_pick:.3f})"
        )

    panels: list[tuple[str, tuple[int, ...]]] = []
    panels.append(
        (
            f"GROUND TRUTH\n"
            f"diag rank {gt_rank}/{len(scored)}, "
            f"rmsd={gt_rmsd:.3f} Å\n"
            f"won {fn_picks.get(gt_match, 0)}/{N_TRIALS} fn-trials",
            gt_match,
        )
    )
    for i, (match, rmsd, max_disp) in enumerate(scored[:TOP_K]):
        marker = ""
        if match == gt_match:
            marker = "  ← GT"
        won = fn_picks.get(match, 0)
        panels.append(
            (
                f"#{i + 1}: rmsd={rmsd:.3f} Å, "
                f"max_disp={max_disp:.3f} Å{marker}\n"
                f"won {won}/{N_TRIALS} fn-trials",
                match,
            )
        )

    cols = 3
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes_flat = np.array(axes).reshape(-1)

    for ax, (title, match) in zip(axes_flat, panels):
        img = _render_2d_image(plain_smiles, match, atom_names)
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes_flat[len(panels) :]:
        ax.axis("off")

    fig.suptitle(
        f"{case_id}\nplain SMILES: {plain_smiles}\n"
        f"all RMSDs scored on a single shared diagnostic conformer; "
        f"`fn-trials` is how often the function picked that match across "
        f"{N_TRIALS} independent runs.",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()

    out = OUTPUT_DIR / f"{case_id}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        _diagnose(*case)


if __name__ == "__main__":
    main()
