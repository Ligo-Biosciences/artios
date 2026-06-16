# recentPDB OOD protein–ligand benchmark

This document describes how the recentPDB out-of-distribution (OOD) protein–ligand
benchmark for **ArtiosPotts** sequence design is generated — the exact filtering,
why each step exists, and the two scripts that produce it.

The benchmark is a set of single-chain protein–ligand complexes, each cropped to
the longest fully-resolved backbone segment plus its bound ligand(s), packed into
the ArtiosPotts natural-backbone NPZ format.

## Source data

- **Structures:** `<release>/recentPDB_bioassembly/<pdb_id>.npz` — AF3-style all-atom
  bioassembly featurisations (one per PDB).
- **Manifest:** `<release>/indices/recentPDB_low_homology_maxtoken1536.csv` — one row
  per chain / interface, with `eval_type`, `sub_mol_type`, `num_prot_chains`, etc.
- **CCD dictionary** (needed only by the curation step): `<release>/components.v20240608.cif`
  (+ `.rdkit_mol.pkl`), resolved via `configs.configs_data.DATA_ROOT_DIR`.

### Why this is out-of-distribution

recentPDB is a **temporal + low-homology holdout**: all entries were released
**2022-05 → 2023-01** (after the AF3 2021-09-30 training cutoff) and were filtered to
be sequence-dissimilar from the AF3 training set, capped at 1536 tokens. It is OOD by
construction relative to a pre-2022 cutoff.

> **Caveat:** the low-homology filter is defined against AF3's training data, *not*
> against the ArtiosPotts training set. No leakage check against the current
> ArtiosPotts training data has been performed. Dedup against that manifest if a
> strict guarantee is required.

## Filtering funnel

Each step and the count remaining (released benchmark):

| step | filter | remaining | rationale |
|------|--------|-----------|-----------|
| a | unique PDBs in the low-homology / maxtoken-1536 index | 1818 | starting set |
| b | ≥1 **non-bonded** protein–ligand interface (`eval_type == ligand_prot` and the ligand's `sub_mol_type == non_bonded_ligand`) | 503 | excludes glycans, covalent/bonded ligands, and AF3-excluded crystallisation additives — keeps genuine non-covalent binders |
| c | **no nucleic acid** in the assembly | 427 | protein–ligand only; no DNA/RNA |
| d | **exactly one protein chain** (`num_prot_chains == 1`) | 162 | clean single-receptor design targets |
| e | **< 5 total ligand molecules** (organic + ions + copies, counted as distinct `is_ligand` `mol_id`s in the npz) | 157 | drop fragment-screen / heavily-bound structures |
| f | take the **longest contiguous fully-resolved backbone segment** (residues with N/CA/C/O all resolved; broken by an incomplete residue *or* a residue-numbering gap) | — | most experimental structures have unmodelled loops; a contiguous resolved segment is a clean design target. Replaces a whole-structure reject-on-internal-gap policy, salvaging the largest gap-free piece. |
| g | keep only if **every organic ligand is still pocketed** by that segment: ≥4 contacting residues within 5 Å **OR** ≥0.4 SASA burial | — | the longest segment can crop the binding site away; this guarantees the ligand is still in contact with the retained backbone |
| h | **segment length ≥ 100 residues** | — | exclude tiny fragments |
| i | **most-buried organic ligand ≥ 0.52 SASA burial** | — | drop structures whose ligand(s) are all highly solvent-exposed — surface/additive binders, or shallow placements unlikely to be reproducible |
| j | **globular**: normalised radius of gyration `Rg / (2.2·N^0.38) < 1.3` | **104** | drop elongated/non-globular chains (the cropped segment's Cα trace) |
| — | **curate to NPZ** (`curate_natural_backbones.py --prebuilt-cif`) | **94** | pack into ArtiosPotts format; 10 drop on structural quirks (below) |

Steps (f)–(j) are applied together as a single per-structure gate, so only the
combined survivor count (104 CIFs) is shown. The (i) burial floor (0.52) and the
(j) globularity filter were tuned to remove a hand-picked set of structures with
non-reproducible ligand placement and clearly non-globular folds.

Ions and cofactors are kept throughout (e.g. ADP/ATP/NAD/SAM/FAD/CoA/SF4, Mg/Zn/Mn);
repeated ligand copies are allowed. "Organic ligand" for the pocket/exposure gates
means a non-ion `is_ligand` molecule (ions identified by the index `ions` set plus a
monatomic metal/halide fallback). SASA burial is `1 − SASA(ligand in segment+ligand) /
SASA(ligand alone)`, biotite `sasa`, probe 1.4 Å.

## Two-step generation

### Step 1 — build the segment CIFs

`evaluation/scripts/build_recentpdb_ligand_benchmark.py` runs filters (a)–(j) and writes
one `<pdb_id>.cif` per kept structure: the segment's protein residues (resolved atoms
only, original PDB numbering) plus every `is_ligand` molecule with a resolved atom within
6 Å of the segment. CIFs are written via `AnnotatedStructure.load → to_atom_array →
save_atoms_to_cif` (bonds included).

```bash
uv run python -m evaluation.scripts.build_recentpdb_ligand_benchmark \
    --index-csv  <release>/indices/recentPDB_low_homology_maxtoken1536.csv \
    --npz-dir    <release>/recentPDB_bioassembly \
    --output-dir <out>/recentPDB_ligand_cifs
```

Output: `<out>/recentPDB_ligand_cifs/<pdb_id>.cif` (104) + `manifest.csv`.

### Step 2 — curate to ArtiosPotts NPZs

These CIFs are **already resolved and pre-curated** (only resolved atoms, no
`entity_poly` metadata, original numbering). Feeding them through the default
`curate_natural_backbones.py` (deposited-mmCIF) path fails for ~92% of them, because
`load_bioassembly` re-derives `is_resolved` via CCD residue-repair (manufacturing
internal gaps) and infers polymer types from residue names (mis-typing chains with
modified residues). A dedicated `--prebuilt-cif` workflow was added:

- `load_resolved_structure()` — parse the CIF, infer polymer type from **backbone
  presence** (robust to MSE/SEP/CSO/… unlike the strict standard-AA rule), mark every
  present atom resolved, and skip CCD residue-repair.
- `_assign_unique_residue_ids()` — give each residue a unique id (handles insertion
  codes / duplicate numbering).
- A bug fix in `_with_canonical_design_topology` (applies to the deposited path too):
  non-protein residues were flat-reset to `res_id = 1`, which collapsed multi-residue
  ligands (oligosaccharides); they are now renumbered contiguously per molecule.

```bash
uv run python -m evaluation.scripts.curate_natural_backbones \
    --input-dir  <out>/recentPDB_ligand_cifs \
    --output-dir <out>/recentPDB_ligand_npzs \
    --prebuilt-cif --overwrite
```

Output: `<out>/recentPDB_ligand_npzs/<pdb_id>.npz` (94, depth-1 `AtomArrayStack`:
design chain backbone + ligand chains, sidechains stripped) + `manifest.parquet` +
`skipped.parquet`. Run from a working directory where the CCD `DATA_ROOT_DIR` resolves.

## The 10 not curated

Of the 104 CIFs, 10 fail curation on genuine structural quirks (not the systematic
round-trip bug, which the `--prebuilt-cif` path fixes):

- **6 `multi_protein`** — a free amino-acid ligand (GLU/ASP/TYR/MET) bound in the pocket
  carries a protein backbone, so it is counted as a second "protein molecule", tripping
  the single-protein-molecule requirement.
- **3 `no_protein`** — a real protein chain carrying one modified residue (FME / SEP /
  CSO) is mis-typed as a ligand by the type-mask logic.
- **1 `rewrite_failed`** — chain-id assignment edge case with an unusual ligand layout.

## Final result

**104 segment CIFs → 94 curated NPZs.** Protein backbone length ≥100 residues; 1–4
ligand molecules per structure; ions/cofactors/sugars and drug-like small molecules.
The globularity (j) and 0.52 burial (i) filters tightened an earlier 127→114 build,
removing 20 structures with non-globular folds and/or shallow, non-reproducible ligand
placement.
