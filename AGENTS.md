# Artios for agents

Artios designs ligand-binding proteins. ArtiosBackbone generates backbones around a motif, and ArtiosPotts designs sequences for any backbone or PDB/mmCIF structure. Use the `artios` command (`artios COMMAND --help`, or `uv run artios` in a source checkout); the Python API is `artios.potts`.

## Before running

- Install with `uv tool install artios`, `uv pip install artios` or `pip install artios` (Python 3.11 or later; on 3.14 and later, building the `biotraj` dependency needs a C compiler), or with `uv sync` in a source checkout. `artios --version` checks the install. Weights and CCD files (about 1.1 GB) download from Hugging Face on first use and are cached.
- The model runs on a CUDA GPU when one is available, otherwise on the CPU. Apple MPS is not supported. On a laptop CPU, `design` takes about 3 minutes for 4 sequences of a 128-residue scaffold backbone and 5 minutes for a 250-residue protein, and `scaffold` about 1.5 minutes for 5 backbones of 128 residues. Design time depends on the size of the whole input, not on how much of it is designed. Run long jobs in the background and wait for the process to exit; output files are filled in per target as each one finishes.
- For a quick pipeline check, add `--set ensemble_size=2 --set n_sweeps=4`. This runs in seconds but gives poor sequences, so never report results from it.

## Commands

| Goal | Command |
|---|---|
| Design sequences for a structure | `artios design complex.pdb -n 8 -o out/design` |
| Design one chain of a complex (e.g. a binder) | add `--design-chains B`; other chains keep their sequence |
| Keep some residues | add `--fixed "B10-14 B42"` |
| Design only a region (loop, pocket) | add `--redesign "A30-41"`; `--fixed` still applies |
| Drop chains from the input | add `--chains A,B`; ligands go with their author chain |
| Backbones around a motif, then sequences | `artios scaffold input.json -n 5 -o out/bb`, then `artios design out/bb/NAME_backbone.npz -o out/seq` |
| Score sequences on a structure | `artios score complex.pdb seqs.fasta -o out/score`; chains left out of a record keep their native sequence, and `designs.fasta` works as is |
| Score the native sequence | `artios score complex.pdb -o out/native` |
| Potts fields and couplings | `artios params complex.pdb -o out/params` |

- **Selecting residues.** Selections use author chain IDs and residue numbers, as in the file and PyMOL. Write `A` (a whole chain), `A42`, `A52B` (insertion code), `A10-20` (inclusive) or `1:42` (for chain IDs that are not letters). Separate several with spaces or commas.
- **What is designed.** Only protein residues are designed. Ligands, ions and nucleic acids are context. In scaffold outputs, motif residues are context and only chain `0` is designed.
- **Inputs.** Inputs can be PDB, mmCIF or scaffold NPZ files, or directories of them. Target names come from file stems, which must be unique. A scaffold NPZ holds several backbones, and every command treats each as its own target, named `STEM_0`, `STEM_1` and so on (e.g. `odcase_backbone_0`).
- **Pocket residues.** To select residues near a ligand (here within 6 Å of `LIG`):

```python
import numpy as np, biotite.structure as struc
from biotite.structure.io import pdbx
cif = pdbx.CIFFile.read("complex.cif")  # for a PDB file: pdb.PDBFile.read(path).get_structure(model=1)
atoms = pdbx.get_structure(cif, model=1, use_author_fields=True)  # author chains and numbers
print(sorted(set(atoms.res_name[atoms.hetero])))  # ligand names
ligand = atoms[atoms.res_name == "LIG"]
protein = atoms[struc.filter_amino_acids(atoms)]
hits = struc.CellList(protein, 6.0).get_atoms(ligand.coord, 6.0)
idx = np.unique(hits[hits >= 0])
print(" ".join(dict.fromkeys(f"{c}{r}" for c, r in zip(protein.chain_id[idx], protein.res_id[idx]))))
```

## Outputs

**`design`** writes three files to `-o`:

- `designs.fasta` holds the designed chains of each design, joined with `:` in the order given by `chains=` in the header.
- `designs.jsonl` holds one record per design: the full `sequences` of every chain (`X` marks a native residue with no standard amino-acid parent), a per-chain `designed` mask (`1` = designed), `energy`, `residue_energy` (one value per designed position) and `native_identity` (the fraction of designed positions that keep the native residue).
- `settings.json` records the settings of the run.

Stdout has the FASTA path; progress and errors go to stderr.

Sequence positions follow each chain's resolved residues in file order. To map them to residue numbers and native residues, for example to list mutations:

```python
import json
from artios.potts import load_target
target = load_target("complex.cif")
native = target.native_sequences()  # {chain: sequence}, as the model sees it
labels = {c: [r.label for r in target.residues if r and r.chain == c] for c in native}
for d in map(json.loads, open("out/design/designs.jsonl")):
    print(d["id"], [f"{c}:{n}{lab[len(c):]}{s}" for c in d["sequences"] for lab, n, s, m in
          zip(labels[c], native[c], d["sequences"][c], d["designed"][c]) if m == "1" and n != s])
```

**`score`** writes `scores.jsonl` and prints a TSV on stdout with `energy` and `pseudo_log_likelihood` (the mean log-probability per residue; higher is better).

**`params`** writes one `TARGET.npz` per target with these arrays:

- `h` [N, 32] and `J` [N, K, 32, 32] cover all N tokens, including ligand and motif tokens. `labels` holds the residue label of each token (e.g. `A42`, or `042` for chain `0`) and is empty for ligand, ion and motif tokens.
- `J[i, k]` couples token `i` with token `edge_index[i, k]` (its K nearest neighbours, where `edge_mask` is true). The graph is directed, so a pair can appear in either list or both.
- `alphabet` lists the 32 states; the first 20 are the amino acids.

The energy of a sequence is the sum over tokens of `h[i, s_i] + 1/2 sum_k J[i, k, s_i, s_edge_index[i, k]]`.

**Energies.** Potts energy: lower is better. It is only comparable between sequences on the same target.

- A design's `energy` sums over designed positions only, and includes any `--set energy_bias` terms.
- `artios score` sums over all protein positions without biases. To compare designs with the native sequence, or designs made with different settings, score them all with `artios score` on the same structure.

**Exit codes.** 0 means success. 1 means some target failed; the others are still written. 2 means bad input, with the reason on stderr.

## Behaviour

- **Rerunning.** Rerunning into the same `-o` skips finished targets. A changed setting into a directory that already has designs is refused, so use `--overwrite` or a new directory. That includes a new `--seed` for extra designs.
- **Reproducibility.** Results are reproducible for the same seed, settings and `--batch-size`.
- **Defaults.** 8 sequences per target, temperature 0.1, no cysteine or methionine (`--allow-cys`, `--allow-met`), and a low-complexity penalty. Designs at 0.1 are often close to each other; raise `--temperature` (e.g. 0.2) for more diversity.
- **Steering.** `--set KEY=VALUE` sets any `artios_mpnn.potts.sampling` option, and an unknown key lists them all. For example, `--set energy_bias.E=0.5` discourages Glu (negative values favour), `--set group_energy_bias.aromatic=-0.3` favours aromatics, and `--set cfg_scale=2.0` strengthens ligand conditioning.

## Developing

- **Tests.** `uv run pytest tests/potts` covers the CLI and API in about 2 minutes on CPU. The full suite is `uv run pytest -m "not slow"`; it needs `uv sync --all-extras` and downloads the CCD files on first run.
- **Formatting.** Run `uvx ruff@0.6.3 format .`; CI checks this exact version.
- **Code map.** `artios/cli.py` is a thin layer over `artios/potts/` (`PottsDesigner`, `load_target`). The model is in `artios/model/artios_potts.py`, the sampler in `artios/model/generator.py` and the steering penalties in `artios/model/modules/penalty.py`. `python -m artios.runner.inference` is the older entry point, which `evaluation/` still uses.
