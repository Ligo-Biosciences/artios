# Sequence Design Input Format

This document describes the JSON input format for sequence design inference, where an amino-acid sequence is predicted for a backbone structure (plus ligands) produced upstream by backbone inference.

## Overview

Sequence design takes an NPZ file containing a serialised `AtomArrayStack` from backbone inference. That NPZ already carries everything the sequence-design model needs:

- Protein backbone coordinates (poly-glycine)
- Ligand coordinates and annotations
- Motif residue annotations (`is_motif`, `is_motif_related`)
- Chain IDs for all entities

You can additionally:

1. Pin amino-acid identity at specific positions via `fixed_residues`
2. Control which backbones in the stack are used (and how many output sequences to generate per backbone) via `backbone_index_map`

The model predicts a sequence that folds into the given backbone and binds any ligands present.

## JSON Format

The top-level shape is a task envelope that the inference dispatcher uses to select the sequence-design pipeline. The task-specific data lives under `spec`:

```json
{
  "name": "my_sequence_design",
  "task": "sequence_design",
  "spec": {
    "npz_file": "path/to/backbone_output.npz",
    "fixed_residues": {
      "0": [
        {"res_id": 94,  "res_name": "HIS"},
        {"res_id": 96,  "res_name": "H"},
        {"res_id": 119, "res_name": "HIS"}
      ]
    },
    "backbone_index_map": [0, 0, 1, 1, 2]
  }
}
```

### Top-level fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Job identifier |
| `task` | Yes | Must be the literal string `"sequence_design"` |
| `spec` | Yes | Task-specific payload (see below) |

### `spec` fields

| Field | Required | Description |
|-------|----------|-------------|
| `npz_file` | Yes | Path to the backbone NPZ file. May be a string or `{"path": "..."}` |
| `fixed_residues` | No | Per-sample map from backbone stack index → list of fixed residues |
| `backbone_index_map` | No | Per-output-position NPZ stack index. Controls which backbones produce sequences and how many per backbone |

### `fixed_residues[<sample_idx>][]` fields

| Field | Required | Description |
|-------|----------|-------------|
| `res_id` | Yes | Residue number (1-based) |
| `res_name` | Yes | Amino-acid code (1-letter or 3-letter) |

`fixed_residues` keys are integers — the backbone stack position in the NPZ. Residues pinned under key `N` apply only to outputs generated from backbone `N`. The stack index `N` is the same index used by `backbone_index_map` entries (see below).

## Backbone NPZ File

The NPZ file is the saved `AtomArrayStack` from backbone inference, loaded via `artios.utils.file_io.load_atomarray_stack`. A single NPZ can contain many backbones (stack depth `m`), each sharing the same atom annotations but with its own coordinates.

Because this file comes straight out of backbone inference, no separate ligand mapping, chain assignment, or motif annotation is required at the sequence-design stage — all of that is already baked into the NPZ.

## Fixed Residues

Fixed residues constrain specific positions to known amino-acid identities. Common sources:

- **Stitched motif residues**: Catalytic/binding residues carried through motif scaffolding
- **Pocket residues**: Residues inherited from a successful folded ancestor in the redesign pipeline
- **External constraints**: Conserved residues from evolutionary analysis or experimental data

In the evaluation pipeline, all sources are merged by `make_potts_input_json` into a single per-sample-index list before this JSON is written. If the same `res_id` appears in multiple sources for the same sample index, stitched-motif identity wins — see `evaluation/utils/pocket_residues.py::merge_fixed_residues`.

### Residue Code Format

Both 1-letter and 3-letter amino-acid codes are accepted. Codes are case-insensitive and normalised to 3-letter format internally.

| 1-letter | 3-letter | Amino Acid |
|----------|----------|------------|
| A | ALA | Alanine |
| R | ARG | Arginine |
| N | ASN | Asparagine |
| D | ASP | Aspartic acid |
| C | CYS | Cysteine |
| Q | GLN | Glutamine |
| E | GLU | Glutamic acid |
| G | GLY | Glycine |
| H | HIS | Histidine |
| I | ILE | Isoleucine |
| L | LEU | Leucine |
| K | LYS | Lysine |
| M | MET | Methionine |
| F | PHE | Phenylalanine |
| P | PRO | Proline |
| S | SER | Serine |
| T | THR | Threonine |
| W | TRP | Tryptophan |
| Y | TYR | Tyrosine |
| V | VAL | Valine |

## `backbone_index_map`

`backbone_index_map` is an optional list of integers that controls the mapping from output sequence index to NPZ stack index.

- **Omitted / `null`**: one output per backbone, in stack order — equivalent to `list(range(stack_depth))`. This is the gen0 default.
- **Provided**: entry `i` is the NPZ stack position that output sequence `i` should be generated from. Multiple entries may point to the same stack position, and group sizes need not be uniform. This is used in later redesign generations where filter survival rates vary per backbone.

The ordering contract (in `evaluation/functions/artios_potts_function.py` and consumed by `artios/data/infer_data_pipeline.py::MPNNDataset`): output position `i` in the FASTA corresponds to the UID whose `batch_sample_idx == i`, and `backbone_index_map[i]` is that UID's `backbone_sample_idx`.

## Complete Examples

### Default one-sequence-per-backbone run

```json
{
  "name": "full_redesign",
  "task": "sequence_design",
  "spec": {
    "npz_file": "backbones/target_backbones.npz"
  }
}
```

### Zinc-binding site with fixed histidines

Pin the three catalytic histidines for every backbone in the stack (stack depth 2 in this example):

```json
{
  "name": "zinc_binding_redesign",
  "task": "sequence_design",
  "spec": {
    "npz_file": "backbones/carbonic_anhydrase_backbones.npz",
    "fixed_residues": {
      "0": [
        {"res_id": 94,  "res_name": "HIS"},
        {"res_id": 96,  "res_name": "HIS"},
        {"res_id": 119, "res_name": "HIS"}
      ],
      "1": [
        {"res_id": 94,  "res_name": "HIS"},
        {"res_id": 96,  "res_name": "HIS"},
        {"res_id": 119, "res_name": "HIS"}
      ]
    }
  }
}
```

### Redesign generation with uneven group sizes

Three surviving backbones (stack positions 0, 1, 2) produce 2, 3, and 1 sequences respectively. Only backbone 1 has pinned residues:

```json
{
  "name": "gen1_redesign",
  "task": "sequence_design",
  "spec": {
    "npz_file": "backbones/gen0_stack.npz",
    "fixed_residues": {
      "1": [
        {"res_id": 52,  "res_name": "K"},
        {"res_id": 168, "res_name": "D"}
      ]
    },
    "backbone_index_map": [0, 0, 1, 1, 1, 2]
  }
}
```

## Validation

The parser (`artios.data.parsing.json.SequenceDesignInput`) validates:

1. **Task envelope**: `task` must be `"sequence_design"`; `name` and `spec` are required
2. **File format**: `npz_file` must end in `.npz`
3. **Residue codes**: `res_name` must be a standard amino acid (20 natural amino acids); 1-letter and 3-letter forms both accepted

## Python Usage

```python
from artios.data.parsing.json import (
    FixedResidue,
    NPZFile,
    SequenceDesignInput,
    parse_input_dict,
)

# Parse from a task-wrapped dict (inference pipeline entry point)
input_data = parse_input_dict({
    "name": "my_design",
    "task": "sequence_design",
    "spec": {"npz_file": "backbone.npz"},
})

# Access fields
print(input_data.name)
print(input_data.npz_file.path)

for sample_idx, residues in input_data.fixed_residues.items():
    for fr in residues:
        print(f"Sample {sample_idx}: res_id {fr.res_id} pinned to {fr.res_name}")

# Create programmatically
input_data = SequenceDesignInput(
    name="my_design",
    npz_file=NPZFile(path="backbone.npz"),
    fixed_residues={
        0: [FixedResidue(res_id=94, res_name="HIS")],
    },
    backbone_index_map=[0, 0, 1],
)
```
