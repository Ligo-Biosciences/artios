# Sequence Design Input Format

This document describes the JSON input format for sequence design inference, where an amino acid sequence is predicted for a given backbone structure with ligand(s).

## Overview

Sequence design allows you to:
1. Provide a poly-glycine backbone CIF file with ligand(s)
2. Specify which residue positions have fixed (constrained) amino acid identities
3. Predict the optimal sequence for the remaining designable positions

The model will predict a sequence that folds into the given backbone structure and binds the ligand.

## JSON Format

```json
{
  "name": "my_sequence_design",
  "backbone": {
    "file": "path/to/backbone.cif",
    "ligands": [
      {
        "cif_res_name": "ZN",
        "full_ligand": {"type": "ccd", "codes": ["ZN"]}
      }
    ]
  },
  "fixed_residues": [
    {"chain_id": "A", "res_id": 94, "residue": "HIS"},
    {"chain_id": "A", "res_id": 96, "residue": "H"},
    {"chain_id": "A", "res_id": 119, "residue": "HIS"}
  ]
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | No | Identifier for this design job |
| `backbone` | Yes | Backbone structure specification |
| `backbone.file` | Yes | Path to the backbone CIF file (.cif or .mmcif) |
| `backbone.ligands` | No | List of ligand mappings for molecules in the CIF |
| `backbone.ligands[].cif_res_name` | Yes | Residue name as it appears in the CIF file |
| `backbone.ligands[].full_ligand` | Yes | Full ligand specification (see below) |
| `fixed_residues` | No | List of residues with constrained amino acid identity |
| `fixed_residues[].chain_id` | Yes | Chain identifier |
| `fixed_residues[].res_id` | Yes | Residue number (1-based) |
| `fixed_residues[].residue` | Yes | Amino acid code (1-letter or 3-letter) |

## Backbone CIF File

The backbone CIF file should contain:

- **Poly-glycine protein chain(s)**: Backbone atoms only (N, CA, C, O) with all residues as GLY
- **Ligand(s)**: The target molecule(s) to design binding for

Example backbone structure:
```
ATOM      1  N   GLY A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  GLY A   1      11.458  10.000  10.000  1.00  0.00           C
ATOM      3  C   GLY A   1      12.009  11.420  10.000  1.00  0.00           C
ATOM      4  O   GLY A   1      11.251  12.400  10.000  1.00  0.00           O
ATOM      5  N   GLY A   2      13.320  11.540  10.000  1.00  0.00           N
...
HETATM  500  ZN  ZN  B   1      15.000  15.000  15.000  1.00  0.00          ZN
```

The backbone can be obtained from:
- An existing structure with sidechains stripped
- A de novo designed backbone from structure prediction
- Any source providing backbone coordinates

## Fixed Residues

Fixed residues constrain specific positions to known amino acid identities. Common use cases:

- **Catalytic residues**: Essential for enzyme function (e.g., HIS in zinc-binding sites)
- **Binding site residues**: Known to interact with the ligand
- **Conserved residues**: From evolutionary analysis or experimental data

### Residue Code Format

Both 1-letter and 3-letter amino acid codes are accepted:

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

Codes are case-insensitive and normalized to 3-letter format internally.

## Ligand Specification

The `full_ligand` field uses the same format as [motif input](motif_input_format.md#ligand-specification). Two types are supported:

### 1. CCD Ligand

For ligands in the Chemical Component Dictionary:

```json
{
  "cif_res_name": "ZN",
  "full_ligand": {"type": "ccd", "codes": ["ZN"]}
}
```

For multi-component ligands:
```json
{
  "cif_res_name": "HEM",
  "full_ligand": {"type": "ccd", "codes": ["HEM", "FE"]}
}
```

### 2. SMILES Ligand

For custom ligands not in CCD:

```json
{
  "cif_res_name": "LIG",
  "full_ligand": {
    "type": "smiles",
    "smiles": "Cc1ccc(S(=O)(=O)N)cc1",
    "atom_map": {"S1": 1, "O1": 2, "O2": 3, "N1": 4}
  }
}
```

See [motif input format](motif_input_format.md#smiles-ligand) for details on SMILES atom mapping.

## Complete Examples

### Zinc-Binding Protein Redesign

Redesigning a carbonic anhydrase-like zinc-binding site:

```json
{
  "name": "zinc_binding_redesign",
  "backbone": {
    "file": "backbones/carbonic_anhydrase_polyG.cif",
    "ligands": [
      {
        "cif_res_name": "ZN",
        "full_ligand": {"type": "ccd", "codes": ["ZN"]}
      }
    ]
  },
  "fixed_residues": [
    {"chain_id": "A", "res_id": 94, "residue": "HIS"},
    {"chain_id": "A", "res_id": 96, "residue": "HIS"},
    {"chain_id": "A", "res_id": 119, "residue": "HIS"}
  ]
}
```

### ATP-Binding Protein

Designing a sequence for an ATP-binding protein with key binding residues fixed:

```json
{
  "name": "atp_binding_design",
  "backbone": {
    "file": "backbones/kinase_fold_polyG.cif",
    "ligands": [
      {
        "cif_res_name": "ATP",
        "full_ligand": {"type": "ccd", "codes": ["ATP"]}
      },
      {
        "cif_res_name": "MG",
        "full_ligand": {"type": "ccd", "codes": ["MG"]}
      }
    ]
  },
  "fixed_residues": [
    {"chain_id": "A", "res_id": 52, "residue": "K"},
    {"chain_id": "A", "res_id": 168, "residue": "D"},
    {"chain_id": "A", "res_id": 184, "residue": "D"}
  ]
}
```

### Multi-Chain Complex

Designing sequences for a homodimer with a shared ligand:

```json
{
  "name": "homodimer_design",
  "backbone": {
    "file": "backbones/homodimer_polyG.cif",
    "ligands": [
      {
        "cif_res_name": "NAD",
        "full_ligand": {"type": "ccd", "codes": ["NAD"]}
      }
    ]
  },
  "fixed_residues": [
    {"chain_id": "A", "res_id": 45, "residue": "GLY"},
    {"chain_id": "A", "res_id": 47, "residue": "GLY"},
    {"chain_id": "B", "res_id": 45, "residue": "GLY"},
    {"chain_id": "B", "res_id": 47, "residue": "GLY"}
  ]
}
```

### No Fixed Residues

If no residues are constrained, all positions are designable:

```json
{
  "name": "full_redesign",
  "backbone": {
    "file": "backbones/target_polyG.cif",
    "ligands": [
      {
        "cif_res_name": "ZN",
        "full_ligand": {"type": "ccd", "codes": ["ZN"]}
      }
    ]
  }
}
```

## Validation

The parser validates:

1. **File format**: Backbone file must be `.cif` or `.mmcif`
2. **Residue codes**: Must be valid standard amino acids (20 natural amino acids)
3. **Residue numbers**: Must be positive integers (1-based)
4. **No duplicates**: Each (chain_id, res_id) pair can only appear once in `fixed_residues`
5. **Ligand codes**: CCD codes must exist in the Chemical Component Dictionary
6. **SMILES validity**: SMILES strings must be parseable by RDKit

## Python Usage

```python
from protenix.data.json_parser_new import (
    parse_sequence_design_json,
    SequenceDesignInput,
    FixedResidue,
)

# Parse from file
input_data = parse_sequence_design_json("my_design.json")

# Access fields
print(input_data.name)
print(input_data.backbone.file)

for fr in input_data.fixed_residues:
    # residue is always normalized to 3-letter code
    print(f"Chain {fr.chain_id}, Res {fr.res_id}: {fr.residue}")

# Create programmatically
input_data = SequenceDesignInput(
    name="my_design",
    backbone={"file": "backbone.cif", "ligands": []},
    fixed_residues=[
        {"chain_id": "A", "res_id": 94, "residue": "H"},  # normalized to "HIS"
    ],
)
```
