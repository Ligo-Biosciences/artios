# Motif Input Format

This document describes the JSON input format for motif-scaffolding inference, where a protein is designed around a structural motif containing partial sidechains and ligand fragments.

## Overview

Motif scaffolding allows you to:
1. Specify a protein length to design
2. Provide a motif CIF file containing partial residue sidechains and ligand fragments with 3D coordinates
3. Describe the full ligands (which may be partially represented in the motif)

The model will design a protein that accommodates the motif geometry.

## JSON Format

```json
{
  "name": "my_motif_design",
  "design": [
    {
      "length": 260
    }
  ],
  "motif": {
    "cif_file": "path/to/motif.cif",
    "ligands": [
      {
        "cif_res_name": "ZN",
        "chain_id": "C",
        "full_ligand": {"type": "ccd", "codes": ["ZN"]}
      },
      {
        "cif_res_name": "AL5",
        "chain_id": "D",
        "full_ligand": {"type": "ccd", "codes": ["AL5"]}
      }
    ]
  }
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | No | Identifier for this design job |
| `design` | Yes | List of protein chains to design |
| `design[].length` | Yes | Number of residues in the designed protein (1-2000) |
| `motif` | No | Motif specification (omit for pure backbone design) |
| `motif.cif_file` | Yes | Path to the motif CIF file (.cif or .mmcif) |
| `motif.ligands` | No | List of ligand mappings (protein residues are auto-detected) |
| `motif.ligands[].cif_res_name` | Yes | Residue name as it appears in the motif CIF file |
| `motif.ligands[].chain_id` | Yes | Chain ID as it appears in the motif CIF file |
| `motif.ligands[].full_ligand` | Yes | Full ligand specification (see below) |


## Motif CIF File

The motif CIF file contains partial structural information extracted from a reference structure. It typically includes:

- **Partial protein sidechains**: Key residues (e.g., catalytic residues, binding site residues) with only the functionally important atoms
- **Ligand fragments**: Metal ions, cofactors, or small molecule fragments that define the binding geometry

### CIF Formatting Rules

The motif CIF must follow these exact rules for the parser to work correctly:

1. **Protein residues use chain ID `0`.** All ATOM records must have `label_asym_id` and `auth_asym_id` set to `0`.
2. **Protein residue IDs are sequential from 1.** Each unique residue gets a sequential integer in `label_seq_id`, starting at 1. Atoms belonging to the same residue share the same ID.
3. **Each ligand gets its own chain ID.** Distinct ligands (HETATM) use chain IDs `1`, `2`, `3`, etc. in `label_asym_id` and `auth_asym_id`. For example, a peptide ligand gets chain `1` and a Zn ion gets chain `2`.
4. **Residue IDs start from 1 and increase.** Across protein residues, IDs go 1, 2, 3, ...
5. **All atoms in the same ligand share residue ID 1.** Every HETATM atom within a single ligand chain has `label_seq_id = 1`.
6. **Entity IDs:** Protein = `1`, first ligand = `2`, second ligand = `3`, etc.

A conversion script is provided at `evaluation/scripts/convert_motif_cif.py` to automatically apply these rules to a raw PyMOL-exported CIF:

```bash
python evaluation/scripts/convert_motif_cif.py input.cif output.cif --name my_motif
```

### Example

Correctly formatted motif CIF (carbonic anhydrase II active site with a sulfonamide inhibitor and zinc, from PDB 1BN1):
```
# Protein residues: chain 0, sequential res IDs, entity 1
ATOM    1  N NE2 . HIS 0 1 1 ? -7.571 -0.471 13.996 1.00  2.54 0 0 1
ATOM    2  N NE2 . HIS 0 1 2 ? -5.940 -3.270 14.688 1.00  8.52 0 0 1
ATOM    3  N NE2 . HIS 0 1 3 ? -8.937 -3.688 18.212 1.00  7.99 0 0 1
ATOM    4  O OG1 . THR 0 1 4 ? -3.913 -2.138 17.799 1.00 10.73 0 0 1

# Ligand AL5: chain 1, entity 2, all res_id = 1
HETATM  5  S  S1 . AL5 1 2 1 ? -5.923  0.680 16.887 1.00 24.28 0 1 1
HETATM  6  N N21 . AL5 1 2 1 ? -5.426 -0.670 16.241 1.00 20.67 0 1 1
...

# Ligand ZN: chain 2, entity 3, res_id = 1
HETATM  9 ZN  ZN .  ZN 2 3 1 ? -6.844 -1.668 15.437 1.00 10.83 0 2 1
```

The parser automatically:
1. Identifies protein residues (hetero=False) vs ligands (hetero=True)
2. Expands partial residues to full residues using CCD reference data
3. Marks atoms with `is_motif=True` for those with coordinates from the motif
4. Marks entire residues with `is_motif_related=True`

## Ligand Specification

The `full_ligand` field is an object with a `type` discriminator. Two types are supported:

### 1. CCD Ligand

For ligands in the Chemical Component Dictionary, use `type: "ccd"`:

```json
{
  "cif_res_name": "ZN",
  "chain_id": "C",
  "full_ligand": {"type": "ccd", "codes": ["ZN"]}
}
```

For multi-component ligands, include multiple codes:
```json
{
  "cif_res_name": "HEM",
  "chain_id": "B",
  "full_ligand": {"type": "ccd", "codes": ["HEM", "FE"]}
}
```

**Atom matching**: CIF atom names (e.g., `O1A`, `S1`, `N21`) are matched directly to CCD atom names.

### 2. SMILES Ligand

For custom ligands, use `type: "smiles"` with an `atom_map` to specify how motif atoms map to the SMILES:

```json
{
  "cif_res_name": "AL5",
  "chain_id": "D",
  "full_ligand": {
    "type": "smiles",
    "smiles": "Cc1ccc([S:1](=[O:2])(=[O:3])[N:4]c2nc(cs2)S(=O)(=O)N)cc1",
    "atom_map": {
      "S1": 1,
      "O1A": 2,
      "O2A": 3,
      "N21": 4
    }
  }
}
```

The `atom_map` field maps CIF atom names to SMILES atom map numbers (the `:N` notation in the SMILES string).

## SMILES Atom Map Notation

Atom map notation uses `[atom:N]` syntax to label specific atoms with integer identifiers.

### Syntax

```
[<element><hydrogens><charge>:<map_number>]
```

| Notation | Meaning |
|----------|---------|
| `[C:1]` | Carbon with map number 1 |
| `[CH3:1]` | Carbon with 3 explicit hydrogens, map number 1 |
| `[N:2]` | Nitrogen with map number 2 |
| `[O-:3]` | Negatively charged oxygen, map number 3 |
| `[nH:4]` | Aromatic nitrogen with hydrogen, map number 4 |

### Example

Standard SMILES for methanesulfonamide:
```
CS(=O)(=O)N
```

With atom maps labeling specific atoms:
```
[CH3:1][S:2](=[O:3])(=[O:4])[NH2:5]
```

### Usage in Motif JSON

Only label atoms that appear in your motif CIF:

```json
{
  "cif_res_name": "AL5",
  "chain_id": "D",
  "full_ligand": {
    "type": "smiles",
    "smiles": "Cc1ccc([S:1](=[O:2])(=[O:3])[N:4]c2nc(cs2)S(=O)(=O)N)cc1",
    "atom_map": {
      "S1": 1,
      "O1A": 2,
      "O2A": 3,
      "N21": 4
    }
  }
}
```

The `atom_map` dictionary maps:
- **Key**: Atom name from the motif CIF file (e.g., `S1`, `O1A`)
- **Value**: Atom map number in the SMILES string

Atoms without map numbers in the SMILES (map number 0) are not part of the motif and will have their coordinates generated by the model.

### Advantages of Atom Map Notation

1. **Explicit**: Labels are visible in the SMILES string
2. **Stable**: Doesn't depend on SMILES atom ordering
3. **Sparse**: Only label the atoms you need (motif atoms)
4. **Verifiable**: Easy to check in RDKit

### Verifying Atom Maps in Python

```python
from rdkit import Chem

smiles = "[CH3:1][S:2](=[O:3])(=[O:4])[NH2:5]"
mol = Chem.MolFromSmiles(smiles)

for atom in mol.GetAtoms():
    map_num = atom.GetAtomMapNum()
    if map_num > 0:
        print(f"Map {map_num}: {atom.GetSymbol()} (idx={atom.GetIdx()})")
```

Output:
```
Map 1: C (idx=0)
Map 2: S (idx=1)
Map 3: O (idx=2)
Map 4: O (idx=3)
Map 5: N (idx=4)
```

## Complete Example

### Carbonic Anhydrase Active Site

A design targeting the carbonic anhydrase active site with a zinc ion and sulfonamide inhibitor:

```json
{
  "name": "carbonic_anhydrase_design",
  "design": [
    {
      "length": 260
    }
  ],
  "motif": {
    "cif_file": "motifs/carbonic_anhydrase_motif.cif",
    "ligands": [
      {
        "cif_res_name": "ZN",
        "chain_id": "C",
        "full_ligand": {"type": "ccd", "codes": ["ZN"]}
      },
      {
        "cif_res_name": "AL5",
        "chain_id": "D",
        "full_ligand": {"type": "ccd", "codes": ["AL5"]}
      }
    ]
  }
}
```

### Custom Ligand with SMILES

Designing with a custom sulfonamide not in CCD:

```json
{
  "name": "custom_sulfonamide_design",
  "design": [
    {
      "length": 260
    }
  ],
  "motif": {
    "cif_file": "motifs/binding_site_motif.cif",
    "ligands": [
      {
        "cif_res_name": "ZN",
        "chain_id": "C",
        "full_ligand": {"type": "ccd", "codes": ["ZN"]}
      },
      {
        "cif_res_name": "LIG",
        "chain_id": "D",
        "full_ligand": {
          "type": "smiles",
          "smiles": "CC(C)c1ccc([S:1](=[O:2])(=[O:3])[N:4])cc1",
          "atom_map": {
            "S": 1,
            "O1": 2,
            "O2": 3,
            "N": 4
          }
        }
      }
    ]
  }
}
```

## Annotations Generated

The motif parser generates an AtomArray with these key annotations:

| Annotation | Description |
|------------|-------------|
| `is_motif` | `True` for atoms with coordinates from the motif CIF |
| `is_motif_related` | `True` for all atoms in residues/ligands that contain any motif atoms |
| `coord` | 3D coordinates (from motif for `is_motif=True`, zeros or reference for others) |

These annotations guide the model to preserve motif geometry during design.
