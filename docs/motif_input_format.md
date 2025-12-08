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
  "task": "motif",
  "spec": {
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
          "full_ligand": {"type": "ccd", "codes": ["ZN"]}
        },
        {
          "cif_res_name": "AL5",
          "full_ligand": {"type": "ccd", "codes": ["AL5"]}
        }
      ]
    }
  }
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Identifier for this design job |
| `task` | Yes | Task type, must be `"motif"` for motif scaffolding |
| `spec` | Yes | Task-specific input data (see below) |
| `spec.design` | Yes | List of protein chains to design |
| `spec.design[].length` | Yes | Number of residues in the designed protein (1-2000) |
| `spec.motif` | Yes | Motif specification |
| `spec.motif.cif_file` | Yes | Path to the motif CIF file (.cif or .mmcif) |
| `spec.motif.ligands` | No | List of ligand mappings (protein residues are auto-detected) |
| `spec.motif.ligands[].cif_res_name` | Yes | Residue name as it appears in the motif CIF file |
| `spec.motif.ligands[].full_ligand` | Yes | Full ligand specification (see below) |


## Motif CIF File

The motif CIF file contains partial structural information extracted from a reference structure. It typically includes:

- **Partial protein sidechains**: Key residues (e.g., catalytic residues, binding site residues) with only the functionally important atoms
- **Ligand fragments**: Metal ions, cofactors, or small molecule fragments that define the binding geometry

Example content from a carbonic anhydrase motif:
```
# Partial HIS sidechain (imidazole ring only)
ATOM   883 C CG  . HIS A 1 94 ? -8.390 -0.591 11.905 ...
ATOM   885 C CD2 . HIS A 1 94 ? -7.724 -1.319 12.926 ...
ATOM   884 N ND1 . HIS A 1 94 ? -8.599  0.626 12.402 ...
ATOM   886 C CE1 . HIS A 1 94 ? -8.111  0.729 13.670 ...
ATOM   887 N NE2 . HIS A 1 94 ? -7.571 -0.471 13.996 ...

# Zinc ion
HETATM 2492 ZN ZN  . ZN  C 3 262 ? -6.844 -1.668 15.437 ...

# Partial ligand (sulfonamide group)
HETATM 2505 S S1  . AL5 D 4 555 ? -5.923  0.680 16.887 ...
HETATM 2503 O O1A . AL5 D 4 555 ? -5.452  0.890 18.209 ...
...
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
  "full_ligand": {"type": "ccd", "codes": ["ZN"]}
}
```

For multi-component ligands, include multiple codes:
```json
{
  "full_ligand": {"type": "ccd", "codes": ["HEM", "FE"]}
}
```

**Atom matching**: CIF atom names (e.g., `O1A`, `S1`, `N21`) are matched directly to CCD atom names.

### 2. SMILES Ligand

For custom ligands, use `type: "smiles"` with an `atom_map` to specify how motif atoms map to the SMILES:

```json
{
  "cif_res_name": "AL5",
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
  "task": "motif",
  "spec": {
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
          "full_ligand": {"type": "ccd", "codes": ["ZN"]}
        },
        {
          "cif_res_name": "AL5",
          "full_ligand": {"type": "ccd", "codes": ["AL5"]}
        }
      ]
    }
  }
}
```

### Custom Ligand with SMILES

Designing with a custom sulfonamide not in CCD:

```json
{
  "name": "custom_sulfonamide_design",
  "task": "motif",
  "spec": {
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
          "full_ligand": {"type": "ccd", "codes": ["ZN"]}
        },
        {
          "cif_res_name": "LIG",
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
