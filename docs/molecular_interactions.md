# Molecular Interaction Detection

This document describes the `artios.contacts` module for detecting non-covalent molecular interactions in protein-ligand structures. The implementation follows Molstar's algorithms as ground truth.

## Overview

The module detects three types of interactions:

| Interaction | Function | Key Criteria |
|------------|----------|--------------|
| Hydrogen bonds | `detect_hydrogen_bonds()` | Distance + angle geometry |
| Pi-stacking | `detect_pi_stacking()` | Ring centroid distance + perpendicular offset |
| Cation-pi | `detect_cation_pi()` | Cation-ring distance + perpendicular offset |

All detection functions accept parameters directly (no config objects) with angles specified in degrees.

## Module Structure

```
artios/contacts/
├── hydrogen_bonds.py   # H-bond donor/acceptor detection
├── aromatic.py         # Pi-stacking and cation-pi detection
├── geometry.py         # Geometric helper functions
└── valence.py          # Valence model (charge, H-count, geometry)
```

---

## Hydrogen Bonds

### Algorithm

Hydrogen bonds are detected between donor-acceptor pairs satisfying:

1. **Distance**: D-A distance ≤ 3.5 Å (4.1 Å for sulfur)
2. **Donor angle**: X-D-A angles within tolerance of ideal geometry
3. **Acceptor angle**: X-A-H angles satisfy one-sided constraint (large angles OK)
4. **Out-of-plane**: For trigonal geometry, acceptor must be near the plane

### Usage

```python
from biotite.structure.io import load_structure
from artios.contacts.hydrogen_bonds import detect_hydrogen_bonds

atoms = load_structure("complex.cif", include_bonds=True)
hbonds = detect_hydrogen_bonds(
    atoms,
    distance_max=3.5,           # Å
    don_angle_dev_max=45.0,     # degrees
    acc_angle_dev_max=45.0,     # degrees
    include_backbone=False,     # skip backbone-backbone
)

for hb in hbonds:
    print(f"{atoms.res_name[hb.donor_idx]} -> {atoms.res_name[hb.acceptor_idx]}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `distance_max` | 3.5 | Max donor-acceptor distance (Å) |
| `sulfur_distance_max` | 4.1 | Max distance when S involved (Å) |
| `don_angle_dev_max` | 45.0 | Max donor angle deviation (degrees) |
| `acc_angle_dev_max` | 45.0 | Max acceptor angle deviation (degrees) |
| `don_out_of_plane_max` | 45.0 | Max donor out-of-plane (degrees) |
| `acc_out_of_plane_max` | 90.0 | Max acceptor out-of-plane (degrees) |
| `ignore_hydrogens` | True | Use implicit H positions |
| `include_backbone` | False | Include backbone-backbone H-bonds |
| `include_water` | False | Include water-water H-bonds |

### Donor/Acceptor Detection

Donors and acceptors are identified using the valence model:

**Donors** (atoms that donate H):
- N, O, S atoms with `total_h > 0`
- Histidine ring nitrogens (ND1, NE2) — always included due to ambiguous protonation

**Acceptors** (atoms that accept H):
- All oxygen atoms
- Neutral nitrogen with lone pairs available
- Sulfur in CYS/MET or with charge -1

---

## Pi-Stacking

### Algorithm (Molstar)

Pi-stacking uses an **offset-based criterion** rather than shift-angle:

1. **Centroid distance**: ≤ 5.5 Å between ring centroids
2. **Perpendicular offset**: ≤ 2.0 Å (minimum from both ring perspectives)
3. **Plane angle**: Parallel (< 30°) or perpendicular (90° ± 30°)

The offset criterion projects one ring centroid onto the other ring's plane and measures lateral displacement. This is more robust than angle-based methods.

### Usage

```python
from artios.contacts.aromatic import detect_pi_stacking, StackingType

interactions = detect_pi_stacking(
    atoms,
    centroid_cutoff=5.5,      # Å
    offset_max=2.0,           # Å
    plane_angle_tol=30.0,     # degrees
    infer_bonds=True,         # for ligands without bond annotations
)

for pi in interactions:
    ring1_res = atoms.res_name[pi.ring1_indices[0]]
    ring2_res = atoms.res_name[pi.ring2_indices[0]]
    stacking = "parallel" if pi.stacking_type == StackingType.PARALLEL else "T-shaped"
    print(f"{ring1_res} - {ring2_res}: {stacking}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `centroid_cutoff` | 5.5 | Max centroid-centroid distance (Å) |
| `offset_max` | 2.0 | Max perpendicular offset (Å) |
| `plane_angle_tol` | 30.0 | Tolerance for parallel/perpendicular (degrees) |
| `infer_bonds` | False | Infer bonds via distances (for ligands) |

### Aromatic Ring Detection

Rings are found using two methods (combined and deduplicated):

1. **Bond-flag detection**: Uses biotite's `find_aromatic_rings()` which looks for aromatic bond types
2. **Planarity detection**: Falls back to SVD-based planarity check for structures without aromatic bond annotations (common in predicted structures)

The `infer_bonds=True` parameter calls `connect_via_distances()` to add missing bonds — critical for ligands that often lack bond annotations in mmCIF files.

---

## Cation-Pi

### Algorithm (Molstar)

Cation-pi interactions are detected between aromatic rings and cationic groups:

1. **Distance**: Ring centroid to cation group centroid ≤ 6.0 Å
2. **Perpendicular offset**: ≤ 2.0 Å

### Cation Identification

Cations are identified by combining two approaches:

1. **Valence model charges**: Atoms with computed `charge > 0` (catches ligands, unusual chemistry)
2. **Residue patterns**: Non-backbone nitrogens in ARG/LYS/HIS (captures physiological protonation that formal charge calculations miss)

### Cation Grouping

Cations are **grouped by residue** to avoid duplicate interactions. For example, ARG's guanidinium group (NE, NH1, NH2) becomes one cation group with its centroid used for distance calculations. This matches Molstar's behavior.

### Usage

```python
from artios.contacts.aromatic import detect_cation_pi

interactions = detect_cation_pi(
    atoms,
    distance_max=6.0,    # Å
    offset_max=2.0,      # Å
    infer_bonds=True,
)

for cp in interactions:
    ring_res = atoms.res_name[cp.ring_indices[0]]
    cation_res = atoms.res_name[cp.cation_indices[0]]
    cation_atoms = [atoms.atom_name[i] for i in cp.cation_indices]
    print(f"{ring_res} - {cation_res}:{cation_atoms} (offset={cp.offset:.2f}Å)")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `distance_max` | 6.0 | Max ring-cation distance (Å) |
| `offset_max` | 2.0 | Max perpendicular offset (Å) |
| `infer_bonds` | False | Infer bonds via distances |

---

## Valence Model

The valence model computes per-atom properties needed for interaction detection:

```python
from artios.contacts.valence import compute_valence_model

model = compute_valence_model(atoms)
# model.charge      - formal charge (int8)
# model.implicit_h  - missing hydrogens (int8)
# model.total_h     - total H count (int8)
# model.ideal_geometry - AtomGeometry enum value (int8)
```

The model is computed once and passed to detection functions to avoid redundant calculation.

### Atom Geometry

Geometry is assigned based on coordination number (VSEPR theory):

| Geometry | Coordination | Ideal Angle | Example |
|----------|--------------|-------------|---------|
| Terminal | 1 | — | H in H₂O |
| Linear | 2 | 180° | CO₂ carbon |
| Trigonal | 3 | 120° | sp² carbon |
| Tetrahedral | 4 | 109.5° | sp³ carbon |

---

## Geometry Utilities

Key functions in `geometry.py`:

### `point_plane_offset(point, plane_origin, plane_normal)`

Projects a point onto a plane and returns the lateral distance from the plane origin. Used for both pi-stacking and cation-pi detection.

### `minimum_angle(v1, v2)`

Returns the minimum angle between vectors (range [0, π/2]). Used for ring normal comparisons where direction is ambiguous.

### `triangle_normal(coords)`

Computes the unit normal from three or more coplanar points. Used for ring plane calculations.

### `calc_angles(atoms, index_a, index_b)`

Calculates X-A-B angles for all atoms X bonded to A. Returns separate lists for heavy atoms and hydrogens.

### `calc_plane_angle(atoms, index_a, index_b)`

Measures the out-of-plane angle from B to the plane at A. Used for trigonal geometry constraints.

---

## Common Patterns

### Processing a Structure

```python
from biotite.structure.io.pdbx import CIFFile, get_structure
from biotite.structure import connect_via_residue_names

from artios.contacts.valence import compute_valence_model
from artios.contacts.hydrogen_bonds import detect_hydrogen_bonds
from artios.contacts.aromatic import detect_pi_stacking, detect_cation_pi

# Load structure with bonds
cif = CIFFile.read("complex.cif")
atoms = get_structure(cif, model=1, include_bonds=True)

# Ensure bonds are set (critical for interaction detection)
if atoms.bonds is None:
    atoms.bonds = connect_via_residue_names(atoms)

# Compute valence model once
valence_model = compute_valence_model(atoms)

# Detect all interactions
hbonds = detect_hydrogen_bonds(atoms, valence_model=valence_model)
pi_stacking = detect_pi_stacking(atoms, infer_bonds=True)
cation_pi = detect_cation_pi(atoms, infer_bonds=True, valence_model=valence_model)
```

### Handling Ligands

Ligands in mmCIF files often lack bond annotations. Use `infer_bonds=True` to detect bonds via atomic distances:

```python
# Without infer_bonds: ligand rings may not be detected
pi_stacking = detect_pi_stacking(atoms, infer_bonds=False)  # May miss ligand interactions

# With infer_bonds: bonds are inferred from distances
pi_stacking = detect_pi_stacking(atoms, infer_bonds=True)   # Detects ligand rings
```

Note: `infer_bonds=True` **always** calls `connect_via_distances()`, even if some bonds exist. This ensures ligand bonds are added even when protein bonds are already annotated.

---

## References

- **Molstar**: https://github.com/molstar/molstar (ground truth implementation)
- **HBPLUS**: McDonald & Thornton (1994) https://doi.org/10.1006/jmbi.1994.1334
- **Sulfur H-bonds**: Zhou et al. (2009) https://doi.org/10.1002/prot.22327
- **OpenEye valence model**: https://docs.eyesopen.com/toolkits/python/oechemtk/valence.html
