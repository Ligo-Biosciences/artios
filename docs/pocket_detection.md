# Pocket Detection: A Design Document

## What Pockets Are and Why We Need Them

A binding pocket is a concave region on a protein's surface where a small molecule can sit and form favorable interactions. For the data generation pipeline, we need to identify these pockets in AlphaFold-predicted structures so we can generate good motif-scaffolding examples (or dock ligands in them etc.).

The problem: geometric pocket detection algorithms (like pocketeer) find every cavity on a protein surface, including surface grooves, crystal packing artifacts, and non-functional clefts. On a typical protein, fewer than 1 in 4 detected pockets correspond to a real ligand binding site. For a pipeline processing millions of structures, this is a lot of noise.

We use a two-stage approach:
1. **Geometric detection** via alpha-sphere tessellation (pocketeer)
2. **Binding-likelihood scoring** via a logistic model trained on 130 PDB structures with known ligand positions

The combined pipeline is `find_realistic_pockets()` in `artios/data/pocket.py`.

---

## Stage 1: Alpha-Sphere Pocket Detection

Pocket detection is based on the **alpha-sphere** method from computational geometry. The core idea: the voids inside a protein can be described by the set of spheres that fit between atoms without overlapping any of them.

### Delaunay tessellation

Given N atom coordinates, the **Delaunay tessellation** partitions 3D space into tetrahedra such that no atom lies inside the circumsphere of any tetrahedron. Each tetrahedron is defined by exactly 4 atoms.

For each tetrahedron, we compute its **circumsphere** (the unique sphere passing through all 4 vertices). The circumsphere center and radius describe a void between those 4 atoms. If the circumsphere is "empty" (no other atoms strictly inside it), it's an **alpha-sphere**.

### Filtering by radius

Alpha-spheres with very small radii (< 3.0 A) correspond to the tight packing between bonded atoms. Very large radii (> 6.0 A) correspond to the protein exterior. We keep only spheres in the range [r_min, r_max] = [3.0, 6.0] A, which captures the voids relevant to small-molecule binding.

### Buried vs surface labeling

Not all interior voids are pockets. Some alpha-spheres sit near the protein surface and would be solvent-exposed. We label each sphere as **buried** or **surface** using solvent-accessible surface area (SASA):

1. Compute SASA for every atom using a probe radius of 5.0 A (larger than the standard 1.4 A water probe, to be more aggressive about detecting buried regions)
2. For each alpha-sphere, look at the SASA values of its 4 defining atoms
3. If all 4 atoms have SASA <= 10.0 A^2, the sphere is buried

Only buried spheres proceed to clustering.

### Clustering into pockets

Buried alpha-spheres are grouped into pockets by spatial proximity:

1. Build a KD-tree over sphere centers
2. Connect spheres within `merge_distance` = 1.75 A of each other
3. Find connected components (depth-first search)
4. Discard clusters with fewer than `min_spheres` = 35 spheres

Each connected component becomes a `Pocket` object with a centroid (mean of sphere centers), volume (voxel grid counting), and a basic druggability score.

### pLDDT confidence filtering

For AlphaFold structures, pockets in low-confidence regions are unreliable. When `plddt_cutoff` is set, spheres whose defining atoms all have pLDDT below the cutoff are discarded before clustering. This is used in `find_confident_pockets()` and optionally in `find_realistic_pockets()`.

---

## Stage 2: Binding-Likelihood Scoring

### The problem with raw pocket detection

On 130 PDB structures with annotated ligand positions (`pdb-samples/tracker.json`), pocketeer detects 1768 pockets. Only 379 (21%) overlap with a known ligand binding site. The rest are surface clefts, inter-domain crevices, or geometrically valid but biologically meaningless cavities.

We need features that distinguish the 379 true binding sites from the 1389 false positives.

### Ground truth construction

Each PDB in `pdb-samples/` has two annotations in `tracker.json`:
- **candidates**: ligands that are real substrates/inhibitors/cofactors (chain, resname, resid)
- **excluded**: crystallization artifacts, buffer molecules, ions, etc.

For each detected pocket, we compute the distance from its centroid to the nearest candidate ligand centroid. If the distance is < 8.0 A, the pocket is labeled as a true positive.

### Feature exploration

We computed 16 candidate features for each of the 1768 pockets and compared distributions between true positives and false positives. The most discriminating features (by TP/FP mean ratio):

| Feature | TP mean | FP mean | Ratio | What it measures |
|---------|---------|---------|-------|------------------|
| n_spheres | 99.5 | 63.2 | 1.57x | Pocket size |
| lining_residues | 21.5 | 15.8 | 1.36x | Contact complexity |
| volume | 1572 | 1166 | 1.35x | Cavity volume (A^3) |
| aromatic_ratio | 0.20 | 0.15 | 1.31x | Aromatic contacts |
| charged_ratio | 0.24 | 0.29 | 0.84x | Surface character (lower is better) |

Several features were explored but not included in the final model:
- **mean_bfactor** (B-factor of lining atoms): slightly discriminating (0.86x ratio) but excluded because B-factors are not available for AlphaFold structures used in the production pipeline. pLDDT is stored in the B-factor field for AlphaFold models, so using it as a "structural quality" proxy would conflate two different signals.
- **volume**: highly correlated with n_spheres. The logistic model implicitly captured volume information through n_spheres and mean_radius.
- **atom_density_8/12** (local atom count): marginal discriminating power.

### The 8 selected features

Each feature captures a distinct aspect of what makes a real binding site:

**Size & geometry:**
- `n_spheres` — number of alpha-spheres in the pocket. Real binding sites are larger.
- `mean_radius` — average radius of alpha-spheres. Larger radii indicate deeper voids between atoms.
- `enclosure` — fraction of 100 uniformly-distributed rays from the centroid that hit protein within 10 A. A real binding pocket is surrounded by protein on most sides.

**Contact environment:**
- `lining_residues` — number of unique residues within 4.5 A of any sphere center. More residues = more complex binding environment.
- `residue_diversity` — number of unique amino acid types among lining residues.

**Chemistry:**
- `aromatic_ratio` — fraction of lining residues that are aromatic (PHE, TYR, TRP, HIS). Aromatic residues are enriched at binding sites due to pi-stacking and hydrophobic interactions.
- `charged_ratio` — fraction of lining residues that are charged (ASP, GLU, LYS, ARG, HIS). Excess charge indicates a solvent-exposed surface rather than a binding cleft.
- `carbon_atom_ratio` — fraction of contact atoms that are carbon. A proxy for hydrophobic contact surface area.

### Enclosure computation

The enclosure score uses Fibonacci sphere sampling to distribute 100 rays uniformly over the unit sphere. For each ray direction, we step outward from the pocket centroid at 0.5 A intervals up to 10.0 A and query the protein KD-tree. If any step point comes within 2.0 A of a protein atom, the ray "hits" protein.

```
enclosure = (number of rays that hit protein) / 100
```

Real binding pockets typically have enclosure > 0.85. Surface grooves score lower.

### Model training

We fit a logistic regression on the 8 features:

```
logit = w . features + bias
binding_score = sigmoid(logit)
```

**Coordinate noise augmentation.** To prevent the model from overfitting to precise crystallographic positions (and to ensure robustness for AlphaFold-predicted coordinates which have ~1 A error), we added 0.2 A Gaussian noise to protein atom coordinates during feature computation. Each structure was augmented 5 times with independent noise realizations, yielding 8840 training samples from 1768 base pockets.

**Train/test split.** Split by PDB (not by pocket) to prevent data leakage: 90 PDBs for training, 39 for testing. This ensures the model is evaluated on entirely unseen protein structures.

**Regularization.** Standard L2 regularization (C=1.0). Features were standardized before fitting.

### Final model weights

The raw (unstandardized) weights for direct computation:

| Feature | Weight | Direction |
|---------|--------|-----------|
| n_spheres | +0.004 | More spheres = better |
| lining_residues | +0.021 | More contacts = better |
| enclosure | +6.495 | More enclosed = better |
| mean_radius | +2.550 | Deeper voids = better |
| aromatic_ratio | +3.697 | Aromatic contacts = better |
| charged_ratio | -1.462 | Fewer charges = better |
| residue_diversity | +0.123 | More diverse = better |
| carbon_atom_ratio | +4.379 | Hydrophobic surface = better |
| bias | -21.797 | |

The dominant contributors per standard deviation of variation are `mean_radius` and `enclosure`. These capture the most fundamental distinction: real binding sites are deep, enclosed cavities rather than shallow surface features.

### Validation results

**Held-out test set (39 PDBs, never seen during training):**

| Threshold | Precision | Recall | F1 | FP reduction |
|-----------|-----------|--------|-----|--------------|
| 0.10 | 0.31 | 0.93 | 0.47 | 40% |
| 0.15 (default) | 0.36 | 0.83 | 0.50 | 58% |
| 0.20 | 0.40 | 0.74 | 0.52 | 70% |
| 0.25 | 0.46 | 0.64 | 0.54 | 79% |

Cross-validated results on all 130 PDBs are consistent, confirming the model generalizes.

**Baseline comparison:** without the binding score, raw pocketeer detection has 21.4% precision (379 TP out of 1768 total pockets). The default threshold of 0.15 raises precision to ~36% while retaining 83% of true binding sites.

**Performance overhead:** the scoring adds ~60ms per protein on top of pocketeer's ~1s detection time (~5% overhead).

---

## The Full Pipeline

```
GIVEN: a protein AtomArray

1. Atom filtering
   Remove hydrogens, water, hetero atoms (ligands)

2. Alpha-sphere computation
   Delaunay tessellation -> circumspheres -> filter by radius [3.0, 6.0] A

3. Buried/surface labeling
   SASA with 5.0 A probe -> label spheres where all 4 defining atoms have SASA <= 10.0 A^2

4. (Optional) pLDDT confidence filtering
   Remove spheres in low-confidence regions

5. Clustering
   KD-tree proximity graph -> connected components -> filter by min_spheres=35

6. Pocket construction
   Compute centroid, volume (voxel grid), druggability score

7. Binding-likelihood scoring
   Compute 8 features per pocket -> logistic model -> binding_score in [0, 1]

8. Filtering
   Keep pockets with binding_score >= min_binding_score (default 0.15)

9. Atom density filtering
   Count protein atoms within 15A and 20A of pocket centroid
   Keep pockets with >= 425 atoms at 15A AND >= 900 atoms at 20A
```

---

## Parameters

### Pocket detection (pocketeer)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r_min` | 3.0 | Minimum alpha-sphere radius (A) |
| `r_max` | 6.0 | Maximum alpha-sphere radius (A) |
| `polar_probe_radius` | 5.0 | SASA probe radius for buried/surface labeling (A) |
| `sasa_cutoff` | 10.0 | SASA threshold for buried atoms (A^2) |
| `merge_distance` | 1.75 | Max sphere-sphere distance for clustering (A) |
| `min_spheres` | 35 | Minimum cluster size to form a pocket |

### Binding score

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_binding_score` | 0.15 | Minimum score to keep a pocket (0-1) |
| `plddt_cutoff` | None | pLDDT threshold for confidence filtering |

### Atom density

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_atoms_in_radius_15` | 425 | Minimum protein atoms within 15A of centroid |
| `min_atoms_in_radius_20` | 900 | Minimum protein atoms within 20A of centroid |

The 20A filter was tuned on 500 manually classified AFDB pockets (1057 good, 108 bad) and validated against 244 true-positive enzyme binding sites from PDB structures. The threshold of 900 was chosen to maximize enzyme TP retention:

| | Kept | Removed |
|---|---|---|
| Enzyme TP (244 PDB pockets) | 95.1% | 4.9% |
| AFDB good (1057 classified) | 81.1% | 18.9% |
| AFDB bad (108 classified) | 51.9% | 48.1% |

The 20A radius captures whether the pocket sits in a sufficiently large globular domain. Bad pockets (surface clefts, inter-domain crevices) tend to have fewer atoms at this radius. The 15A filter catches locally sparse regions while the 20A filter catches pockets in small or elongated structural contexts.

### Choosing `min_binding_score`

- **0.10** for maximum recall (93%): use when you cannot afford to miss pockets
- **0.15** (default) for balanced filtering (83% recall, 58% FP reduction): good for large-scale data generation
- **0.25** for higher precision (64% recall, 79% FP reduction): use when downstream processing is expensive

---

## Entry Points

| Function | Description |
|----------|-------------|
| `find_pockets(atomarray)` | Raw alpha-sphere pocket detection |
| `find_confident_pockets(atomarray)` | Pocket detection with pLDDT filtering |
| `find_realistic_pockets(atomarray)` | Pocket detection + binding-likelihood scoring (recommended) |
| `find_globular_pockets(atomarray)` | Pocket detection + local atom density filter |
| `score_binding_pocket(pocket, protein_atoms, tree)` | Score a single pocket |

---

## File Reference

| File | Role |
|------|------|
| `artios/data/pocket.py` | All pocket detection functions, binding score model |
| `artios/data/engineering/scripts/retrain_pocket_binding_model.py` | Retraining script with noise augmentation |
| `pdb-samples/tracker.json` | Ground truth: candidate ligands and excluded artifacts for 130 PDBs |
| `pdb-samples/enzyme/` | 79 enzyme structures with annotated ligands |
| `pdb-samples/non-enzyme/` | 56 non-enzyme structures with annotated ligands |
