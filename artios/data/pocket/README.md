# Pocket Detection

Two-stage ML pocket detector trained on 3,161 holo crystal structures from the [holo4k dataset](https://github.com/rdk/p2rank) with MOAD-validated ligand positions as ground truth.

## Usage

```python
from artios.data.pocket import find_pockets

pockets = find_pockets(protein_atoms)                       # balanced (default)
pockets = find_pockets(protein_atoms, mode="high_recall")   # ~86% recall, ~49% precision
pockets = find_pockets(protein_atoms, mode="high_precision") # ~63% recall, ~73% precision
pockets = find_pockets(protein_atoms, plddt_cutoff=70.0)    # for AlphaFold models
```

Returns a list of `pocketeer.core.Pocket` objects sorted by `binding_score` (highest first). Each pocket has `.centroid`, `.volume`, `.n_spheres`, `.spheres`, and `.binding_score` attributes.

## How it works

### Stage 1: Per-sphere random forest

Alpha spheres are geometric primitives from Delaunay tessellation — they sit in void spaces between protein atoms and mark potential cavities. For each sphere, we compute 22 features:

- **Geometric (3):** sphere radius, is_buried (from SASA), burial count of defining atoms
- **Protrusion (4):** protein atom counts within 6/8/10/12 A (inspired by [P2Rank](https://github.com/rdk/p2rank)'s most important feature)
- **Chemistry within 8A (11):** hydrophobic/hydrophilic/charged/aromatic residue fractions, H-bond donor/acceptor fractions, element fractions (C/N/O/S), polar atom fraction
- **Local sphere context (3):** nearby sphere density at 3/5 A, mean neighbor sphere radius
- **Atom density (1):** atom count / sphere volume at 8A

A random forest (100 trees, depth 8) classifies each sphere as near a ligand atom (within 3A) or not. AUROC: 0.938 on held-out structures.

Top features by importance: `sphere_density_5A` (31%), `protrusion_12A` (16%), `mean_neighbor_radius_5A` (10%) — the model primarily uses geometric density features to identify spheres in deep, well-packed cavities.

### Stage 2: Per-pocket gradient boosting

Buried spheres are clustered into pockets (same clustering as pocketeer). Each pocket gets 59 features:

- **Binding geometry (8):** n_spheres, lining_residues, enclosure (ray casting), mean_radius, aromatic/charged/diversity ratios, carbon_atom_ratio — these are the features from the old logistic model, with enclosure being the most informative pocket-level geometric feature
- **Aggregated sphere features (44):** mean and std of all 22 sphere features across the pocket's spheres
- **Stage 1 RF score statistics (7):** mean, median, 90th percentile, std, and fraction above 0.3/0.5/0.7 of per-sphere binding probabilities

A gradient boosting classifier (500 trees, depth 7, lr 0.03) scores each pocket. The RF score statistics (especially `rf_p90` at 39% importance) dominate, with chemistry features providing secondary signal.

## Training data

**Source:** [holo4k(mlig)](https://github.com/rdk/p2rank) — 3,448 PDB structures with ligand codes validated by the [MOAD 2013](http://bindingmoad.org/) database.

**Filtering:**
- Excluded membrane proteins (15,641 PDB IDs from `excluded_membrane_pdbs.txt`)
- Excluded ribosomes (2,661 PDB IDs from `excluded_ribosome_pdbs.txt`)
- Excluded fibrils (1,316 PDB IDs from `excluded_fibril_pdbs.txt`)
- Excluded artifact ligands, crystallization aids, and glycans using CCD code sets from `artios/common/ccd_sets.py` (1,127 codes)
- Excluded metal ions (`artios/common/elements.py:METALS`) from both protein atoms and ligand labeling
- Held out 78 enzyme structures (`pdb-samples/enzyme/`) for validation

**After filtering:** 3,161 structures, 14.1M alpha spheres (5.2% positive), 23.5K pockets (24% positive).

**Labeling:**
- Sphere label: positive if center is within **3A** of any non-artifact ligand heavy atom
- Pocket label: positive if centroid is within **3A** of any ligand atom
- 0.1A Gaussian noise added to protein coordinates for robustness

**Train/test split:** 90/10 by PDB structure (not by sphere/pocket) to prevent leakage.

## Performance

### On holo4k test set (317 structures)

| Stage | Metric | Value |
|-------|--------|-------|
| Stage 1 (sphere) | AUROC | 0.938 |
| Stage 1 (sphere) | Avg Precision | 0.498 |
| Stage 2 (pocket) | AUROC | 0.933 |
| Stage 2 (pocket) | Avg Precision | 0.822 |

Stage 2 pocket-level precision-recall:

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.10 | 48% | 95% | 0.64 |
| 0.20 | 58% | 89% | 0.70 |
| 0.30 | 64% | 84% | 0.73 |
| 0.40 | 69% | 75% | 0.72 |
| 0.50 | 75% | 67% | 0.70 |
| 0.70 | 83% | 47% | 0.60 |

### On enzyme validation set (78 held-out structures)

Compared to the old logistic model (`find_realistic_pockets`):

| Method | Precision | Recall | F1 | Structures with TP |
|--------|-----------|--------|-----|-------------------|
| Old logistic @ 0.15 | 35% | 100% | 0.51 | 61/78 |
| **Two-stage balanced** | **54%** | **82%** | **0.65** | **65/78** |
| **Two-stage high_precision** | **73%** | **63%** | **0.68** | - |

## Background: why not P2Rank directly?

We initially benchmarked [P2Rank](https://github.com/rdk/p2rank) (a Java-based pocket predictor using random forests on Connolly surface points). P2Rank's rescoring of our pocketeer pockets showed strong TP/FP separation (per-pocket P2Rank probability: TP median 0.546 vs FP 0.203). However:

1. P2Rank is a heavy Java dependency requiring JDK 21
2. Its rescore mode requires fpocket-format output files with specific directory structures
3. Its key insight — per-point RF classification using protrusion and chemistry features — is simple enough to reimplement on alpha spheres

So we adopted P2Rank's core idea (per-point random forest with geometric + chemical features) but applied it to pocketeer's alpha spheres instead of Connolly surface points. The protrusion feature (atom count within a radius) turned out to be the most discriminative, consistent with P2Rank's findings.

## Files

| File | Description |
|------|-------------|
| `detect.py` | Main API: `find_pockets()` with mode presets |
| `tessellation.py` | Vectorized alpha sphere computation (4x faster than pocketeer's Python loop) |
| `legacy.py` | Old `pocket.py` API: `find_realistic_pockets()`, `find_globular_pockets()` |
| `sphere_rf.joblib` | Stage 1 model (3.9 MB) |
| `pocket_rf.joblib` | Stage 2 model (5.0 MB) |

## Retraining

```bash
# Step 1: Compute alpha spheres for all holo4k structures (parallel, ~5 min)
python -m artios.data.engineering.scripts.train_pocket_classifier prepare --workers 8

# Step 2: Compute features from cached spheres (~10 min)
python -m artios.data.engineering.scripts.train_pocket_classifier features

# Step 3: Train models from pre-computed features (~2 min per config)
python -m artios.data.engineering.scripts.train_pocket_classifier train \
    --s1-estimators 100 --s1-depth 8 \
    --s2-estimators 500 --s2-depth 7 --s2-lr 0.03

# Hyperparameter sweep (loads features once, sweeps stage 2 configs)
python -m artios.data.engineering.scripts.tune_pocket_classifier
python -m artios.data.engineering.scripts.tune_pocket_classifier --subset 10  # fast test on 1/10th
```

Requires the [holo4k dataset](https://github.com/rdk/p2rank) cloned at `../p2rank-datasets/`.
