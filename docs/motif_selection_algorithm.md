# The Motif Selection Algorithm: A Design Document

## The Problem We're Actually Solving

At first glance, "pick some atoms to be the motif" sounds trivial. Just grab a ligand and some nearby residues, right?

The reality is more subtle. We're training a model to scaffold around motifs, which means:

1. **The motif must be learnable** — if it's always complete sidechains, the model never learns to work with partial information
2. **The motif must be diverse** — if it's always ATP, the model won't generalize to other ligands
3. **The motif must be realistic** — random atom soup isn't useful; real motifs have spatial coherence
4. **The motif must have context** — a ligand floating in space teaches nothing; we need surrounding protein

This tension between diversity (sample everything!) and realism (sample coherent things!) drives most of the complexity.

---

## The High-Level Decision Tree

Every training sample goes through this cascade:

```
GIVEN: a protein structure (maybe with ligands, maybe with pre-computed pockets)

IF structure has ligands AND coin_flip(85%):
    → use LIGAND-CENTERED motif generation

ELSE IF structure has pocket annotations:  # 100% when available
    → use POCKET-CENTERED motif generation

ELSE:
    → use CHAIN-BASED fallback (compute pockets on-the-fly)
```

Why this hierarchy?

**Ligands are gold.** When we have a ligand-bound structure, we know exactly where the binding site is. The ligand itself provides rich geometric information. We use this 85% of the time for such structures.

**But not always.** That 15% where we ignore the ligand? It teaches the model that scaffolds exist independently of ligands. If we always center on ligands, the model might learn to "cheat" by only looking at ligand atoms.

**Pre-computed pockets are silver.** For both AFDB and PDB structures, we run pocket detection offline during preprocessing. This is expensive (alpha-sphere computation via Delaunay tessellation), so we cache it. When available, we *always* use it (100%).

**On-the-fly computation is bronze.** When pocket annotations aren't available for some reason, we compute a simple "pocket score" at runtime. It's not as good as proper alpha-sphere pocket detection, but it's fast and better than random.

---

## Ligand-Centered Motifs: The Happy Path

When we have a ligand, we face several non-obvious decisions:

### Which ligand?

Many structures have multiple ligands. A naive approach picks uniformly at random. But this creates a problem: if a structure has one ATP and five zinc ions, uniform sampling gives zinc 83% of the time.

Metal ions are boring. They're single atoms. We want to learn from complex ligands with interesting geometry.

**Our solution: two-stage selection**

```
CLASSIFY each ligand:
    IF atom_count >= 5:  → "complex" (ATP, heme, drug-like molecules)
    ELSE:                → "simple" (metal ions, water-like)

IF complex ligands exist:
    WITH probability 80%:
        → sample from complex ligands only
    WITH probability 20%:
        → sample uniformly from all ligands
ELSE:
    → sample from simple ligands (fallback)
```

We heavily prefer complex ligands (80%) because they provide richer geometric signal. But that 20% where we might pick a simple ligand ensures the model doesn't completely forget about metal coordination sites and small molecule binding.

### Always partial ligands

Here's a key insight: at inference time, users will almost always give us incomplete ligand specifications. "I want to bind something that looks like *this part* of ATP." If we train on whole ligands, the model learns to expect complete molecules and fails on partial specifications.

**Our solution: always use partial ligands (100%).** We never include the whole ligand in the motif. This forces the model to learn from fragments, making it robust to incomplete specifications at inference time.

### Partial ligands: multi-region BFS

A ligand isn't just "atoms" — it's a molecular graph with bonds. We can't just pick random atoms; that would create chemically nonsensical fragments.

**The key insight:** real partial ligand specifications often describe multiple functional groups. "I want something with a phosphate group AND an aromatic ring." These are typically non-adjacent in the molecular graph.

```
SAMPLE number of regions from {1: 50%, 2: 40%, 3: 7%, 4: 3%}

FOR each region:
    SELECT a seed atom (ensuring seeds are graph-distance >= 2 apart)
    SAMPLE region size from {1: 15%, 2: 25%, 3: 30%, 4: 20%, 5: 10%} atoms
    PERFORM BFS from seed, collecting that many atoms

COMBINE all regions into final partial ligand
```

Why require seeds to be non-adjacent? If they could be neighbors, multiple regions might just merge into one contiguous blob. Forcing distance >= 2 ensures we actually get distinct functional groups.

Why those specific distributions? They're empirically tuned. Most partial specifications are 1-2 regions (90%), but we occasionally see 3-4 region specs (10%), so we train on those too.

### Supplementary ligands

Plot twist: sometimes multiple ligands cooperate in a binding site. ATP might sit next to a magnesium ion. If we're centering on ATP, should we include the magnesium?

```
FIND all other ligand chains within 5Å of our seed point

FOR each nearby ligand chain:
    WITH probability 50%:
        PERFORM partial BFS from closest atom
        ADD to motif
```

This teaches the model that binding sites can involve multiple molecular entities.

### The "drop ligand" trick

Sometimes we want to generate scaffolds that *could* bind a ligand but don't currently show it. The motif is just protein residues shaped for binding.

```
WITH probability 10%:
    → generate residues around ligand, but exclude ligand atoms from final motif
```

This is subtle but important: we use the ligand to *find* the binding site, then pretend it's not there. The model learns to recognize binding-competent pockets from protein shape alone.

---

## Residue Selection: The Shared Machinery

All three paths (ligand/pocket/chain) eventually call the same core function: given a 3D point, select nearby protein residues.

### The radius dilemma

How far from the seed point should we look? Too small = not enough context. Too large = half the protein becomes "motif."

**Solution:** sample radius uniformly from [5Å, 8Å] each time. This creates natural variation in motif size while ensuring we capture enough of the binding site geometry.

### Tip atoms vs all atoms

Here's a non-obvious design choice. Sidechains have many atoms, but not all are equally informative:

- **Backbone atoms (N, CA, C, O):** Same for every residue. Not very distinctive.
- **Beta carbon (CB):** First sidechain atom. Indicates direction but not chemistry.
- **Tip atoms (terminal atoms of sidechain):** These define the functional groups that actually interact with ligands.

```
WITH probability 95%:
    → only include TIP ATOMS in motif (CB, CG, CD, etc. depending on residue type)
WITH probability 5%:
    → include all atoms
```

We almost always train on tip atoms because these are what actually matter for chemistry. The rare "all atoms" case (5%) keeps the model from completely forgetting about non-tip atoms, but it's a minor consideration.

### Partial sidechains: BFS on bonds

Even tip atoms can be too much information. A phenylalanine has 7 heavy sidechain atoms. Sometimes we want to specify "there's *something aromatic* here" without full atomic detail.

```
FOR each residue near seed point:
    WITH probability 60%:
        → skip this residue entirely (aggressive dropout!)

    # For the 40% of residues we keep:
    → ALWAYS use PARTIAL sidechain via BFS (never whole):
        PICK seed atom: ALWAYS from atoms already in sphere (100%)
        SAMPLE depth from geometric(p=0.5) distribution
        COLLECT atoms within that many bonds of seed
```

Two key design choices here:

1. **Aggressive residue dropout (60%).** We skip most residues! This forces the model to work with sparse motif information, which is realistic — users won't specify every residue in a binding site.

2. **Always partial sidechains.** We never include whole sidechains. Combined with the tip-atoms-only preference, this creates very sparse motifs: just a few atoms from a few residues, representing the functional groups that matter.

The geometric distribution for depth creates a mix of very partial (1-2 atoms, high probability) and slightly more complete (4+ atoms, low probability) sidechain fragments.

---

## Chain-Based Fallback: When All Else Fails

For structures without ligands or pre-computed pockets, we need to find pocket-like regions on the fly.

### The pocket score heuristic

The intuition: atoms in pockets are "surrounded" by other atoms. Surface atoms have neighbors mostly on one side.

```
FOR each atom in the chain:
    FIND 100 nearest neighbors
    COMPUTE centroid of neighbors
    pocket_score = distance from atom to neighbor centroid

Atoms where pocket_score <= 0.18Å are likely in pockets
```

Why does this work? Surface atoms have their neighbor centroid shifted toward the protein interior. Pocket atoms have neighbors surrounding them, so the centroid is close.

We use 100 neighbors (not 25) to get a more robust estimate of local geometry. This isn't as accurate as alpha-sphere detection (used in preprocessing), but it's fast enough to run at training time.

---

## The is_motif_related Mask

We generate two masks:
- `is_motif`: the actual atoms the model must reconstruct
- `is_motif_related`: all atoms in residues that have any motif atoms

Why both? Partial sidechains create a problem: if the motif includes CB and CG of a phenylalanine, what about CD1, CD2, CE1, CE2, CZ? They're not in the motif, but they're clearly "related" to motif atoms.

The loss function can weight these differently. Motif atoms get high weight (must reconstruct accurately). Motif-related atoms get medium weight (should be reasonable). Everything else gets base weight.

---

## Why All These Probabilities?

You might notice the code is full of probability distributions rather than hard rules. This is intentional:

1. **Robustness**: The model sees diverse combinations during training
2. **No sharp boundaries**: Hard cutoffs create weird behavior at edge cases
3. **Sparsity as default**: Our training config pushes hard toward sparse motifs (partial everything, aggressive dropout)

The specific probability values are hyperparameters. Our training configuration goes further than the code defaults — we *never* use whole ligands or sidechains, and we drop 60% of residues. This creates very sparse motifs that force the model to learn from minimal information.

---

## Summary: The Full Algorithm

```
FUNCTION make_motif(structure):

    # === STEP 1: Choose strategy ===
    IF has_ligand AND random() < 0.85:
        strategy = LIGAND
    ELSE IF has_pocket_annotation:  # always use if available
        strategy = POCKET
    ELSE:
        strategy = CHAIN

    # === STEP 2: Find seed point ===
    IF strategy == LIGAND:
        ligand = select_ligand_with_complexity_bias()  # 80% complex preference
        # Always use partial ligand (never whole)
        ligand_atoms = multi_region_bfs(ligand)
        seed = random_atom_in(ligand_atoms)

    ELSE IF strategy == POCKET:
        pocket = random_choice(pre_computed_pockets)
        seed = centroid(pocket)

    ELSE:  # CHAIN
        chain = random_polymer_chain()
        pocket_scores = compute_pocket_scores(chain, n_neighbors=100)
        pocket_atoms = atoms_where(pocket_scores <= 0.18)
        seed = random_atom_in(pocket_atoms)

    # === STEP 3: Select surrounding residues ===
    radius = uniform(5.0, 8.0)
    nearby_residues = residues_within(seed, radius)

    tip_only = random() < 0.95  # almost always tip atoms only

    FOR each residue in nearby_residues:
        IF random() > 0.4:  # aggressive 60% dropout!
            CONTINUE

        # Always partial sidechain (never whole)
        seed_atom = random_atom_in_sphere(residue)  # always from sphere
        depth = geometric(0.5)
        partial = bfs_on_residue_bonds(residue, seed_atom, depth)
        ADD partial TO motif

    IF tip_only:
        FILTER motif TO tip_atoms_only

    # === STEP 4: Handle ligand inclusion ===
    IF strategy == LIGAND:
        IF random() < 0.1:  # drop ligand
            REMOVE ligand_atoms FROM motif
        ELSE:
            ADD ligand_atoms TO motif
            ADD supplementary_ligands_nearby(seed) TO motif  # 50% per nearby ligand

    # === STEP 5: Validate ===
    IF motif.atom_count == 1:
        RETURN empty_motif  # single atoms aren't useful
    IF motif.residue_count > 20:
        RETURN empty_motif  # too large, probably a bug

    RETURN motif
```

---

## Training Configuration

The code has sensible defaults, but our actual training configuration is more aggressive about sparsity:

| Parameter | Training Value | Philosophy |
|-----------|----------------|------------|
| `ligand_motif_prob` | 0.85 | Usually center on ligands when available |
| `pocket_motif_prob` | 1.0 | Always use pre-computed pockets |
| `whole_ligand_prob` | 0.0 | Never whole ligands — always partial |
| `whole_sidechain_prob` | 0.0 | Never whole sidechains — always partial |
| `residue_inclusion_prob` | 0.4 | Aggressive 60% residue dropout |
| `use_tip_atoms_only_prob` | 0.95 | Almost always just tip atoms |
| `complex_ligand_preference_prob` | 0.8 | Prefer complex ligands 80% of time |
| `r_min` / `r_max` | 5.0 / 8.0 Å | Larger radius than defaults |
| `pocket_n_neighbours` | 100 | More neighbors for robust pocket scoring |
| `sidechain_seed_is_in_sphere_prob` | 1.0 | Always seed from in-sphere atoms |

The theme is **sparsity**: partial ligands, partial sidechains, tip atoms only, aggressive dropout. This forces the model to work with minimal information, making it robust to incomplete motif specifications at inference time.

---

## File Reference

The implementation lives in `protenix/data/motif_maker.py`:

| Function | Purpose |
|----------|---------|
| `MotifMaker.make_motif()` | Main entry point |
| `_generate_ligand_motif_mask()` | Ligand-centered path |
| `_generate_pocket_motif_mask()` | Pocket-centered path |
| `_generate_chain_motif_mask()` | Fallback path |
| `_generate_motif_around_point_mask()` | Core residue selection |
| `_get_partial_residue_motif_mask()` | BFS on residue bonds |
| `_add_supplementary_ligands()` | Multi-ligand handling |
| `compute_pocket_scores()` | On-the-fly pocket detection |
| `bfs_nodes_by_count()` | BFS utility for partial selection |
| `select_non_adjacent_seeds()` | Multi-region seed selection |
