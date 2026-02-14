# Structural Mechanics Metrics for Protein Evaluation

Physics-based, interpretable metrics inspired by mechanical/structural engineering — complementing our existing deep-learning-based folding scores.

---

## Overview

We evaluate generated structures at three levels:

| Level | What we have | What we generate |
|-------|-------------|-----------------|
| **Backbone** | Cα trace, secondary structure | Backbone model output |
| **Sequence + backbone** | Residue identities on a scaffold | MPNN / sequence design output |
| **All-atom** | Full heavy-atom coordinates + ligand | All-atom model output |

**Design principle: self-contained implementations.** Prefer scipy/numpy/biotite over external tools wherever possible. The only external library we allow for a non-trivial algorithm is TRAMbio for rigidity analysis (the pebble game is too complex to reimplement well).

---

## Phase 1: Backbone Metrics

### 1.1 Anisotropic Network Model (ANM) — Predicted B-Factors

**Level:** Backbone, All-atom

**Engineering analogy:** Finite element analysis (FEA). The protein is a network of springs; normal mode analysis gives natural frequencies and displacement patterns.

**Algorithm:**
1. Place nodes at Cα atoms (backbone) or all heavy atoms (all-atom)
2. Connect nodes within a cutoff distance (15 Å for Cα, ~4–5 Å for all-atom) with uniform harmonic springs (spring constant γ)
3. Build the 3N × 3N Hessian matrix H, where off-diagonal 3×3 blocks encode anisotropic spring interactions via the outer product of the distance vector, normalized by distance squared. Diagonal blocks enforce row-sum-to-zero (translational invariance).
4. Diagonalize H. First 6 eigenvalues are zero (rigid-body modes). Mean-square displacement of node i comes from summing `(1/λ_k) · |u_k^(i)|²` over the first ~20 non-trivial modes.
5. B-factor: `B_i = (8π²/3) · ⟨ΔR_i²⟩`

**Implementation:** Self-contained with scipy + numpy. We already build kNN graphs and do eigendecomposition in `fragment.py` — the ANM Hessian is a natural extension. Use `scipy.spatial.KDTree.query_pairs(cutoff)` for pair finding, `numpy.linalg.eigh` for diagonalization (dense is fine for Cα-level, ~hundreds of nodes).

**Including ligands:** Yes — add ligand heavy-atom coordinates to the node array alongside Cα atoms. Use a shorter cutoff (~5 Å) for ligand–protein contacts to reflect tighter interaction distances. Compute both with and without ligand, report ΔB-factor at binding-site residues. This reveals whether the binding site is preorganized (already stiff before ligand) and how much the ligand constrains local dynamics.

**Outputs (per residue):**

| Metric | Description |
|--------|-------------|
| `anm_bfactor` | Predicted B-factor from ANM (mobility) |
| `anm_bfactor_with_ligand` | B-factor with ligand in the network |
| `anm_delta_bfactor` | Change in B-factor upon ligand inclusion |
| `anm_collectivity` | Collectivity of the dominant mode (0–1) |

**Validation:** Run on CIF files in `notebooks/PDBs/`. Compare ANM-predicted B-factors against experimental B-factors. Mobile residues (loops) should have high ANM B-factors; core/active site residues should have low.

---

### 1.2 Network / Graph Metrics

**Level:** Backbone

**Engineering analogy:** Structural topology analysis. A building's load-bearing frame is a graph; its spectral properties reveal weak points, redundancy, and connectivity.

We already build Gaussian-weighted kNN graphs over Cα atoms in `protenix/data/fragment.py` (spectral bisection for domain detection, lines 130–165). The same graph construction yields additional metrics.

**Implementation:** Reuse the kNN graph + normalized Laplacian construction from `fragment.py`. Fully self-contained with scipy sparse + numpy.

**Metrics from the graph Laplacian:**

| Metric | Formula | Engineering interpretation |
|--------|---------|--------------------------|
| **Fiedler value (λ₂)** | 2nd smallest eigenvalue of normalized Laplacian | Algebraic connectivity — how well-connected the structure is. Our spectral_split threshold is 0.05; generated proteins should be well above this. |
| **Spectral gap** | λ₂ / λ_max | Normalized connectivity. Compares weakest link to strongest. |
| **Fiedler vector entropy** | −Σ p_i log(p_i) where p_i = v_i² / Σv² | How uniform the structure is. High entropy = no clear partition; low = two-part division. |
| **Graph effective resistance** | Σ 1/λ_k terms from Laplacian | Total "compliance" (inverse stiffness) of the network. |

**Applicability of Fiedler vector:** Yes, appropriate here. For domain detection we threshold λ₂ and cut; for evaluation we use it as a continuous quality metric. A generated protein with λ₂ ≈ 0.01 has a near-disconnection that probably shouldn't be there.

---

### 1.3 Rigidity Analysis — Pebble Game Algorithm

**Level:** Backbone, All-atom

**Engineering analogy:** Structural determinacy analysis. A truss is rigid if it has enough bars (constraints) in the right places. Maxwell's rule gives a necessary count (3N − 6); the pebble game checks whether constraints are properly distributed locally.

For proteins, the appropriate model is the **(6,6)-body-bar framework**: each atom with its covalent neighbors forms a rigid body (6 DOF), single bonds contribute 5 constraints (1 remaining torsional DOF), double bonds contribute 6 (fully rigid), and H-bonds/salt bridges/hydrophobic contacts add further constraints.

The algorithm assigns 6 "pebbles" (DOF) per body, inserts constraints one at a time, and uses BFS pebble searches to determine whether each constraint is independent or redundant. After processing all constraints, bodies that cannot exchange pebbles belong to the same rigid cluster.

**Implementation:** TRAMbio (2025, Python, `pip install trambio`). This is the one external library we accept here — the (6,6)-pebble game is algorithmically complex and TRAMbio handles all biologically relevant constraint types (covalent bonds, H-bonds, salt bridges, hydrophobic contacts, pi-stacking). Reads PDB files directly.

**Fallback:** Simplified (3,3)-pebble game on the Cα contact graph (bar-joint framework). Less rigorous but captures essential rigidity topology in ~100 lines.

**Outputs:**

| Metric | Description |
|--------|-------------|
| `rigidity_largest_cluster_fraction` | Fraction of atoms in the largest rigid cluster |
| `rigidity_n_clusters` | Number of distinct rigid clusters |
| `rigidity_binding_site_in_single_cluster` | Whether binding-site residues belong to one rigid cluster |
| `rigidity_floppy_mode_count` | Internal DOF remaining: F = 6N − 6 − E_independent |
| `rigidity_redundancy` | Number of redundant constraints |

---

## Phase 2: Packing & Energetic Metrics

### 2.1 Voronoi Packing Efficiency

**Level:** All-atom

**Engineering analogy:** Material density / void fraction analysis. Packing fraction determines load-bearing capacity in granular mechanics.

**Algorithm:**
1. Compute Voronoi tessellation over all heavy atoms using `scipy.spatial.Voronoi`
2. For each atom: packing density = V_vdW / V_Voronoi (van der Waals sphere volume / Voronoi cell volume)
3. Classify atoms as buried (SASA = 0 with 1.4 Å probe) or exposed
4. Report mean packing density for buried atoms only

Surface atoms get unbounded Voronoi cells (open to solvent) — detected by the presence of the `-1` vertex index in `scipy.spatial.Voronoi.regions` — and are excluded. For bounded cells, compute volume via `scipy.spatial.ConvexHull`.

**No waters:** We don't model explicit waters. This is fine — we only care about the protein interior (buried atoms), which are water-excluded by definition.

**Implementation:** Self-contained with `scipy.spatial.Voronoi` + `scipy.spatial.ConvexHull`. vdW radii from a standard lookup table (element → radius).

**Reference values:**

| Region | Packing fraction | Notes |
|--------|-----------------|-------|
| Protein cores | 0.56 ± 0.02 | Consistent across soluble proteins |
| Protein-protein interfaces | 0.56 ± 0.02 | Similar to cores |
| Random close-packed spheres | ~0.64 | Higher than proteins |
| FCC crystal | 0.74 | The old (incorrect) claim for proteins |

Protein interiors pack like amorphous, glassy materials near the jamming threshold — not like crystals.

**Outputs:**

| Metric | Description |
|--------|-------------|
| `voronoi_packing_core` | Mean packing density of buried atoms |
| `voronoi_packing_binding_site` | Packing density near the ligand |
| `voronoi_void_volume` | Total void volume in the core |

---

### 2.2 Residue-Level Strain Energy (Ramachandran)

**Level:** Sequence + Backbone

**Engineering analogy:** Checking if each structural member is loaded within its yield envelope.

**Algorithm:**
1. Compute backbone dihedrals (φ, ψ) for each residue using `biotite.structure.dihedral`
2. Look up statistical energy from a Ramachandran distribution (e.g., MolProbity contour data or a kernel density estimate from a high-quality PDB subset)
3. Flag residues in disallowed or generously-allowed regions

**Implementation:** Biotite for dihedral computation. For region classification, use the standard Ramachandran contour boundaries (favored / allowed / generously allowed / disallowed) as a lookup on the (φ, ψ) grid. Can be stored as a precomputed 2D array.

**Outputs (per residue):**

| Metric | Description |
|--------|-------------|
| `rama_energy` | Statistical energy from φ/ψ position |
| `rama_region` | Favored / allowed / generously allowed / disallowed |
| `rama_outlier_fraction` | Fraction of residues in disallowed regions (per structure) |

---

### 2.3 Electrostatic Complementarity

**Level:** All-atom (requires ligand)

**Engineering analogy:** Thermal stress analysis — mismatched electrostatics at an interface cause internal stress, like thermal expansion mismatches.

**Established algorithm:** Yes. The gold standard uses Poisson-Boltzmann solvers (APBS + PDB2PQR). For our pipeline, a coarse-grained Coulombic approach is sufficient for relative ranking and much faster.

**Coarse-grained approach (recommended):**
1. Assign partial charges — protein: AMBER ff14SB lookup by (residue_name, atom_name); ligand: Gasteiger charges via RDKit or MMFF94
2. Define interface: atoms within 6 Å of the binding partner
3. Compute Coulombic potential at each interface point from each molecule, using distance-dependent dielectric ε(r) = 4r (Skolnick-Lazaridis approximation) for crude solvent screening
4. Complementarity score: Pearson correlation between φ_protein and −φ_ligand at interface points. Also per-point: EC(v) = 1 − |φ_L(v) + φ_P(v)| / max(|φ_L(v)|, |φ_P(v)|). Global EC > 0.3 is decent, > 0.6 is excellent.

**Implementation:** Self-contained with numpy + RDKit (for ligand charges). AMBER ff14SB protein charges stored as a dict lookup keyed by (residue_name, atom_name). No external PB solver needed for the coarse version.

**Optional validation:** Compare coarse EC against full APBS results on a few structures to calibrate.

**Outputs:**

| Metric | Description |
|--------|-------------|
| `ec_pearson` | Pearson correlation of φ_protein vs −φ_ligand at interface |
| `ec_mean` | Mean per-point electrostatic complementarity |
| `ec_fraction_complementary` | Fraction of interface points with EC > 0 |

---

## Phase 3: Interaction & Stress Metrics

### 3.1 H-Bond Network Truss Analysis

**Level:** All-atom

**Engineering analogy:** Truss determinacy and redundancy analysis. H-bonds are tension/compression members; the network's topology determines structural robustness.

**H-bond detection:** Use our existing `protenix/contacts/hydrogen_bonds.py` — specifically `detect_hydrogen_bonds()`. This is a Molstar-calibrated implementation with proper VSEPR-based donor/acceptor identification, implicit hydrogen placement, and geometric validation (D–A distance < 3.5 Å, sulfur < 4.1 Å, angle deviations < 45°). It handles protonation ambiguity (histidine) and lone-pair geometry correctly. **Do not use biotite or MDAnalysis for H-bond detection.**

**Graph analysis:** Build a NetworkX graph from the detected H-bonds (nodes = residues, edges = H-bonds). Compute:

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Cyclomatic number (μ)** | E − V + C | Number of independent cycles = redundant constraints. Each cycle means one H-bond can be removed without disconnecting the network. |
| **Edge connectivity (λ)** | Min-cut | Minimum H-bonds to disconnect network. Higher = more robust. |
| **Betweenness centrality** | Per-edge, via shortest paths | Identifies "keystone" H-bonds — structural bottlenecks. |
| **Clustering coefficient** | Average local triangle density | Local redundancy of the H-bond neighborhood. |

**Binding-site subgraph:** Extract residues within 6 Å of the ligand, compute the same metrics. This tells us whether the binding site is well-braced specifically.

**Implementation:** `protenix/contacts/hydrogen_bonds.py` for detection + NetworkX for graph analysis. Both already available as dependencies.

**Outputs:**

| Metric | Description |
|--------|-------------|
| `hbond_cyclomatic_number` | Global H-bond network redundancy |
| `hbond_edge_connectivity` | Min-cut of H-bond network |
| `hbond_mean_betweenness` | Average edge betweenness (lower = more distributed) |
| `hbond_site_cyclomatic` | Redundancy in binding-site subgraph |

---

### 3.2 Clash Analysis

**Level:** All-atom

**Engineering analogy:** Geometric interference fit analysis. van der Waals overlaps are the molecular analog of parts that don't fit.

**Algorithm:**
1. For all heavy-atom pairs (i, j) separated by ≥ 3 covalent bonds: compute overlap δ = (r_vdW_i + r_vdW_j) − d_ij
2. If δ > 0.4 Å: clash (MolProbity definition)
3. Beyond counting, compute strain energy of deformation: E_clash = Σ (1/2) k_LJ δ² (quadratic approximation of the LJ repulsive wall)

**Implementation:** Self-contained with biotite for distance calculations + KDTree for neighbor finding. vdW radii from a standard element lookup table. Bond connectivity from biotite's `BondList` to exclude 1-2 and 1-3 pairs.

**Outputs:**

| Metric | Description |
|--------|-------------|
| `clash_count` | Number of severe clashes (δ > 0.4 Å) |
| `clashscore` | Clashes per 1000 atoms (MolProbity standard) |
| `clash_strain_energy` | Total deformation energy from clashes |

---

### 3.3 Per-Residue Virial Stress

**Level:** All-atom

**Engineering analogy:** FEA stress tensor analysis. Compute the stress state at each "element" (residue) from all forces acting on it. Flag residues exceeding a threshold — like yield-stress checking.

For a static structure, the virial stress tensor at atom i sums `f_ij ⊗ r_ij` over all interacting neighbors j, divided by a characteristic atomic volume. Forces come from Lennard-Jones and Coulombic potentials. The von Mises equivalent stress collapses the 3×3 tensor to a single scalar for easy thresholding.

**Implementation approach — two options:**

**Option A (recommended): Simplified LJ + Coulomb in numpy.** Assign AMBER ff14SB charges and LJ parameters via lookup tables. Compute pairwise forces within 10 Å cutoff using KDTree. Build virial tensor per atom. ~200 lines of numpy, no external MD engine. Omits bonded terms but captures the dominant nonbonded strain (packing clashes + electrostatic frustration).

**Option B: Full force field via OpenMM.** More accurate (includes bonded terms) but adds a heavy dependency. OpenMM provides per-atom forces directly; the virial tensor must be assembled manually from forces × positions.

We start with Option A. If we need bonded-term decomposition later, upgrade to Option B.

**Outputs (per residue):**

| Metric | Description |
|--------|-------------|
| `stress_von_mises` | Von Mises equivalent stress |
| `stress_hydrostatic` | Hydrostatic pressure (+ve = compression) |
| `stress_max_principal` | Largest eigenvalue of σ (max tension/compression) |

---

## Implementation Plan

### Phase 1: Backbone (implement first, visualize on `notebooks/PDBs/`)

1. **ANM B-factors** — scipy + numpy, self-contained. Prototype in notebook on natural proteins, compare to experimental B-factors, visualize ΔB-factors with/without ligand.
2. **Graph metrics** — scipy sparse, reuse `fragment.py` graph construction patterns. Fiedler value, spectral gap, effective resistance. Compute on PDBs/, check distributions.
3. **Rigidity analysis** — TRAMbio. Install and test on PDBs/, rigid cluster decomposition, binding-site rigidity.

### Phase 2: Packing & Energetics

4. **Voronoi packing** — `scipy.spatial.Voronoi` + `ConvexHull`, self-contained
5. **Ramachandran strain** — biotite dihedral computation + precomputed contour lookup
6. **Electrostatic complementarity** — numpy + RDKit (ligand charges), AMBER charge lookup table

### Phase 3: Interaction & Stress

7. **H-bond truss analysis** — `protenix/contacts/hydrogen_bonds.py` for detection + NetworkX for graph metrics
8. **Clash analysis** — biotite + KDTree + vdW radius table
9. **Virial stress** — numpy LJ + Coulomb implementation with AMBER parameters

### Integration with Evaluation Pipeline

Each metric becomes a `BatchComponent` following the pattern in `evaluation/batch_components/`. Constants in `evaluation/utils/constants.py`, Modal functions in `batch_functions/`, deploy via `deploy_all_batched.py`.

---

## Dependencies Summary

| Metric | Implementation | New Dep? |
|--------|---------------|----------|
| ANM B-factors | scipy + numpy | No |
| Graph metrics | scipy.sparse | No |
| Rigidity / pebble game | TRAMbio | **Yes** |
| Voronoi packing | scipy.spatial | No |
| Ramachandran strain | biotite | No |
| Electrostatic comp. | numpy + RDKit | No |
| H-bond truss | protenix/contacts + NetworkX | No (networkx already available) |
| Clash analysis | biotite + scipy.spatial | No |
| Virial stress | numpy + AMBER param tables | No |

Only one new dependency: `trambio` for rigidity analysis. Everything else uses existing libraries.
