# Fragment Detection: A Design Document

## What Fragments Are and Why We Need Them

AlphaFold DB structures are full-length protein predictions. Many are multi-domain proteins connected by long, disordered linkers. We can't train on entire 1000+ residue chains — we need compact, well-folded regions with ligand-binding potential. These are **fragments**.

The `FragmentMaker` class (in `protenix/data/fragment.py`) takes a predicted structure and carves it into training-ready fragments. Each fragment gets a `fragment_id` annotation on the AtomArray:
- `-1` = filtered out (disordered, linker, too small, etc.)
- `1, 2, 3, ...` = distinct fragment regions

---

## The Problem with Naive Fragmentation

The original approach was simple: remove low-pLDDT residues, then group whatever contiguous residues remain. This works when disorder conveniently sits between domains.

But it fails on multi-domain proteins connected by **high-pLDDT coily linkers**. AlphaFold can predict a long alpha-helical linker with high confidence (pLDDT > 70), even though it's structurally just a connector between two independent domains. The pLDDT filter lets these through, and two distinct domains merge into one oversized fragment.

This is a real problem in the AFDB dataset. The failure mode looks like:

```
Domain A (200 residues) --- helical linker (30 residues, pLDDT ~80) --- Domain B (250 residues)

Naive result:  1 fragment of 480 residues (too large, high Rg, filtered out entirely)
Desired result: 2 fragments of ~200 and ~250 residues (both kept)
```

We lose both domains instead of keeping two good training examples.

---

## The Solution: Recursive Spectral Bisection

The core insight: linker residues are **topological bottlenecks**. In a spatial graph of the protein, all shortest paths between the two domains must pass through the linker. This is exactly what spectral graph theory detects.

### The Fiedler vector

Given a weighted adjacency matrix W of a graph, the **normalized graph Laplacian** is:

```
L_sym = I - D^{-1/2} W D^{-1/2}
```

where D is the diagonal degree matrix. The eigenvectors of L_sym reveal graph structure:

- The smallest eigenvalue is always 0 (constant eigenvector)
- The **second smallest eigenvalue** (lambda_2, the "Fiedler value") measures algebraic connectivity
- The corresponding **Fiedler vector** gives the optimal 2-way partition

The Fiedler vector assigns each node a real value. Nodes in one cluster get positive values, nodes in the other get negative values, and nodes at the boundary between clusters get values near zero. This is exactly the linker.

### Why recursive?

A single bisection only finds one cut. Multi-domain proteins (3+ domains) need multiple cuts. We recurse: after each bisection, check each partition separately. If a partition's Fiedler value is high (lambda_2 > threshold), it's well-connected internally — it's a single domain, stop. If lambda_2 is low, there's another bottleneck to find.

---

## The Full Pipeline

```
GIVEN: a protein AtomArray from an AlphaFold CIF file

1. filter_low_plddt_residues
   Mark residues with pLDDT < 65 as fragment_id = -1

2. spectral_split
   Detect and remove linker residues between domains

3. assign_fragment_ids
   Group remaining contiguous residues into numbered fragments

4. spatially_cluster_fragments
   Re-merge fragments that are in spatial contact (Union-Find)

4.5. fill_spectral_gaps
     Reclaim boundary residues that create internal gaps after re-merging

5. filter_fragments
   Remove fragments that fail quality criteria (length, pLDDT, Rg)
```

---

## Step 2 in Detail: `spectral_split`

### Graph construction

We build a Gaussian-weighted k-nearest-neighbor graph over CA atoms:

```
FOR each CA atom i (where fragment_id != -1):
    FIND k nearest CA neighbors
    FOR each neighbor j at distance d:
        edge weight w(i,j) = exp(-d^2 / (2 * sigma^2))
```

**Why CA atoms only?** They're one per residue, reducing the graph from thousands of all-atoms to hundreds of nodes. The CA trace captures the backbone topology we care about.

**Why Gaussian weights?** Close contacts (within a domain) get weight ~1.0. Distant contacts (across a linker) get low weight. This makes the spectral decomposition sensitive to the spatial structure, not just graph connectivity.

**Why kNN instead of a distance cutoff?** kNN graphs adapt to local density. In tightly-packed beta sheets, 8A captures many neighbors. In extended helices, 8A might miss important contacts. k=15 ensures every residue has the same number of connections regardless of local packing.

### Recursive bisection

```
FUNCTION split(node_subset):
    IF |node_subset| < min_partition_size:
        RETURN  # too small to split further

    COMPUTE normalized Laplacian of subgraph
    COMPUTE Fiedler value (lambda_2) and Fiedler vector

    IF lambda_2 > connectivity_threshold:
        RETURN  # well-connected, this is a single domain

    # Identify linker residues: those closest to the sign change
    boundary = boundary_size residues with smallest |fiedler_vector|
    MARK boundary residues as linkers (fragment_id = -1)

    # Partition by Fiedler sign
    partition_A = {nodes where fiedler_vector < 0} - boundary
    partition_B = {nodes where fiedler_vector >= 0} - boundary

    split(partition_A)  # recurse on each half
    split(partition_B)
```

### Key design decisions

**The connectivity threshold (lambda_2 > 0.05).** This controls sensitivity. The Fiedler value of the normalized Laplacian ranges from 0 (disconnected graph) to ~2. Values below 0.05 indicate a near-bottleneck that we should cut. Higher values mean the subgraph is well-connected. This threshold was tuned on the failure examples — too low and we miss obvious linkers, too high and we start cutting within compact domains.

**Boundary size (4 residues).** We remove exactly 4 residues at each cut point. This is enough to cleanly sever a linker (typically 2-4 residues at the actual junction) without losing too much of the flanking domains. The removed residues are the ones with the smallest absolute Fiedler vector values — they sit right at the sign change, which is the topological boundary.

**Minimum partition size (50 residues).** We don't attempt to bisect small groups. Below ~50 residues, the spectral decomposition becomes noisy and the resulting fragments would be too small for training anyway.

### Implementation: sparse graph + partial eigendecomposition

The spectral bisection is implemented with two performance tricks that give a ~4x speedup over a naive dense approach:

**Sparse adjacency matrix.** The kNN graph is naturally sparse — each of the n nodes has at most k=15 neighbors, so only ~15n of the n² possible entries are nonzero. We store W as a `scipy.sparse.csr_matrix` built directly from the KDTree output in vectorized numpy (no Python loops):

```python
row = np.repeat(np.arange(n), k)
col = neighbors[:, 1:].ravel()
weights = np.exp(-(distances[:, 1:] ** 2).ravel() / (2 * sigma**2))
W = csr_matrix((weights, (row, col)), shape=(n, n))
W = W.maximum(W.T)  # symmetrize
```

**Partial eigendecomposition.** We only need 2 eigenvalues (the trivial one and the Fiedler value), not all n. Instead of `np.linalg.eigh(L_sym)` which computes all n eigenvalues in O(n³), we use `scipy.sparse.linalg.eigsh` with k=2 on the sparse matrix, which uses Lanczos iteration in roughly O(n·k_neighbors) time.

The mathematical trick: finding the 2 *smallest* eigenvalues of L_sym is numerically hard for iterative solvers, but finding the 2 *largest* eigenvalues of L_norm = D^{-1/2} W D^{-1/2} = I - L_sym is easy (standard Lanczos). The Fiedler value is `1 - λ₂(L_norm)` and the eigenvectors are identical:

```python
L_norm = diags(d_inv_sqrt) @ sub_W @ diags(d_inv_sqrt)
evals, evecs = eigsh(L_norm, k=2, which="LM")
fiedler_value = 1.0 - evals[0]
fiedler_vector = evecs[:, 0]
```

### Gap filling after spatial re-clustering

Spectral bisection removes `boundary_size` residues at each cut point (marks them as fragment_id = -1). When spatial re-clustering later merges two fragments that were on either side of a cut, the removed boundary residues leave a small internal gap.

`fill_spectral_gaps` (step 4.5) patches this by scanning each fragment for internal gaps of at most `2 * spectral_boundary_size` residues where all gap residues have pLDDT above the disorder threshold. These are residues that were only removed because they sat at a spectral sign change, not because they were disordered — so they are reclaimed into the surrounding fragment.

### The safety net: spatial re-clustering

Spectral bisection can sometimes over-split. Two lobes of the same domain might have a thin connection that looks like a bottleneck to the algorithm. Step 4 (`spatially_cluster_fragments`) handles this: if two fragments have at least `min_contact_residues` residues with atoms within `contact_distance_threshold` of each other, they get merged back together via Union-Find.

This split-then-merge design is intentional. It's better to over-split (spectral) and then re-merge (spatial contacts) than to under-split and miss real linkers. The spectral step is aggressive; the spatial clustering step is conservative.

---

## Parameters

All parameters are exposed via the `FragmentMaker` constructor:

### pLDDT filtering

| Parameter | Default | Description |
|-----------|---------|-------------|
| `plddt_disorder_threshold` | 65 | Residues below this pLDDT are marked disordered |

### Spectral bisection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spectral_k` | 15 | Neighbors per node in kNN graph |
| `spectral_sigma` | 8.0 | Gaussian kernel bandwidth (Angstroms) |
| `spectral_connectivity_threshold` | 0.05 | Lambda_2 above which we stop splitting |
| `spectral_boundary_size` | 4 | Residues removed per cut |
| `spectral_min_partition_size` | 50 | Minimum subgraph size to attempt bisection |

### Spatial clustering

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contact_distance_threshold` | 4.0 | Max atom-atom distance for contact (Angstroms) |
| `min_contact_residues` | 4 | Min residues with contacts to merge fragments |

### Fragment quality filters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_fragment_length` | 150 | Minimum residues in a kept fragment |
| `min_fragment_plddt` | 70 | Minimum average pLDDT of a kept fragment |
| Rg upper bound | `0.0298 * length + 14.28` | Length-dependent max radius of gyration |

---

## Data Types

Fragment metrics are returned as `Fragment` dataclass instances:

```python
@dataclass(frozen=True)
class Fragment:
    fragment_id: int
    avg_plddt: float
    length: int
    rg: float
    sasa: float
    boundaries: str   # e.g. "15-200_210-350"
```

The `boundaries` field encodes which residue ID ranges belong to the fragment, with non-contiguous segments separated by underscores.

---

## Entry Points

| Method | Returns | Use case |
|--------|---------|----------|
| `get_fragment_annotation(atom_array)` | `AtomArray` with `fragment_id` | Training data pipeline (`parser.py`) |
| `calculate_fragment_metrics(atom_array)` | `list[Fragment]` | Programmatic access to per-fragment metrics |

---

## File Reference

| File | Role |
|------|------|
| `protenix/data/fragment.py` | `FragmentMaker` class, `Fragment` dataclass |
| `protenix/data/parser.py` | Calls `FragmentMaker.get_fragment_annotation()` during CIF parsing |
| `notebooks/prototype_fragment_detection.py` | Original prototype comparing 6 linker detection approaches |
| `notebooks/visualize_spectral_fragments.py` | Visualization script for parameter sweeps |
