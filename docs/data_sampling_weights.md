# Data Sampling Weights

## Indices DataFrame

The training indices CSV (`weightedPDB_indices_*.csv`) describes training examples from PDB structures. Each row is one training example — either a single chain (monomer) or a pairwise interface.

All statistics below are from the training split: `weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_4.csv.gz` (1.71M rows, 164K PDB structures). The full unfiltered indices CSV is larger (~2.4M rows, ~168K structures).

### Columns

| Column | Type | Description |
|---|---|---|
| `pdb_id` | str | PDB structure ID |
| `assembly_id` | int | Biological assembly number (usually 1) |
| `entity_1_id` | int | Entity ID of the first molecule |
| `chain_1_id` | str | Chain ID of the first molecule |
| `mol_1_type` | str | Molecule type: `prot`, `ligand`, `nuc` |
| `cluster_1_id` | str | Cluster ID for molecule 1 (see Clustering) |
| `sub_mol_1_type` | str | Finer classification (e.g., `glycosylation_prot`, `non_bonded_ligand`, `rna`) |
| `entity_2_id` | float | Entity ID of second molecule (NaN for chains) |
| `chain_2_id` | str | Chain ID of second molecule (NaN for chains) |
| `mol_2_type` | str | Molecule type of second molecule (NaN for chains) |
| `cluster_2_id` | str | Cluster ID for molecule 2 (NaN for chains) |
| `sub_mol_2_type` | str | Finer classification of molecule 2 (NaN for chains) |
| `cluster_id` | str | Combined cluster ID used for sampling weights |
| `type` | str | `chain` (monomer) or `interface` (pairwise interaction) |
| `mol_type_group` | str | Sorted pair of mol types (e.g., `ligand_prot`, `prot_prot`, `intra_prot`) |
| `eval_type` | str | Evaluation category (same as `mol_type_group` for proteins; splits `nuc` into `dna`/`rna`) |
| `num_tokens` | int | Total tokens in the bioassembly |
| `num_prot_chains` | int | Number of protein chains in the bioassembly |
| `resolution` | float | Structure resolution in Angstroms |
| `release_date` | str | PDB release date |
| `ligand_1_category` | str | PLINDER ligand category for side 1: `proper`, `artifact`, `unknown`, or null |
| `ligand_2_category` | str | PLINDER ligand category for side 2: `proper`, `artifact`, `unknown`, or null |

The `ligand_*_category` columns are added by `enrich_indices_with_plinder.py` using PLINDER's annotation table. They are null when that side is not a ligand. See [PLINDER Enrichment](#plinder-enrichment) below.

### Row Types

**`type=chain`** (742K rows): Single-molecule entries. Only side-1 columns populated; side-2 are NaN. One row per physical chain in the assembly (e.g., a homodimer creates 2 `intra_prot` rows).

**`type=interface`** (970K rows): Pairwise interaction entries. Both sides populated. One row per interacting chain pair across entities.

A single PDB generates many rows due to combinatorial interface enumeration. Median: 5 rows per PDB. Max: 851.

### Distribution by Interaction Type

| mol_type_group | Rows | % |
|---|---|---|
| `prot_prot` | 427K | 25.0% |
| `ligand_prot` | 412K | 24.1% |
| `intra_prot` | 402K | 23.5% |
| `intra_ligand` | 316K | 18.4% |
| `ligand_ligand` | 64K | 3.7% |
| `nuc_prot` | 45K | 2.6% |
| `intra_nuc` | 24K | 1.4% |
| `nuc_nuc` | 14K | 0.8% |
| `ligand_nuc` | 8K | 0.5% |

## Clustering

Cluster IDs determine sampling weights. The format differs by molecule type:

| Molecule type | cluster_id format | Example |
|---|---|---|
| Protein | `pdb_cluster_{representative}_{n}` (40% seq identity) | `pdb_cluster_1A00_1` |
| Nucleic acid | Full sequence string | `GUGCGAC...` |
| Ligand | CCD code | `NAG`, `IRI` |

For chains: `cluster_id = cluster_1_id`.
For interfaces: `cluster_id = sorted(cluster_1_id, cluster_2_id)` joined by `:`.

There are 36,261 unique protein clusters (from `cluster_1_id`) and 225,306 unique combined cluster IDs total.

## Base Weight Calculation

Implemented in `artios/data/data_weights.py:calc_weights_for_df`.

### Step 1: Identify symmetric copies

A key `pdb_sorted_entity_id = "{pdb_id}_{assembly_id}_{sorted_entity_ids}"` identifies each unique entity-pair within an assembly. If a homodimer has 3 instances of the same entity-pair, `member_num = 3`.

### Step 2: Compute cluster size

For each `cluster_id`, count the number of unique `pdb_sorted_entity_id` values. This is the sequence-similarity cluster size.

### Step 3: Per-row weight

```
weight = beta * alpha_sum / cluster_size
```

- `beta`: 0.5 for chains, 1.0 for interfaces
- `alpha_sum = 3 * n_prot + 3 * n_nuc + 1 * n_ligand`

### Step 4: Divide by symmetric copies

```
final_weight = weight / member_num
```

### Sampling

`WeightedSampler` / `DistributedWeightedSampler` draws `epoch_size` samples with replacement using `torch.multinomial` on the merged weights. Multiple datasets can be combined via `WeightedMultiDataset` with per-dataset weights.

## Problem: Oversampling of Common Protein Families

Interface clusters are defined as (protein cluster, ligand CCD code) pairs. Each distinct ligand creates a new cluster. Extensively studied protein families (kinases, immunoglobulins, HIV protease, carbonic anhydrase, etc.) bind many different ligands, creating many interface clusters — each at near-maximum weight (4.0).

### Scale of the problem

- 11,627 unique protein clusters participate in ligand-protein interfaces
- Top protein cluster (`pdb_cluster_1AGW_1`) has **803** distinct interface clusters
- p50 interfaces per protein cluster: 2, p90: 8, p99: 60, max: 803
- 71% of ligand-protein interface clusters are size-1 singletons

### Weight concentration

Computing `W_k` (total interface weight per protein cluster, over `ligand_prot` + `nuc_prot` groups):

- 13,616 protein clusters participate in balanced interface groups
- Mean W_k: 24.6, median: 8.0, max: 3,212
- The top 10% of protein clusters (by W_k) account for **62%** of all balanced interface weight
- The top cluster has W_k = 3,212 — a single protein family gets 76x the weight of the p90 cluster

## Protein-Cluster-Balanced Sampling

Implemented in `artios/data/data_weights.py:apply_protein_cluster_balancing`. Caps the total interface weight per protein cluster so that overrepresented families don't dominate. All other rows (monomers, prot-prot interfaces, etc.) keep their original weights.

### Algorithm

1. Compute base interface weights as before: `w_i = alpha_t / |P(i)|`
2. For each protein cluster k, sum the balanced-interface weights: `W_k = sum(w_i for i in I(k))`
3. Compute cap `tau = percentile_90({W_k})` (= 18.0 on training split with production config)
4. For protein clusters where `W_k > tau`, scale all their interface weights by `tau / W_k`

Only applies to interfaces with exactly one protein side (`ligand_prot`, `nuc_prot` by default). `prot_prot` interfaces keep their original weights. Monomer weights are not modified — the base weight formula already equalizes per-cluster monomer weight (every protein cluster contributes exactly 0.5 total monomer weight).

### Impact on training split

With production config (alpha all 1.0):

- 1,362 / 13,616 protein clusters (10%) are capped
- Total balanced interface weight drops 45% (151K -> 83K)
- The top cluster (`pdb_cluster_1AGW_1`, W_k=1606) is scaled by 0.011x
- No zero-weight rows, max weight remains 2.0
- `ligand_prot` share: 40% -> 29%, `intra_prot` stays at ~8%, `prot_prot` rises to ~32%

### Reduction factors for top clusters

| Percentile | W_k | Scale factor |
|---|---|---|
| p90 (tau) | 18.0 | 1.0 (no reduction) |
| p95 | 36.0 | 0.50 |
| p99 | 130.0 | 0.14 |
| p99.9 | 513.4 | 0.035 |
| max | 1606.0 | 0.011 |

## PLINDER Enrichment

The [PLINDER dataset](https://plinder.sh) provides curated annotations for protein-ligand interactions in the PDB. We use it to classify ligands in our indices as proper (real biological interactions) vs artifacts (crystallization junk).

### Ligand Categories

| Category | Description | Examples |
|---|---|---|
| `proper` | Real small molecules, cofactors, drugs | HEM, NAD, NAG, kinase inhibitors |
| `artifact` | Crystallization buffer, solvents, precipitants | GOL, SO4, EDO, PEG, DMS, ACT, PO4 |
| `unknown` | CCD code not found in PLINDER | Rare or novel ligands |

### Enrichment Script

`artios/data/engineering/scripts/enrich_indices_with_plinder.py` adds `ligand_1_category` and `ligand_2_category` columns to the indices CSV by matching ligand CCD codes against PLINDER's annotation table.

```bash
python -m artios.data.engineering.scripts.enrich_indices_with_plinder \
    --indices /path/to/indices.csv.gz \
    --plinder /path/to/annotation_table.parquet \
    --output /path/to/enriched_indices.csv.gz
```

### Artifact Filtering

`filter_artifact_ligands()` in `data_weights.py` removes rows where any ligand side is an artifact before computing sampling weights. This is called at the top of `calc_weights_for_df`.

Impact on training split:
- 197K rows (11.5%) removed across all ligand interaction types
- 96K artifact `ligand_prot` interfaces removed (23.4% of all ligand-protein interfaces)
- Artifact rows no longer inflate cluster sizes or symmetric copy counts
- Backward compatible: if the columns are absent, no filtering occurs
