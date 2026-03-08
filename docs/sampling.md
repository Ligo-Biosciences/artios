# Training Data Sampling

## Indices DataFrame

The training indices CSV (`weightedPDB_indices_*.csv`) contains ~2.4M rows across ~168K PDB structures. Each row is one training example — either a single chain (monomer) or a pairwise interface.

### Columns

| Column | Type | Description |
|---|---|---|
| `pdb_id` | str | PDB structure ID |
| `assembly_id` | int | Biological assembly number (usually 1) |
| `entity_1_id` | int | Entity ID of the first molecule |
| `chain_1_id` | str | Chain ID of the first molecule |
| `mol_1_type` | str | Molecule type: `prot`, `ligand`, `ions`, `nuc` |
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
| `ligand_1_category` | str | PLINDER ligand category for side 1: `proper`, `artifact`, `ion`, `unknown`, or null |
| `ligand_2_category` | str | PLINDER ligand category for side 2: `proper`, `artifact`, `ion`, `unknown`, or null |

The `ligand_*_category` columns are added by `enrich_indices_with_plinder.py` using PLINDER's annotation table. They are null when that side is not a ligand. See [PLINDER Enrichment](#plinder-enrichment) below.

### Row Types

**`type=chain`** (1.02M rows): Single-molecule entries. Only side-1 columns populated; side-2 are NaN. One row per physical chain in the assembly (e.g., a homodimer creates 2 `intra_prot` rows).

**`type=interface`** (1.41M rows): Pairwise interaction entries. Both sides populated. One row per interacting chain pair across entities.

A single PDB generates many rows due to combinatorial interface enumeration. Median: 7 rows per PDB. Max: 859.

### Distribution by Interaction Type

| mol_type_group | Rows | % |
|---|---|---|
| `prot_prot` | 481K | 19.8% |
| `intra_prot` | 433K | 17.8% |
| `ligand_prot` | 428K | 17.6% |
| `intra_ligand` | 328K | 13.5% |
| `ions_prot` | 253K | 10.4% |
| `intra_ions` | 200K | 8.2% |
| `ions_ligand` | 120K | 4.9% |
| `ligand_ligand` | 66K | 2.7% |
| other (nuc combinations, ions_ions) | ~123K | 5.1% |

## Clustering

Cluster IDs determine sampling weights. The format differs by molecule type:

| Molecule type | cluster_id format | Example |
|---|---|---|
| Protein | `pdb_cluster_{representative}_{n}` (40% seq identity) | `pdb_cluster_1A00_1` |
| Nucleic acid | Full sequence string | `GUGCGAC...` |
| Ligand | CCD code | `NAG`, `IRI` |
| Ion | CCD code | `MG`, `ZN` |

For chains: `cluster_id = cluster_1_id`.
For interfaces: `cluster_id = sorted(cluster_1_id, cluster_2_id)` joined by `:`.

There are 24,789 unique protein clusters (from `cluster_1_id`) and 269,503 unique combined cluster IDs total.

## Current Sampling Weight Calculation

Implemented in `artios/data/dataset.py:calc_weights_for_df`.

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

### Scale of the problem (full dataset)

- 14,875 unique protein clusters participate in ligand-protein interfaces
- Top protein cluster (`pdb_cluster_1A0J_1`) has **790** distinct interface clusters
- p50 interfaces per protein cluster: 2, p90: 8, p99: 56, max: 823
- 75% of ligand-protein interface clusters are size 1 (singletons)

### Weight concentration

Computing `W_k` (total interface weight per protein cluster):

- Mean W_k: 45.8, median: 16.0, max: 3,819
- The top 10% of protein clusters (by W_k) account for **60%** of all ligand-protein interface weight
- The top cluster has W_k = 3,819 — a single protein family gets 40x the weight budget it would under uniform-by-protein sampling

## Proposed Fix: Protein-Cluster-Balanced Sampling

Based on the approach from the literature. The idea: cap the total interface weight per protein cluster so that overrepresented families don't dominate, then redistribute the freed budget to monomer entries.

### Algorithm

1. Compute base interface weights as before: `w_i = alpha_t / |P(i)|`
2. Sum interface weights per protein cluster: `W_k = sum(w_i for i in I(k))`
3. Compute cap `tau = percentile_90({W_k})` (= 95.2 on our data)
4. For protein clusters where `W_k > tau`, scale all their interface weights by `tau / W_k`
5. Set monomer weight for each entry j in protein cluster k: `w_j = (tau - W_tilde_k) / |M_k|`

This ensures every protein cluster gets total weight ~= tau (interfaces + monomers), achieving approximately uniform sampling across protein families.

### Expected impact on our data

- 1,487 / 14,875 protein clusters (10%) would be capped
- Total ligand-protein interface weight drops 39% (681K -> 415K)
- The top cluster's interfaces are scaled by 0.025x (from W_k=3,819 to tau=95.2)
- Freed weight redistributed to monomer entries of those same proteins

## PLINDER Enrichment

The [PLINDER dataset](https://plinder.sh) provides curated annotations for protein-ligand interactions in the PDB. We use it to classify ligands in our indices as proper (real biological interactions) vs artifacts (crystallization junk).

### Ligand Categories

| Category | Description | Examples |
|---|---|---|
| `proper` | Real small molecules, cofactors, drugs | HEM, NAD, NAG, kinase inhibitors |
| `artifact` | Crystallization buffer, solvents, precipitants | GOL, SO4, EDO, PEG, DMS, ACT, PO4 |
| `ion` | Metal ions (handled separately from ligands) | MG, ZN, CA, NA |
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

`filter_artifact_ligands()` in `dataset.py` removes rows where any ligand side is an artifact before computing sampling weights. This is called at the top of `calc_weights_for_df`.

Impact on the full dataset:
- ~207K rows (8.5%) removed across all ligand interaction types
- 97K artifact `ligand_prot` interfaces removed (22.7% of all ligand-protein interfaces)
- Artifact rows no longer inflate cluster sizes or symmetric copy counts
- Backward compatible: if the columns are absent, no filtering occurs
