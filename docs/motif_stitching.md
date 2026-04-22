# Motif Representation and Stitching

A guide to how motifs flow through the design pipeline — from the user's CIF input, through ArtiosBackbone, into ArtiosPotts, out to the folding models, and back as rebased structures. The non-obvious parts are (a) the deliberate atom-level duplication in the `AtomArray` that ArtiosBackbone/ArtiosPotts consume, (b) why motif residues live on their own chains, and (c) what "stitching" actually does.

---

## TL;DR

- The user provides a motif CIF with one or more chains: protein sidechain residues and/or ligands.
- ArtiosBackbone's `AtomArray` stores the design chain as a **full-length glycine-backbone placeholder** and adds each motif residue/ligand as a **separate chain** (one chain per motif residue, one chain per ligand).
- This means at positions where a motif sidechain will eventually land, there are **two sets of atoms** — the gly placeholder backbone on the design chain, and the full motif residue (backbone + sidechain) on its own motif chain. This duplication is intentional.
- **Stitching** is a post-hoc RMSD-based placement step that figures out *where* each motif residue belongs on the design chain, then rewrites the atom array to move the motif atoms onto the design chain and drop the placeholder backbone. Stitched outputs are what we hand to folding models (AF2/Chai/Protenix).
- **ArtiosPotts consumes the UN-stitched form** — it treats the gly-placeholder backbone and the motif residue as two separate tokens, designs over the gly slot, and fixes the motif-chain residue's identity via `fixed_residues`. The duplicate backbones work out because both tokens resolve to the same physical location.
- The `ChainTracker` (`evaluation/utils/chain_tracker.py`) records the chain-id transitions at every stage so downstream components can keep their lookups straight.

---

## 1. The motif CIF input

The user supplies a motif CIF via `motif.cif_file` in the input JSON (see `motif_input_format.md`). A CIF can contain any mix of:

- **Protein sidechain residues** — e.g. a His/Cys/Asp catalytic triad, each as an individual residue on its own chain or a shared chain.
- **Ligands** — e.g. a CCD-code molecule like `ZN` or a SMILES-defined small molecule.

Each motif entity in the CIF has a `chain_id` (we call this the **`canonical_chain_id`**) and a `res_id` (**`canonical_res_id`**). These are the ground-truth identifiers — they never change, and they're what the chain tracker keys off of.

---

## 2. What ArtiosBackbone builds

`backbone_input_to_atom_array()` (`evaluation/functions/artios_backbone_function.py`) constructs an `AtomArray` with two conceptually distinct parts:

### 2a. The design chain (full-length gly placeholder)

A glycine-only protein of length `design_length`, with exactly `{N, CA, C, O}` per residue (no `OXT`, no sidechains). Chain ID = `design_chain.chain_id` (conventionally `"0"` via `DESIGN_CHAIN_ID`).

```
chain "0":  res 1(GLY)  res 2(GLY)  ...  res 50(GLY)  ...  res N(GLY)
            [N CA C O]  [N CA C O]  ...  [N CA C O]   ...  [N CA C O]
```

This is a placeholder — ArtiosBackbone will predict coordinates for every residue here, and downstream ArtiosPotts will predict identities for every residue here (except at positions that end up "occupied" by a motif residue).

### 2b. Motif residues/ligands — each on its own chain

Every motif entity from the CIF becomes a separate chain in the atom array. This is the first non-obvious choice: protein motif residues are **not** merged into the design chain; they sit on their own chains.

```
chain "B":  res 42(HIS)  [N CA C O CB CG ND1 CE1 NE2 ...]   ← motif residue 1
chain "C":  res 85(ASP)  [N CA C O CB CG OD1 OD2]           ← motif residue 2
chain "D":  res 120(CYS) [N CA C O CB SG]                   ← motif residue 3
chain "E":  res 1(UNL)   [C1 O1 O C2 C3 ... ]               ← ligand
```

Each motif chain has its own **`unindexed_chain_id`** (a new sequential ID assigned during unindexing to avoid collisions). The chain tracker records the mapping `canonical_chain_id → unindexed_chain_id`.

**Why separate chains?** A motif residue on a separate chain is trivially addressable as a token (one chain, one residue). If we merged it into the design chain, we'd need to commit to a specific `res_id` on the design chain at input-processing time — but we don't *know* which design-chain residue each motif should occupy until after ArtiosBackbone has predicted backbone coordinates. Separate chains defer the placement decision.

### 2c. The resulting duplication

Because the design chain has a full-length gly placeholder and the motif chains hold the real motif residues separately, at the eventual stitch positions there are **two sets of atoms**:

- Design chain `"0"` at `res_id=50`: gly placeholder backbone (`N, CA, C, O`).
- Motif chain `"B"` at `res_id=42` (its unindexed_res_id): the real HIS residue (backbone + sidechain).

Both will end up representing the same physical location after stitching — the design's res 50 *is* the motif HIS. But at the ArtiosBackbone/ArtiosPotts input stage, they're represented twice.

---

## 3. Stitching: finding the right placement

After ArtiosBackbone produces coordinates, we need to figure out *which* design-chain residues the motif residues should land on. This is the job of `MotifStitchingComponent` (`evaluation/functions/motif_stitching_function.py`).

**Algorithm sketch** (see `compute_motif_stitching_for_sample`):

1. For each motif protein chain, extract its backbone atoms (`N, CA, C, O`).
2. For each design-chain residue's backbone, compute backbone RMSD to the motif residue's backbone.
3. Pick the design-chain residue with the lowest RMSD as the **`stitched_res_id`** for that motif.
4. Greedy: once a design-chain residue is assigned, remove it from the pool so two motif residues don't stitch to the same slot.

The `stitched_res_id` is recorded per motif island on the chain tracker. Because each sample has slightly different predicted coordinates, `stitched_res_id` is **per-uid** — uid 1001's HIS might stitch to res 50 while uid 1002's stitches to res 67.

`perform_stitching_for_sample` then produces the "stitched" atom array:
- Drop the gly-placeholder backbone at the stitched positions on the design chain.
- Move the motif-chain atoms into the design chain, relabelling their `chain_id` and `res_id` to the stitched values.
- Ligand chains are NOT merged — they stay on their own chains with identity preserved.

Stitched outputs are what we write to CIF/PDB for the folding models. Folders (AF2/Chai/Protenix) expect a clean, single-chain protein + ligand chains — they wouldn't know what to do with duplicate backbones.

---

## 4. Why ArtiosPotts consumes the UN-stitched form

This is the crux of the design.

ArtiosPotts reads the **raw ArtiosBackbone NPZ** (unstitched), not the stitched CIF. This is intentional:

- **Design chain gly-placeholder backbones** become tokens at every position, including stitch positions. ArtiosPotts samples an identity for each of these tokens.
- **Motif-chain residues** also become tokens, with their identity **fixed** via `fixed_residues` (the motif identity must be preserved — it's the whole point of the motif).
- At stitch position 50: the design-chain gly token is NOT fixed (Potts designs the gly position), and the motif-chain HIS token IS fixed.

So Potts sees two tokens at the same physical location, one it designs freely and one pinned to HIS. Why is this OK?

The MPNN featurizer treats them as independent tokens in the neighbour graph, so their h/J contributions add up for their respective neighbourhoods. Because the backbone coords at the design's res 50 and the motif chain's res 42 are (approximately) identical — ArtiosBackbone placed them there — the two tokens' atomic representations are in the same place. Effectively you get the motif residue "voting twice" at that position, but since its identity is fixed it doesn't affect sequence sampling; it just provides context.

The alternative would be to stitch before Potts — which would require picking a canonical representation and dropping the other. Since the choice would be arbitrary (which atoms to keep? which chain_id to adopt?) and the MPNN/Potts code was originally written against the unstitched form, we keep things as they are. It's ugly but it works.

The `fixed_residues` input to Potts is keyed by `stitched_res_id` for the motif residues and is built from the chain tracker by `get_stitched_residues_from_ct()` — this tells Potts "pin the motif identity at this specific design-chain `res_id`" so the gly-placeholder token in the design chain gets the motif identity (not a free Potts-designed identity) whenever the user enables pocket/motif residue fixing.

---

## 5. Chain-ID lifecycle

A motif chain's ID transitions through several stages as the sample moves through the pipeline. The `ChainTracker` (`evaluation/utils/chain_tracker.py`) records each one per motif island:

| Stage | Field | Example (protein motif) | Example (ligand motif) |
|---|---|---|---|
| Original motif CIF | `canonical_chain_id` | `"A"` | `"C"` |
| ArtiosBackbone NPZ (unstitched) | `unindexed_chain_id` | `"B"` (sequential) | `"C"` (unchanged for ligands) |
| Stitched CIF | `stitched_chain_id` | `"0"` (merged into design chain) | `"C"` (unchanged) |
| AF2 output | `af2_chain_id` | `"A"` (design chain only) | n/a (AF2 is protein-only) |
| Chai output | `chai_chain_id` | `"A"` | `"B"` |
| Protenix output | `protenix_chain_id` | `"A"` | `"B"` |

Analogously, `canonical_res_id` → `unindexed_res_id` → `stitched_res_id` records residue-ID transitions. `stitched_res_id` is per-uid (varies by sample); the others are shared per-run.

Key invariants downstream components rely on:
- **Protein motif residues each live on their OWN `unindexed_chain_id`** — one motif residue per chain. This is why `protein_island_lookup` keys by `unindexed_chain_id` in the rebase code.
- **`canonical_res_id` of a ligand never changes** (unless the CIF numbering is unusual). For protein residues, `canonical_res_id` is opaque metadata; what matters is `stitched_res_id`.
- **Chain tracker lookups use `backbone_uid`, not `uid`** (see `CLAUDE.md`): the tracker is only indexed by gen0 uids. Gen1+ redesigns inherit their parent's chain tracker entry.

---

## 6. Implications for the Protenix-rebase component

`ProtenixRebaseComponent` (`evaluation/components/protenix_rebase_component.py`) produces a new AtomArrayStack NPZ where coordinates come from Protenix but topology (atom order, chain/res-id annotations, bonds) is inherited from the parent ArtiosBackbone NPZ. This is where the duplication story becomes load-bearing.

**Per-atom replacement rule** — keyed by the parent atom's chain:

| Parent chain | Lookup key in Protenix |
|---|---|
| `design_chain.chain_id` (design chain) | `(design_chain.protenix_chain_id, parent.res_id, parent.atom_name)` |
| `protein_island.unindexed_chain_id` (motif residue) | `(design_chain.protenix_chain_id, protein_island.stitched_res_id, parent.atom_name)` |
| `ligand_island.unindexed_chain_id` (ligand) | `(ligand_island.protenix_chain_id, parent.res_id, parent.atom_name)` after atom-name mapping |

Critically, at stitch positions **the same Protenix atom gets looked up twice** — once to fill the design-chain backbone slot (via its own `res_id`), and once to fill the motif-chain residue slot (via `stitched_res_id`). Both writes hit the same coordinates, so the two "duplicate" backbones in the rebased NPZ are numerically identical. This is intentional and correct: Potts will still see two tokens at that position, now both with Protenix-self-consistent coords.

If we'd instead built the rebased NPZ from a stitched structure, we'd have had to invent atoms for the design chain at non-stitch positions (which was never a problem since the rebase just copies parent topology). Staying in the unstitched representation keeps the mapping trivial.

---

## 7. Worked example

Design a length-260 protein with a 3-residue catalytic motif (HIS/ASP/CYS) and one UNK ligand.

**Input motif CIF** (user-supplied):
```
chain "A" res 42 HIS  [backbone + sidechain]
chain "A" res 85 ASP  [backbone + sidechain]
chain "A" res 120 CYS [backbone + sidechain]
chain "B" res 1 UNK   [ligand heavy atoms]
```

**ArtiosBackbone input AtomArray** (after `backbone_input_to_atom_array`):
```
chain "0" res 1..260            GLY placeholders  [N CA C O]×260
chain "B" res 42                HIS              [N CA C O CB CG ND1 CE1 NE2]
chain "C" res 85                ASP              [N CA C O CB CG OD1 OD2]
chain "D" res 120               CYS              [N CA C O CB SG]
chain "E" res 1                 UNK              [ligand atoms]
```

Each motif protein residue got its own unindexed chain (B, C, D). The ligand's canonical chain was preserved (B → E after unindexing to avoid collision).

**ArtiosBackbone output NPZ**: same `AtomArray` topology, but every atom now has predicted coordinates. Suppose the backbone-RMSD stitch picks `stitched_res_id = {HIS→50, ASP→147, CYS→183}` for this sample.

**Chain tracker for this uid**:
```
design_chain: chain_id="0"
motif_islands:
  - HIS: canonical=("A", 42),   unindexed=("B", 42),   stitched=("0", 50)
  - ASP: canonical=("A", 85),   unindexed=("C", 85),   stitched=("0", 147)
  - CYS: canonical=("A", 120),  unindexed=("D", 120),  stitched=("0", 183)
  - UNK: canonical=("B", 1),    unindexed=("E", 1)      (ligand — no stitch)
```

**Stitched CIF** (output to AF2/Chai/Protenix):
```
chain "0": designed residues at 1..260, with HIS at 50, ASP at 147, CYS at 183
           (gly backbone replaced with real motif atoms at those positions)
chain "E": UNK ligand (unchanged)
```

**ArtiosPotts input**: uses the ArtiosBackbone NPZ directly (unstitched). Sees:
- 260 design-chain tokens (gly placeholders) — Potts designs identity for all of them.
- 3 motif-chain tokens (HIS/ASP/CYS) — their identity is fixed.
- 1 ligand-chain token.
- `fixed_residues = [{res_id: 50, res_name: HIS}, {res_id: 147, res_name: ASP}, {res_id: 183, res_name: CYS}]` — pins the design-chain positions that "shadow" the motif residues.

**Rebased NPZ** (produced by `ProtenixRebaseComponent` after Protenix folds the gen0 sequence):
- Same topology as the ArtiosBackbone NPZ.
- Design chain `"0"` at res 50 gets Protenix's coords at (Protenix design chain, res 50) — i.e., Protenix's HIS backbone there. Same for res 147, 183.
- Motif chain `"B"` at res 42 gets Protenix's coords at (Protenix design chain, res 50) — the *same* coordinates as the design chain's res 50. Ditto for C/res 85 → Protenix (..., 147), D/res 120 → Protenix (..., 183).
- Ligand chain `"E"` gets Protenix's ligand coords, with atom names remapped via the atom-name mapping so they match our canonical names.

Result: the rebased NPZ's duplicate backbones are coincident (numerically identical coords), and Potts can condition on the Protenix-self-consistent structure when redesigning.

---

## Related files

- `evaluation/functions/artios_backbone_function.py` — `backbone_input_to_atom_array`, the entry point that builds the duplicated AtomArray.
- `evaluation/functions/motif_stitching_function.py` — `compute_motif_stitching_for_sample`, `perform_stitching_for_sample`.
- `evaluation/utils/chain_tracker.py` — `ChainTracker`, `MotifIsland`, `get_stitched_residues_from_ct`.
- `evaluation/utils/protenix_rebase.py` — `rebase_atomarray_on_protenix` (uses the chain tracker to decide per-atom lookups).
- `artios/data/features.py` — `FullContext.without_ligand` (how ligand atoms are zeroed for classifier-free guidance), `FixedResidueContext`.
- `artios/model/modules/mpnn.py` / `artios_potts.py` — how the tokenizer builds tokens from the atom array, including the duplicate backbones.
- `docs/motif_input_format.md` — user-facing JSON/CIF schema.
- `docs/motif_selection_algorithm.md` — training-time motif selection (different concern, but related).
