"""Atom-map inference for SMILES ligands with disambiguation.

Given a partial-ligand fragment from a motif CIF and a full-ligand
SMILES, this module enumerates the top-K substructure matches —
ranked by Kabsch RMSD between motif coordinates and a UFF-relaxed
conformer of the full SMILES — and labels each one with `[X:N]`
notation in canonical-SMILES traversal order.

Two public entry points:

- `enumerate_atom_map_candidates(...)` — returns the top-K
  `AtomMapResult`s. Used by the eval-bot tool that presents
  multiple candidates for operator review (the operator picks one).
- `infer_atom_map_with_disambiguation(...)` — convenience wrapper
  returning just the best candidate. Used by callers that want a
  best-guess single answer with no human-in-the-loop.

`infer_atom_map_from_substructure` (in `rdkit.py`) is unrelated
and kept as-is — it's the path used internally by
`array_builder.build_motif_ligand` when an artios JSON omits its
`atom_map` and the build pipeline auto-infers at structure-build
time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import permutations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AdjustQueryParameters, AdjustQueryProperties

from artios.data.chem.rdkit import atoms_to_rdkit_mol, smiles_to_rdkit_mol

_BRACKET_LABEL_RE = re.compile(r"\[[^\]]*:(\d+)\]")

# Tokeniser for atom positions in a SMILES string. Used by
# `_label_match_preserving_input_smiles` to inject `[X:N]` notation
# directly into the operator's input string without round-tripping
# through `Chem.MolToSmiles` (which would canonicalise, aromatise,
# and emit `[cH]` artefacts the operator doesn't want).
_ATOM_TOKEN_RE = re.compile(
    r"\[[^\]]*\]"  # bracketed atom (any contents)
    r"|Cl|Br"  # common two-letter organic elements
    r"|[BCNOPSFI]"  # single-letter uppercase organic atoms
    r"|[bcnops]"  # lowercase aromatic atoms
)
_TRAILING_MAP_RE = re.compile(r":\d+$")


@dataclass(frozen=True)
class AtomMapResult:
    """One candidate atom-mapping for a SMILES ligand.

    Fields:
      matched_indices: heavy-atom indices in the full SMILES mol, one
        per motif atom (in motif order).
      labelled_smiles: the input SMILES with `[Element:N]` notation
        injected on the matched atoms. `N` ascends in canonical-
        SMILES traversal order — `:1` appears first, `:2` next, etc.
      atom_map: `{cif_atom_name: N}` — drop into a SMILES-ligand
        artios JSON entry's `atom_map` field.
      rmsd: Kabsch-aligned RMSD over matched atoms (Å).
      max_atom_displacement: worst single-atom distance after
        alignment (Å). Useful for spotting "one atom badly placed"
        vs. "uniformly off due to torsion mismatch".
      n_candidate_matches: total number of substructure embeddings
        the search returned. >1 means the motif fragment is symmetric
        or repeated in the SMILES; the operator should sanity-check
        that this candidate is the right one.
    """

    matched_indices: tuple[int, ...]
    labelled_smiles: str
    atom_map: dict[str, int]
    rmsd: float
    max_atom_displacement: float
    n_candidate_matches: int


def _kabsch_align(motif: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Optimal rigid alignment of `target` onto `motif`. No scaling.

    Returns `(rmsd, max_per_atom_distance)`. Both inputs must be
    `(n, 3)` and same shape. Plain numpy SVD — kept independent of
    `artios.metrics.rmsd.align_pred_to_true` (which is torch-based)
    so this module is callable outside training contexts.
    """
    motif_c = motif - motif.mean(axis=0)
    target_c = target - target.mean(axis=0)

    H = target_c.T @ motif_c
    U, _, Vt = np.linalg.svd(H)

    # Correct for reflection so we always get a proper rotation.
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    aligned = target_c @ R.T
    per_atom = np.linalg.norm(aligned - motif_c, axis=1)
    rmsd = float(np.sqrt((per_atom**2).mean()))
    max_disp = float(per_atom.max())
    return rmsd, max_disp


def _label_match_preserving_input_smiles(
    full_smiles: str,
    match: tuple[int, ...],
    motif_atom_names: tuple[str, ...],
) -> tuple[str, dict[str, int]]:
    """Inject `[Element:N]` notation directly into the input SMILES
    string, preserving its formatting exactly.

    Round-tripping through `Chem.MolToSmiles` would canonicalise the
    string (aromatising kekulé forms, reordering atoms, emitting
    `[cH]` artefacts that the operator did not write). Instead, we
    tokenise the input character-by-character, find each matched
    atom's character span, and rewrite only those tokens.

    Label assignment: matched atoms are sorted by their position in
    the input string (left-to-right) and labelled 1..n in that order.
    This keeps `:1` appearing before `:2` etc. as you read the
    output, while preserving every non-matched character of the
    input verbatim.

    Returns `(labelled_smiles, atom_map)` keyed by CIF atom name.
    """
    mol = Chem.MolFromSmiles(full_smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {full_smiles!r}")

    token_matches = list(_ATOM_TOKEN_RE.finditer(full_smiles))
    if len(token_matches) != mol.GetNumAtoms():
        raise ValueError(
            f"SMILES tokeniser found {len(token_matches)} atom tokens but "
            f"RDKit parsed {mol.GetNumAtoms()} atoms in {full_smiles!r} — "
            f"the tokeniser does not cover this SMILES dialect (likely an "
            f"exotic element symbol). Add it to _ATOM_TOKEN_RE."
        )

    # Sort matched atoms by their position in the input string.
    by_position = sorted(
        (token_matches[full_idx].start(), full_idx, motif_order)
        for motif_order, full_idx in enumerate(match)
    )
    full_idx_to_label: dict[int, int] = {}
    for new_label, (_, full_idx, _) in enumerate(by_position, start=1):
        full_idx_to_label[full_idx] = new_label

    # Walk the input, replacing each matched atom's token with the
    # bracketed form. Non-matched characters pass through unchanged.
    out_parts: list[str] = []
    cursor = 0
    for _, full_idx, _ in by_position:
        tm = token_matches[full_idx]
        start, end = tm.start(), tm.end()
        token = full_smiles[start:end]
        label = full_idx_to_label[full_idx]
        if token.startswith("["):
            # Already bracketed — strip any prior atom-map suffix and
            # append the new one before the closing `]`.
            inner = _TRAILING_MAP_RE.sub("", token[1:-1])
            new_token = f"[{inner}:{label}]"
        else:
            new_token = f"[{token}:{label}]"
        out_parts.append(full_smiles[cursor:start])
        out_parts.append(new_token)
        cursor = end
    out_parts.append(full_smiles[cursor:])
    labelled_smiles = "".join(out_parts)

    atom_map = {
        motif_atom_names[motif_order]: full_idx_to_label[match[motif_order]]
        for motif_order in range(len(match))
    }
    return labelled_smiles, atom_map


def enumerate_atom_map_candidates(
    motif_elements: tuple[str, ...],
    motif_coords: np.ndarray,
    motif_atom_names: tuple[str, ...],
    full_smiles: str,
    *,
    top_k: int = 6,
    random_seed: int | None = None,
) -> list[AtomMapResult]:
    """Top-K substructure matches sorted by Kabsch RMSD ascending.

    Each result carries an independently-computed labelled SMILES and
    atom_map dict, so any single result is directly drop-in to a
    JSON. Operator review picks one — see
    `eval_bot.tools.atom_map.enumerate_atom_map_candidates`.

    Algorithm:
      1. Build a query mol from motif elements+coords with bond types
         made generic (matches `infer_atom_map_from_substructure`).
      2. `GetSubstructMatches(uniquify=False)` enumerates every
         isomorphic embedding (including ones that swap chemically-
         equivalent atoms — load-bearing for ester orientations).
      3. Generate one 3D conformer of the full SMILES (ETKDGv3 + UFF).
      4. Kabsch-align each candidate's matched-atom subset to the
         motif coords; sort ascending by RMSD; take top-K.
      5. For each, inject `[Element:N]` labels in canonical-SMILES
         traversal order and build the `atom_map` dict.

    Raises:
      ValueError: if the SMILES is unparseable, no substructure match
        exists, or conformer generation fails.
    """
    motif_query_mol = atoms_to_rdkit_mol(motif_elements, motif_coords)
    params = AdjustQueryParameters()
    params.makeBondsGeneric = True
    query_mol = AdjustQueryProperties(motif_query_mol, params)

    full_mol_2d = Chem.MolFromSmiles(full_smiles)
    if full_mol_2d is None:
        raise ValueError(f"Failed to parse SMILES: {full_smiles!r}")

    # uniquify=False is load-bearing: with the default (True), RDKit
    # collapses graph-symmetric matches into one canonical ordering,
    # silently dropping motif-to-target mappings that swap two
    # equivalent atoms (e.g. carbonyl-O and linkage-O of an ester
    # under makeBondsGeneric). Without it, only one ester orientation
    # ever reaches the Kabsch step.
    matches = full_mol_2d.GetSubstructMatches(query_mol, uniquify=False)
    if not matches:
        raise ValueError(
            f"Substructure match failed: motif fragment "
            f"({len(motif_elements)} atoms) not found in full ligand "
            f"SMILES {full_smiles!r}."
        )

    full_mol_3d = smiles_to_rdkit_mol(
        full_smiles, generate_conformers=True, random_seed=random_seed
    )
    if full_mol_3d.GetNumConformers() == 0:
        raise ValueError(
            f"Failed to generate a 3D conformer for SMILES: {full_smiles!r}"
        )
    conformer_coords = full_mol_3d.GetConformer().GetPositions()

    motif_coords_arr = np.asarray(motif_coords, dtype=float)
    scored = []
    for match in matches:
        target = conformer_coords[list(match)]
        rmsd, max_disp = _kabsch_align(motif_coords_arr, target)
        scored.append((match, rmsd, max_disp))
    scored.sort(key=lambda x: x[1])

    results: list[AtomMapResult] = []
    for match, rmsd, max_disp in scored[:top_k]:
        labelled_smiles, atom_map = _label_match_preserving_input_smiles(
            full_smiles, match, motif_atom_names
        )
        results.append(
            AtomMapResult(
                matched_indices=tuple(int(i) for i in match),
                labelled_smiles=labelled_smiles,
                atom_map=atom_map,
                rmsd=rmsd,
                max_atom_displacement=max_disp,
                n_candidate_matches=len(matches),
            )
        )
    return results


def infer_atom_map_with_disambiguation(
    motif_elements: tuple[str, ...],
    motif_coords: np.ndarray,
    motif_atom_names: tuple[str, ...],
    full_smiles: str,
) -> AtomMapResult:
    """Single-best-match wrapper around `enumerate_atom_map_candidates`.

    Returns the lowest-RMSD candidate. For inspection of multiple
    candidates (e.g. operator review), use `enumerate_atom_map_candidates`
    directly.
    """
    return enumerate_atom_map_candidates(
        motif_elements,
        motif_coords,
        motif_atom_names,
        full_smiles,
        top_k=1,
    )[0]


def enumerate_pick_permutation_candidates(
    motif_elements: tuple[str, ...],
    motif_coords: np.ndarray,
    motif_atom_names: tuple[str, ...],
    full_smiles: str,
    pick_indices: tuple[int, ...],
    *,
    top_k: int = 9,
    n_conformers: int = 10,
    random_seed: int | None = None,
) -> list[AtomMapResult]:
    """Rank permutations of operator-picked SMILES atoms by Kabsch RMSD.

    Replaces the substructure-search-based
    `enumerate_atom_map_candidates` for the human-in-the-loop flow
    where the operator has already selected which SMILES atoms ARE
    the motif atoms (just not which-is-which CIF name). The combina-
    torial space is `len(picks)!`, and we score every permutation
    rather than searching subsets — RMSD is meaningful here because
    the search ambiguity (which subset of SMILES atoms is the motif?)
    has been resolved by the operator pick.

    Algorithm:
      1. Validate that `pick_indices` element multiset matches the
         motif's. (Wrong elements = wrong picks.)
      2. Generate `n_conformers` independent ETKDG conformers of the
         full SMILES.
      3. Enumerate all `len(picks)!` permutations. For each
         permutation `perm`, treat `perm[i]` as the SMILES atom
         intended for `motif_atom_names[i]`. For each conformer,
         Kabsch-align the perm's coords to the motif coords and
         record the RMSD; take the min RMSD across conformers as the
         permutation's score (best-case conformer match).
      4. Sort permutations by score ascending, take top-K.
      5. For each top-K permutation, build the labelled SMILES +
         atom_map via `_label_match_preserving_input_smiles`.

    Args:
      motif_elements / motif_coords / motif_atom_names: from the
        motif CIF's ligand atoms, in CIF order.
      full_smiles: the operator-provided SMILES.
      pick_indices: RDKit heavy-atom indices in the full SMILES,
        operator-supplied, in any order. Must be the same length as
        `motif_atom_names`.
      top_k: number of permutation candidates to return.
      n_conformers: how many ETKDG conformers to score against.
      random_seed: if set, seeds the first conformer; subsequent
        conformers are seeded `seed+1`, `seed+2`, … for reproducibility.

    Raises:
      ValueError on input mismatches (wrong pick count, element
      mismatch, unparseable SMILES, conformer generation failure).
    """
    n_motif = len(motif_atom_names)
    if len(motif_elements) != n_motif or motif_coords.shape != (n_motif, 3):
        raise ValueError(
            f"Inconsistent motif inputs: elements={len(motif_elements)}, "
            f"names={n_motif}, coords={motif_coords.shape}"
        )
    if len(pick_indices) != n_motif:
        raise ValueError(
            f"Operator picked {len(pick_indices)} atoms but the motif has "
            f"{n_motif}. Counts must match."
        )

    full_mol_2d = Chem.MolFromSmiles(full_smiles)
    if full_mol_2d is None:
        raise ValueError(f"Failed to parse SMILES: {full_smiles!r}")

    n_atoms = full_mol_2d.GetNumAtoms()
    for idx in pick_indices:
        if idx < 0 or idx >= n_atoms:
            raise ValueError(
                f"Pick atom index {idx} out of range for SMILES with "
                f"{n_atoms} heavy atoms."
            )

    pick_elements = tuple(
        full_mol_2d.GetAtomWithIdx(int(i)).GetSymbol() for i in pick_indices
    )
    if sorted(pick_elements) != sorted(motif_elements):
        raise ValueError(
            f"Element mismatch: operator picks have elements "
            f"{sorted(pick_elements)}, motif has {sorted(motif_elements)}. "
            f"The operator selected atoms of the wrong element(s)."
        )

    conformer_coords_list: list[np.ndarray] = []
    for i in range(n_conformers):
        seed = None if random_seed is None else random_seed + i
        mol = smiles_to_rdkit_mol(
            full_smiles, generate_conformers=True, random_seed=seed
        )
        if mol.GetNumConformers() == 0:
            continue
        conformer_coords_list.append(mol.GetConformer().GetPositions())
    if not conformer_coords_list:
        raise ValueError(
            f"All {n_conformers} ETKDG attempts failed for SMILES: {full_smiles!r}"
        )

    motif_coords_arr = np.asarray(motif_coords, dtype=float)
    pick_indices_list = list(int(i) for i in pick_indices)

    scored: list[tuple[tuple[int, ...], float, float]] = []
    for perm in permutations(pick_indices_list):
        # Element check per-permutation: motif_atom_names[i] must have
        # the same element as the SMILES atom at perm[i]. Skip
        # permutations that violate (the multiset matches globally,
        # but a particular ordering may still pair O→N etc.).
        valid = True
        for i in range(n_motif):
            if full_mol_2d.GetAtomWithIdx(perm[i]).GetSymbol() != motif_elements[i]:
                valid = False
                break
        if not valid:
            continue

        per_conf: list[tuple[float, float]] = []
        for conf_coords in conformer_coords_list:
            target = conf_coords[list(perm)]
            per_conf.append(_kabsch_align(motif_coords_arr, target))
        rmsd, max_disp = min(per_conf, key=lambda t: t[0])
        scored.append((perm, rmsd, max_disp))

    if not scored:
        raise ValueError(
            "No permutation paired the operator picks to the motif "
            "atom names with element matches — this shouldn't happen "
            "after the multiset check above."
        )

    scored.sort(key=lambda x: x[1])
    n_total = len(scored)
    results: list[AtomMapResult] = []
    for perm, rmsd, max_disp in scored[:top_k]:
        labelled_smiles, atom_map = _label_match_preserving_input_smiles(
            full_smiles, perm, motif_atom_names
        )
        results.append(
            AtomMapResult(
                matched_indices=perm,
                labelled_smiles=labelled_smiles,
                atom_map=atom_map,
                rmsd=rmsd,
                max_atom_displacement=max_disp,
                n_candidate_matches=n_total,
            )
        )
    return results
