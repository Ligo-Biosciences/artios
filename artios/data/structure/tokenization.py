"""Atom-to-token assignment (AF3 SI 2.6).

``TokenGrouping`` is the single source of truth for the standard-residue vs
per-atom tokenization decision. ``compute_token_grouping`` builds one from a
cleaned AtomArray plus the mol_type and (optional) motif-related mask; the
vectorized variant is the production path. ``reindex_grouping`` remaps the
token indices after an atom selection so cropped structures stay dense.

Decoupled from ``_AnnotationGroup`` on purpose: the only per-atom information
needed to make the decision is ``mol_type`` and ``is_motif_related``. Taking
bare numpy arrays instead of ``TypeMasks`` / ``MotifAnnotation`` avoids a
circular import with ``structure.py`` and makes the inputs self-documenting.
"""

from __future__ import annotations

from dataclasses import dataclass

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from artios.common.constants import STD_RESIDUES
from artios.utils.typing import Bool, Int, Shaped, typecheck


@typecheck
@dataclass(frozen=True)
class TokenGrouping:
    """How atoms partition into tokens.

    Single source of truth for the standard-vs-per-atom decision.
    Replaces both centre_atom_mask computation and TokenArray pre-computation.
    """

    atom_token_index: Int[np.ndarray, "n_atom"]
    centre_atom_mask: Bool[np.ndarray, "n_atom"]
    n_tokens: int


def _find_centre_atom(
    atom_array: AtomArray, start: int, stop: int, mol_type: str
) -> int:
    """Find the centre atom index for a standard residue.

    CA for protein, C1' for nucleotide. Falls back to first atom.
    """
    names = atom_array.atom_name[start:stop]
    target = "CA" if mol_type == "protein" else "C1'"
    matches = np.where(names == target)[0]
    if len(matches) > 0:
        return start + matches[0]
    return start


def compute_token_grouping(
    atom_array: AtomArray,
    mol_type: Shaped[np.ndarray, "n_atom"],
    is_motif_related: Int[np.ndarray, "n_atom"] | None = None,
) -> TokenGrouping:
    """Compute how atoms partition into tokens.

    Rules (AF3 SI 2.6):
    - Standard residue, not ligand, not motif-related: 1 token per residue.
      Centre atom: CA (protein), C1' (nucleotide).
    - Everything else: 1 token per atom, each atom is its own centre.
    """
    n_atoms = len(atom_array)
    centre_atom_mask = np.zeros(n_atoms, dtype=bool)
    atom_token_index = np.full(n_atoms, -1, dtype=np.int64)
    token_counter = 0

    if is_motif_related is None:
        is_motif_related = np.zeros(n_atoms, dtype=int)

    # Running counter mirrors the old AtomArrayTokenizer.tokenize() pattern.
    # residue_iter yields sub-arrays in array order, so len(res) advances us.
    start = 0
    for res in struc.residue_iter(atom_array):
        stop = start + len(res)

        res_name = atom_array.res_name[start]
        mt = mol_type[start]

        is_standard = (
            res_name in STD_RESIDUES and mt != "ligand" and not is_motif_related[start]
        )

        if is_standard:
            atom_token_index[start:stop] = token_counter
            centre_idx = _find_centre_atom(atom_array, start, stop, mt)
            centre_atom_mask[centre_idx] = True
            token_counter += 1
        else:
            for i in range(start, stop):
                atom_token_index[i] = token_counter
                centre_atom_mask[i] = True
                token_counter += 1

        start = stop

    assert (atom_token_index >= 0).all(), "Some atoms were not assigned to a token"

    return TokenGrouping(
        atom_token_index=atom_token_index,
        centre_atom_mask=centre_atom_mask,
        n_tokens=token_counter,
    )


def compute_token_grouping_vectorized(
    atom_array: AtomArray,
    mol_type: Shaped[np.ndarray, "n_atom"],
    is_motif_related: Int[np.ndarray, "n_atom"] | None = None,
) -> TokenGrouping:
    """Vectorized equivalent of ``compute_token_grouping`` (same output, no Python loops)."""
    n_atoms = len(atom_array)
    if is_motif_related is None:
        is_motif_related = np.zeros(n_atoms, dtype=int)

    # -- Residue-level arrays ------------------------------------------------
    starts = struc.get_residue_starts(atom_array)
    n_res = len(starts)
    sizes = np.diff(np.append(starts, n_atoms))
    residue_index = np.repeat(np.arange(n_res), sizes)  # [n_atoms]

    # Per-residue properties (sampled from first atom of each residue).
    res_names = atom_array.res_name[starts]
    mol_types_by_res = mol_type[starts]

    is_standard_res = (
        np.isin(res_names, list(STD_RESIDUES))
        & (mol_types_by_res != "ligand")
        & ~is_motif_related[starts].astype(bool)
    )

    # -- Token index ----------------------------------------------------------
    tokens_per_res = np.where(is_standard_res, 1, sizes)  # [n_res]
    token_offset = np.zeros(n_res, dtype=np.int64)
    if n_res > 1:
        np.cumsum(tokens_per_res[:-1], out=token_offset[1:])

    is_standard_atom = is_standard_res[residue_index]  # [n_atoms]
    within_res_offset = np.arange(n_atoms) - starts[residue_index]
    within_res_offset[is_standard_atom] = 0

    atom_token_index = token_offset[residue_index] + within_res_offset

    # -- Centre atom mask -----------------------------------------------------
    # Non-standard: every atom is its own centre.
    centre_atom_mask = ~is_standard_atom

    # Standard residues: CA (protein) or C1' (nucleotide), fallback first atom.
    per_atom_mol = mol_types_by_res[residue_index]
    target_match = is_standard_atom & (
        ((per_atom_mol == "protein") & (atom_array.atom_name == "CA"))
        | ((per_atom_mol != "protein") & (atom_array.atom_name == "C1'"))
    )
    matched_atoms = np.where(target_match)[0]
    if len(matched_atoms) > 0:
        _, first_idx = np.unique(residue_index[matched_atoms], return_index=True)
        centre_atom_mask[matched_atoms[first_idx]] = True

    # Fallback: standard residues with no name-matched centre use first atom.
    std_res_indices = np.where(is_standard_res)[0]
    matched_res = (
        np.unique(residue_index[matched_atoms])
        if len(matched_atoms) > 0
        else np.array([], dtype=int)
    )
    fallback_res = np.setdiff1d(std_res_indices, matched_res)
    if len(fallback_res) > 0:
        centre_atom_mask[starts[fallback_res]] = True

    assert (atom_token_index >= 0).all(), "Some atoms were not assigned to a token"

    return TokenGrouping(
        atom_token_index=atom_token_index,
        centre_atom_mask=centre_atom_mask,
        n_tokens=int(tokens_per_res.sum()),
    )


def reindex_grouping(grouping: TokenGrouping, indices: np.ndarray) -> TokenGrouping:
    """Remap token indices to a dense 0..n_new_tokens-1 range after atom selection."""
    old_token_idx = grouping.atom_token_index[indices]
    unique_old, inverse = np.unique(old_token_idx, return_inverse=True)
    return TokenGrouping(
        atom_token_index=inverse.astype(np.int64),
        centre_atom_mask=grouping.centre_atom_mask[indices],
        n_tokens=len(unique_old),
    )
