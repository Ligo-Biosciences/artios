"""Cropping strategies for training data augmentation.

All functions operate on numpy arrays — no torch.
AnnotatedStructure is the input for index selection.
Structure cropping (``structure.select_tokens()``) is the caller's job.
Feature cropping (MSA/template) is a separate concern.
"""

from __future__ import annotations

import random

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from artios.data.structure import AnnotatedStructure

# TODO: switch to jaxtyping for numpy arrays here.
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.floating]

METHODS = [
    "ContiguousCropping",
    "SpatialCropping",
    "SpatialInterfaceCropping",
    "PocketCropping",
]


# ── Top-level orchestrator ──────────────────────────────────────────────


def crop(
    structure: AnnotatedStructure,
    crop_size: int,
    ref_chain_ids: list[int],
    method_weights: list[float],
    *,
    contiguous_complete_molecules: bool = False,
    spatial_complete_molecules: bool = False,
    drop_last: bool = False,
    remove_metal: bool = False,
) -> tuple[str, np.ndarray]:
    """Pick a random crop method and return selected token indices.

    Args:
        structure: The full (uncropped) annotated structure.
        crop_size: Maximum number of tokens to keep.
        ref_chain_ids: ``asym_id_int`` values for the reference chain(s).
        method_weights: Weights for [contiguous, spatial, spatial_interface, pocket].
        contiguous_complete_molecules: Preserve complete molecules in contiguous crop.
        spatial_complete_molecules: Drop incomplete molecules in spatial crops.
        drop_last: In contiguous crop, drop last incomplete molecule to stay in budget.
        remove_metal: In contiguous crop, skip single-atom metal/ion chains.

    Returns:
        (method_name, sorted token indices).
    """
    # Zero out PocketCropping weight if the structure has no pockets.
    has_pockets = (
        structure.training.pocket_id is not None
        and (structure.training.pocket_id > 0).any()
    )
    if not has_pockets:
        method_weights = list(method_weights)
        method_weights[3] = 0.0

    method = random.choices(METHODS, weights=method_weights, k=1)[0]

    centre = structure.centre_atom_indices
    coords = structure.token_centre_coords
    is_resolved = structure.training.is_resolved[centre]
    chain_ids = structure.entity_ids.asym_id_int[centre]
    uid = structure.reference.ref_space_uid[centre]
    n_tok = structure.grouping.n_tokens

    if method == "ContiguousCropping":
        atom_counts = _compute_token_molecule_atom_counts(structure)
        indices = contiguous_crop_indices(
            n_tokens=n_tok,
            chain_ids=chain_ids,
            ref_space_uid=uid,
            token_atom_counts=atom_counts,
            crop_size=crop_size,
            complete_molecules=contiguous_complete_molecules,
            drop_last=drop_last,
            remove_metal=remove_metal,
        )
    elif method == "SpatialCropping":
        indices = spatial_crop_indices(
            coords=coords,
            is_resolved=is_resolved,
            chain_ids=chain_ids,
            ref_space_uid=uid,
            ref_chain_ids=ref_chain_ids,
            crop_size=crop_size,
            complete_molecules=spatial_complete_molecules,
        )
    elif method == "SpatialInterfaceCropping":
        indices = spatial_interface_crop_indices(
            coords=coords,
            is_resolved=is_resolved,
            chain_ids=chain_ids,
            ref_space_uid=uid,
            ref_chain_ids=ref_chain_ids,
            crop_size=crop_size,
            complete_molecules=spatial_complete_molecules,
        )
    elif method == "PocketCropping":
        indices = pocket_crop_indices(
            structure=structure,
            crop_size=crop_size,
            complete_molecules=spatial_complete_molecules,
        )
    else:
        raise ValueError(f"Unknown crop method: {method}")

    return method, indices


# ── Spatial crop variants ───────────────────────────────────────────────


def spatial_crop_indices(
    coords: FloatArray,
    is_resolved: BoolArray,
    chain_ids: IntArray,
    ref_space_uid: IntArray,
    ref_chain_ids: list[int],
    crop_size: int,
    complete_molecules: bool = False,
) -> np.ndarray:
    """Spatial crop: pick a random resolved ref-chain token, take k nearest."""
    ref_token_indices = np.where(np.isin(chain_ids, ref_chain_ids) & is_resolved)[0]
    return _spatial_crop_from_refs(
        coords,
        is_resolved,
        ref_space_uid,
        ref_token_indices,
        crop_size,
        len(chain_ids),
        complete_molecules,
    )


def spatial_interface_crop_indices(
    coords: FloatArray,
    is_resolved: BoolArray,
    chain_ids: IntArray,
    ref_space_uid: IntArray,
    ref_chain_ids: list[int],
    crop_size: int,
    complete_molecules: bool = False,
    interface_threshold: float = 15.0,
) -> np.ndarray:
    """Spatial interface crop: reference tokens are cross-chain contacts.

    Falls back to all resolved ref-chain tokens if no interface found.
    """
    interface = _find_interface_tokens(
        coords,
        is_resolved,
        chain_ids,
        ref_chain_ids,
        interface_threshold,
    )
    if len(interface) == 0 and len(ref_chain_ids) == 1:
        # Single chain with no interface — use all resolved ref tokens.
        interface = np.where(np.isin(chain_ids, ref_chain_ids) & is_resolved)[0]
    ref_token_indices = interface
    return _spatial_crop_from_refs(
        coords,
        is_resolved,
        ref_space_uid,
        ref_token_indices,
        crop_size,
        len(chain_ids),
        complete_molecules,
    )


def pocket_crop_indices(
    structure: AnnotatedStructure,
    crop_size: int,
    complete_molecules: bool = False,
) -> np.ndarray:
    """Pocket crop: pick a random pocket, use its tokens as reference.

    Raises:
        ValueError: If no pockets are annotated.
    """
    if structure.training.pocket_id is None:
        raise ValueError("No pocket_id annotation on structure")
    pocket_mask = structure.training.pocket_id > 0
    pocket_ids = np.unique(structure.training.pocket_id[pocket_mask])
    if len(pocket_ids) == 0:
        raise ValueError("No pockets annotated (all pocket_id <= 0)")
    chosen = np.random.choice(pocket_ids)
    ref_token_indices = _get_pocket_token_indices(structure, int(chosen))

    centre = structure.centre_atom_indices
    return _spatial_crop_from_refs(
        structure.token_centre_coords,
        structure.training.is_resolved[centre],
        structure.reference.ref_space_uid[centre],
        ref_token_indices,
        crop_size,
        structure.grouping.n_tokens,
        complete_molecules,
    )


# ── Contiguous crop ─────────────────────────────────────────────────────


def contiguous_crop_indices(
    n_tokens: int,
    chain_ids: IntArray,
    ref_space_uid: IntArray,
    token_atom_counts: IntArray,
    crop_size: int,
    complete_molecules: bool = False,
    drop_last: bool = False,
    remove_metal: bool = False,
) -> np.ndarray:
    """AF-Multimer-style contiguous crop across shuffled chains.

    Iterates through randomly shuffled chains, selects a contiguous range
    of tokens from each, adjusting boundaries to preserve molecule integrity
    when *complete_molecules* is True.
    """
    unique_chains = np.unique(chain_ids)
    chain_lengths = np.bincount(
        chain_ids.astype(np.intp), minlength=int(unique_chains.max()) + 1
    )

    # Per-chain offsets (first token index for each chain).
    chain_offsets = np.array([np.where(chain_ids == c)[0][0] for c in unique_chains])

    # Molecule boundary info.
    mol_first, mol_last = _molecule_spans(ref_space_uid)
    is_metal = _detect_metal_tokens(chain_ids, ref_space_uid, token_atom_counts)

    # Shuffle chains.
    shuffle_order = np.random.permutation(len(unique_chains))

    selected: list[np.ndarray] = []
    n_added = 0
    n_remaining = n_tokens
    if remove_metal:
        n_remaining -= int(is_metal.sum())

    for idx in shuffle_order:
        if n_added >= crop_size:
            break

        chain = unique_chains[idx]
        chain_len = int(chain_lengths[chain])

        # Skip metals.
        if remove_metal and is_metal[chain_offsets[idx]]:
            continue

        n_remaining -= chain_len

        crop_min = min(chain_len, max(0, crop_size - (n_added + n_remaining)))
        crop_max = min(crop_size - n_added, chain_len)
        if crop_min > crop_max:
            crop_min = crop_max

        chain_crop_size = np.random.randint(crop_min, crop_max + 1)
        chain_crop_start = np.random.randint(0, chain_len - chain_crop_size + 1)

        offset = chain_offsets[idx]
        start = offset + chain_crop_start
        end = offset + chain_crop_start + chain_crop_size

        if complete_molecules:
            start, end = _adjust_boundaries_for_molecules(
                start,
                end,
                mol_first,
                mol_last,
                crop_min,
                n_added,
                crop_size,
                drop_last,
            )
            chain_crop_size = end - start

        if start < end:
            selected.append(np.arange(start, end))
        n_added += chain_crop_size

        if complete_molecules and drop_last and start < end:
            assert mol_first[start] == start
            assert mol_last[end - 1] == end - 1

    if not selected:
        return np.array([], dtype=np.int64)

    result = np.sort(np.concatenate(selected))
    if drop_last:
        assert (
            len(result) <= crop_size
        ), f"Contiguous crop selected {len(result)} > {crop_size} tokens"
    return result


# ── Internal: spatial core ──────────────────────────────────────────────


def _spatial_crop_from_refs(
    coords: FloatArray,
    is_resolved: BoolArray,
    ref_space_uid: IntArray,
    ref_token_indices: np.ndarray,
    crop_size: int,
    n_tokens: int,
    complete_molecules: bool,
) -> np.ndarray:
    """Shared core for all spatial crop variants.

    Picks a random reference token, selects k nearest by distance.
    """
    assert len(ref_token_indices) > 0, "No resolved reference tokens"

    dist_matrix, dist_mask = _build_ref_distance_matrix(
        coords,
        is_resolved,
        ref_token_indices,
    )
    random_idx = np.random.randint(0, len(ref_token_indices))

    # Verify the chosen reference token has at least one valid distance.
    assert dist_mask[random_idx].any(), "Selected an unresolved reference token"

    k = min(crop_size, n_tokens)
    indices = _select_k_nearest(random_idx, dist_matrix, dist_mask, k)

    if complete_molecules:
        indices = _drop_incomplete_molecules(indices, ref_space_uid)

    return np.sort(indices)


def _build_ref_distance_matrix(
    coords: FloatArray,
    is_resolved: BoolArray,
    ref_token_indices: np.ndarray,
) -> tuple[FloatArray, BoolArray]:
    """Distance matrix from reference tokens to all tokens.

    Returns:
        distances: [n_ref, n_tokens] euclidean distances.
        mask: [n_ref, n_tokens] True where both ref and target are resolved.
    """
    ref_coords = coords[ref_token_indices]
    distances = cdist(ref_coords, coords, "euclidean")
    ref_resolved = is_resolved[ref_token_indices]
    mask = ref_resolved[:, None] & is_resolved[None, :]
    return distances, mask


def _select_k_nearest(
    ref_idx: int,
    dist_matrix: FloatArray,
    dist_mask: BoolArray,
    k: int,
) -> np.ndarray:
    """Select k nearest tokens to reference, with tie-breaking noise."""
    distances = dist_matrix[ref_idx].copy()
    mask = dist_mask[ref_idx]

    # Unresolved pairs → infinite distance.
    distances[~mask] = np.inf

    # Small index-based noise to break ties deterministically.
    noise = np.arange(len(distances), dtype=np.float64) * 1e-4
    distances += noise

    # k smallest distances (argsort for deterministic tie-breaking).
    indices = np.argsort(distances)[:k]
    return np.sort(indices)


def _find_interface_tokens(
    coords: FloatArray,
    is_resolved: BoolArray,
    chain_ids: IntArray,
    ref_chain_ids: list[int],
    threshold: float,
) -> np.ndarray:
    """Find ref-chain tokens with cross-chain neighbors within threshold.

    Returns global token indices of interface tokens.
    """
    in_ref = np.isin(chain_ids, ref_chain_ids)
    ref_indices = np.where(in_ref)[0]
    if len(ref_indices) == 0:
        return np.array([], dtype=np.int64)

    # Build distance matrix: ref tokens → all tokens.
    dist, mask = _build_ref_distance_matrix(coords, is_resolved, ref_indices)

    # For each ref token, check if any token in a DIFFERENT chain is close.
    diff_chain = chain_ids[None, :] != chain_ids[ref_indices, None]  # [n_ref, n_all]
    close = dist < threshold
    contact = close & diff_chain & mask  # [n_ref, n_all]

    has_contact = contact.any(axis=1)  # [n_ref]
    return ref_indices[has_contact]


def _drop_incomplete_molecules(
    selected: np.ndarray,
    ref_space_uid: IntArray,
) -> np.ndarray:
    """Remove tokens from molecules that are only partially selected."""
    all_indices = np.arange(len(ref_space_uid))

    selected_uids = set(ref_space_uid[selected].tolist())
    unselected_mask = ~np.isin(all_indices, selected)
    unselected_uids = set(ref_space_uid[unselected_mask].tolist())

    # UIDs that appear in both selected and unselected = incomplete.
    incomplete = selected_uids & unselected_uids

    if not incomplete:
        return selected

    keep_mask = ~np.isin(ref_space_uid[selected], list(incomplete))
    return selected[keep_mask]


def _get_pocket_token_indices(
    structure: AnnotatedStructure,
    pocket_id_value: int,
) -> np.ndarray:
    """Get token indices belonging to a specific pocket.

    Maps from atom-level pocket_id to token-level via ref_space_uid.
    """
    pocket_atoms = structure.training.pocket_id == pocket_id_value
    pocket_uids = np.unique(structure.reference.ref_space_uid[pocket_atoms])
    token_uid = structure.reference.ref_space_uid[structure.centre_atom_indices]
    return np.where(np.isin(token_uid, pocket_uids))[0]


# ── Internal: contiguous crop ───────────────────────────────────────────


def _molecule_spans(ref_space_uid: IntArray) -> tuple[IntArray, IntArray]:
    """For each token, find the first and last token index of its molecule.

    Returns:
        first: [n_tokens] first token index with the same ref_space_uid.
        last: [n_tokens] last token index with the same ref_space_uid.
    """
    n = len(ref_space_uid)
    first = np.empty(n, dtype=np.int64)
    last = np.empty(n, dtype=np.int64)

    uid_first: dict[int, int] = {}
    uid_last: dict[int, int] = {}
    for i in range(n):
        uid = int(ref_space_uid[i])
        if uid not in uid_first:
            uid_first[uid] = i
        uid_last[uid] = i

    for i in range(n):
        uid = int(ref_space_uid[i])
        first[i] = uid_first[uid]
        last[i] = uid_last[uid]

    return first, last


def _detect_metal_tokens(
    chain_ids: IntArray,
    ref_space_uid: IntArray,
    token_atom_counts: IntArray,
) -> BoolArray:
    """Detect single-atom metal/ion tokens.

    A token is metal if its molecule (ref_space_uid) appears exactly once,
    its chain has exactly one token, and the molecule has exactly one atom.
    """
    # How many tokens per molecule.
    unique_uids, inverse = np.unique(ref_space_uid, return_inverse=True)
    uid_token_counts = np.bincount(inverse)
    tokens_per_mol = uid_token_counts[inverse]

    # How many tokens per chain.
    chain_token_counts = np.bincount(chain_ids.astype(np.intp))
    tokens_per_chain = chain_token_counts[chain_ids.astype(np.intp)]

    return (tokens_per_mol == 1) & (tokens_per_chain == 1) & (token_atom_counts == 1)


def _adjust_boundaries_for_molecules(
    start: int,
    end: int,
    mol_first: IntArray,
    mol_last: IntArray,
    crop_size_min: int,
    n_added: int,
    crop_size: int,
    drop_last: bool,
) -> tuple[int, int]:
    """Adjust contiguous crop boundaries to preserve molecule integrity.

    When a boundary falls in the middle of a molecule (a run of tokens with the
    same ref_space_uid), decide whether to expand or truncate.

    Four cases based on whether start/end split a molecule:

    1. Neither splits → no adjustment.
    2. Both split → force start to left boundary, handle end as case 3.
    3. Start OK, end splits → choose left (truncate) or right (extend) for end.
    4. Start splits, end OK → choose left (extend) or right (truncate) for start.

    The choice respects: crop_size_min ≤ result_size ≤ (crop_size - n_added).
    """
    if start == end:
        return start, end

    # Detect if start is in the middle of a molecule.
    start_in_middle = mol_first[start] != start
    left_start = int(mol_first[start])
    right_start = int(mol_last[start]) + 1

    # Detect if end-1 is in the middle of a molecule.
    end_in_middle = end > 0 and mol_last[end - 1] != end - 1
    left_end = int(mol_first[end - 1]) if end > 0 else end
    right_end = int(mol_last[end - 1]) + 1 if end > 0 else end

    # Case 1: clean boundaries.
    if not start_in_middle and not end_in_middle:
        return start, end

    # Case 2: both split → force start to left, then handle end.
    if start_in_middle and end_in_middle:
        start_in_middle = False
        start = left_start

    def _fits(size: int) -> bool:
        return crop_size_min <= size and size + n_added <= crop_size

    # Case 3: start OK, end splits.
    if not start_in_middle and end_in_middle:
        left_size = left_end - start
        right_size = right_end - start
        left_ok = _fits(left_size)
        right_ok = _fits(right_size)

        if left_ok and right_ok:
            end = left_end if random.random() < 0.5 else right_end
        elif left_ok:
            end = left_end
        elif right_ok:
            end = right_end
        elif drop_last:
            # Walk backward to find a clean boundary that fits.
            end = left_end
            while end - start + n_added > crop_size:
                if end > start:
                    end = int(mol_first[end - 1])
                else:
                    break
        else:
            # Truncate to fit.
            size = min(end - start, crop_size - n_added)
            end = start + size
        return start, end

    # Case 4: start splits, end OK.
    if start_in_middle and not end_in_middle:
        left_size = end - left_start
        right_size = end - right_start
        left_ok = _fits(left_size)
        right_ok = _fits(right_size)

        if left_ok and right_ok:
            start = left_start if random.random() < 0.5 else right_start
        elif left_ok:
            start = left_start
        elif right_ok:
            start = right_start
        elif drop_last:
            start = right_start
        else:
            pass  # keep original start
        return start, end

    return start, end


def _compute_token_molecule_atom_counts(
    structure: AnnotatedStructure,
) -> IntArray:
    """Per-token total atom count of the molecule the token belongs to.

    For each token, sums atoms across all tokens sharing the same
    ref_space_uid. Used for metal/ion detection.
    """
    atoms_per_token = np.bincount(
        structure.grouping.atom_token_index,
        minlength=structure.grouping.n_tokens,
    )
    uid = structure.reference.ref_space_uid[structure.centre_atom_indices]
    unique_uids, inverse = np.unique(uid, return_inverse=True)
    atoms_per_molecule = np.bincount(
        inverse,
        weights=atoms_per_token.astype(np.float64),
    )
    return atoms_per_molecule[inverse].astype(np.int64)
