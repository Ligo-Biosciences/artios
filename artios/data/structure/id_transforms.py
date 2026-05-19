"""Validation and construction helpers for structure id rewrites."""

from __future__ import annotations

import numpy as np
from biotite.structure import AtomArray, get_residue_starts


def coerce_atom_chain_ids(chain_id: np.ndarray, *, n_atoms: int) -> np.ndarray:
    """Validate shape and return a private string copy."""
    new_chain_id = np.asarray(chain_id).astype(str)
    if new_chain_id.ndim != 1:
        raise ValueError("chain_id must be a 1D atom-indexed array")
    if len(new_chain_id) != n_atoms:
        raise ValueError(
            f"chain_id length {len(new_chain_id)} does not match n_atom {n_atoms}"
        )
    if np.isin(new_chain_id, ["", ".", "?"]).any():
        raise ValueError("chain_id must not contain empty or missing labels")
    return new_chain_id.copy()


def coerce_atom_residue_ids(res_id: np.ndarray, *, n_atoms: int) -> np.ndarray:
    """Validate shape/dtype and return a private integer copy."""
    new_res_id = np.asarray(res_id)
    if new_res_id.ndim != 1:
        raise ValueError("res_id must be a 1D atom-indexed array")
    if len(new_res_id) != n_atoms:
        raise ValueError(
            f"res_id length {len(new_res_id)} does not match n_atom {n_atoms}"
        )
    if not np.issubdtype(new_res_id.dtype, np.integer):
        raise ValueError("res_id must contain integers")
    return new_res_id.astype(int, copy=True)


def validate_chain_id_rewrite(atoms: AtomArray, chain_id: np.ndarray) -> None:
    """Reject chain-id rewrites that would corrupt residue or entity structure."""
    residue_starts = get_residue_starts(atoms, add_exclusive_stop=True)
    _validate_constant_within_residue(
        chain_id,
        residue_starts,
        field_name="chain_id",
    )
    _validate_chain_ids_are_contiguous(chain_id)
    _validate_chain_ids_have_single_entity(atoms, chain_id)
    _validate_chain_residue_keys_unique(atoms, chain_id, residue_starts)


def validate_residue_id_rewrite(
    atoms: AtomArray, asym_id_int: np.ndarray, res_id: np.ndarray
) -> None:
    """Reject residue-id rewrites that would split or merge residues."""
    residue_starts = get_residue_starts(atoms, add_exclusive_stop=True)
    _validate_constant_within_residue(
        res_id,
        residue_starts,
        field_name="res_id",
    )
    _validate_residue_ids_unique_per_asym(asym_id_int, res_id, residue_starts)


def make_contiguous_residue_ids(
    atoms: AtomArray, asym_id_int: np.ndarray, *, start: int, by_chain: bool
) -> np.ndarray:
    """Create dense atom-indexed residue ids, optionally resetting per asym chain."""
    residue_starts = get_residue_starts(atoms, add_exclusive_stop=True)
    residue_ids = _contiguous_ids_by_residue(
        asym_id_int,
        residue_starts,
        start=start,
        by_chain=by_chain,
    )
    return np.repeat(residue_ids, np.diff(residue_starts))


def _validate_constant_within_residue(
    values: np.ndarray, residue_starts: np.ndarray, *, field_name: str
) -> None:
    """Ensure each existing residue receives one and only one value."""
    for start, stop in zip(residue_starts[:-1], residue_starts[1:]):
        if np.unique(values[start:stop]).size != 1:
            raise ValueError(
                f"{field_name} must be constant within each existing residue"
            )


def _validate_chain_ids_are_contiguous(chain_id: np.ndarray) -> None:
    """Reject repeated chain labels in disjoint atom-array runs."""
    if len(chain_id) == 0:
        return

    run_starts = np.concatenate([[0], np.where(chain_id[1:] != chain_id[:-1])[0] + 1])
    run_labels = chain_id[run_starts]
    if np.unique(run_labels).size != len(run_labels):
        raise ValueError("chain_id labels must form contiguous runs")


def _validate_chain_ids_have_single_entity(
    atoms: AtomArray, chain_id: np.ndarray
) -> None:
    """Ensure each new chain run has one label_entity_id."""
    if len(chain_id) == 0:
        return

    run_starts = np.concatenate(
        [[0], np.where(chain_id[1:] != chain_id[:-1])[0] + 1, [len(chain_id)]]
    )
    for start, stop in zip(run_starts[:-1], run_starts[1:]):
        if np.unique(atoms.label_entity_id[start:stop]).size != 1:
            raise ValueError(
                "chain_id would merge atoms from multiple label_entity_id values"
            )


def _validate_chain_residue_keys_unique(
    atoms: AtomArray, chain_id: np.ndarray, residue_starts: np.ndarray
) -> None:
    """Ensure the new chain labels do not merge existing residues."""
    residue_chain_ids = chain_id[residue_starts[:-1]]
    if len(residue_chain_ids) == 0:
        return

    residue_res_ids = atoms.res_id[residue_starts[:-1]].astype(str)
    residue_keys = np.stack([residue_chain_ids, residue_res_ids], axis=1)
    if np.unique(residue_keys, axis=0).shape[0] != len(residue_keys):
        raise ValueError("chain_id would collapse two residues with the same res_id")


def _validate_residue_ids_unique_per_asym(
    asym_id_int: np.ndarray, res_id: np.ndarray, residue_starts: np.ndarray
) -> None:
    """Ensure the new ids do not merge residues inside an asym chain."""
    residue_new_ids = res_id[residue_starts[:-1]]
    if len(residue_new_ids) == 0:
        return

    residue_asym_ids = asym_id_int[residue_starts[:-1]]
    residue_keys = np.stack([residue_asym_ids, residue_new_ids], axis=1)
    if np.unique(residue_keys, axis=0).shape[0] != len(residue_keys):
        raise ValueError(
            "res_id would collapse two residues within the same asym chain"
        )


def _contiguous_ids_by_residue(
    asym_id_int: np.ndarray, residue_starts: np.ndarray, *, start: int, by_chain: bool
) -> np.ndarray:
    """Create one dense id per residue, optionally resetting per asym chain."""
    n_residues = max(len(residue_starts) - 1, 0)
    if not by_chain:
        return start + np.arange(n_residues, dtype=int)

    residue_ids = np.empty(n_residues, dtype=int)
    residue_asym_ids = asym_id_int[residue_starts[:-1]]
    for asym_id in np.unique(residue_asym_ids):
        mask = residue_asym_ids == asym_id
        residue_ids[mask] = start + np.arange(mask.sum(), dtype=int)
    return residue_ids
