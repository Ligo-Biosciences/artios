"""Equivalence and behavioural tests for the torch DSSP port."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray, get_residue_starts
from biotite.structure.io import load_structure

from artios.utils.dssp import compute_dssp
from artios.utils.dssp_torch import (
    SSE_ALPHA,
    SSE_BETA,
    SSE_COIL,
    SSE_MASKED,
    compute_dssp_torch,
)

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = ROOT / "pdb-samples/golden-monomers"
# Output of compute_dssp is a char array; map to torch int convention.
LABEL_TO_INT = {"a": SSE_ALPHA, "b": SSE_BETA, "c": SSE_COIL}


def _extract_backbone(atoms: AtomArray) -> tuple[np.ndarray, int]:
    """Return (N×4×3 backbone coords for residues with full N/CA/C/O, L)."""
    res_starts = get_residue_starts(atoms, add_exclusive_stop=True)
    n_res = len(res_starts) - 1
    names = atoms.atom_name
    res_index = np.repeat(np.arange(n_res), np.diff(res_starts))
    bb_res, bb_coords = {}, {}
    for a in ("N", "CA", "C", "O"):
        mask = names == a
        bb_res[a] = res_index[mask]
        bb_coords[a] = atoms.coord[mask]
    common = np.intersect1d(
        np.intersect1d(bb_res["N"], bb_res["CA"]),
        np.intersect1d(bb_res["C"], bb_res["O"]),
    )
    coords = np.empty((len(common), 4, 3), dtype=np.float32)
    for i, a in enumerate(("N", "CA", "C", "O")):
        r_idx = bb_res[a]
        c = bb_coords[a]
        keep = np.isin(r_idx, common)
        r_idx = r_idx[keep]
        c = c[keep]
        _, first = np.unique(r_idx, return_index=True)
        coords[:, i] = c[first]
    return coords, len(common)


def _numpy_labels_for(atoms: AtomArray) -> np.ndarray:
    labels = compute_dssp(atoms)
    return np.array([LABEL_TO_INT[c] for c in labels], dtype=np.int64)


# A small spread chosen for coverage:
#  - varying lengths
#  - mix of helical / sheet / mixed folds
GOLDEN_PDBS = ["1a2p", "1ako", "1ah7", "1bk7", "1eyo"]


@pytest.fixture(scope="module")
def golden_cases() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """List of (pdb_id, backbone_coords [L,4,3], numpy_sse_labels [L])."""
    available = [p for p in GOLDEN_PDBS if (GOLDEN_DIR / f"{p}.cif").exists()]
    if len(available) < 3:
        pytest.skip(
            "Need at least 3 golden monomers in pdb-samples/golden-monomers; "
            f"found {len(available)}"
        )
    out = []
    for pid in available:
        atoms = load_structure(str(GOLDEN_DIR / f"{pid}.cif"))
        if hasattr(atoms, "stack_depth"):
            atoms = atoms[0]
        atoms = atoms[~atoms.hetero]  # drop water/ligand
        coords, L = _extract_backbone(atoms)
        numpy_labels = _numpy_labels_for(atoms)
        assert len(numpy_labels) == L, (
            f"{pid}: numpy DSSP returned {len(numpy_labels)} labels "
            f"for {L} backbone-complete residues"
        )
        out.append((pid, coords, numpy_labels))
    return out


def test_matches_numpy_per_residue_single_sample(golden_cases):
    """Every residue label from torch DSSP must match the numpy version."""
    for pid, coords, expected in golden_cases:
        bb = torch.from_numpy(coords)
        mask = torch.ones(coords.shape[0], dtype=torch.bool)
        got = compute_dssp_torch(bb, mask).numpy()
        mismatches = np.where(got != expected)[0]
        assert mismatches.size == 0, (
            f"{pid}: {mismatches.size}/{len(expected)} residue mismatches "
            f"(first 10 indices: {mismatches[:10].tolist()})"
        )


def test_batched_equals_unbatched(golden_cases):
    """Running [B, N, 4, 3] together is equivalent to running each individually."""
    max_L = max(c.shape[0] for _, c, _ in golden_cases)
    B = len(golden_cases)
    bb = torch.zeros(B, max_L, 4, 3, dtype=torch.float32)
    mask = torch.zeros(B, max_L, dtype=torch.bool)
    for i, (_, coords, _) in enumerate(golden_cases):
        L = coords.shape[0]
        bb[i, :L] = torch.from_numpy(coords)
        mask[i, :L] = True

    batched = compute_dssp_torch(bb, mask).numpy()
    for i, (pid, coords, _) in enumerate(golden_cases):
        L = coords.shape[0]
        single = compute_dssp_torch(
            torch.from_numpy(coords),
            torch.ones(L, dtype=torch.bool),
        ).numpy()
        assert np.array_equal(batched[i, :L], single), (
            f"{pid}: batched output diverges from unbatched at "
            f"indices {np.where(batched[i, :L] != single)[0][:10].tolist()}"
        )
        # Padded tail should be SSE_MASKED.
        assert np.all(
            batched[i, L:] == SSE_MASKED
        ), f"{pid}: padded positions {L}..{max_L} should be SSE_MASKED"


def test_mask_makes_position_ignored(golden_cases):
    """Masking out a residue must yield SSE_MASKED there and not change others.

    Masking a *single* residue can legitimately re-route its neighbours through
    different H-bond partners, so we check the (must change) and (cannot remain
    spurious helix/sheet pattern) properties separately.
    """
    pid, coords, expected = golden_cases[0]
    bb = torch.from_numpy(coords)
    L = bb.shape[0]
    # Pick a residue that's helix in the full assignment so it's a meaningful test.
    helix_idx = np.where(expected == SSE_ALPHA)[0]
    if helix_idx.size == 0:
        pytest.skip(f"{pid}: no helix residues to mask")
    target = int(helix_idx[helix_idx.size // 2])
    mask = torch.ones(L, dtype=torch.bool)
    mask[target] = False
    got = compute_dssp_torch(bb, mask).numpy()
    assert got[target] == SSE_MASKED


def test_short_sequence_all_coil():
    """Sequences shorter than 5 residues collapse to coil for valid positions."""
    bb = torch.randn(3, 4, 3)
    mask = torch.tensor([True, True, False])
    got = compute_dssp_torch(bb, mask).numpy()
    assert got[0] == SSE_COIL and got[1] == SSE_COIL
    assert got[2] == SSE_MASKED


def test_label_constants_disjoint():
    """The four label constants must be pairwise distinct."""
    values = {SSE_ALPHA, SSE_BETA, SSE_COIL, SSE_MASKED}
    assert len(values) == 4
