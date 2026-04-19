"""3-state DSSP secondary structure assignment (helix/sheet/coil).

Vectorized implementation of the Kabsch-Sander algorithm for backbone
H-bond detection and DSSP rules for SSE assignment. No external binary
required.

Reference: Kabsch & Sander (1983) Biopolymers 22:2577-2637
"""

import numpy as np
from biotite.structure import AtomArray, get_residue_starts

_KS_Q1Q2 = 0.084
_KS_F = 332
_KS_CUTOFF = -0.5
_KS_MARGIN = 1.0


def _extract_backbone_coords(atom_array: AtomArray) -> np.ndarray:
    """Extract N, CA, C, O coordinates per residue as (L, 4, 3) array.

    Residues missing any backbone atom are excluded.
    """
    res_starts = get_residue_starts(atom_array, add_exclusive_stop=True)
    n_residues = len(res_starts) - 1
    names = atom_array.atom_name

    # Per-atom residue index (handles multi-chain correctly)
    res_index = np.repeat(np.arange(n_residues), np.diff(res_starts))

    # For each backbone atom type, find which residues have it
    bb_res = {}
    bb_coords = {}
    for atom_name in ("N", "CA", "C", "O"):
        mask = names == atom_name
        bb_res[atom_name] = res_index[mask]
        bb_coords[atom_name] = atom_array.coord[mask]

    # Residues with all 4 backbone atoms
    common = np.intersect1d(
        np.intersect1d(bb_res["N"], bb_res["CA"]),
        np.intersect1d(bb_res["C"], bb_res["O"]),
    )
    if len(common) == 0:
        return np.empty((0, 4, 3))

    coords = np.empty((len(common), 4, 3))
    for i, atom_name in enumerate(("N", "CA", "C", "O")):
        r_idx = bb_res[atom_name]
        c = bb_coords[atom_name]
        keep = np.isin(r_idx, common)
        r_idx = r_idx[keep]
        c = c[keep]
        _, first_idx = np.unique(r_idx, return_index=True)
        coords[:, i, :] = c[first_idx]

    return coords


def _estimate_h_positions(coord: np.ndarray) -> np.ndarray:
    """Estimate amide H positions from backbone geometry.

    H is placed 1.01 Å from N along the bisector of C_prev→N and CA→N.
    First residue gets H = N (no preceding residue).

    Args:
        coord: (L, 4, 3) backbone coordinates [N, CA, C, O].

    Returns:
        (L-1, 3) estimated H positions for residues 1..L-1.
    """
    vec_cn = coord[1:, 0] - coord[:-1, 2]  # C_{i-1} -> N_i
    vec_cn = vec_cn / np.linalg.norm(vec_cn, axis=-1, keepdims=True)
    vec_can = coord[1:, 0] - coord[1:, 1]  # CA_i -> N_i
    vec_can = vec_can / np.linalg.norm(vec_can, axis=-1, keepdims=True)
    vec_nh = vec_cn + vec_can
    vec_nh = vec_nh / np.linalg.norm(vec_nh, axis=-1, keepdims=True)
    return coord[1:, 0] + 1.01 * vec_nh


def _unfold(a: np.ndarray, window: int, axis: int) -> np.ndarray:
    """Sliding window view along an axis (like torch.Tensor.unfold)."""
    idx = np.arange(window)[:, None] + np.arange(a.shape[axis] - window + 1)[None, :]
    unfolded = np.take(a, idx, axis=axis)
    return np.moveaxis(unfolded, axis - 1, -1)


def _compute_hbond_map(coord: np.ndarray) -> np.ndarray:
    """Compute soft H-bond map from backbone coordinates.

    Uses Kabsch-Sander electrostatic energy with a continuous sin-based
    relaxation for smooth bridge detection.

    Args:
        coord: (L, 4, 3) backbone coordinates [N, CA, C, O].

    Returns:
        (L, L) soft H-bond map where hbond[i,j] > 0 indicates a bond
        from donor i (N-H) to acceptor j (C=O).
    """
    l = len(coord)
    h = _estimate_h_positions(coord)

    # Pairwise distance maps between donor (i>=1) and acceptor (j<L-1) atoms
    n = coord[1:, 0]  # (L-1, 3) donor N
    c = coord[:-1, 2]  # (L-1, 3) acceptor C
    o = coord[:-1, 3]  # (L-1, 3) acceptor O

    # Compute reciprocal distances via squared distances (avoids 4x linalg.norm)
    def _recip_dist(a, b):
        """1/||a_i - b_j|| for all pairs, shape (len(a), len(b))."""
        diff = a[:, None, :] - b[None, :, :]
        return 1.0 / np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

    r_on = _recip_dist(n, o)
    r_ch = _recip_dist(h, c)
    r_oh = _recip_dist(h, o)
    r_cn = _recip_dist(n, c)

    # Kabsch-Sander energy, padded back to (L, L)
    e = _KS_Q1Q2 * _KS_F * (r_on + r_ch - r_oh - r_cn)
    e = np.pad(e, [[1, 0], [0, 1]])

    # Mask local pairs: (i,i), (i,i+1), (i,i+2)
    local_mask = ~np.eye(l, dtype=bool)
    local_mask &= ~np.diag(np.ones(l - 1, dtype=bool), k=-1)
    local_mask &= ~np.diag(np.ones(l - 2, dtype=bool), k=-2)

    # Continuous H-bond map (sin-based soft threshold)
    hbond_map = np.clip(
        _KS_CUTOFF - _KS_MARGIN - e, a_min=-_KS_MARGIN, a_max=_KS_MARGIN
    )
    hbond_map = (np.sin(hbond_map / _KS_MARGIN * np.pi / 2) + 1.0) / 2.0
    hbond_map = hbond_map * local_mask

    return hbond_map


def _assign_sse(hbmap: np.ndarray) -> np.ndarray:
    """Assign 3-state SSE from an H-bond map.

    Args:
        hbmap: (L, L) H-bond map in "i:C=O, j:N-H" convention
               (hbmap[i,j] > 0 means residue j donates to residue i).

    Returns:
        (L, 3) one-hot array with columns [coil, helix, sheet].
    """
    # --- Helices ---
    # n-turn at i: H-bond from CO of i to NH of i+n → hbmap[i, i+n] > 0
    turn3 = np.diagonal(hbmap, offset=3) > 0.0
    turn4 = np.diagonal(hbmap, offset=4) > 0.0
    turn5 = np.diagonal(hbmap, offset=5) > 0.0

    # Helix of type n: two consecutive n-turns
    h3 = np.pad(turn3[:-1] * turn3[1:], [1, 3])
    h4 = np.pad(turn4[:-1] * turn4[1:], [1, 4])
    h5 = np.pad(turn5[:-1] * turn5[1:], [1, 5])

    # Alpha helix (4-turn) takes priority
    helix4 = h4 | np.roll(h4, 1) | np.roll(h4, 2) | np.roll(h4, 3)
    h3 = h3 & ~np.roll(helix4, -1) & ~helix4
    h5 = h5 & ~np.roll(helix4, -1) & ~helix4
    helix3 = h3 | np.roll(h3, 1) | np.roll(h3, 2)
    helix5 = h5 | np.roll(h5, 1) | np.roll(h5, 2) | np.roll(h5, 3) | np.roll(h5, 4)

    helix = helix3 | helix4 | helix5

    # --- Beta bridges ---
    unfoldmap = _unfold(_unfold(hbmap, 3, -2), 3, -2) > 0.0
    unfoldmap_rev = unfoldmap.transpose(1, 0, 2, 3)

    # Parallel bridge
    p_bridge = (unfoldmap[:, :, 0, 1] & unfoldmap_rev[:, :, 1, 2]) | (
        unfoldmap_rev[:, :, 0, 1] & unfoldmap[:, :, 1, 2]
    )
    p_bridge = np.pad(p_bridge, [[1, 1], [1, 1]])

    # Antiparallel bridge
    a_bridge = (unfoldmap[:, :, 1, 1] & unfoldmap_rev[:, :, 1, 1]) | (
        unfoldmap[:, :, 0, 2] & unfoldmap_rev[:, :, 0, 2]
    )
    a_bridge = np.pad(a_bridge, [[1, 1], [1, 1]])

    # Strand: residue participates in any bridge
    strand = (p_bridge | a_bridge).any(axis=-1)

    # --- Combine ---
    coil = ~helix & ~strand
    return np.stack([coil, helix, strand], axis=-1)


def compute_dssp(atom_array: AtomArray) -> np.ndarray:
    """Compute 3-state secondary structure assignment using the DSSP algorithm.

    Uses vectorized Kabsch-Sander backbone H-bond energies to detect
    helices and beta sheets. No external binary required.

    Args:
        atom_array: Biotite AtomArray (protein atoms only).

    Returns:
        Array of 'a' (helix), 'b' (sheet), 'c' (coil), one per residue
        that has complete backbone atoms (N, CA, C, O).
    """
    coord = _extract_backbone_coords(atom_array)
    n_res = len(coord)

    if n_res < 5:
        return np.full(n_res, "c")

    hbmap = _compute_hbond_map(coord)

    # Transpose to "i:C=O, j:N-H" convention for SSE assignment
    hbmap = hbmap.T

    onehot = _assign_sse(hbmap)

    # Convert one-hot [coil, helix, sheet] to character array
    labels = np.array(["c", "a", "b"])
    return labels[np.argmax(onehot, axis=-1)]
