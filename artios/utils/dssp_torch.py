"""Pure-torch port of 3-state DSSP secondary structure assignment.

Vectorized Kabsch-Sander DSSP, batched over a leading batch dim. Matches
the numpy implementation in :mod:`artios.data.engineering.metrics.dssp`
per-residue on inputs with finite backbone coordinates.

Convention:
    SSE_ALPHA = 0   # helix  ('a' in the numpy version)
    SSE_BETA  = 1   # sheet  ('b')
    SSE_COIL  = 2   # coil   ('c')
    SSE_MASKED = -1 # padded / invalid / sequence too short

Reference: Kabsch & Sander (1983) Biopolymers 22:2577-2637.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from artios.utils.typing import Bool, Float, Int

_KS_Q1Q2 = 0.084
_KS_F = 332.0
_KS_CUTOFF = -0.5
_KS_MARGIN = 1.0

SSE_ALPHA = 0
SSE_BETA = 1
SSE_COIL = 2
SSE_MASKED = -1


def _estimate_h_positions(
    coord: Float[Tensor, "*batch n_residue 4 3"],
) -> Float[Tensor, "*batch n_minus_one 3"]:
    """Place amide H at 1.01 Å from N along the bisector of C_prev→N and CA→N.

    First residue has no preceding C; we drop it (output length = L - 1).
    """
    n_cur = coord[..., 1:, 0, :]
    ca_cur = coord[..., 1:, 1, :]
    c_prev = coord[..., :-1, 2, :]

    vec_cn = n_cur - c_prev
    vec_cn = vec_cn / (vec_cn.norm(dim=-1, keepdim=True) + 1e-8)
    vec_can = n_cur - ca_cur
    vec_can = vec_can / (vec_can.norm(dim=-1, keepdim=True) + 1e-8)
    vec_nh = vec_cn + vec_can
    vec_nh = vec_nh / (vec_nh.norm(dim=-1, keepdim=True) + 1e-8)
    return n_cur + 1.01 * vec_nh


def _recip_dist(
    a: Float[Tensor, "*batch m 3"],
    b: Float[Tensor, "*batch k 3"],
) -> Float[Tensor, "*batch m k"]:
    """1/‖a_i − b_j‖ for all pairs."""
    diff = a.unsqueeze(-2) - b.unsqueeze(-3)
    sq = (diff * diff).sum(dim=-1)
    return 1.0 / (sq.sqrt() + 1e-8)


def _compute_hbond_map(
    coord: Float[Tensor, "*batch n_residue 4 3"],
) -> Float[Tensor, "*batch n_residue n_residue"]:
    """Soft Kabsch-Sander H-bond map.

    ``hbmap[..., i, j] > 0`` indicates residue *j* donates (N-H) to residue
    *i* (C=O), i.e. the (donor, acceptor) convention of the numpy ``compute_dssp``
    before its final transpose.
    """
    L = coord.shape[-3]
    h = _estimate_h_positions(coord)  # [..., L-1, 3]
    n_d = coord[..., 1:, 0, :]  # donor N at i ≥ 1
    c_a = coord[..., :-1, 2, :]  # acceptor C at j ≤ L-2
    o_a = coord[..., :-1, 3, :]  # acceptor O at j ≤ L-2

    r_on = _recip_dist(n_d, o_a)
    r_ch = _recip_dist(h, c_a)
    r_oh = _recip_dist(h, o_a)
    r_cn = _recip_dist(n_d, c_a)

    e = _KS_Q1Q2 * _KS_F * (r_on + r_ch - r_oh - r_cn)  # [..., L-1, L-1]
    # Match numpy np.pad(e, [[1, 0], [0, 1]]):
    #   F.pad order is (last_left, last_right, second_last_left, second_last_right).
    e_full = F.pad(e, (0, 1, 1, 0))

    device = coord.device
    idx = torch.arange(L, device=device)
    diff = idx.unsqueeze(-1) - idx.unsqueeze(-2)
    # Match numpy ``local_mask = ~eye & ~diag(k=-1) & ~diag(k=-2)``:
    # this excludes (i, i), (i, i-1), (i, i-2) — note it's NOT symmetric.
    local_mask = ~((diff == 0) | (diff == 1) | (diff == 2))

    soft = torch.clamp(
        _KS_CUTOFF - _KS_MARGIN - e_full, min=-_KS_MARGIN, max=_KS_MARGIN
    )
    hbond = (torch.sin(soft / _KS_MARGIN * (math.pi / 2)) + 1.0) / 2.0
    return hbond * local_mask.to(hbond.dtype)


def _assign_sse(
    hbmap: Float[Tensor, "*batch n_residue n_residue"],
) -> Int[Tensor, "*batch n_residue"]:
    """SSE assignment from an H-bond map in (i:C=O, j:N-H) convention.

    Returns int labels in {SSE_ALPHA=0, SSE_BETA=1, SSE_COIL=2}. Helix takes
    priority over sheet to match the numpy ``argmax([coil, helix, sheet])`` rule.
    """
    turn3 = torch.diagonal(hbmap, offset=3, dim1=-2, dim2=-1) > 0.0  # [..., L-3]
    turn4 = torch.diagonal(hbmap, offset=4, dim1=-2, dim2=-1) > 0.0  # [..., L-4]
    turn5 = torch.diagonal(hbmap, offset=5, dim1=-2, dim2=-1) > 0.0  # [..., L-5]

    def _consec_pad(turn: Tensor, n: int) -> Tensor:
        """Two consecutive n-turns starting at i, padded to length L.

        numpy form: ``np.pad(t[:-1] & t[1:], [1, n])``.
        """
        consec = (turn[..., :-1] & turn[..., 1:]).to(torch.uint8)
        return F.pad(consec, (1, n)).bool()

    h3 = _consec_pad(turn3, 3)
    h4 = _consec_pad(turn4, 4)
    h5 = _consec_pad(turn5, 5)

    def _shift(x: Tensor, n: int) -> Tensor:
        return torch.roll(x, shifts=n, dims=-1)

    # Alpha helix (4-turn) wins over 3- and 5-turn.
    helix4 = h4 | _shift(h4, 1) | _shift(h4, 2) | _shift(h4, 3)
    h3 = h3 & ~_shift(helix4, -1) & ~helix4
    h5 = h5 & ~_shift(helix4, -1) & ~helix4
    helix3 = h3 | _shift(h3, 1) | _shift(h3, 2)
    helix5 = h5 | _shift(h5, 1) | _shift(h5, 2) | _shift(h5, 3) | _shift(h5, 4)
    helix = helix3 | helix4 | helix5

    # Bridge detection via 3×3 sliding windows over the H-bond map.
    # ``Tensor.unfold`` appends a new last dim of the window size, and doesn't
    # accept bool, so we run it on uint8 and re-cast.
    hb_bool_u8 = (hbmap > 0.0).to(torch.uint8)
    unfoldmap = hb_bool_u8.unfold(-2, 3, 1).unfold(-2, 3, 1).bool()
    # shape: [..., L-2, L-2, 3, 3]
    unfoldmap_rev = unfoldmap.transpose(-4, -3)

    p_bridge = (unfoldmap[..., 0, 1] & unfoldmap_rev[..., 1, 2]) | (
        unfoldmap_rev[..., 0, 1] & unfoldmap[..., 1, 2]
    )
    a_bridge = (unfoldmap[..., 1, 1] & unfoldmap_rev[..., 1, 1]) | (
        unfoldmap[..., 0, 2] & unfoldmap_rev[..., 0, 2]
    )
    # Pad (L-2, L-2) → (L, L) on the last two dims.
    p_bridge = F.pad(p_bridge.to(torch.uint8), (1, 1, 1, 1)).bool()
    a_bridge = F.pad(a_bridge.to(torch.uint8), (1, 1, 1, 1)).bool()
    strand = (p_bridge | a_bridge).any(dim=-1)  # [..., L]

    sse = torch.full(helix.shape, SSE_COIL, dtype=torch.long, device=hbmap.device)
    sse = torch.where(strand, torch.full_like(sse, SSE_BETA), sse)
    sse = torch.where(helix, torch.full_like(sse, SSE_ALPHA), sse)
    return sse


def compute_dssp_torch(
    backbone_coords: Float[Tensor, "*batch n_residue 4 3"],
    token_mask: Bool[Tensor, "*batch n_residue"],
) -> Int[Tensor, "*batch n_residue"]:
    """Pure-torch 3-state DSSP equivalent to ``compute_dssp`` (biotite numpy).

    Args:
        backbone_coords: N, CA, C, O coordinates per residue. The leading
            ``*batch`` dims are forwarded through; either unbatched ``[L, 4, 3]``
            or batched ``[B, L, 4, 3]`` works.
        token_mask: bool mask over residues. Masked positions are removed
            from the H-bond graph and labelled ``SSE_MASKED`` in the output.

    Returns:
        ``[*batch, L]`` int tensor with values in
        ``{SSE_ALPHA=0, SSE_BETA=1, SSE_COIL=2, SSE_MASKED=-1}``.
    """
    N = backbone_coords.shape[-3]
    device = backbone_coords.device

    if N < 5:
        # numpy version returns all "c" for short inputs; we mirror that for
        # valid positions and label masked positions as SSE_MASKED.
        out = torch.full(
            backbone_coords.shape[:-2], SSE_COIL, dtype=torch.long, device=device
        )
        return torch.where(token_mask, out, torch.full_like(out, SSE_MASKED))

    hbmap = _compute_hbond_map(backbone_coords)
    # numpy ``compute_dssp`` transposes to "i:C=O, j:N-H" convention.
    hbmap = hbmap.transpose(-2, -1)

    # Zero out hbonds involving masked positions.
    pair_mask = token_mask.unsqueeze(-1) & token_mask.unsqueeze(-2)
    hbmap = hbmap * pair_mask.to(hbmap.dtype)

    sse = _assign_sse(hbmap)
    return torch.where(token_mask, sse, torch.full_like(sse, SSE_MASKED))
