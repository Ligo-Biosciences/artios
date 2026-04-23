"""Unit tests for MSA/MSAQueryInfo (sampler.py) and MSAContext plumbing
through FullContext."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from artios.common.constants import GAP
from artios.data.features.context import FullContext, MSAContext
from artios.data.features.featurizer import ContextGenerator
from artios.data.msa import MSA, MSAQueryInfo

_GAP = GAP["-"]  # 31


# ---------------------------------------------------------------------------
# MSA dataclass invariants
# ---------------------------------------------------------------------------


def _make_msa(n_msa: int = 4, l_chain: int = 5, chain_asym: int = 0) -> MSA:
    tokens = np.zeros((n_msa, l_chain), dtype=np.int8)
    tokens[1, 0] = _GAP  # one gap to exercise gap_mask
    gap_mask = tokens == _GAP
    return MSA(
        tokens=tokens,
        gap_mask=gap_mask,
        msa_chain_asym_id=chain_asym,
        chain_token_indices=np.arange(l_chain, dtype=np.int32),
    )


def test_msa_construction_passes_invariants():
    msa = _make_msa()
    assert msa.tokens.shape == (4, 5)
    assert msa.gap_mask[1, 0]
    assert not msa.gap_mask[0, 0]


def test_msa_raises_on_non_2d_tokens():
    with pytest.raises(ValueError, match="2D"):
        MSA(
            tokens=np.zeros(4, dtype=np.int8),  # 1-D — bad
            gap_mask=np.zeros(4, dtype=bool),
            msa_chain_asym_id=0,
            chain_token_indices=np.zeros(4, dtype=np.int32),
        )


def test_msa_raises_on_gap_mask_shape_mismatch():
    with pytest.raises(ValueError, match="gap_mask"):
        MSA(
            tokens=np.zeros((4, 5), dtype=np.int8),
            gap_mask=np.zeros((4, 6), dtype=bool),  # wrong shape
            msa_chain_asym_id=0,
            chain_token_indices=np.arange(5, dtype=np.int32),
        )


def test_msa_raises_when_chain_token_indices_length_wrong():
    with pytest.raises(ValueError, match="chain_token_indices"):
        MSA(
            tokens=np.zeros((4, 5), dtype=np.int8),
            gap_mask=np.zeros((4, 5), dtype=bool),
            msa_chain_asym_id=0,
            chain_token_indices=np.arange(7, dtype=np.int32),  # len 7 != l_chain 5
        )


def test_msa_raises_when_gap_mask_inconsistent_with_tokens():
    tokens = np.zeros((4, 5), dtype=np.int8)
    # gap_mask says [0, 0] is a gap, but tokens[0, 0] == 0 (ALA), not 31.
    gap_mask = np.zeros((4, 5), dtype=bool)
    gap_mask[0, 0] = True
    with pytest.raises(ValueError, match="gap_mask must equal"):
        MSA(
            tokens=tokens,
            gap_mask=gap_mask,
            msa_chain_asym_id=0,
            chain_token_indices=np.arange(5, dtype=np.int32),
        )


# ---------------------------------------------------------------------------
# MSA.null_for — sentinel for samples with no protein chain
# ---------------------------------------------------------------------------


def test_null_msa_construction_passes_invariants():
    """Null MSA uses l_chain_cropped=0 (empty chain columns). All invariants
    should pass vacuously."""
    null = MSA.null_for(n_msa=64)
    assert null.tokens.shape == (64, 0)
    assert null.gap_mask.shape == (64, 0)
    assert null.chain_token_indices.shape == (0,)
    assert null.msa_chain_asym_id == -1
    assert null.tokens.dtype == np.int8


def test_null_msa_build_context_fills_rows_with_gt_residues(
    load_structure,
):
    """A null MSA run through _build_msa_context must yield:
    - tokens filled with the gt residue id at every (token, row) — every
      row is a copy of the query so the MSA loss degrades to the
      single-sequence composite NLL,
    - gap_mask True only where a gt residue id genuinely equals 31
      (i.e. nowhere in a clean protein fixture),
    - msa_chain_mask all False (no reference-chain data was supplied).
    """
    s = load_structure("2g1k")
    n_token = s.grouping.n_tokens
    null = MSA.null_for(n_msa=8)

    ctx = ContextGenerator(s, msa=null, pdb_id="2g1k").get_full_context(
        include_arrays=False
    )
    assert ctx.msa is not None
    assert ctx.msa.tokens.shape == (n_token, 8)
    # Every row is identical to row 0 (the gt residue id per token).
    row0 = ctx.msa.tokens[:, 0]
    assert (ctx.msa.tokens == row0.unsqueeze(-1)).all()
    # gap_mask is exactly `tokens == 31`; no gt tokens should be gap in
    # this fixture, but the invariant is consistency, not absence.
    assert torch.equal(ctx.msa.gap_mask, ctx.msa.tokens == _GAP)
    assert not ctx.msa.msa_chain_mask.any()
    assert ctx.msa.msa_chain_asym_id == -1


# ---------------------------------------------------------------------------
# MSAQueryInfo extraction
# ---------------------------------------------------------------------------


def test_query_info_extracts_protein_chain(load_structure):
    s = load_structure("2g1k")  # chain A is pure protein
    info = MSAQueryInfo.from_pre_crop_structure(s, "A")
    assert info is not None
    assert info.chain_id == "A"
    assert set(info.sequence) <= set("ACDEFGHIKLMNPQRSTVWYX")
    # One sequence entry per centre atom of chain A.
    n_centre_a = int((s.grouping.centre_atom_mask & (s.atoms.chain_id == "A")).sum())
    assert len(info.sequence) == n_centre_a
    assert info.res_ids.shape == (n_centre_a,)
    assert info.res_ids.dtype == np.int64


def test_query_info_returns_none_for_missing_chain(load_structure):
    s = load_structure("2g1k")
    assert MSAQueryInfo.from_pre_crop_structure(s, "ZZZ_BOGUS") is None


def test_query_info_returns_none_for_non_protein_chain(load_structure):
    # Chain G of 2g1k is pure non-protein (ligand/sugar).
    s = load_structure("2g1k")
    assert MSAQueryInfo.from_pre_crop_structure(s, "G") is None


def test_query_info_falls_back_to_second_candidate(load_structure):
    """First candidate is non-protein (chain G); fallback picks chain A."""
    s = load_structure("2g1k")
    info = MSAQueryInfo.from_any_protein_chain(s, ["G", "A"])
    assert info is not None
    assert info.chain_id == "A"


def test_query_info_scans_complex_when_no_candidate_resolves(load_structure):
    """Neither candidate is a protein chain (bogus + non-protein); scan
    finds chain A (the only protein chain in 2g1k)."""
    s = load_structure("2g1k")
    info = MSAQueryInfo.from_any_protein_chain(s, ["ZZZ_BOGUS", "G"])
    assert info is not None
    assert info.chain_id == "A"


def test_query_info_empty_preferred_list_scans_from_scratch(load_structure):
    """Empty preferred list → scan finds the first protein chain."""
    s = load_structure("2g1k")
    info = MSAQueryInfo.from_any_protein_chain(s, [])
    assert info is not None
    assert info.chain_id == "A"


def test_query_info_first_candidate_wins(load_structure):
    """Don't waste time on the scan when chain_1 is already protein."""
    s = load_structure("2g1k")
    info = MSAQueryInfo.from_any_protein_chain(s, ["A", "G"])
    assert info is not None
    assert info.chain_id == "A"


def test_query_info_handles_non_string_candidates(load_structure):
    """Candidates containing nan / None / non-strings are skipped cleanly."""
    s = load_structure("2g1k")
    info = MSAQueryInfo.from_any_protein_chain(
        s,
        [None, float("nan"), "G", "A"],  # type: ignore[list-item]
    )
    assert info is not None
    assert info.chain_id == "A"


# ---------------------------------------------------------------------------
# MSAContext plumbing through FullContext
# ---------------------------------------------------------------------------


def _build_full_context(structure) -> FullContext:
    return ContextGenerator(
        structure,
        pdb_id="2g1k",
        bioassembly_dict_fpath="<fixture>",
    ).get_full_context(include_arrays=False)


def _make_msa_context(n_token: int, n_msa: int = 4) -> MSAContext:
    # Realistic MSAContext with gap-fill everywhere and a few chain tokens
    # flagged. msa_chain_mask must be bool, fields must be the right dtype.
    tokens = torch.full((n_token, n_msa), _GAP, dtype=torch.int8)
    tokens[0, :] = 0  # one token with all AAs=0 (ALA), gap_mask[0] all False
    gap_mask = tokens == _GAP
    msa_chain_mask = torch.zeros(n_token, dtype=torch.bool)
    msa_chain_mask[0] = True
    return MSAContext(
        tokens=tokens,
        gap_mask=gap_mask,
        msa_chain_mask=msa_chain_mask,
        msa_chain_asym_id=0,
    )


def test_full_context_msa_none_by_default(load_structure):
    s = load_structure("2g1k")
    ctx = _build_full_context(s)
    assert ctx.msa is None


def test_full_context_pad_pads_msa_on_token_axis(load_structure):
    s = load_structure("2g1k")
    ctx = _build_full_context(s)
    n_token = ctx.basic.n_token
    n_atom = ctx.basic.n_atom
    ctx = dataclasses.replace(ctx, msa=_make_msa_context(n_token, n_msa=4))

    padded = ctx.pad(n_max_atoms=n_atom + 10, n_max_tokens=n_token + 7)

    assert padded.msa is not None
    assert padded.msa.tokens.shape == (n_token + 7, 4)
    assert padded.msa.gap_mask.shape == (n_token + 7, 4)
    assert padded.msa.msa_chain_mask.shape == (n_token + 7,)
    # msa_chain_asym_id is a plain int, not a tensor — passes through.
    assert padded.msa.msa_chain_asym_id == 0
    # Original token 0 was chain; padded positions (tail) are not chain.
    assert padded.msa.msa_chain_mask[0]
    assert not padded.msa.msa_chain_mask[n_token + 6]
    # Padded columns are zero-filled (torch.nn.functional.pad default); chain
    # column at row 0 still holds the original zeros.
    assert padded.msa.tokens[0, 0] == 0


def test_full_context_to_unconditional_preserves_msa(load_structure):
    s = load_structure("2g1k")
    ctx = _build_full_context(s)
    msa = _make_msa_context(ctx.basic.n_token, n_msa=4)
    ctx = dataclasses.replace(ctx, msa=msa)
    assert ctx.to_unconditional().msa is msa


def test_full_context_without_ligand_preserves_msa(load_structure):
    s = load_structure("2g1k")
    ctx = _build_full_context(s)
    msa = _make_msa_context(ctx.basic.n_token, n_msa=4)
    ctx = dataclasses.replace(ctx, msa=msa)
    assert ctx.without_ligand().msa is msa


def test_full_context_to_dict_round_trip_preserves_msa(load_structure):
    s = load_structure("2g1k")
    ctx = _build_full_context(s)
    n_token = ctx.basic.n_token
    msa = _make_msa_context(n_token, n_msa=4)
    ctx = dataclasses.replace(ctx, msa=msa)

    d = ctx.to_dict()
    assert "msa" in d

    rebuilt = FullContext.from_dict(d)
    assert rebuilt.msa is not None
    torch.testing.assert_close(rebuilt.msa.tokens, msa.tokens)
    torch.testing.assert_close(rebuilt.msa.gap_mask, msa.gap_mask)
    torch.testing.assert_close(rebuilt.msa.msa_chain_mask, msa.msa_chain_mask)
    assert rebuilt.msa.msa_chain_asym_id == msa.msa_chain_asym_id


def test_full_context_to_dict_omits_msa_when_none(load_structure):
    s = load_structure("2g1k")
    ctx = _build_full_context(s)
    assert "msa" not in ctx.to_dict()


def test_context_generator_build_msa_context_scatters_and_gt_fills(load_structure):
    """_build_msa_context turns [n_msa, l_chain] MSA into a token-wide
    MSAContext: scatter chain columns into their token positions, fill
    every other (token, row) cell with the gt residue id from the
    reference sequence. Non-reference-chain positions no longer carry the
    gap sentinel — they carry their true residue id so the Potts loss
    contributes a meaningful NLL at those tokens."""
    s = load_structure("2g1k")
    n_token = s.grouping.n_tokens

    # Hand-build an MSA covering 3 tokens on the reference chain with a
    # known pattern so we can assert the scatter.
    chain_a_centre = s.grouping.centre_atom_mask & (s.atoms.chain_id == "A")
    chain_a_token_indices = s.grouping.atom_token_index[np.where(chain_a_centre)[0]]
    # Pick 3 arbitrary token positions of chain A for the MSA columns.
    token_positions = np.array(
        [
            int(chain_a_token_indices[0]),
            int(chain_a_token_indices[5]),
            int(chain_a_token_indices[10]),
        ],
        dtype=np.int32,
    )
    tokens = np.array(
        [
            [0, 1, 2],  # row 0 (query): ALA, ARG, ASN
            [3, _GAP, 5],  # row 1: ASP, gap, GLN
        ],
        dtype=np.int8,
    )
    gap_mask = tokens == _GAP
    msa = MSA(
        tokens=tokens,
        gap_mask=gap_mask,
        msa_chain_asym_id=int(
            np.unique(s.entity_ids.asym_id_int[s.atoms.chain_id == "A"])[0]
        ),
        chain_token_indices=token_positions,
    )

    ctx = ContextGenerator(s, msa=msa, pdb_id="2g1k").get_full_context(
        include_arrays=False
    )

    assert ctx.msa is not None
    assert ctx.msa.tokens.shape == (n_token, 2)
    assert ctx.msa.gap_mask.shape == (n_token, 2)
    assert ctx.msa.msa_chain_mask.shape == (n_token,)

    # Chain columns carry the original MSA values (transposed).
    assert ctx.msa.tokens[int(token_positions[0]), 0] == 0  # row 0, col 0
    assert ctx.msa.tokens[int(token_positions[1]), 0] == 1  # row 0, col 1
    assert ctx.msa.tokens[int(token_positions[1]), 1] == _GAP  # row 1, col 1

    # Non-chain positions are gt residue ids — both rows identical there,
    # and none of them equal the gap sentinel in a clean protein fixture.
    non_chain = torch.ones(n_token, dtype=torch.bool)
    non_chain[torch.from_numpy(token_positions).long()] = False
    assert (ctx.msa.tokens[non_chain, 0] == ctx.msa.tokens[non_chain, 1]).all()
    assert (ctx.msa.tokens[non_chain] != _GAP).all()

    # gap_mask is exactly `tokens == 31` everywhere.
    assert torch.equal(ctx.msa.gap_mask, ctx.msa.tokens == _GAP)

    # msa_chain_mask is True exactly on chain_token_indices.
    assert ctx.msa.msa_chain_mask.sum().item() == 3
    for pos in token_positions:
        assert ctx.msa.msa_chain_mask[int(pos)]
