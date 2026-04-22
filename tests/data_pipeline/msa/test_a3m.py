"""Unit tests for artios.data.msa.a3m."""

from pathlib import Path

import numpy as np
import pytest

from artios.common.constants import (
    GAP,
    PRO_STD_RESIDUES,
    PROT_STD_RESIDUES_ONE_TO_THREE,
)
from artios.data.msa import encode_rows, parse_a3m


def test_parse_query_only_returns_single_row():
    content = ">query\nMKWVTFISLLFL\n"
    query, rows = parse_a3m(content)
    assert query == "MKWVTFISLLFL"
    assert rows == ["MKWVTFISLLFL"]


def test_parse_strips_lowercase_insertions_and_preserves_length():
    content = ">query\nMKWVTFISLLFL\n>hit\nMaaKWVTFISLLFL\n"
    query, rows = parse_a3m(content)
    assert len(query) == 12
    assert [len(r) for r in rows] == [12, 12]
    assert rows[1] == "MKWVTFISLLFL"


def test_parse_keeps_gap_character_and_encodes_to_31():
    content = ">query\nMKWVTFISLLFL\n>hit\nMKW-TFISLLFL\n"
    _, rows = parse_a3m(content)
    tokens, gap_mask = encode_rows(rows)
    assert tokens[1, 3] == GAP["-"]
    assert gap_mask[1, 3]
    assert not gap_mask[0].any()


def test_encode_aa_order_matches_repo_constants():
    # A..V in one-letter order → the integer map derived from the repo's
    # PRO_STD_RESIDUES (alphabetical by three-letter code).
    row = "ARNDCQEGHILKMFPSTWYV"
    tokens, _ = encode_rows([row])
    expected = [PRO_STD_RESIDUES[PROT_STD_RESIDUES_ONE_TO_THREE[c]] for c in row]
    assert tokens[0].tolist() == expected
    # Sanity: the first 20 positions are a permutation of 0..19.
    assert sorted(expected) == list(range(20))


@pytest.mark.parametrize("unknown_char", ["X", "B", "Z", "U", "O", "J"])
def test_encode_non_canonical_chars_map_to_unk(unknown_char):
    row = f"M{unknown_char}W"
    tokens, gap_mask = encode_rows([row])
    assert tokens[0, 1] == PRO_STD_RESIDUES["UNK"]
    assert not gap_mask[0, 1]


def test_parse_raises_on_length_mismatch_after_strip():
    content = (
        ">query\nMKWVTFISLLFL\n"
        ">hit_ok\nMaaKWVTFISLLFL\n"  # strips to length 12
        ">hit_corrupt\nM\n"  # length 1 — mismatch
    )
    with pytest.raises(ValueError, match="length"):
        parse_a3m(content)


def test_parse_raises_on_empty_content():
    with pytest.raises(ValueError):
        parse_a3m("")


def test_parse_raises_when_content_has_no_header():
    with pytest.raises(ValueError, match="header"):
        parse_a3m("MKWVTFISLLFL\n")


def test_parse_raises_on_empty_query():
    with pytest.raises(ValueError, match="empty"):
        parse_a3m(">query\n\n")


def test_fixture_query_short_parses_and_encodes(fixtures_dir: Path):
    content = (fixtures_dir / "msa" / "query_short.a3m").read_text()
    query, rows = parse_a3m(content)
    tokens, gap_mask = encode_rows(rows)
    assert len(rows) == 5
    assert all(len(r) == len(query) for r in rows)
    assert tokens.shape == (5, len(query))
    assert tokens.dtype == np.int8
    # hit_gap (row 2) has a gap in column 3.
    assert gap_mask[2, 3]
    # hit_identical (row 4) matches the query exactly.
    np.testing.assert_array_equal(tokens[4], tokens[0])


def test_fixture_query_shallow_has_three_rows(fixtures_dir: Path):
    content = (fixtures_dir / "msa" / "query_shallow.a3m").read_text()
    _, rows = parse_a3m(content)
    assert len(rows) == 3


def test_fixture_query_empty_is_query_only(fixtures_dir: Path):
    content = (fixtures_dir / "msa" / "query_empty.a3m").read_text()
    _, rows = parse_a3m(content)
    assert len(rows) == 1


def test_fixture_corrupt_raises(fixtures_dir: Path):
    content = (fixtures_dir / "msa" / "corrupt.a3m").read_text()
    with pytest.raises(ValueError):
        parse_a3m(content)


def test_real_a3m_fixtures_parse_cleanly(fixtures_dir: Path):
    """Smoke check on any real non_pairing.a3m supplied by the user.

    Skipped until at least one real fixture is dropped under
    tests/fixtures/msa/real/. Catches classes of parse failure that the
    handcrafted fixtures can't anticipate (long rows, unusual headers,
    Windows line endings, etc.).
    """
    real_dir = fixtures_dir / "msa" / "real"
    paths = sorted(real_dir.glob("*.a3m")) if real_dir.exists() else []
    if not paths:
        pytest.skip(f"no real a3m fixtures under {real_dir}")
    for path in paths:
        query, rows = parse_a3m(path.read_text())
        assert len(query) > 0
        assert all(len(r) == len(query) for r in rows)
        tokens, gap_mask = encode_rows(rows)
        assert tokens.shape == (len(rows), len(query))
        assert tokens.dtype == np.int8
        # Query (row 0) is canonical protein — no gaps.
        assert not gap_mask[0].any()
