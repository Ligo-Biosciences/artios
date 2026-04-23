"""Unit tests for MSASampler (artios/data/msa/sampler.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from artios.data.msa import MSA, MSAQueryInfo, MSASampler

# ---------------------------------------------------------------------------
# helpers — build a tiny on-disk MSA root inside pytest's tmp_path
# ---------------------------------------------------------------------------


def _write_a3m(a3m_path: Path, query: str, n_rows: int) -> None:
    """Create an a3m with the query followed by (n_rows - 1) identical homologs."""
    a3m_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [">query", query]
    for i in range(n_rows - 1):
        lines.append(f">hit_{i}")
        lines.append(query)
    a3m_path.write_text("\n".join(lines) + "\n")


def _setup_msa_root(
    tmp_path: Path, sequence: str, idx: int = 0, n_rows: int = 10
) -> Path:
    msa_root = tmp_path / "msa_root"
    _write_a3m(msa_root / str(idx) / "non_pairing.a3m", sequence, n_rows)
    return msa_root


def _write_seq_to_idx(tmp_path: Path, mapping: dict[str, int]) -> Path:
    path = tmp_path / "seq_to_idx.json"
    path.write_text(json.dumps(mapping))
    return path


@pytest.fixture
def sample_sampler_setup(load_structure, tmp_path):
    """Structure, pre-crop query_info, and a helper to build a sampler.

    Returns (structure, query_info, build_sampler) where build_sampler takes
    a3m_n_rows and optional n_msa/seed/sequence/index overrides.
    """
    structure = load_structure("2g1k")
    query_info = MSAQueryInfo.from_pre_crop_structure(structure, "A")
    assert query_info is not None

    def build(
        *,
        a3m_n_rows: int = 10,
        n_msa: int = 4,
        seed: int | None = None,
        index_sequence: str | None = None,
        a3m_sequence: str | None = None,
        idx: int = 0,
    ) -> MSASampler:
        """Write a3m + seq_to_idx keyed on index_sequence (defaults to
        query_info.sequence) and point an MSASampler at them."""
        index_sequence = index_sequence or query_info.sequence
        a3m_sequence = a3m_sequence or query_info.sequence
        _write_a3m(
            tmp_path / "msa_root" / str(idx) / "non_pairing.a3m",
            a3m_sequence,
            a3m_n_rows,
        )
        seq_to_idx_path = _write_seq_to_idx(tmp_path, {index_sequence: idx})
        return MSASampler(
            msa_root=tmp_path / "msa_root",
            seq_to_idx_path=seq_to_idx_path,
            n_msa=n_msa,
            seed=seed,
        )

    return structure, query_info, build


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_missing_chain_returns_none(sample_sampler_setup):
    structure, _, build = sample_sampler_setup
    sampler = build(a3m_n_rows=6, n_msa=3)
    bogus_info = MSAQueryInfo(
        chain_id="ZZZ_BOGUS",
        sequence="AAAA",
        res_ids=np.arange(4, dtype=np.int64),
    )
    assert sampler.sample(structure, bogus_info) is None


def test_sequence_not_in_index_returns_none(sample_sampler_setup):
    structure, info, build = sample_sampler_setup
    # Index a different sequence — info.sequence will miss.
    sampler = build(
        a3m_n_rows=6, n_msa=3, index_sequence="MKDUMMY", a3m_sequence="MKDUMMY"
    )
    assert sampler.sample(structure, info) is None


def test_depth_below_n_msa_returns_none(sample_sampler_setup):
    structure, info, build = sample_sampler_setup
    # a3m has 4 rows; sampler wants 10. Should drop.
    sampler = build(a3m_n_rows=4, n_msa=10)
    assert sampler.sample(structure, info) is None


def test_success_shapes_and_query_at_row_0(sample_sampler_setup):
    structure, info, build = sample_sampler_setup
    sampler = build(a3m_n_rows=10, n_msa=4, seed=0)
    msa = sampler.sample(structure, info)

    assert msa is not None
    l_chain = len(info.sequence)
    assert msa.tokens.shape == (4, l_chain)
    assert msa.gap_mask.shape == (4, l_chain)
    assert msa.chain_token_indices.shape == (l_chain,)
    assert msa.tokens.dtype == np.int8

    # Row 0 is the query; homologs are identical in this fixture.
    for row in range(1, 4):
        np.testing.assert_array_equal(msa.tokens[row], msa.tokens[0])

    # msa_chain_asym_id resolves to the unique asym_id_int of chain A in the
    # fixture.
    chain_a_mask = structure.atoms.chain_id == "A"
    expected_asym = int(np.unique(structure.entity_ids.asym_id_int[chain_a_mask])[0])
    assert msa.msa_chain_asym_id == expected_asym


def test_chain_token_indices_point_to_reference_chain(sample_sampler_setup):
    structure, info, build = sample_sampler_setup
    sampler = build(a3m_n_rows=10, n_msa=3)
    msa = sampler.sample(structure, info)

    assert msa is not None
    centre_atoms = structure.centre_atom_indices
    chains = structure.atoms.chain_id[centre_atoms[msa.chain_token_indices]]
    assert np.all(chains == "A")


def test_deterministic_with_same_seed(sample_sampler_setup, tmp_path):
    structure, info, _ = sample_sampler_setup

    # Build an a3m with DISTINCT homologs so sampling ambiguity is visible.
    l = len(info.sequence)
    query = info.sequence
    distinct_rows = [query] + [
        ("A" * i + "-" + "A" * (l - i - 1)) for i in range(1, 20)
    ]
    a3m_path = tmp_path / "msa_root" / "0" / "non_pairing.a3m"
    a3m_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [line for i, row in enumerate(distinct_rows) for line in (f">r_{i}", row)]
    a3m_path.write_text("\n".join(lines) + "\n")
    seq_to_idx_path = _write_seq_to_idx(tmp_path, {query: 0})

    def run(seed: int) -> MSA:
        sampler = MSASampler(
            msa_root=tmp_path / "msa_root",
            seq_to_idx_path=seq_to_idx_path,
            n_msa=5,
            seed=seed,
        )
        m = sampler.sample(structure, info)
        assert m is not None
        return m

    msa_a = run(seed=42)
    msa_b = run(seed=42)
    msa_c = run(seed=7)

    np.testing.assert_array_equal(msa_a.tokens, msa_b.tokens)
    # Different seeds → different homolog selection (probabilistic but
    # effectively certain with 20 distinct rows and 4 samples).
    assert not np.array_equal(msa_a.tokens, msa_c.tokens)


def test_a3m_query_mismatch_with_index_returns_none(sample_sampler_setup):
    """Stale seq_to_idx entry: index points at an a3m whose row 0 doesn't
    match the sequence we looked up by. Sampler must bail."""
    structure, info, build = sample_sampler_setup
    mismatched = "A" * len(info.sequence)
    sampler = build(
        a3m_n_rows=10,
        n_msa=3,
        index_sequence=info.sequence,
        a3m_sequence=mismatched,
    )
    assert sampler.sample(structure, info) is None


def test_corrupt_a3m_returns_none(sample_sampler_setup, tmp_path):
    """A parse-ValueError on the a3m side is a soft skip."""
    structure, info, _ = sample_sampler_setup

    # Write a corrupt a3m: second row shorter than the query after lowercase strip.
    a3m_path = tmp_path / "msa_root" / "0" / "non_pairing.a3m"
    a3m_path.parent.mkdir(parents=True, exist_ok=True)
    a3m_path.write_text(f">query\n{info.sequence}\n>hit_short\nM\n")
    seq_to_idx_path = _write_seq_to_idx(tmp_path, {info.sequence: 0})

    sampler = MSASampler(
        msa_root=tmp_path / "msa_root",
        seq_to_idx_path=seq_to_idx_path,
        n_msa=3,
    )
    assert sampler.sample(structure, info) is None


def test_missing_a3m_file_returns_none(sample_sampler_setup, tmp_path):
    """Index resolves to an idx whose a3m file does not exist on disk."""
    structure, info, _ = sample_sampler_setup

    seq_to_idx_path = _write_seq_to_idx(tmp_path, {info.sequence: 99})
    # Note: /msa_root/99/non_pairing.a3m does NOT exist.
    msa_root = tmp_path / "msa_root"
    msa_root.mkdir(parents=True, exist_ok=True)

    sampler = MSASampler(msa_root=msa_root, seq_to_idx_path=seq_to_idx_path, n_msa=3)
    assert sampler.sample(structure, info) is None


# ---------------------------------------------------------------------------
# Per-homolog quality filters: min_homolog_identity + max_gap_rate
# ---------------------------------------------------------------------------


def _write_custom_a3m(a3m_path: Path, query: str, homologs: list[str]) -> None:
    """Write an a3m with explicit per-homolog sequences (no synthesized hits)."""
    a3m_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [">query", query]
    for i, h in enumerate(homologs):
        lines.append(f">hit_{i}")
        lines.append(h)
    a3m_path.write_text("\n".join(lines) + "\n")


def _setup_custom(tmp_path: Path, query: str, homologs: list[str]):
    a3m_path = tmp_path / "msa_root" / "0" / "non_pairing.a3m"
    _write_custom_a3m(a3m_path, query, homologs)
    seq_to_idx_path = _write_seq_to_idx(tmp_path, {query: 0})
    return tmp_path / "msa_root", seq_to_idx_path


def test_max_gap_rate_filters_out_gappy_homologs(sample_sampler_setup, tmp_path):
    """Gappy homologs above max_gap_rate are dropped from the subsample pool."""
    structure, info, _ = sample_sampler_setup
    q = info.sequence
    # 5 clean homologs (copy of query = 0 gaps), 5 all-gap homologs.
    homologs = [q] * 5 + ["-" * len(q)] * 5
    msa_root, seq_to_idx_path = _setup_custom(tmp_path, q, homologs)

    sampler = MSASampler(
        msa_root=msa_root,
        seq_to_idx_path=seq_to_idx_path,
        n_msa=6,  # needs 5 homologs
        max_gap_rate=0.5,  # drops the 5 all-gap rows
        seed=0,
    )
    msa = sampler.sample(structure, info)
    assert msa is not None
    # All sampled homolog rows must be below the threshold.
    homolog_gap_frac = (msa.tokens[1:] == 31).mean(axis=1)
    assert (homolog_gap_frac <= 0.5).all()


def test_max_gap_rate_returns_none_when_too_few_survive(sample_sampler_setup, tmp_path):
    """Tight max_gap_rate plus a mostly-gappy a3m → insufficient depth."""
    structure, info, _ = sample_sampler_setup
    q = info.sequence
    # Only 2 clean + 8 all-gap → 2 survive the filter.
    homologs = [q] * 2 + ["-" * len(q)] * 8
    msa_root, seq_to_idx_path = _setup_custom(tmp_path, q, homologs)

    sampler = MSASampler(
        msa_root=msa_root,
        seq_to_idx_path=seq_to_idx_path,
        n_msa=5,  # needs 4 homologs, only 2 survive
        max_gap_rate=0.5,
    )
    assert sampler.sample(structure, info) is None


def test_min_homolog_identity_filters_out_low_identity_rows(
    sample_sampler_setup, tmp_path
):
    """Low-identity homologs below min_homolog_identity are dropped."""
    structure, info, _ = sample_sampler_setup
    q = info.sequence
    # 5 identical-to-query (identity=1.0), 5 all-X (identity≈0 on non-gap).
    homologs = [q] * 5 + ["X" * len(q)] * 5
    msa_root, seq_to_idx_path = _setup_custom(tmp_path, q, homologs)

    sampler = MSASampler(
        msa_root=msa_root,
        seq_to_idx_path=seq_to_idx_path,
        n_msa=6,
        min_homolog_identity=0.5,
        seed=0,
    )
    msa = sampler.sample(structure, info)
    assert msa is not None
    # With only 5 identity=1.0 homologs surviving and n_msa-1=5, the sampler
    # must pick all of them — every sampled row should equal the query row.
    for row in range(1, 6):
        np.testing.assert_array_equal(msa.tokens[row], msa.tokens[0])


def test_min_homolog_identity_returns_none_when_too_few_survive(
    sample_sampler_setup, tmp_path
):
    """Tight min_homolog_identity + mostly-divergent a3m → insufficient depth."""
    structure, info, _ = sample_sampler_setup
    q = info.sequence
    homologs = [q] * 2 + ["X" * len(q)] * 8  # only 2 high-identity
    msa_root, seq_to_idx_path = _setup_custom(tmp_path, q, homologs)

    sampler = MSASampler(
        msa_root=msa_root,
        seq_to_idx_path=seq_to_idx_path,
        n_msa=5,
        min_homolog_identity=0.5,
    )
    assert sampler.sample(structure, info) is None


def test_init_rejects_thresholds_outside_unit_interval():
    with pytest.raises(ValueError, match="min_homolog_identity"):
        MSASampler(msa_root="/x", seq_to_idx_path="/y", min_homolog_identity=-0.1)
    with pytest.raises(ValueError, match="max_gap_rate"):
        MSASampler(msa_root="/x", seq_to_idx_path="/y", max_gap_rate=1.5)
