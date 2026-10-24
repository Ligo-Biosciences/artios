"""Contracts for the ``artios`` command's design, score and params subcommands."""

import json
from pathlib import Path

import numpy as np
import pytest

from artios.cli import main

FIXTURES = Path(__file__).parents[1] / "fixtures" / "cif"
BACKBONES = Path(__file__).parents[2] / "examples" / "sequence_design" / "backbone.npz"
# Small ensemble and few sweeps keep the tests fast on CPU.
FAST = ["--set", "ensemble_size=2", "--set", "n_sweeps=4", "--device", "cpu"]
DESIGN = [
    "design",
    str(FIXTURES / "3iwd.cif"),
    "--design-chains",
    "B",
    "--fixed",
    "B10-14",
    "-n",
    "3",
    "--batch-size",
    "2",
    *FAST,
]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def design_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("design")
    assert main([*DESIGN, "-o", str(out)]) == 0
    return out


def test_design_writes_designed_chains_and_full_records(design_dir):
    records = _jsonl(design_dir / "designs.jsonl")
    fasta = (design_dir / "designs.fasta").read_text().splitlines()
    assert [r["id"] for r in records] == ["3iwd_0", "3iwd_1", "3iwd_2"]
    assert fasta[0].startswith(">3iwd_0 target=3iwd seed=0 ")
    assert "chains=B" in fasta[0]
    assert fasta[1] == records[0]["sequences"]["B"]  # only the designed chain
    assert set(records[0]["sequences"]) == {"A", "B", "C", "D"}
    assert records[0]["designed"]["B"].count("0") == 5


def test_score_round_trips_design_energies(design_dir, tmp_path):
    """Scoring designs.fasta recovers the energies the sampler reported."""
    assert main(
        ["score", str(FIXTURES / "3iwd.cif"), str(design_dir / "designs.fasta"),
         "-o", str(tmp_path), *FAST]
    ) == 0  # fmt: skip
    designs = _jsonl(design_dir / "designs.jsonl")
    scores = _jsonl(tmp_path / "scores.jsonl")
    assert [s["id"] for s in scores] == [d["id"] for d in designs]
    for d, s in zip(designs, scores):
        mask = np.array([m == "1" for c in d["designed"] for m in d["designed"][c]])
        np.testing.assert_allclose(
            np.array(s["residue_energy"])[mask],
            d["residue_energy"],
            rtol=1e-5,
            atol=1e-4,
        )


def test_rerun_resumes_and_refuses_changed_settings(design_dir, tmp_path):
    out = tmp_path / "copy"
    out.mkdir()
    for name in ("designs.jsonl", "designs.fasta", "settings.json"):
        (out / name).write_text((design_dir / name).read_text())
    before = (out / "designs.jsonl").read_text()

    assert main([*DESIGN, "-o", str(out)]) == 0
    assert (out / "designs.jsonl").read_text() == before

    assert main([*DESIGN, "-n", "1", "-o", str(out)]) == 1
    assert main([*DESIGN, "-n", "1", "-o", str(out), "--overwrite"]) == 0
    assert len(_jsonl(out / "designs.jsonl")) == 1


def test_backbone_npz_designs_the_design_chain_only(tmp_path):
    assert main(["design", str(BACKBONES), "-n", "1", "-o", str(tmp_path), *FAST]) == 0
    records = _jsonl(tmp_path / "designs.jsonl")
    assert [r["id"] for r in records] == [f"backbone_{i}_0" for i in range(10)]
    for r in records:
        assert list(r["sequences"]) == ["0"]  # motif residues are context
        assert set(r["designed"]["0"]) == {"1"}
        assert r["native_identity"] is None
    header = (tmp_path / "designs.fasta").read_text().splitlines()[0]
    assert "chains=0" in header and "native_identity" not in header


def test_params_reproduce_the_native_score(tmp_path):
    """h, J and the graph in params NPZ give the energy `artios score` reports."""
    structure = str(FIXTURES / "1m2g.cif")
    assert main(["params", structure, "-o", str(tmp_path), *FAST]) == 0
    assert main(["score", structure, "-o", str(tmp_path), *FAST]) == 0
    p = np.load(tmp_path / "1m2g.npz")
    h, J, nbr, mask, s = p["h"], p["J"], p["edge_index"], p["edge_mask"], p["native"]
    n = np.arange(len(s))
    pair = J[n[:, None], np.arange(nbr.shape[1])[None], s[:, None], s[nbr]]
    per_token = h[n, s] + 0.5 * (pair * mask).sum(-1)
    protein = p["labels"] != ""
    (score,) = _jsonl(tmp_path / "scores.jsonl")
    np.testing.assert_allclose(per_token[protein].sum(), score["energy"], rtol=1e-5)


def test_bad_sequences_fail_with_a_message(tmp_path, capsys):
    fasta = tmp_path / "bad.fasta"
    fasta.write_text(">x chains=A:B\nAAAA\n")
    code = main(
        ["score", str(FIXTURES / "3iwd.cif"), str(fasta), "-o", str(tmp_path), *FAST]
    )
    assert code == 2
    assert "':'-separated chains" in capsys.readouterr().err
