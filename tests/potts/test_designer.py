"""Contracts for ``artios.potts.PottsDesigner`` (uses the released weights)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from artios.potts import (
    PottsDesigner,
    batch_contexts,
    load_backbones,
    load_target,
    parse_selection,
)
from artios.potts.designer import _IDX_TO_ONE

FIXTURES = Path(__file__).parents[1] / "fixtures" / "cif"
# Small ensemble and few sweeps keep the tests fast on CPU; the contracts do not
# depend on sampling quality.
FAST = {"ensemble_size": 2, "n_sweeps": 4}


@pytest.fixture(scope="module")
def designer():
    return PottsDesigner(sampling=FAST)


@pytest.fixture(scope="module")
def multi_chain():
    return load_target(FIXTURES / "3iwd.cif")


def test_single_design_matches_production_inference(designer, multi_chain):
    """design(num_sequences=1) is the same computation as ArtiosPotts.forward."""
    fixed = parse_selection("B10-14")
    (design,) = designer.design(
        multi_chain, design_chains=["B"], fixed=fixed, num_sequences=1, seed=3
    )
    context = batch_contexts(
        [multi_chain.build_context(design_chains=["B"], fixed=fixed)]
    ).to(designer.device)
    with torch.no_grad():
        output = designer.model.main_inference_loop(context, seed=3)
    n = len(multi_chain.residues)
    mutable = multi_chain.mutable_mask(design_chains=["B"], fixed=fixed)
    np.testing.assert_array_equal(
        output.energy[0, :n].cpu().numpy()[mutable], design.residue_energy
    )


def test_fixed_chains_and_residues_keep_native_identity(designer, multi_chain):
    native = multi_chain.native_sequences()
    designs = designer.design(
        multi_chain,
        design_chains=["B"],
        fixed=parse_selection("B10-14"),
        num_sequences=3,
        batch_size=2,
        seed=0,
    )
    for d in designs:
        for chain in ("A", "C", "D"):
            assert d.sequences[chain] == native[chain]
            assert set(d.designed[chain]) == {"0"}
        kept = [m == "0" for m in d.designed["B"]]
        assert sum(kept) == 5
        assert all(a == b for a, b, k in zip(d.sequences["B"], native["B"], kept) if k)
        designed = [a for a, k in zip(d.sequences["B"], kept) if not k]
        assert not set(designed) & {"C", "M"}  # excluded by default


def test_design_is_reproducible_per_seed(designer, multi_chain):
    a = designer.design(multi_chain, design_chains=["A"], num_sequences=2, seed=11)
    b = designer.design(multi_chain, design_chains=["A"], num_sequences=2, seed=11)
    c = designer.design(multi_chain, design_chains=["A"], num_sequences=2, seed=12)
    assert [d.sequences for d in a] == [d.sequences for d in b]
    assert [d.sequences for d in a] != [d.sequences for d in c]


def test_score_reproduces_sampler_energies(designer, multi_chain):
    """Scoring a design recovers the per-residue energies the sampler reported."""
    (design,) = designer.design(multi_chain, design_chains=["A"], seed=5)
    (score,) = designer.score(multi_chain, [design.sequences], seed=5)
    designed = np.array([m == "1" for c in design.designed for m in design.designed[c]])
    np.testing.assert_allclose(
        np.array(score.residue_energy)[designed],
        design.residue_energy,
        rtol=1e-5,
        atol=1e-4,
    )
    assert score.pseudo_log_likelihood < 0


def test_score_rejects_mismatched_sequences(designer, multi_chain):
    native = multi_chain.native_sequences()
    with pytest.raises(ValueError, match="mapping"):
        designer.score(multi_chain, [native["A"]])
    with pytest.raises(ValueError, match="residues"):
        designer.score(multi_chain, [{**native, "A": native["A"][:-1]}])
    with pytest.raises(ValueError, match="Unsupported"):
        designer.score(multi_chain, [{**native, "A": "B" * len(native["A"])}])


def test_backbone_npz_matches_the_runner_dataset(designer):
    """ArtiosBackbone NPZ targets give the runner's (MPNNDataset) result exactly."""
    from artios.data.infer_data_pipeline import MPNNDataset
    from artios.data.parsing.json import parse_input_dict

    npz = Path(__file__).parents[2] / "examples" / "sequence_design" / "backbone.npz"
    dataset = MPNNDataset(
        parse_input_dict(
            {"name": "bb", "task": "sequence_design", "spec": {"npz_file": str(npz)}}
        )
    )
    targets = load_backbones(npz)
    assert len(targets) == len(dataset)
    for i in (0, len(targets) - 1):
        with torch.no_grad():
            runner = designer.model.main_inference_loop(
                batch_contexts([dataset[i]]).to(designer.device), seed=4
            )
        (design,) = designer.design(targets[i], seed=4)
        n = len(targets[i].residues)
        mutable = targets[i].mutable_mask()
        assert mutable.sum() == len(design.sequences["0"])  # motif atoms excluded
        np.testing.assert_array_equal(
            runner.energy[0, :n].cpu().numpy()[mutable], design.residue_energy
        )
        sampled = runner.sampled_sequence[0, :n].argmax(-1).cpu().numpy()[mutable]
        assert "".join(_IDX_TO_ONE[int(s)] for s in sampled) == design.sequences["0"]
