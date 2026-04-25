"""Golden-parity tests for the inference pipelines in ``infer_data_pipeline.py``.

Pins the tensor output of each inference dataset on a fixed input. Any
divergence names the field that silently broke. See
``tests/integration/CLAUDE.md`` for regeneration ergonomics.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from artios.data.parsing.json import (
    BackboneInput,
    FastaFile,
    NPZFile,
    PackingInput,
    SequenceDesignInput,
)
from artios.utils.seed import seed_everything

from .conftest import assert_full_context_matches, dump_full_context

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
MOTIF_INPUT_DIR = FIXTURES_DIR / "motifs"
GOLDEN_DIR = FIXTURES_DIR / "integration"

# Seed applied before every dataset call so ``ref_pos_augment=True`` produces
# deterministic output. ``random_transform`` uses ``np.random.uniform`` and
# ``scipy.spatial.transform.Rotation.random``; the latter falls back to
# numpy's legacy RandomState singleton when no explicit generator is passed,
# so seeding numpy is sufficient. We use ``seed_everything`` (which also
# seeds ``random`` and torch) for defence in depth — ``array_builder``'s
# ``remove_leaving_atoms`` path calls ``random.sample`` and future featurizer
# changes may pull torch randomness.
RNG_SEED = 0


def _seed() -> None:
    seed_everything(RNG_SEED, deterministic=False)


def _load_backbone_input(json_name: str) -> BackboneInput:
    """Load a BackboneInput from ``tests/fixtures/motifs/`` and absolutise paths.

    The fixture JSONs use relative ``cif_file`` strings (and the human-friendly
    string form, not the ``{"path": ...}`` dict). Normalise both so the test is
    invariant to the working directory.
    """
    src = MOTIF_INPUT_DIR / json_name
    data = json.loads(src.read_text())
    data = copy.deepcopy(data)
    motif = data.get("motif")
    if motif is not None and isinstance(motif.get("cif_file"), str):
        motif["cif_file"] = {"path": str(MOTIF_INPUT_DIR / motif["cif_file"])}
    return BackboneInput(**data)


def _assert_round_trip(case, ctx, regenerate_golden: bool) -> None:
    """Write or assert against the golden for *case*.

    Shared by every dataset class so the pattern stays consistent across
    entry points.
    """
    if regenerate_golden or not case.exists():
        dump_full_context(ctx, case)
        if not regenerate_golden:
            pytest.skip(
                f"wrote new golden at {case}; re-run without "
                "--regenerate-golden to assert parity"
            )
        return
    assert_full_context_matches(ctx, case)


def _assert_deterministic(run_fn) -> None:
    """Assert two seeded runs of *run_fn* produce byte-identical tensors."""
    from .conftest import full_context_to_flat_tensors

    a = full_context_to_flat_tensors(run_fn())
    b = full_context_to_flat_tensors(run_fn())
    assert set(a.keys()) == set(b.keys()), (
        f"dataset produced different field sets across seeded runs: "
        f"{sorted(set(a.keys()) ^ set(b.keys()))}"
    )
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


class TestMotifDatasetParity:
    """Pin ``MotifDataset[0]`` FullContext tensors on the carbonic-anhydrase motif.

    The motif-critical inference path: exercises ``build_motif`` →
    ``add_atom_array_attributes`` → ``AtomArrayTokenizer`` → old
    ``ContextGenerator``. PR 1 of the P7 refactor replaces this with
    ``AnnotatedStructure.annotate()`` → ``.with_unindexed_motif()`` → new
    ``ContextGenerator``; the golden stays the same.
    """

    GOLDEN = GOLDEN_DIR / "backbone_motif_ccd" / "context.npz"

    def _run(self) -> Any:
        from artios.data.infer_data_pipeline import MotifDataset

        backbone_input = _load_backbone_input("motif_input.json")
        _seed()
        return MotifDataset(backbone_input)[0]

    def test_motif_dataset_matches_golden(self, regenerate_golden):
        _assert_round_trip(self.GOLDEN, self._run(), regenerate_golden)

    def test_run_is_deterministic(self):
        """Seed + same input → byte-identical tensors across calls.

        Catches the case where something in the pipeline consumes randomness
        in a path the seed doesn't cover (e.g. a scipy call that bypasses the
        singleton RNG). Without this, a flaky golden would look like a
        legitimate refactor regression.
        """
        _assert_deterministic(self._run)


class TestMPNNDatasetParity:
    """Pin ``MPNNDataset[0]`` FullContext tensors on a real backbone stack.

    Input: a ``batch_0_backbone.npz`` pulled from the ``artios-evaluation``
    modal volume (v2.6 finetune sweep). Exercises the non-motif inference
    featurizer path — no ``build_motif`` involvement, just
    ``AtomArrayTokenizer`` → old ``ContextGenerator`` on an already-annotated
    AtomArrayStack. PR 1 replaces it with ``AnnotatedStructure.from_atom_array``
    → new ``ContextGenerator``.
    """

    NPZ = GOLDEN_DIR / "mpnn" / "backbone.npz"
    GOLDEN = GOLDEN_DIR / "mpnn_no_fix" / "context.npz"

    def _run(self) -> Any:
        from artios.data.infer_data_pipeline import MPNNDataset

        sequence_input = SequenceDesignInput(
            name="mpnn-parity",
            npz_file=NPZFile(path=self.NPZ),
            fixed_residues={},
            backbone_index_map=[0],  # pin to first model in the 100-stack
        )
        _seed()
        return MPNNDataset(sequence_input)[0]

    def test_fixture_present(self):
        """Skip downstream tests with a clear message if the npz is missing."""
        if not self.NPZ.exists():
            pytest.skip(
                f"MPNN fixture {self.NPZ} not present; pull with "
                "`modal volume get artios-evaluation "
                "v2.6-afdb-mgnify-fine-tune-ckpt-sweep/batches/batch_0/"
                "batch_0_backbone.npz " + str(self.NPZ) + "`"
            )

    def test_mpnn_dataset_matches_golden(self, regenerate_golden):
        if not self.NPZ.exists():
            pytest.skip(f"MPNN fixture {self.NPZ} not present")
        _assert_round_trip(self.GOLDEN, self._run(), regenerate_golden)

    def test_run_is_deterministic(self):
        if not self.NPZ.exists():
            pytest.skip(f"MPNN fixture {self.NPZ} not present")
        _assert_deterministic(self._run)


class TestPackingDatasetParity:
    """Pin ``PackingDataset[0]`` FullContext tensors on a real backbone stack.

    Shares the MPNN stack npz but pairs it with a committed deterministic
    FASTA (``sequences.fasta`` — 10 identical ACDE...VWY x8 sequences). The
    pre-refactor golden is generated with the legacy ``packing_input_to_atom_array_v2``
    → ``AtomArrayTokenizer`` → old ``ContextGenerator`` path; post-refactor
    (P7) it must match byte-for-byte.
    """

    NPZ = GOLDEN_DIR / "mpnn" / "backbone.npz"
    FASTA = GOLDEN_DIR / "packing" / "sequences.fasta"
    GOLDEN = GOLDEN_DIR / "packing" / "context.npz"

    def _run(self) -> Any:
        from artios.data.infer_data_pipeline import PackingDataset

        packing_input = PackingInput(
            name="packing-parity",
            npz_file=NPZFile(path=self.NPZ),
            fasta_file=FastaFile(path=self.FASTA),
        )
        _seed()
        return PackingDataset(packing_input)[0]

    def test_fixture_present(self):
        if not self.NPZ.exists():
            pytest.skip(
                f"Packing test reuses MPNN fixture {self.NPZ}; pull with "
                "`modal volume get artios-evaluation "
                "v2.6-afdb-mgnify-fine-tune-ckpt-sweep/batches/batch_0/"
                "batch_0_backbone.npz " + str(self.NPZ) + "`"
            )

    def test_packing_dataset_matches_golden(self, regenerate_golden):
        if not self.NPZ.exists():
            pytest.skip(f"Packing test needs MPNN fixture {self.NPZ}")
        _assert_round_trip(self.GOLDEN, self._run(), regenerate_golden)

    def test_run_is_deterministic(self):
        if not self.NPZ.exists():
            pytest.skip(f"Packing test needs MPNN fixture {self.NPZ}")
        _assert_deterministic(self._run)
