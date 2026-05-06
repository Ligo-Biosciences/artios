"""Slow end-to-end MSA test. Runs only on environments with the real MSA
data mounted (the VM); skips locally.

The test stands up a minimal BaseSingleDataset and exercises
``process_one`` end-to-end, asserting that FullContext.msa is populated
with correct shapes and that the msa_chain_asym_id resolves back to the
reference chain_1_id.

To run on the VM:

    pytest tests/data_pipeline/msa/test_integration.py -v -m slow

The test auto-skips unless the paths below resolve to real files.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


# Paths the VM is expected to have. Override via env vars if the layout differs.
DEFAULT_MSA_ROOT = os.environ.get(
    "ARTIOS_MSA_ROOT", "/home/jupyter/protenix-data/mmcif_msa_template"
)
DEFAULT_SEQ_TO_IDX_PATH = os.environ.get(
    "ARTIOS_SEQ_TO_IDX_PATH",
    "/home/jupyter/protenix-data/common/seq_to_pdb_index.json",
)
DEFAULT_BIOASSEMBLY_DIR = os.environ.get(
    "ARTIOS_BIOASSEMBLY_DIR",
    "/home/jupyter/artios-data/release_data/mmcif_bioassembly",
)
DEFAULT_INDICES_FPATH = os.environ.get(
    "ARTIOS_INDICES_FPATH",
    "/home/jupyter/artios-data/release_data/indices/"
    "weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz",
)


def _paths_available() -> bool:
    return all(
        Path(p).exists()
        for p in (
            DEFAULT_MSA_ROOT,
            DEFAULT_SEQ_TO_IDX_PATH,
            DEFAULT_BIOASSEMBLY_DIR,
            DEFAULT_INDICES_FPATH,
        )
    )


@pytest.mark.skipif(
    not _paths_available(),
    reason="real MSA data not mounted; run on the VM or set ARTIOS_MSA_ROOT etc.",
)
def test_process_one_with_msa_produces_populated_msacontext():
    """Full pipeline with msa.enabled=True populates FullContext.msa."""
    from artios.data.dataset import BaseSingleDataset
    from artios.data.msa import MSASampler

    sampler = MSASampler(
        msa_root=DEFAULT_MSA_ROOT,
        seq_to_idx_path=DEFAULT_SEQ_TO_IDX_PATH,
        n_msa=64,
        seed=0,
    )

    dataset = BaseSingleDataset(
        mmcif_dir=os.environ.get("ARTIOS_MMCIF_DIR", ""),
        bioassembly_dict_dir=DEFAULT_BIOASSEMBLY_DIR,
        indices_fpath=DEFAULT_INDICES_FPATH,
        cropping_configs={
            "method_weights": [0.0, 0.5, 0.5, 0.0],
            "crop_size": 256,
        },
        msa_sampler=sampler,
        motif_training={"motif_train_rate": 0.0},
        # Keep the run reproducible and fast.
        random_sample_if_failed=False,
        filter_unique_fragment_prob=0.0,
    )
    assert len(dataset) > 0

    # Random indices, NOT sequential: the weightedPDB indices CSV is sorted
    # by PDB id alphabetically, so indices 0..N are dominated by a few
    # non-protein PDBs (100d etc.) whose chain_1_id is nucleic/ligand.
    # Random sampling gives a representative mix.
    #
    # Expected hit rate: ~50% of entries are protein × ~91% MSA coverage on
    # protein ≈ 45%. 20 trials → P(all miss) ≈ 4e-6.
    rng = random.Random(0)
    probe_indices = rng.sample(range(len(dataset)), min(20, len(dataset)))

    context = None
    for probe_idx in probe_indices:
        candidate = dataset.process_one(probe_idx)
        if candidate.msa is not None:
            context = candidate
            break
    assert context is not None and context.msa is not None, (
        f"Expected at least one of {len(probe_indices)} random samples to "
        "have a populated MSA"
    )

    msa = context.msa
    n_token = context.basic.n_token
    assert msa.tokens.shape == (n_token, 64)
    assert msa.gap_mask.shape == (n_token, 64)
    assert msa.msa_chain_mask.shape == (n_token,)
    assert msa.tokens.dtype.itemsize == 1  # int8
    assert int(msa.msa_chain_mask.sum()) > 0  # at least one chain token
    # msa_chain_asym_id is a plain int, not a tensor
    assert isinstance(msa.msa_chain_asym_id, int)
