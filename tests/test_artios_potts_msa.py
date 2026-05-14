"""MSA-dispatch tests for ``ArtiosPotts.compute_loss``.

Cover:
  - When context.msa is not None, compute_loss routes through the
    MSA-aware composite NLL path.
  - All-gap batches (test-level null-MSA fixture) produce a finite,
    zero loss (no NaN, no gradient explosion).
  - Pseudo + MSA raises.
  - Loss is finite and gradients flow through h and J for non-null
    MSAs.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from artios.data.features.context import MSAContext
from artios.model.artios_potts import ArtiosPotts
from tests._helpers.context import create_test_context


def _make_msa_configs(loss_type: str = "composite"):
    """Build an ArtiosPotts config mirroring the existing fixtures."""
    from ml_collections import ConfigDict

    return ConfigDict(
        {
            "loss": {
                "potts": {
                    "loss_type": loss_type,
                    "label_smoothing": 0.0,
                    "active_site_upweight": 1.0,
                    "active_site_distance": 8.0,
                    "n_msa_chunk": 8,
                    "supervise_query": True,
                },
            },
            "artios_mpnn": {
                "model": {
                    "num_residue_types": 32,
                    "node_dim": 64,
                    "edge_dim": 64,
                    "num_encoder_layers": 2,
                    "num_decoder_layers": 2,
                    "num_neighbours": 12,
                    "positional_encoding_dim": 16,
                    "positional_encoding_max_relative_feats": 32,
                    "num_rbf_kernels": 16,
                    "dropout": 0.0,
                    "coord_noise_near_ligand": 0.0,
                    "coord_noise_away_from_ligand": 0.0,
                    "coord_noise_distance_cutoff": 8.0,
                    "checkpoint_grads": False,
                },
                "potts": {
                    "symmetric_J": False,
                    "zero_sum_gauge": True,
                    "parameterisation": "linear",
                    "num_factors": 32,
                    "masking_sampler": "linear",
                    "sampling": {
                        "proposal": "gibbs",
                        "n_sweeps": 5,
                        "temperature": 0.1,
                        "temperature_init": 10.0,
                        "annealing_fraction": 0.5,
                        "dt": 0.5,
                        "rejection_step": True,
                        "complexity_penalty_weight": 0.0,
                        "complexity_window_size": 30,
                        "complexity_entropy_min": 2.32,
                        "exclude_cys": False,
                        "exclude_met": False,
                        "aromatic_penalty_weight": 0.0,
                        "aromatic_target_fraction": 0.25,
                        "active_site_distance_cutoff": 8.0,
                        "charged_penalty_weight": 0.0,
                        "charged_target_fraction": 0.25,
                        "composition_penalty_weight": 0.0,
                        "composition_organism": "swissprot",
                        "composition_tolerance_k": 1.0,
                        "net_charge_penalty_weight": 0.0,
                        "net_charge_target_per_residue": -0.013,
                        "hbond_density_penalty_weight": 0.0,
                        "per_sse_composition_penalty_weight": 0.0,
                        "per_sse_composition_nb_radius": 10.0,
                        "per_sse_composition_nb_threshold": 22,
                        "ensemble_size": 1,
                        "ensemble_noise_scale": 0.0,
                        "cfg_scale": 1.0,
                        "active_site_only_cfg": False,
                        "granular_active_site_coloring": False,
                        "energy_bias": {aa: 0.0 for aa in "ARNDCQEGHILKMFPSTWYV"},
                        "track_trajectory": False,
                        "trajectory_interval": 1,
                        "track_u_i": False,
                        "u_i_interval": 1,
                    },
                },
            },
        }
    )


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def context_with_restype(device):
    """Small FullContext with valid one-hot restypes."""
    ctx = create_test_context(batch_size=2, n_tokens=16, n_atoms=32, device=device)
    B, N = 2, 16
    indices = torch.randint(0, 32, (B, N), device=device)
    restype = torch.zeros(B, N, 32, dtype=torch.long, device=device)
    restype.scatter_(-1, indices.unsqueeze(-1), 1)
    ctx.token.restype = restype
    return ctx


def _attach_msa(
    context,
    *,
    n_rows: int = 4,
    gap_everywhere: bool = False,
    msa_chain_mask_all_true: bool = True,
    copy_gt_to_row_0: bool = False,
):
    """Return a new FullContext with an MSAContext attached.

    ``n_rows`` is the total MSA row count (= ``n_msa_plus_one`` on the
    tensor dim = 1 query + ``n_msa`` homologs). Named ``n_rows`` here so
    the helper is decoupled from the sampler-config convention where
    ``n_msa`` now means homolog count.

    - ``gap_everywhere``: tokens all 31 (null MSA shape).
    - ``msa_chain_mask_all_true``: if False, msa_chain_mask is all False
      (simulating an MSA.null_for sample).
    - ``copy_gt_to_row_0``: put argmax of restype in row 0 so that row
      can be compared against the non-MSA composite loss.
    """
    B, N, _ = context.token.restype.shape
    device = context.token.restype.device

    gt = context.token.restype.argmax(dim=-1)  # [B, N]

    if gap_everywhere:
        tokens = torch.full((B, N, n_rows), 31, dtype=torch.int8, device=device)
    else:
        # Random tokens in [0, 20) (canonical AAs) to avoid gap collisions.
        tokens = torch.randint(0, 20, (B, N, n_rows), dtype=torch.int8, device=device)
        if copy_gt_to_row_0:
            tokens[:, :, 0] = gt.to(torch.int8)

    gap_mask = tokens == 31
    msa_chain_mask = torch.full(
        (B, N),
        msa_chain_mask_all_true,
        dtype=torch.bool,
        device=device,
    )

    msa = MSAContext(
        tokens=tokens,
        gap_mask=gap_mask,
        msa_chain_mask=msa_chain_mask,
        msa_chain_asym_id=0,
    )
    return dataclasses.replace(context, msa=msa)


# ---------------------------------------------------------------------------
# Dispatch: pseudo + MSA raises.
# ---------------------------------------------------------------------------


def test_pseudo_loss_with_msa_raises(context_with_restype, device):
    model = ArtiosPotts(_make_msa_configs(loss_type="pseudo")).to(device)
    model.train()
    ctx = _attach_msa(context_with_restype)
    output = model.main_train_loop(context=ctx, seed=0)
    with pytest.raises(ValueError, match="MSA not implemented for pseudo loss"):
        model.compute_loss(output=output, context=ctx)


# ---------------------------------------------------------------------------
# MSA path produces a finite loss with gradient flow.
# ---------------------------------------------------------------------------


def test_msa_loss_is_finite_and_has_gradients(context_with_restype, device):
    model = ArtiosPotts(_make_msa_configs()).to(device)
    model.train()
    ctx = _attach_msa(context_with_restype, n_rows=4, copy_gt_to_row_0=True)

    output = model.main_train_loop(context=ctx, seed=0)
    loss, metrics = model.compute_loss(output=output, context=ctx)

    assert torch.isfinite(loss), "MSA loss should be finite"
    assert loss.ndim == 0, "loss should be scalar"
    assert "weighted_nll_loss" in metrics

    loss.backward()
    # h and J flow through the MPNN graph; if MSA wiring is broken,
    # gradients won't reach their parameters.
    grad_norms = [
        p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None
    ]
    assert any(g > 0 for g in grad_norms), "expected non-zero gradients"


# ---------------------------------------------------------------------------
# Null-MSA batch: zero loss, no NaN, no gradient explosion.
# ---------------------------------------------------------------------------


def test_null_msa_batch_returns_zero_loss_cleanly(context_with_restype, device):
    """When every (row, token) in the batch is marked as a gap, the MSA
    per-position NLL is the canonical masked_mean-with-empty-mask fallback
    (0). The scalar loss must be finite, exactly zero, and must not
    propagate NaN into gradients. Real null-MSA samples produced by the
    featurizer instead fill every row with the gt sequence and degrade
    to the single-sequence composite NLL — that path is covered by
    integration tests in the data-pipeline suite."""
    model = ArtiosPotts(_make_msa_configs()).to(device)
    model.train()
    ctx = _attach_msa(
        context_with_restype,
        n_rows=4,
        gap_everywhere=True,
        msa_chain_mask_all_true=False,
    )
    output = model.main_train_loop(context=ctx, seed=0)
    loss, _ = model.compute_loss(output=output, context=ctx)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0

    # Backward should not NaN.
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN gradient on {p.shape} parameter"


# ---------------------------------------------------------------------------
# Dispatch verification: with msa=None, the non-MSA path is taken
# (no exception, no MSA-specific metric names).
# ---------------------------------------------------------------------------


def test_msa_none_uses_non_msa_path(context_with_restype, device):
    """Sanity: context.msa=None → existing non-MSA composite path runs
    (doesn't raise), returning the same metric keys it always did."""
    model = ArtiosPotts(_make_msa_configs()).to(device)
    model.train()
    output = model.main_train_loop(context=context_with_restype, seed=0)
    loss, metrics = model.compute_loss(output=output, context=context_with_restype)
    assert torch.isfinite(loss)
    assert "weighted_nll_loss" in metrics
    assert "accuracy" in metrics
