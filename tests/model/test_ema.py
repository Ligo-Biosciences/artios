"""EMA update — correctness vs the reference Python-loop formula, and
speedup from the batched ``torch._foreach_lerp_`` implementation.

The production ``EMAWrapper.update`` uses ``_foreach_lerp_`` to update all
shadow parameters in a single fused kernel call instead of a per-parameter
Python loop. These tests pin:

1. Batched output matches the reference ``(1-decay)*param + decay*shadow``
   formula within float32 rounding tolerance.
2. Batched is strictly faster than the old loop. The test prints the
   measured speedup so the regression shows up in CI logs.
"""

from __future__ import annotations

import copy
import time

import pytest
import torch
import torch.nn as nn

from artios.runner.ema import EMAWrapper


def _reference_update(wrapper: EMAWrapper) -> None:
    """Per-parameter Python-loop reference. Mirrors the formula that lived
    in EMAWrapper before the ``_foreach_lerp_`` rewrite."""
    for name, param in wrapper.model.named_parameters():
        if not wrapper._mutable(name):
            continue
        new_avg = (1.0 - wrapper.decay) * param.data + wrapper.decay * wrapper.shadow[
            name
        ]
        wrapper.shadow[name] = new_avg.clone()


def _paired_wrappers(decay: float, seed: int = 0) -> tuple[EMAWrapper, EMAWrapper]:
    """Two wrappers around identically-initialised copies of the same model."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    new_ema = EMAWrapper(copy.deepcopy(model), decay=decay, mutable_param_keywords=[""])
    old_ema = EMAWrapper(copy.deepcopy(model), decay=decay, mutable_param_keywords=[""])
    for (_, p_new), (_, p_old) in zip(
        new_ema.model.named_parameters(), old_ema.model.named_parameters()
    ):
        p_old.data.copy_(p_new.data)
    new_ema.register()
    old_ema.register()
    return new_ema, old_ema


@pytest.mark.parametrize("decay", [0.999, 0.99, 0.9999, 0.5])
def test_update_matches_reference_formula(decay: float) -> None:
    """Batched update must match ``(1-decay)*param + decay*shadow`` across
    50 steps, within float32 rounding tolerance."""
    new_ema, old_ema = _paired_wrappers(decay)
    for _ in range(50):
        # Apply the same random perturbation to both models so their parameter
        # trajectories stay aligned before the shadow updates diverge on math.
        for (_, p_new), (_, p_old) in zip(
            new_ema.model.named_parameters(), old_ema.model.named_parameters()
        ):
            delta = torch.randn_like(p_new.data) * 0.01
            p_new.data += delta
            p_old.data += delta
        new_ema.update()
        _reference_update(old_ema)
    torch.testing.assert_close(new_ema.shadow, old_ema.shadow, rtol=1e-5, atol=1e-6)


def test_update_is_faster_than_python_loop(capsys: pytest.CaptureFixture) -> None:
    """Batched ``_foreach_lerp_`` update must beat the per-param Python loop."""
    big = nn.Sequential(*[nn.Linear(512, 512) for _ in range(20)])
    w_new = EMAWrapper(copy.deepcopy(big), decay=0.999, mutable_param_keywords=[""])
    w_old = EMAWrapper(copy.deepcopy(big), decay=0.999, mutable_param_keywords=[""])
    w_new.register()
    w_old.register()

    # Warmup to stabilise timing.
    for _ in range(5):
        w_new.update()
        _reference_update(w_old)

    n_steps = 200

    t0 = time.perf_counter()
    for _ in range(n_steps):
        w_new.update()
    dt_new = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_steps):
        _reference_update(w_old)
    dt_old = time.perf_counter() - t0

    speedup = dt_old / dt_new
    with capsys.disabled():
        print(
            f"\n  EMA update: {speedup:.2f}x faster "
            f"(loop {dt_old * 1000:.0f} ms, _foreach_lerp_ {dt_new * 1000:.0f} ms, "
            f"{n_steps} steps, 40 params of 512x512)"
        )
    assert speedup > 1.0, f"batched update should be faster; got {speedup:.2f}x"
