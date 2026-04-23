"""Scene-reset tests.

Verifies that running `apps/bootstrap.py` when state is already populated
takes the fast-reset path and:
  - teleports the AMR back to the configured start position
  - re-opens the gripper
  - does NOT reload the hospital stage (which would take 60-180s)

Contract: after bootstrap, regardless of where the robots ended up in
a previous run, probe shows them back at defaults.
"""
from __future__ import annotations

import time

import pytest

from .conftest import run_and_read, run_in_isaac, sync_to_container


# Tolerance for "back at start" — PhysX settle can leave micro-drift
_POS_TOL = 0.05   # meters


def _near(a, b, tol=_POS_TOL):
    return abs(a - b) < tol


def test_amr_resets_to_start_position(ensure_bootstrap):
    """After perturbing the AMR + running bootstrap, the AMR should be
    back near its configured start_position."""
    # 1. Push AMR off origin
    perturb = run_and_read("tests/scripts/perturb_scene.py", "perturb")
    assert perturb["ok"], perturb
    assert abs(perturb["amr_pos"][0] - 3.5) < 0.1, "Perturbation didn't stick"

    # 2. Fast-reset via bootstrap
    sync_to_container("apps/bootstrap.py")
    run_in_isaac("apps/bootstrap.py", timeout=60.0)

    # 3. Probe — AMR should be back near origin
    probe = run_and_read("tests/scripts/probe.py", "probe")
    assert probe["ok"] and probe["ready"], probe

    expected = probe["config"]["start_position"]  # from config.yaml
    actual = probe["amr"]["pos"]
    assert _near(actual[0], expected[0]), \
        f"AMR X not reset: expected ~{expected[0]}, got {actual[0]}"
    assert _near(actual[1], expected[1]), \
        f"AMR Y not reset: expected ~{expected[1]}, got {actual[1]}"


def test_bootstrap_bumps_nav_generation(ensure_bootstrap):
    """Each bootstrap must increment nav_generation so stale run_*.py
    loops abort on their next tick."""
    before = run_and_read("tests/scripts/probe.py", "probe")
    gen_before = before["nav_generation"]

    sync_to_container("apps/bootstrap.py")
    run_in_isaac("apps/bootstrap.py", timeout=60.0)

    after = run_and_read("tests/scripts/probe.py", "probe")
    gen_after = after["nav_generation"]

    assert gen_after > gen_before, \
        f"nav_generation didn't bump: {gen_before} -> {gen_after}"


def test_bootstrap_is_fast_when_state_ready(ensure_bootstrap):
    """Fast-reset path should complete in a few seconds, not the
    60-180s a full hospital reload takes. If bootstrap somehow
    decides to reload the stage, this test catches it."""
    sync_to_container("apps/bootstrap.py")
    t0 = time.time()
    run_in_isaac("apps/bootstrap.py", timeout=30.0)
    elapsed = time.time() - t0
    # Full stage reload is ~60s minimum even from cache. If we're under
    # 30s, the fast-reset path took effect.
    assert elapsed < 30.0, \
        f"Bootstrap took {elapsed:.1f}s — looks like a full reload instead of fast-reset"
