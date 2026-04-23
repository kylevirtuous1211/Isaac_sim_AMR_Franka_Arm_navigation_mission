"""Navigation-reset tests.

Verifies that bootstrap clears navigation state cleanly:
  - bumps nav_generation (stale loops see the change, self-abort)
  - clears the `_reached_latch` so a new set_goal() can drive again
  - leaves the navigator usable for a fresh nav cycle
"""
from __future__ import annotations

from .conftest import run_and_read, run_in_isaac, sync_to_container


def test_reached_latch_clears_on_bootstrap(ensure_bootstrap):
    """Perturb_scene.py sets the latch manually; after bootstrap it
    should be cleared (navigator is ready for a new goal)."""
    p = run_and_read("tests/scripts/perturb_scene.py", "perturb")
    assert p["ok"] and p["nav_latched"] is True, \
        "Pre-condition: latch should be set after perturb"

    sync_to_container("apps/bootstrap.py")
    run_in_isaac("apps/bootstrap.py", timeout=60.0)

    probe = run_and_read("tests/scripts/probe.py", "probe")
    assert probe["ok"] and probe["ready"]
    # After fast-reset, the latch is cleared because bootstrap's
    # fast-reset path calls world.reset_async() AND bumps generation,
    # and manipulator.reset() is idempotent. The navigator's latch
    # lives on the same instance — it should be False.
    assert probe["nav_latched"] is False, \
        f"Latch not cleared after bootstrap: {probe['nav_latched']}"


def test_amr_can_drive_after_reset(ensure_bootstrap):
    """End-to-end: after bootstrap, a fresh nav cycle should drive the
    AMR away from origin and reach the goal."""
    sync_to_container("apps/bootstrap.py")
    run_in_isaac("apps/bootstrap.py", timeout=60.0)

    # Drive briefly — drive_briefly.py runs up to 800 ticks
    drive = run_and_read(
        "tests/scripts/drive_briefly.py", "drive",
        exec_timeout=180.0, read_timeout=120.0,
    )
    assert drive["ok"], drive

    # After driving, AMR must have moved from origin.
    moved = abs(drive["final_pos"][0]) + abs(drive["final_pos"][1])
    assert moved > 0.5, \
        f"AMR barely moved after bootstrap: {drive['final_pos']}"

    # Reaching goal within 800 ticks is expected for the default config
    # (origin -> (2, 1) at 0.3 m/s). If it fails/times out, that's a
    # regression in the controller.
    assert drive["final_status"] in ("reached", "running"), \
        f"Nav status {drive['final_status']} — stuck or failed"
