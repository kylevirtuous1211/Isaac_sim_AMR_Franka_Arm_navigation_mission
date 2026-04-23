"""Arm-mount tests.

Verifies that Franka is rigidly attached to Nova Carter's chassis:
  - After bootstrap, Franka XY ≈ AMR XY and Franka Z ≈ AMR Z + offset.Z
  - After the AMR drives somewhere, Franka has followed it
  - Mount config is actually active (`mount_to` set in config.yaml)
"""
from __future__ import annotations

from .conftest import run_and_read, run_in_isaac, sync_to_container


# Tolerance for "Franka directly above AMR"
_XY_TOL = 0.10   # m, generous — mount offset is applied in local frame so
                 # a rotated chassis shifts the XY a bit
_Z_TOL = 0.05    # m


def test_mount_is_configured(ensure_bootstrap):
    """Sanity: config.yaml must have `mount_to` set for mobile-manip
    mode. If someone reverts to the 'arm station' fallback, this test
    flags it."""
    p = run_and_read("tests/scripts/probe.py", "probe")
    assert p["ok"] and p["ready"]
    mount_to = p["config"]["mount_to"]
    assert mount_to and "NovaCarter" in mount_to, \
        f"Expected mount_to to reference Nova Carter, got {mount_to!r}"


def test_franka_sits_above_amr_at_start(ensure_bootstrap):
    """Right after bootstrap: Franka should be at (AMR.x, AMR.y, AMR.z + offset_z)."""
    sync_to_container("apps/bootstrap.py")
    run_in_isaac("apps/bootstrap.py", timeout=60.0)

    p = run_and_read("tests/scripts/probe.py", "probe")
    assert p["ok"] and p["ready"]

    amr = p["amr"]["pos"]
    fra = p["franka"]["pos"]
    offset_z = p["config"]["mount_local_offset"][2]

    assert abs(fra[0] - amr[0]) < _XY_TOL, \
        f"Franka X drifted from AMR X: AMR={amr[0]:.3f}, Franka={fra[0]:.3f}"
    assert abs(fra[1] - amr[1]) < _XY_TOL, \
        f"Franka Y drifted from AMR Y: AMR={amr[1]:.3f}, Franka={fra[1]:.3f}"
    assert abs(fra[2] - (amr[2] + offset_z)) < _Z_TOL, \
        f"Franka Z off: AMR.z={amr[2]:.3f}, offset={offset_z}, Franka.z={fra[2]:.3f}"


def test_franka_follows_amr_after_driving(ensure_bootstrap):
    """After the AMR drives somewhere, Franka must still be directly
    above it. This is the real mobile-manipulation contract — the
    pose-sync callback tracks the chassis every tick."""
    # Reset first so we start from a known state
    sync_to_container("apps/bootstrap.py")
    run_in_isaac("apps/bootstrap.py", timeout=60.0)

    # Drive briefly
    drive = run_and_read(
        "tests/scripts/drive_briefly.py", "drive",
        exec_timeout=180.0, read_timeout=120.0,
    )
    assert drive["ok"]

    # Immediately probe — Franka should be at the AMR's new location
    p = run_and_read("tests/scripts/probe.py", "probe")
    assert p["ok"]
    amr = p["amr"]["pos"]
    fra = p["franka"]["pos"]
    offset_z = p["config"]["mount_local_offset"][2]

    # AMR must have actually moved (sanity)
    assert abs(amr[0]) + abs(amr[1]) > 0.5, \
        f"AMR didn't move ({amr}) — can't verify mount-during-motion"

    # Franka tracked the motion
    assert abs(fra[0] - amr[0]) < _XY_TOL, \
        f"After driving, Franka X={fra[0]:.3f} but AMR X={amr[0]:.3f}"
    assert abs(fra[1] - amr[1]) < _XY_TOL, \
        f"After driving, Franka Y={fra[1]:.3f} but AMR Y={amr[1]:.3f}"
    assert abs(fra[2] - (amr[2] + offset_z)) < _Z_TOL, \
        f"After driving, Franka Z={fra[2]:.3f} but expected {amr[2] + offset_z:.3f}"
