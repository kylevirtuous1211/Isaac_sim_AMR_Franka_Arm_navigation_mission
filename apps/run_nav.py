# ============================================================
# apps/run_nav.py — drive the AMR from current pose to task.point_a
#
# Requires: apps/bootstrap.py has been run first (populates core.state).
# Reuses the live hospital stage + spawned robots — no scene reload.
#
# Run via: python3 run_in_isaac.py midterm_project/apps/run_nav.py
# Log:     cache/isaac-sim/logs/run_nav.log
# ============================================================
import sys
import traceback

sys.path.insert(0, "/workspace/midterm_project")
sys.modules.pop("apps._common", None)
from apps._common import bootstrap_imports, load_config, make_logger  # noqa: E402

bootstrap_imports()
log, _ = make_logger("run_nav")

try:
    import numpy as np
    import omni.kit.app

    from core import state
    from core.navigator import NavStatus

    state.require_ready()
    log(state.summary())

    # Always re-read config.yaml so hot-tunable values reflect the latest file
    # on disk (bootstrap's cached copy is fine for structure, but edits made
    # between bootstrap and run should take effect immediately).
    CFG = load_config()
    world = state.world
    navigator = state.navigator
    planner = state.planner

    # Hot-apply tunables from config.yaml that may have changed since bootstrap
    # spawned the navigator. Speeds are baked into the controller at setup time,
    # but reach threshold / replan budget are simple attributes we can refresh.
    nav_cfg = CFG["navigator"]
    navigator._reach_tol = float(nav_cfg.get("waypoint_reach_threshold", 0.2))
    navigator._stuck_limit = int(nav_cfg.get("stuck_threshold_ticks", 240))
    navigator._max_replans = int(nav_cfg.get("max_replans", 3))
    log(f"Hot-applied: reach_tol={navigator._reach_tol} "
        f"stuck_limit={navigator._stuck_limit} max_replans={navigator._max_replans}")

    # Target: Point A from config (override via apps/run_nav_to.py for arbitrary goals)
    goal_xy = np.array(CFG["task"]["point_a"], dtype=float)[:2]
    log(f"Goal: {goal_xy.tolist()}")
    log(f"Goal valid? {planner.is_valid(tuple(goal_xy))}")

    # Clean up any callbacks left over from prior script runs (e.g. if a
    # previous script crashed before its own cleanup).
    for name in ("run_nav_step", "nav_step", "sim_step"):
        try:
            world.remove_physics_callback(name)
        except Exception:
            pass

    await world.play_async()
    navigator.set_goal(goal_xy)

    # ── Main control loop — step the navigator inline each tick ──
    # (Inline instead of a physics-callback: easier to reason about,
    # no stale-callback risk across script runs, all debug goes through
    # the same logger.)
    max_ticks = int(CFG["simulation"]["max_ticks"])
    status = NavStatus.RUNNING
    tick = 0

    # Capture the current nav generation — if bootstrap / teardown bumps it
    # while we're running, bail out cleanly. Prevents this loop from driving
    # the AMR after the user has started a new script.
    # getattr() fallback because core.state is preserved across script runs;
    # an older cached version may not have this attribute yet.
    my_gen = getattr(state, "nav_generation", 0)
    log(f"nav_generation captured = {my_gen}")

    while tick < max_ticks:
        await omni.kit.app.get_app().next_update_async()
        cur_gen = getattr(state, "nav_generation", 0)
        if cur_gen != my_gen:
            log(f"nav_generation changed ({my_gen} → {cur_gen}) — aborting")
            navigator.stop()
            break
        status = navigator.step()
        tick += 1
        if tick % 200 == 0:
            pos, ori = navigator.get_pose()
            dist = float(np.linalg.norm(pos[:2] - goal_xy))
            # Yaw from quaternion (w, x, y, z): atan2(2(wz+xy), 1-2(y²+z²))
            w, x, y, z = float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])
            yaw_deg = float(np.degrees(np.arctan2(
                2 * (w * z + x * y), 1 - 2 * (y * y + z * z)
            )))
            raw_vels = navigator.robot.get_joint_velocities()
            if raw_vels is None:
                wheel_vels = None
            else:
                vels = np.asarray(raw_vels)
                wheel_idx = navigator._wheel_indices if navigator._wheel_indices else [1, 2]
                wheel_vels = vels[wheel_idx].tolist() if len(vels) >= max(wheel_idx) + 1 else None
            log(f"  t {tick}: pos={[round(p,2) for p in pos[:2].tolist()]} "
                f"yaw={yaw_deg:.1f}° dist={dist:.2f}m wheel={wheel_vels} {status.value}")
        if status.value in ("reached", "failed"):
            break
    if status.value == "reached":
        log(f"REACHED goal at tick {tick}")
    elif status.value == "failed":
        log("FAILED to reach goal (no path / out of replans)")
    else:
        log(f"TIMEOUT at {max_ticks} ticks (still {status.value})")

    navigator.stop()
    await world.pause_async()
    log("run_nav complete.")

except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())
    raise
