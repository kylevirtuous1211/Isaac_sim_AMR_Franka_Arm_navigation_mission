# ============================================================
# apps/run_manip.py — exercise the arm pick & place on the cube.
#
# Assumes:
#   1. apps/bootstrap.py has been run (state populated, cube spawned)
#   2. The AMR is close enough to the cube for the arm to reach it.
#      If not, run apps/run_nav.py first (drives AMR to Point A) OR
#      teleport the AMR with set_world_pose from a helper script.
#
# What this script does:
#   - Locks the AMR in place (navigator.stop + latch) so the pose-sync
#     callback's Franka teleport is a no-op and the arm has a stable
#     world-frame base for its plan.
#   - Reads the cube's current world position (NOT config.task.point_a —
#     after navigation the cube may have shifted a few cm).
#   - Calls manipulator.pick(cube_xyz + pick_z_offset)
#     then manipulator.place(place_xyz + place_z_offset).
#     FrankaRMPflowManipulator chains them via its phase FSM.
#   - Ticks physics until the manipulator reports DONE (or FAILED,
#     or we hit max_ticks).
#
# Run via: python3 run_in_isaac.py midterm_project/apps/run_manip.py
# Log:     cache/isaac-sim/logs/run_manip.log
# ============================================================
import sys
import traceback

sys.path.insert(0, "/workspace/midterm_project")
sys.modules.pop("apps._common", None)
from apps._common import bootstrap_imports, load_config, make_logger  # noqa: E402

bootstrap_imports()
log, _ = make_logger("run_manip")

try:
    import numpy as np
    import omni.kit.app

    from core import state
    from core.manipulator import ManipStatus

    state.require_ready()
    log(state.summary())

    CFG = load_config()
    world = state.world
    manipulator = state.manipulator
    navigator = state.navigator

    # Gate: AMR must be stationary during manipulation. The pose-sync
    # callback teleports Franka's base to AMR chassis every tick, so if
    # the AMR is moving the RMPflow base frame drifts mid-plan.
    navigator.stop()
    log(f"Navigator stopped; AMR at {navigator.get_pose()[0].tolist()}")

    # Clean up any callbacks left over from earlier scripts (same defense
    # as run_nav.py — prevents ghost control loops from driving the arm).
    for name in ("run_manip_step", "manip_step"):
        try:
            world.remove_physics_callback(name)
        except Exception:
            pass

    # Remove the pose-sync callback while the arm operates so the Franka
    # base is stable. RMPflow reads the live base pose each tick, so no
    # rebase is needed.
    mount_to = CFG["manipulator"].get("mount_to")
    if mount_to:
        sync_cb_name = CFG["manipulator"].get("mount_sync_name")
        if sync_cb_name:
            try:
                world.remove_physics_callback(sync_cb_name)
            except Exception:
                pass

    await world.play_async()

    # ── Determine pick & place XYZ ──────────────────────────
    # Cube's CURRENT world pose (not point_a, which is stale if the
    # cube shifted during prior simulation).
    cube = world.scene.get_object("target_cube")
    if cube is None:
        raise RuntimeError("target_cube not found in scene — did bootstrap run?")
    cube_pos, _ = cube.get_world_pose()
    cube_pos = np.asarray(cube_pos, dtype=float)
    log(f"Cube current pos: {cube_pos.tolist()}")

    task = CFG["task"]
    place_xy = np.array(task["point_b"], dtype=float)[:2]
    # Place target Z aligns with the cube's current Z (so we drop at
    # floor height, offset added below).
    place_pos = np.array([place_xy[0], place_xy[1], cube_pos[2]])

    manip_cfg = CFG["manipulator"]
    pick_z_offset = float(manip_cfg.get("pick_z_offset", 0.0))
    place_z_offset = float(manip_cfg.get("place_z_offset", 0.03))

    pick_target = cube_pos + np.array([0.0, 0.0, pick_z_offset])
    place_target = place_pos + np.array([0.0, 0.0, place_z_offset])
    log(f"Pick target:  {pick_target.tolist()}")
    log(f"Place target: {place_target.tolist()}")

    # ── Kick off the pick+place cycle ───────────────────────
    manipulator.reset()                  # idempotent, opens gripper
    manipulator.pick(pick_target)
    manipulator.place(place_target)      # both manipulator impls accept this order
    log(f"Manipulator armed: {type(manipulator).__name__}, "
        f"phase={manipulator.get_phase()}")

    # ── Physics loop ────────────────────────────────────────
    max_ticks = int(CFG["simulation"]["max_ticks"])
    my_gen = getattr(state, "nav_generation", 0)
    status = ManipStatus.RUNNING
    last_logged_phase = manipulator.get_phase()

    for tick in range(max_ticks):
        await omni.kit.app.get_app().next_update_async()
        # Honour the generation-bump kill switch (same pattern as run_nav)
        if getattr(state, "nav_generation", 0) != my_gen:
            log(f"nav_generation bumped — aborting run_manip")
            break

        status = manipulator.step()
        cur_phase = manipulator.get_phase()
        if cur_phase != last_logged_phase:
            pos, _ = cube.get_world_pose()
            log(f"  t {tick}: phase {last_logged_phase} -> {cur_phase}, "
                f"cube_z={float(pos[2]):.3f}")
            last_logged_phase = cur_phase

        if status.value in ("done", "failed"):
            break

        if tick % 500 == 0 and tick > 0:
            pos, _ = cube.get_world_pose()
            log(f"  t {tick}: phase={cur_phase}, status={status.value}, "
                f"cube={[round(float(v), 2) for v in pos]}")
    else:
        log(f"TIMEOUT at {max_ticks} ticks (phase={manipulator.get_phase()})")
        status = ManipStatus.FAILED

    # ── Report final state ──────────────────────────────────
    cube_final, _ = cube.get_world_pose()
    cube_final = np.asarray(cube_final, dtype=float)
    place_dist = float(np.linalg.norm(cube_final[:2] - place_xy))
    log(f"Final cube pos: {cube_final.tolist()}")
    log(f"Distance from cube to place target: {place_dist:.3f} m")

    if status.value == "done":
        log(f"SUCCESS: manipulator cycle DONE (phase={manipulator.get_phase()})")
    else:
        log(f"FAILED: status={status.value}, phase={manipulator.get_phase()}")

    await world.pause_async()
    log("run_manip complete.")

except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())
    raise
