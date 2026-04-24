# ============================================================
# apps/run_pipeline.py — full nav -> pick -> place demo.
#
# Flow:
#   1. Drive the AMR from origin toward Point A (navigation).
#   2. When the AMR stops, REBASE the Franka to the AMR's current
#      pose + mount offset. This teleports the arm AND rebuilds
#      the PickPlaceController so its IK solves in the new root
#      frame. Required because the Isaac Sim PickPlaceController
#      caches the articulation root pose at construction — a mobile
#      arm can't be driven without rebase.
#   3. Pick the cube at its current world position.
#   4. Place it at a reachable dropoff near the AMR (same cell as
#      the pick, offset in Y) — Franka's 0.85 m reach means we
#      can't drop at the far Point B without also driving there
#      and rebasing again.
#   5. Report success/failure with final cube location.
#
# Assumes apps/bootstrap.py has populated state.
#
# Run via: python3 run_in_isaac.py midterm_project/apps/run_pipeline.py
# Log:     cache/isaac-sim/logs/run_pipeline.log
# ============================================================
import sys
import traceback

sys.path.insert(0, "/workspace/midterm_project")
from apps._common import bootstrap_imports, load_config, make_logger  # noqa: E402

bootstrap_imports()
log, _ = make_logger("run_pipeline")

try:
    import numpy as np
    import omni.kit.app

    from core import state
    from core.navigator import NavStatus
    from core.manipulator import ManipStatus

    state.require_ready()
    log(state.summary())

    CFG = load_config()
    world = state.world
    navigator = state.navigator
    manipulator = state.manipulator

    # Hot-apply nav tunables (same as run_nav)
    nav_cfg = CFG["navigator"]
    navigator._reach_tol = float(nav_cfg.get("waypoint_reach_threshold", 0.5))
    navigator._stuck_limit = int(nav_cfg.get("stuck_threshold_ticks", 240))
    navigator._max_replans = int(nav_cfg.get("max_replans", 3))

    # Clean stale callbacks
    for name in ("run_nav_step", "run_manip_step", "nav_step"):
        try:
            world.remove_physics_callback(name)
        except Exception:
            pass

    task = CFG["task"]
    point_a = np.array(task["point_a"], dtype=float)[:2]

    # ── PHASE 1: Navigate AMR to Point A ─────────────────────
    log(f"=== Phase 1: navigate origin -> Point A ({point_a.tolist()}) ===")
    await world.play_async()
    navigator.set_goal(point_a)

    max_nav_ticks = 8000
    my_gen = getattr(state, "nav_generation", 0)
    nav_status = NavStatus.RUNNING

    for tick in range(max_nav_ticks):
        await omni.kit.app.get_app().next_update_async()
        if getattr(state, "nav_generation", 0) != my_gen:
            log(f"nav_generation bumped — aborting")
            break
        nav_status = navigator.step()
        if tick % 500 == 0 and tick > 0:
            pos, _ = navigator.get_pose()
            log(f"  t {tick}: AMR={[round(float(p), 2) for p in pos[:2]]}, "
                f"status={nav_status.value}")
        if nav_status.value in ("reached", "failed"):
            break

    amr_pos, amr_ori = navigator.get_pose()
    log(f"AMR final pose after nav: {amr_pos.tolist()} (status={nav_status.value})")
    if nav_status.value != "reached":
        log(f"FAILED: navigation didn't REACH (status={nav_status.value})")
        await world.pause_async()
        raise SystemExit(0)

    # ── Stop AMR cold before arm operation ───────────────────
    navigator.stop()
    for _ in range(30):
        await omni.kit.app.get_app().next_update_async()

    # ── PHASE 2: arm prep ────────────────────────────────────
    # Only touch pose-sync + rebase when mount_to is configured (mobile
    # manipulation). In station mode Franka is already at a fixed world
    # position that the IK knows about — moving it would BREAK pick.
    mount_to = CFG["manipulator"].get("mount_to")
    if mount_to:
        sync_cb_name = CFG["manipulator"].get("mount_sync_name", "franka_base_sync")
        mount_offset = np.array(
            CFG["manipulator"].get("mount_local_offset", [0, 0, 0.5]),
            dtype=float,
        )
        try:
            world.remove_physics_callback(sync_cb_name)
            log(f"Disabled pose-sync '{sync_cb_name}' for manipulation")
        except Exception:
            pass
        new_base = amr_pos.copy()
        new_base[2] = amr_pos[2] + mount_offset[2]
        log(f"=== Phase 2: rebase Franka to {new_base.tolist()} ===")
        if hasattr(manipulator, "rebase"):
            manipulator.rebase(new_base, world_orientation=amr_ori)
    else:
        log("=== Phase 2: station mode — skipping rebase ===")

    manipulator.reset()
    for _ in range(60):
        await omni.kit.app.get_app().next_update_async()

    # ── PHASE 3: pick cube, place at a reachable nearby spot ─
    cube = world.scene.get_object("target_cube")
    cube_pos, _ = cube.get_world_pose()
    cube_pos = np.asarray(cube_pos, dtype=float)
    log(f"Cube pos before pick: {cube_pos.tolist()}")

    manip_cfg = CFG["manipulator"]
    pick_z_offset = float(manip_cfg.get("pick_z_offset", 0.0))
    place_z_offset = float(manip_cfg.get("place_z_offset", 0.03))

    # Dropoff: 30 cm along the AMR's -Y axis from the cube (still
    # within Franka's reach from the rebased base pose). World frame.
    dropoff_xy = cube_pos[:2] + np.array([0.0, -0.30])
    pick_target = cube_pos + np.array([0.0, 0.0, pick_z_offset])
    place_target = np.array([dropoff_xy[0], dropoff_xy[1], cube_pos[2] + place_z_offset])
    log(f"Pick target:  {pick_target.tolist()}")
    log(f"Place target: {place_target.tolist()}  (reachable from AMR station)")

    manipulator.pick(pick_target)
    manipulator.place(place_target)
    log(f"Manipulator armed (phase={manipulator.get_phase()})")

    max_manip_ticks = 4000
    last_phase = manipulator.get_phase()
    manip_status = ManipStatus.RUNNING

    for tick in range(max_manip_ticks):
        await omni.kit.app.get_app().next_update_async()
        manip_status = manipulator.step()
        cur_phase = manipulator.get_phase()
        if cur_phase != last_phase:
            cp, _ = cube.get_world_pose()
            log(f"  t {tick}: phase {last_phase} -> {cur_phase}, cube_z={float(cp[2]):.3f}")
            last_phase = cur_phase
        if tick % 300 == 0 and tick > 0:
            cp, _ = cube.get_world_pose()
            log(f"  t {tick}: phase={cur_phase}, cube={[round(float(v), 2) for v in cp]}")
        if manip_status.value in ("done", "failed"):
            break

    # ── PHASE 4: report ─────────────────────────────────────
    cube_final, _ = cube.get_world_pose()
    cube_final = np.asarray(cube_final, dtype=float)
    place_err_xy = float(np.linalg.norm(cube_final[:2] - place_target[:2]))
    lifted = float(cube_final[2]) > 0.01   # above floor = was picked
    log(f"Final cube pos: {cube_final.tolist()}")
    log(f"Place error (XY): {place_err_xy:.3f} m")

    if manip_status.value == "done" and place_err_xy < 0.30:
        log(f"SUCCESS: pipeline complete. Cube moved from ~{cube_pos.tolist()} "
            f"to {cube_final.tolist()}")
    else:
        log(f"PARTIAL/FAILED: manip_status={manip_status.value}, place_err={place_err_xy:.3f}")

    await world.pause_async()
    log("run_pipeline complete.")

except SystemExit:
    pass
except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())
    raise
