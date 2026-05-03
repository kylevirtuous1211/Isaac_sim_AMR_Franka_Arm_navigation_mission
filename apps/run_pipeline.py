# ============================================================
# apps/run_pipeline.py — full mobile manipulation demo.
#
# 5-phase flow:
#   1. Navigate AMR from origin to point_a (block location).
#   2. Stop AMR; freeze Franka base; pick + lift the cube. The
#      SurfaceGripper authored at bootstrap creates a D6 joint that
#      keeps the cube attached even when Franka's base teleports.
#   3. (NEW transit) Re-install pose-sync; navigate AMR from point_a
#      to a safe standoff before point_b. The cube rides on Franka
#      via the D6 joint while the AMR drives.
#   4. Stop AMR; freeze Franka base; place + retract the cube at
#      point_b. SurfaceGripper releases first, then ParallelGripper.
#   5. Report success/failure with final cube position.
#
# Assumes apps/bootstrap.py has populated state.
#
# Run via: python3 run_in_isaac.py midterm_project/apps/run_pipeline.py
# Log:     cache/isaac-sim/logs/run_pipeline.log
# ============================================================
import sys
import traceback

sys.path.insert(0, "/workspace/midterm_project")
# Isaac Sim's Kit process caches apps._common across runs — drop the stale
# copy so freshly-edited helpers (e.g. make_stream_logger) are picked up.
sys.modules.pop("apps._common", None)
from apps._common import bootstrap_imports, load_config, make_logger, make_stream_logger  # noqa: E402

bootstrap_imports()
log, _ = make_logger("run_pipeline")
stream, _stream_path = make_stream_logger("positions")
log(f"Streaming positions to {_stream_path} (tail -f to watch)")

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

    # ── Hot-apply tunables so config edits propagate without state.teardown() ──
    nav_cfg = CFG["navigator"]
    navigator._reach_tol = float(nav_cfg.get("waypoint_reach_threshold", 0.5))
    navigator._stuck_limit = int(nav_cfg.get("stuck_threshold_ticks", 240))
    navigator._max_replans = int(nav_cfg.get("max_replans", 3))

    planner = state.planner
    rrt_cfg = CFG["planner"].get("rrt_star", {})
    if hasattr(planner, "goal_tolerance"):
        planner.goal_tolerance = float(rrt_cfg.get("goal_tolerance", 0.2))
    log(f"Planner tunables: goal_tolerance={getattr(planner, 'goal_tolerance', None)}")

    _mcfg = CFG["manipulator"]
    manipulator._approach_tol = float(_mcfg.get("approach_tolerance", 0.04))
    manipulator._clearance_height = float(_mcfg.get("clearance_height", 0.15))
    manipulator._grasp_hold_ticks = int(_mcfg.get("grasp_hold_ticks", 40))
    manipulator._release_hold_ticks = int(_mcfg.get("release_hold_ticks", 30))
    manipulator._phase_timeout_ticks = int(_mcfg.get("phase_timeout_ticks", 800))
    log(f"Manipulator tunables: approach_tol={manipulator._approach_tol}, "
        f"clearance={manipulator._clearance_height}, "
        f"phase_timeout={manipulator._phase_timeout_ticks}")

    # Clean stale callbacks from prior runs of run_nav / run_manip
    for name in ("run_nav_step", "run_manip_step", "nav_step", "manip_step"):
        try:
            world.remove_physics_callback(name)
        except Exception:
            pass

    # ── Helpers ────────────────────────────────────────────
    mount_to = CFG["manipulator"].get("mount_to")
    sync_cb_name = CFG["manipulator"].get("mount_sync_name", "franka_base_sync")
    mount_offset = np.array(
        CFG["manipulator"].get("mount_local_offset", [0.0, 0.0, 0.50]),
        dtype=float,
    )

    def _disable_pose_sync():
        if mount_to:
            try:
                world.remove_physics_callback(sync_cb_name)
            except Exception:
                pass

    def _rebase_franka_to_amr():
        if not mount_to:
            return
        amr_pos, amr_ori = navigator.get_pose()
        manipulator.franka.set_world_pose(
            position=amr_pos + mount_offset,
            orientation=amr_ori,
        )
        log(f"Franka rebased to AMR pose: {(amr_pos + mount_offset).tolist()}")

    def _enable_pose_sync():
        # ensure_mount_sync re-installs the callback idempotently
        manipulator.ensure_mount_sync(world)

    # ── Cube carry-sync ──────────────────────────────────────
    # The SurfaceGripper authored at bootstrap is the "right" way to
    # attach the cube. In practice its D6 joint geometry needs the gripper
    # to be near-touching the candidate, and our long-reach top-down pick
    # plateaus 8+ cm above the cube. So we add a belt-and-braces
    # physics callback that teleports the cube to the gripper EE pose
    # every tick during transit. Cube velocities are zeroed each tick so
    # PhysX doesn't accumulate momentum that would launch it on release.
    def _install_cube_carry_sync():
        franka = manipulator.franka

        # Capture the joint configuration at install time — this is the
        # post-grasp pose with the gripper closed and the arm in lift
        # position. We force-restore these every tick so gravity doesn't
        # flail the arm during transit (which would shove the AMR via
        # collision contact and send it on chaotic trajectories).
        try:
            frozen_joints = np.asarray(franka.get_joint_positions(), dtype=float)
        except Exception:
            frozen_joints = None

        def _carry(_step_size):
            try:
                if frozen_joints is not None:
                    franka.set_joint_positions(frozen_joints)
                    franka.set_joint_velocities(np.zeros_like(frozen_joints))
                ee_pos, ee_ori = franka.end_effector.get_world_pose()
                cube.set_world_pose(position=ee_pos, orientation=ee_ori)
                cube.set_linear_velocity(np.zeros(3))
                cube.set_angular_velocity(np.zeros(3))
            except Exception as _e:
                print(f"[carry_sync] failed: {_e}")

        try:
            world.remove_physics_callback("cube_carry_sync")
        except Exception:
            pass
        world.add_physics_callback("cube_carry_sync", callback_fn=_carry)
        log(f"Cube carry-sync installed "
            f"(froze {len(frozen_joints) if frozen_joints is not None else 0} arm joints)")

    def _remove_cube_carry_sync():
        try:
            world.remove_physics_callback("cube_carry_sync")
            log("Cube carry-sync removed")
        except Exception:
            pass

    def _stream_tick(tick: int, phase_label: str):
        """Emit one telemetry line. Tolerant of missing surface gripper."""
        try:
            cp, _ = cube.get_world_pose()
            ee = manipulator._ee_position()
            bp, _ = manipulator.franka.get_world_pose()
            amr_pos, _ = navigator.get_pose()
            sg = manipulator.surface_gripper
            sg_status = sg.status() if sg is not None else "n/a"
            sg_gripped = sg.gripped() if sg is not None else []
            tgt = manipulator._phase_target()
            if tgt is not None:
                d = float(np.linalg.norm(np.asarray(ee) - np.asarray(tgt)))
                tgt_s = f"{tgt[0]:.3f},{tgt[1]:.3f},{tgt[2]:.3f}"
            else:
                d, tgt_s = -1.0, "-,-,-"
            stream(
                f"t={tick} ph={phase_label}/{manipulator.get_phase()} "
                f"cube={cp[0]:.3f},{cp[1]:.3f},{cp[2]:.3f} "
                f"ee={ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f} "
                f"base={bp[0]:.3f},{bp[1]:.3f},{bp[2]:.3f} "
                f"amr={amr_pos[0]:.3f},{amr_pos[1]:.3f} "
                f"target={tgt_s} dist={d:.3f} "
                f"sg={sg_status} gripped={len(sg_gripped)}"
            )
        except Exception as _e:
            stream(f"t={tick} stream-error: {_e}")

    task = CFG["task"]
    point_a = np.array(task["point_a"], dtype=float)[:2]
    point_b = np.array(task["point_b"], dtype=float)[:2]
    cube_size = float(task["cube_size"])
    cube_half = cube_size / 2.0
    place_standoff = float(task.get("place_standoff", 0.40))
    place_z_offset = float(_mcfg.get("place_z_offset", 0.03))

    cube = world.scene.get_object("target_cube")
    if cube is None:
        raise RuntimeError("target_cube not found in scene — did bootstrap run?")

    # ============================================================
    # PHASE 1 — drive AMR origin → point_a
    # ============================================================
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
    log(f"AMR final pose after Phase 1: {amr_pos.tolist()} (status={nav_status.value})")
    if nav_status.value != "reached":
        log(f"FAILED Phase 1: navigation didn't REACH (status={nav_status.value})")
        await world.pause_async()
        raise SystemExit(0)

    navigator.stop()
    for _ in range(30):
        await omni.kit.app.get_app().next_update_async()

    # ============================================================
    # PHASE 2 — pick + lift (FSM stops at "done" since no place target)
    # ============================================================
    log(f"=== Phase 2: pick + lift cube ===")
    _disable_pose_sync()
    _rebase_franka_to_amr()

    cube_pos, _ = cube.get_world_pose()
    cube_pos = np.asarray(cube_pos, dtype=float)
    log(f"Cube pos before pick: {cube_pos.tolist()}")

    pick_target = cube_pos + np.array(
        [0.0, 0.0, float(_mcfg.get("pick_z_offset", 0.0))]
    )
    log(f"Pick target: {pick_target.tolist()}")

    manipulator.reset()                          # opens gripper(s) (ParallelGripper + SurfaceGripper)
    for _ in range(30):                           # let gripper open settle
        await omni.kit.app.get_app().next_update_async()

    manipulator.pick(pick_target)                # NO place target — FSM stops at "done" after lift
    log(f"Manipulator armed for pick (phase={manipulator.get_phase()})")

    max_pick_ticks = 4000
    last_phase = manipulator.get_phase()
    pick_status = ManipStatus.RUNNING

    for tick in range(max_pick_ticks):
        await omni.kit.app.get_app().next_update_async()
        pick_status = manipulator.step()
        cur_phase = manipulator.get_phase()
        if cur_phase != last_phase:
            cp, _ = cube.get_world_pose()
            sg = manipulator.surface_gripper
            sg_status = sg.status() if sg is not None else "n/a"
            log(f"  t {tick}: phase {last_phase} -> {cur_phase}, "
                f"cube_z={float(cp[2]):.3f}, sg={sg_status}")
            last_phase = cur_phase
        if tick % 10 == 0:
            _stream_tick(tick, "P2")
        if pick_status.value in ("done", "failed"):
            break

    if pick_status.value != "done":
        log(f"FAILED Phase 2: status={pick_status.value}, phase={manipulator.get_phase()}")
        await world.pause_async()
        raise SystemExit(0)

    sg_after_pick = manipulator.surface_gripper.gripped() if manipulator.surface_gripper else []
    log(f"Phase 2 complete. SurfaceGripper gripped: {sg_after_pick}")

    # ============================================================
    # PHASE 3 — TRANSIT: drive AMR point_a → standoff before point_b
    # ============================================================
    log(f"=== Phase 3: transit (carry cube) -> standoff before {point_b.tolist()} ===")

    # Direction from current AMR position toward point_b — park `standoff`
    # meters before point_b along that approach axis, so the arm has reach
    # headroom when we drop in Phase 4.
    approach_vec = point_b - amr_pos[:2]
    dist_to_b = float(np.linalg.norm(approach_vec))
    if dist_to_b > 1e-3:
        unit = approach_vec / dist_to_b
        standoff_xy = point_b - place_standoff * unit
    else:
        standoff_xy = point_b
    log(f"Standoff target: {standoff_xy.tolist()} ({place_standoff:.2f} m before B)")

    # Park the manipulator FSM idle so subsequent place() can transition
    # cleanly from "idle" → "above_place".
    manipulator._phase = "idle"

    # Re-install pose-sync — Franka rides AMR while it drives.
    _enable_pose_sync()

    # NOTE: cube carry-sync intentionally NOT installed for this run —
    # we want to see whether the real gripper (ParallelGripper.close()
    # firing during Phase 2) actually holds the cube via finger friction
    # alone. If the cube falls during transit, we know the grasp didn't
    # take and we'll need a pedestal or different mount tuning.
    # _install_cube_carry_sync()

    navigator.set_goal(standoff_xy)             # set_goal resets nav internal state
    log(f"Navigator armed for standoff: {standoff_xy.tolist()}")

    transit_status = NavStatus.RUNNING
    for tick in range(max_nav_ticks):
        await omni.kit.app.get_app().next_update_async()
        if getattr(state, "nav_generation", 0) != my_gen:
            log(f"nav_generation bumped — aborting")
            break
        transit_status = navigator.step()
        if tick % 200 == 0:
            pos, _ = navigator.get_pose()
            cp, _ = cube.get_world_pose()
            sg = manipulator.surface_gripper
            log(f"  transit t {tick}: AMR={[round(float(p), 2) for p in pos[:2]]}, "
                f"cube_z={float(cp[2]):.3f}, sg={sg.status() if sg else 'n/a'}, "
                f"status={transit_status.value}")
        if tick % 50 == 0:
            _stream_tick(tick, "P3")
        if transit_status.value in ("reached", "failed"):
            break

    amr_pos, amr_ori = navigator.get_pose()
    log(f"AMR final pose after transit: {amr_pos.tolist()} (status={transit_status.value})")
    if transit_status.value != "reached":
        log(f"FAILED Phase 3: transit didn't REACH (status={transit_status.value})")
        await world.pause_async()
        raise SystemExit(0)

    navigator.stop()
    for _ in range(30):
        await omni.kit.app.get_app().next_update_async()

    # ============================================================
    # PHASE 4 — place + retract at point_b
    # ============================================================
    log(f"=== Phase 4: place cube at {point_b.tolist()} ===")
    _disable_pose_sync()
    # carry-sync wasn't installed for this run; no-op cleanup
    # _remove_cube_carry_sync()
    _rebase_franka_to_amr()

    place_target = np.array([point_b[0], point_b[1], cube_half + place_z_offset])
    log(f"Place target: {place_target.tolist()}")

    manipulator.place(place_target)             # FSM goes idle/done → above_place
    log(f"Manipulator armed for place (phase={manipulator.get_phase()})")

    max_place_ticks = 4000
    last_phase = manipulator.get_phase()
    place_status = ManipStatus.RUNNING

    for tick in range(max_place_ticks):
        await omni.kit.app.get_app().next_update_async()
        place_status = manipulator.step()
        cur_phase = manipulator.get_phase()
        if cur_phase != last_phase:
            cp, _ = cube.get_world_pose()
            sg = manipulator.surface_gripper
            sg_status = sg.status() if sg is not None else "n/a"
            log(f"  t {tick}: phase {last_phase} -> {cur_phase}, "
                f"cube_z={float(cp[2]):.3f}, sg={sg_status}")
            last_phase = cur_phase
        if tick % 10 == 0:
            _stream_tick(tick, "P4")
        if place_status.value in ("done", "failed"):
            break

    # ============================================================
    # PHASE 5 — report
    # ============================================================
    cube_final, _ = cube.get_world_pose()
    cube_final = np.asarray(cube_final, dtype=float)
    place_err_xy = float(np.linalg.norm(cube_final[:2] - point_b))
    sg_final = manipulator.surface_gripper.status() if manipulator.surface_gripper else "n/a"
    sg_gripped_final = manipulator.surface_gripper.gripped() if manipulator.surface_gripper else []
    log(f"Final cube pos: {cube_final.tolist()}")
    log(f"Place error (XY) vs point_b={point_b.tolist()}: {place_err_xy:.3f} m")
    log(f"Final SurfaceGripper: status={sg_final}, gripped={sg_gripped_final}")

    if place_status.value == "done" and place_err_xy < 0.50:
        log(f"SUCCESS: pipeline complete. Cube delivered to "
            f"{cube_final.tolist()} (err={place_err_xy:.3f} m vs B).")
    else:
        log(f"PARTIAL/FAILED: place_status={place_status.value}, "
            f"place_err={place_err_xy:.3f}, phase={manipulator.get_phase()}")

    await world.pause_async()
    log("run_pipeline complete.")

except SystemExit:
    pass
except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())
    raise
finally:
    # Defensive cleanup — ensure cube_carry_sync is gone no matter how
    # we exit. Otherwise next bootstrap's fast-reset will see the cube
    # stuck in the gripper because the callback keeps teleporting it.
    try:
        from core import state as _state
        if _state.world is not None:
            _state.world.remove_physics_callback("cube_carry_sync")
            log("Cube carry-sync removed (finally cleanup)")
    except Exception:
        pass
