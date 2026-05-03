# ============================================================
# apps/run_cortex.py — Cortex-style decider network for the
# mobile pick-and-place pipeline.
#
# Replaces run_pipeline.py's hardcoded 5-phase loop with a
# DfNetwork that routes on a per-tick block-state classifier:
#
#   Dispatch
#     ├── go_home      cube reached B
#     ├── fail         3 consecutive pick failures
#     ├── nav_to_block AMR drives to the cube's live position
#     ├── pick_rlds    arm closes around cube (BumpRetry on fail)
#     ├── transit      cube in gripper, AMR drives to B-standoff
#     └── place_rlds   arm drops cube at B
#
# Reactivity comes for free: drag the cube mid-execution and
# the next tick's classifier flips block_state, Dispatch routes
# to a different child, and df_descend tears down the in-flight
# sub-sequence. See `cortex/` package for the implementation.
#
# Run via: python3 run_in_isaac.py midterm_project/apps/run_cortex.py
# Log:     cache/isaac-sim/logs/run_cortex.log
# ============================================================
import sys
import traceback

sys.path.insert(0, "/workspace/midterm_project")
# Drop stale `apps._common` so freshly-edited helpers are picked up.
sys.modules.pop("apps._common", None)
from apps._common import bootstrap_imports, load_config, make_logger, make_stream_logger  # noqa: E402

# Reload `cortex.*` between runs too — without this, edits to the
# decider network don't take effect until Isaac Sim is restarted.
bootstrap_imports(reload_packages=("core", "scenes", "cortex"))
log, _ = make_logger("run_cortex")
stream, _stream_path = make_stream_logger("cortex_positions")
log(f"Streaming positions to {_stream_path} (tail -f to watch)")

try:
    import numpy as np
    import omni.kit.app

    from core import state
    from cortex.context import BlockState, MobileManipContext
    from cortex.network import make_decider_network

    state.require_ready()
    log(state.summary())

    CFG = load_config()
    world = state.world
    navigator = state.navigator
    manipulator = state.manipulator

    # ── Hot-apply tunables (mirrors run_pipeline.py) ─────────
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

    # Clean stale callbacks from prior runs (defensive — bootstrap's
    # fast-reset clears these but a partial run may have left some).
    for name in ("run_nav_step", "run_manip_step", "nav_step", "manip_step",
                 "cube_carry_sync"):
        try:
            world.remove_physics_callback(name)
        except Exception:
            pass

    # Ensure pose-sync is ON before nav_to_block runs — the nav sequence
    # doesn't include EnablePoseSync, so if a prior crash left it off the
    # AMR would drive away from a stationary Franka.
    manipulator.ensure_mount_sync(world)

    # ── Build context + decider network ──────────────────────
    task = CFG["task"]
    point_a = np.array(task["point_a"], dtype=float)[:2]
    point_b = np.array(task["point_b"], dtype=float)[:2]

    cube = world.scene.get_object("target_cube")
    if cube is None:
        raise RuntimeError("target_cube not found in scene — did bootstrap run?")

    ctx = MobileManipContext(navigator, manipulator, cube, point_a, point_b, CFG)
    network = make_decider_network(ctx)
    log(f"Decider network built: point_a={point_a.tolist()} "
        f"point_b={point_b.tolist()} "
        f"place_tol={ctx.place_tolerance} in_grip_tol={ctx.in_gripper_tolerance} "
        f"max_pick_attempts={ctx.MAX_PICK_ATTEMPTS}")

    # ── Per-tick stream telemetry ────────────────────────────
    def _stream_tick(tick: int):
        try:
            cp, _ = cube.get_world_pose()
            ee = manipulator._ee_position()
            amr_pos, _ = navigator.get_pose()
            stream(
                f"t={tick} state={ctx.block_state.value} "
                f"pick_attempts={ctx.pick_attempts} "
                f"manip_phase={manipulator.get_phase()} "
                f"cube={cp[0]:.3f},{cp[1]:.3f},{cp[2]:.3f} "
                f"ee={ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f} "
                f"amr={amr_pos[0]:.3f},{amr_pos[1]:.3f}"
            )
        except Exception as _e:
            stream(f"t={tick} stream-error: {_e}")

    # ── Main loop ────────────────────────────────────────────
    log("=== run_cortex starting decider loop ===")
    await world.play_async()

    max_ticks = int(CFG.get("simulation", {}).get("max_ticks", 50000))
    my_gen = getattr(state, "nav_generation", 0)

    last_state_logged = None
    tick = 0
    for tick in range(max_ticks):
        await omni.kit.app.get_app().next_update_async()
        if getattr(state, "nav_generation", 0) != my_gen:
            log("nav_generation bumped — aborting")
            break

        # 1. Advance physics-coupled controllers. Both are no-ops in
        #    idle/done state. Order matters less than always running
        #    each one before the classifier reads their phase/status.
        ctx.last_manip_status = manipulator.step()
        ctx.last_nav_status = navigator.step()

        # 2. Classify the world state, then run the decider.
        ctx.update_block_state()
        network.step()

        # 3. Log state transitions (one-line per change).
        if ctx.block_state != last_state_logged:
            cp, _ = cube.get_world_pose()
            amr_pos, _ = navigator.get_pose()
            prev_label = last_state_logged.value if last_state_logged is not None else "none"
            log(f"  t {tick}: block_state {prev_label} -> {ctx.block_state.value} "
                f"(cube={[round(float(x), 3) for x in cp]}, "
                f"amr={[round(float(x), 2) for x in amr_pos[:2]]}, "
                f"manip_phase={manipulator.get_phase()}, "
                f"pick_attempts={ctx.pick_attempts})")
            last_state_logged = ctx.block_state

        if tick % 50 == 0:
            _stream_tick(tick)

        # 4. Termination
        if ctx.block_state in (BlockState.DONE, BlockState.FAILED):
            log(f"  t {tick}: terminal state {ctx.block_state.value} reached")
            break

    # ── Final report ─────────────────────────────────────────
    cube_final, _ = cube.get_world_pose()
    cube_final = np.asarray(cube_final, dtype=float)
    err_xy = float(np.linalg.norm(cube_final[:2] - point_b))
    log(f"Final cube pose: {cube_final.tolist()}")
    log(f"Place error vs point_b={point_b.tolist()}: {err_xy:.3f} m")

    if ctx.block_state == BlockState.DONE:
        log(f"SUCCESS: cube delivered to {cube_final.tolist()} "
            f"(err={err_xy:.3f} m, pick_attempts={ctx.pick_attempts})")
    elif ctx.block_state == BlockState.FAILED:
        log(f"FAILED: pick budget exhausted "
            f"({ctx.pick_attempts}/{ctx.MAX_PICK_ATTEMPTS} attempts)")
    else:
        log(f"INCOMPLETE: terminal state={ctx.block_state.value} after {tick + 1} ticks")

    await world.pause_async()
    log("run_cortex complete.")

except SystemExit:
    pass
except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())
    raise
finally:
    # Defensive — same cleanup pattern as run_pipeline.py. The Cortex
    # path doesn't install cube_carry_sync, but if a previous run left
    # one it would override the cube's reset, so clear it here too.
    try:
        from core import state as _state
        if _state.world is not None:
            _state.world.remove_physics_callback("cube_carry_sync")
    except Exception:
        pass
