# ============================================================
# apps/run_cortex.py — Cortex-style decider network for the
# mobile pick-and-place pipeline, wrapped in a domain-randomized
# multi-episode loop.
#
# Each episode:
#   1. Randomizer.sample_episode() picks (start, cube, place).
#   2. apply_episode_to_cfg + reset_world_for_episode teleport
#      AMR/Franka/cube/markers to the new layout.
#   3. ctx.point_a / ctx.point_b are rebound; ctx.reset() clears
#      pick budget + block_state; the decider network is rebuilt.
#   4. The per-tick decider loop runs to terminal state (DONE /
#      FAILED) or to per_episode_max_ticks.
#
# Set `randomization.enabled: false` in config.yaml to fall back
# to a single deterministic episode using the configured
# task.point_a / task.point_b — useful for debugging.
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
    from core.episode import apply_episode_to_cfg, reset_world_for_episode
    from core.randomizer import Randomizer
    from cortex.context import BlockState, MobileManipContext
    from cortex.network import make_decider_network

    state.require_ready()
    log(state.summary())

    CFG = load_config()
    world = state.world
    navigator = state.navigator
    manipulator = state.manipulator
    planner = state.planner

    # ── Hot-apply tunables (mirrors run_pipeline.py) ─────────
    nav_cfg = CFG["navigator"]
    navigator._reach_tol = float(nav_cfg.get("waypoint_reach_threshold", 0.5))
    navigator._stuck_limit = int(nav_cfg.get("stuck_threshold_ticks", 240))
    navigator._max_replans = int(nav_cfg.get("max_replans", 3))

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

    cube = world.scene.get_object("target_cube")
    if cube is None:
        raise RuntimeError("target_cube not found in scene — did bootstrap run?")

    # ── Randomizer + episode plan ────────────────────────────
    rand_cfg = CFG.get("randomization", {})
    rand_enabled = bool(rand_cfg.get("enabled", False))
    n_episodes = int(rand_cfg.get("episodes", 1)) if rand_enabled else 1
    randomizer = Randomizer(planner, rand_cfg) if rand_enabled else None
    sim_cfg = CFG.get("simulation", {})
    per_ep_ticks = int(sim_cfg.get("per_episode_max_ticks", 8000))
    settle_ticks = int(sim_cfg.get("inter_episode_settle_ticks", 60))
    log(f"Episode plan: n={n_episodes} randomized={rand_enabled} "
        f"per_episode_max_ticks={per_ep_ticks}")

    # ── Episode helpers ──────────────────────────────────────
    def _clean_callbacks():
        for name in ("run_nav_step", "run_manip_step", "nav_step", "manip_step",
                     "cube_carry_sync"):
            try:
                world.remove_physics_callback(name)
            except Exception:
                pass

    async def _settle(n):
        for _ in range(n):
            await omni.kit.app.get_app().next_update_async()

    def _stream_tick(ep_idx, tick, ctx):
        try:
            cp, _ = cube.get_world_pose()
            ee = manipulator._ee_position()
            amr_pos, _ = navigator.get_pose()
            stream(
                f"ep={ep_idx} t={tick} state={ctx.block_state.value} "
                f"pick_attempts={ctx.pick_attempts} "
                f"manip_phase={manipulator.get_phase()} "
                f"cube={cp[0]:.3f},{cp[1]:.3f},{cp[2]:.3f} "
                f"ee={ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f} "
                f"amr={amr_pos[0]:.3f},{amr_pos[1]:.3f}"
            )
        except Exception as _e:
            stream(f"ep={ep_idx} t={tick} stream-error: {_e}")

    async def _run_one_episode(ep_idx, point_a_xy, point_b_xy):
        """Run a single episode to terminal block_state or tick budget.
        Returns dict with outcome/place_error/ticks/pick_attempts."""
        # Defensive cleanup — bootstrap clears these on cold-start, but
        # a partial in-process run may have left some live.
        _clean_callbacks()

        # Pose-sync must be ON before nav_to_block. The nav sub-decider
        # never enables it, so a prior crash that left it off would make
        # the AMR drive away from a stationary Franka.
        manipulator.ensure_mount_sync(world)

        ctx = MobileManipContext(navigator, manipulator, cube,
                                 point_a_xy, point_b_xy, CFG)
        network = make_decider_network(ctx)
        log(f"  ep={ep_idx} ctx ready: point_a={point_a_xy.tolist()} "
            f"point_b={point_b_xy.tolist()} "
            f"place_tol={ctx.place_tolerance}")

        my_gen = getattr(state, "nav_generation", 0)
        last_state_logged = None
        tick = 0
        for tick in range(per_ep_ticks):
            await omni.kit.app.get_app().next_update_async()
            if getattr(state, "nav_generation", 0) != my_gen:
                log("  nav_generation bumped — aborting episode")
                break

            ctx.last_manip_status = manipulator.step()
            ctx.last_nav_status = navigator.step()
            ctx.update_block_state()
            network.step()

            if ctx.block_state != last_state_logged:
                cp, _ = cube.get_world_pose()
                amr_pos, _ = navigator.get_pose()
                prev_label = (last_state_logged.value
                              if last_state_logged is not None else "none")
                log(f"  ep={ep_idx} t {tick}: block_state {prev_label} -> "
                    f"{ctx.block_state.value} "
                    f"(cube={[round(float(x), 3) for x in cp]}, "
                    f"amr={[round(float(x), 2) for x in amr_pos[:2]]}, "
                    f"manip_phase={manipulator.get_phase()}, "
                    f"pick_attempts={ctx.pick_attempts})")
                last_state_logged = ctx.block_state

            if tick % 50 == 0:
                _stream_tick(ep_idx, tick, ctx)

            if ctx.block_state in (BlockState.DONE, BlockState.FAILED):
                log(f"  ep={ep_idx} t {tick}: terminal "
                    f"{ctx.block_state.value}")
                break

        cube_final, _ = cube.get_world_pose()
        cube_final = np.asarray(cube_final, dtype=float)
        err_xy = float(np.linalg.norm(cube_final[:2] - point_b_xy))

        if ctx.block_state == BlockState.DONE:
            outcome = "success"
        elif ctx.block_state == BlockState.FAILED:
            outcome = "failed"
        else:
            outcome = "tick_budget_exhausted"

        return {
            "episode": ep_idx,
            "outcome": outcome,
            "block_state": ctx.block_state.value,
            "place_error_m": err_xy,
            "ticks": tick + 1,
            "pick_attempts": ctx.pick_attempts,
            "point_a": point_a_xy.tolist(),
            "point_b": point_b_xy.tolist(),
        }

    # ── Outer loop ──────────────────────────────────────────
    log("=== run_cortex starting episode loop ===")
    await world.play_async()
    results = []

    for ep_idx in range(n_episodes):
        log(f"--- Episode {ep_idx + 1}/{n_episodes} ---")

        if randomizer is not None:
            ep = randomizer.sample_episode()
            log(f"  sampled: start={ep.start_xy.tolist()} "
                f"cube={ep.cube_xyz.tolist()} place={ep.place_xyz.tolist()}")
            apply_episode_to_cfg(CFG, ep)
            reset_world_for_episode(world, CFG, manipulator, navigator)
            await _settle(settle_ticks)

        task = CFG["task"]
        point_a = np.array(task["point_a"], dtype=float)[:2]
        point_b = np.array(task["point_b"], dtype=float)[:2]

        try:
            result = await _run_one_episode(ep_idx, point_a, point_b)
        except Exception as _e:
            log(f"  ep={ep_idx} crashed: {type(_e).__name__}: {_e}")
            log(traceback.format_exc())
            result = {
                "episode": ep_idx,
                "outcome": "crashed",
                "block_state": "n/a",
                "place_error_m": float("nan"),
                "ticks": 0,
                "pick_attempts": 0,
                "point_a": point_a.tolist(),
                "point_b": point_b.tolist(),
            }
        results.append(result)
        log(f"  ep={ep_idx} done: {result}")

    # ── Aggregate report ────────────────────────────────────
    n_success = sum(1 for r in results if r["outcome"] == "success")
    log("=== run_cortex episode summary ===")
    for r in results:
        log(f"  ep={r['episode']}: {r['outcome']:<22} "
            f"err={r['place_error_m']:.3f} m  "
            f"ticks={r['ticks']:<5} "
            f"picks={r['pick_attempts']}  "
            f"a={r['point_a']} b={r['point_b']}")
    log(f"=== TOTAL: {n_success}/{len(results)} episodes succeeded ===")

    await world.pause_async()
    log("run_cortex complete.")

except SystemExit:
    pass
except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())
    raise
finally:
    # Defensive cleanup — same pattern as run_pipeline.py. The Cortex
    # path doesn't install cube_carry_sync at the orchestrator level,
    # but states.py does, and a partial run may have left one alive.
    try:
        from core import state as _state
        if _state.world is not None:
            _state.world.remove_physics_callback("cube_carry_sync")
    except Exception:
        pass
