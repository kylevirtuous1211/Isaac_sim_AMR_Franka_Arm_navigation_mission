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
    from core.diag import diag, truncate as diag_truncate
    from core.episode import apply_episode_to_cfg, reset_world_for_episode
    from core.randomizer import Randomizer
    from cortex.context import BlockState, MobileManipContext
    from cortex.network import make_decider_network

    diag_truncate()
    diag("=== run_cortex started — diag stream truncated ===")

    # ── Mount/articulation attribute probe ───────────────────────
    # The chassis-jam bug fires when /World/Franka has fixedBase=True or
    # the mount FixedJoint is missing excludeFromArticulation=True.
    # Surface the live values so we can rule that out quickly.
    try:
        import omni.usd
        from pxr import UsdPhysics
        _stage = omni.usd.get_context().get_stage()
        _franka_root = "/World/Franka"
        _franka_prim = _stage.GetPrimAtPath(_franka_root)
        _fb = _franka_prim.GetAttribute("physxArticulation:fixedBase") if _franka_prim and _franka_prim.IsValid() else None
        _fb_val = _fb.Get() if _fb and _fb.IsValid() else "<missing>"
        diag(f"[probe] {_franka_root} physxArticulation:fixedBase = {_fb_val}  (must be False or wheels jam)")

        _joint_path = "/World/Franka/franka_mount"
        _joint_prim = _stage.GetPrimAtPath(_joint_path)
        if _joint_prim and _joint_prim.IsValid():
            _exa = _joint_prim.GetAttribute("physics:excludeFromArticulation")
            _exa_val = _exa.Get() if _exa and _exa.IsValid() else "<missing>"
            _b0 = _joint_prim.GetRelationship("physics:body0").GetTargets() if _joint_prim.GetRelationship("physics:body0") else []
            _b1 = _joint_prim.GetRelationship("physics:body1").GetTargets() if _joint_prim.GetRelationship("physics:body1") else []
            diag(f"[probe] {_joint_path} excludeFromArticulation = {_exa_val} body0={list(map(str, _b0))} body1={list(map(str, _b1))}")
        else:
            diag(f"[probe] {_joint_path} prim not found (mount_mode may be pose_sync)")

        # Check for ANY UsdPhysics joint anywhere under /World/Franka or /World/NovaCarter
        # that could be pinning the chassis. Wider net than just franka_mount.
        try:
            for _p in _stage.Traverse():
                _path_str = str(_p.GetPath())
                if not (_path_str.startswith("/World/Franka")
                        or _path_str.startswith("/World/NovaCarter")):
                    continue
                if (_p.IsA(UsdPhysics.Joint) or _p.IsA(UsdPhysics.FixedJoint)
                        or _p.IsA(UsdPhysics.RevoluteJoint)
                        or _p.IsA(UsdPhysics.PrismaticJoint)):
                    _b0 = _p.GetRelationship("physics:body0").GetTargets() if _p.GetRelationship("physics:body0") else []
                    _b1 = _p.GetRelationship("physics:body1").GetTargets() if _p.GetRelationship("physics:body1") else []
                    diag(f"[probe joint] {_path_str} type={_p.GetTypeName()} "
                         f"body0={list(map(str, _b0))} body1={list(map(str, _b1))}")
        except Exception as _je:
            diag(f"[probe joints] failed: {type(_je).__name__}: {_je}")
    except Exception as _probe_e:
        diag(f"[probe] failed: {type(_probe_e).__name__}: {_probe_e}")

    state.require_ready()
    log(state.summary())

    CFG = load_config()
    world = state.world
    navigator = state.navigator
    manipulator = state.manipulator
    planner = state.planner

    # ── Hot-apply tunables ───────────────────────────────────
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
        # The cortex pipeline doesn't install per-tick physics callbacks
        # under named keys (it drives via the main loop), but a partial
        # in-process run from a prior orchestrator may have left some
        # alive. Defensive cleanup.
        for name in ("nav_step", "manip_step"):
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

        # Idempotent mount enforcement: in fixed_joint mode this clears
        # any stale pose_sync callback; in pose_sync mode it re-installs
        # the per-tick teleport.
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
            try:
                network.step()
            except Exception as _net_e:
                diag(f"[run_cortex] network.step() RAISED at tick {tick}: "
                     f"{type(_net_e).__name__}: {_net_e}")
                raise

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

    # Capture the nav_generation we started with. If something else
    # (force_reboot, a new bootstrap) bumps it during our run, we abort
    # the OUTER loop too — without this guard, the per-episode reset
    # would race with whatever script just took over the world and we'd
    # produce interleaved log spam.
    outer_my_gen = getattr(state, "nav_generation", 0)

    for ep_idx in range(n_episodes):
        if getattr(state, "nav_generation", 0) != outer_my_gen:
            log(f"--- nav_generation bumped before ep {ep_idx} — aborting outer loop ---")
            break
        log(f"--- Episode {ep_idx + 1}/{n_episodes} ---")

        if randomizer is not None:
            ep = randomizer.sample_episode()
            log(f"  sampled: start={ep.start_xy.tolist()} "
                f"cube={ep.cube_xyz.tolist()} place={ep.place_xyz.tolist()}")
            apply_episode_to_cfg(CFG, ep)
            # Pause around the multi-body teleport — with mount_mode=
            # fixed_joint and rootJoint disabled, the FixedJoint between
            # chassis_link and panda_link0 is the only constraint holding
            # them together. Any PhysX tick where they're not at the
            # correct relative offset triggers a huge impulse and NaN.
            await world.pause_async()
            reset_world_for_episode(world, CFG, manipulator, navigator)
            await world.play_async()
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
