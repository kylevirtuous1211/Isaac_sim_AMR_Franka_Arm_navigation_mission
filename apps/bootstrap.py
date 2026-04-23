# ============================================================
# apps/bootstrap.py — load hospital + spawn robots ONCE
#
# Subsequent scripts (apps/run_nav.py, apps/run_manip.py, ...) import
# from core.state and skip the 60–180s hospital reload.
#
# Idempotent: re-running this script is a no-op if core.state is populated.
# To force a re-bootstrap, call core.state.teardown() first, or
# send the script with env FORCE_REBOOT=1.
#
# Run via: python3 run_in_isaac.py midterm_project/apps/bootstrap.py
# Log:     cache/isaac-sim/logs/bootstrap.log
# ============================================================
import os
import sys
import traceback

# Must run FIRST — sets up sys.path and purges stale core/* modules.
sys.path.insert(0, "/workspace/midterm_project")
from apps._common import bootstrap_imports, load_config, make_logger  # noqa: E402

bootstrap_imports()
log, _log_path = make_logger("bootstrap")

try:
    import numpy as np
    import omni.kit.app

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
    from isaacsim.core.utils.viewports import set_camera_view

    from core import state
    from core.factory import build_planner, build_navigator, build_manipulator
    from scenes.hospital import load_hospital

    CFG = load_config()
    state.config = CFG

    force = bool(int(os.environ.get("FORCE_REBOOT", "0")))
    if force:
        log("FORCE_REBOOT=1 — tearing down previous state")
        state.teardown()

    if state.is_ready() and not force:
        # Fast-reset path: scene + robots already spawned, just restore poses.
        # world.reset_async() returns every articulation to its default pose
        # (the positions we passed to Navigator/Manipulator.setup). We also
        # re-open the gripper and re-play the sim.
        log(state.summary())
        log("State already populated — fast-reset instead of full reload.")
        world = state.world
        await world.reset_async()
        state.manipulator.reset()
        await world.play_async()

        # Re-aim camera behind the reset AMR position.
        from isaacsim.core.utils.viewports import set_camera_view
        import numpy as np
        start_pos = state.navigator.get_pose()[0]
        set_camera_view(
            eye=start_pos + np.array([-2.5, 0.0, 1.8]),
            target=start_pos + np.array([1.0, 0.0, 0.6]),
        )
        log("Scene reset to defaults (fast path — no stage reload).")
        # Give physics a few ticks to settle
        import omni.kit.app
        for _ in range(30):
            await omni.kit.app.get_app().next_update_async()
        log(f"AMR pose after reset: {state.navigator.get_pose()[0].tolist()}")
    else:
        # ── 1. Hospital stage ────────────────────────────────
        # Always reload the stage here: if state is NOT populated but a
        # previous script left prims in the world (e.g. old `nova_carter`),
        # adding them again would collide. A fresh stage guarantees a
        # clean slate; subsequent run_*.py scripts then reuse via
        # `core.state` without reloading.
        state.scene_loaded_path = await load_hospital(force=True)

        # ── 2. World ─────────────────────────────────────────
        # Clear any stale World singleton left over from earlier runs.
        existing = World.instance()
        if existing is not None:
            existing.clear_instance()
        world = World(stage_units_in_meters=1.0)
        await world.initialize_simulation_context_async()
        state.world = world
        log(f"World: {type(world).__name__}")

        # ── 3. Planner, navigator, manipulator ───────────────
        state.planner = build_planner(CFG["planner"])
        state.planner.build(world)
        log(f"Planner: {type(state.planner).__name__}")

        state.navigator = build_navigator(CFG["navigator"], state.planner, world)
        log(f"Navigator: {type(state.navigator).__name__}")

        state.manipulator = build_manipulator(CFG["manipulator"], world)
        log(f"Manipulator: {type(state.manipulator).__name__}")

        # ── 4. Cube + point markers ──────────────────────────
        task = CFG["task"]
        point_a = np.array(task["point_a"], dtype=float)
        point_b = np.array(task["point_b"], dtype=float)
        cube_size = float(task["cube_size"])
        cube_half = cube_size / 2.0

        world.scene.add(
            DynamicCuboid(
                prim_path="/World/TargetCube",
                name="target_cube",
                position=np.array([point_a[0], point_a[1], cube_half]),
                scale=np.array([cube_size, cube_size, cube_size]),
                color=np.array([1.0, 0.85, 0.0]),
            )
        )
        marker_size = 0.20
        world.scene.add(
            VisualCuboid(
                prim_path="/World/MarkerA",
                name="marker_a",
                position=np.array([point_a[0], point_a[1], 0.01]),
                scale=np.array([marker_size, marker_size, 0.02]),
                color=np.array([0.0, 1.0, 0.0]),
            )
        )
        world.scene.add(
            VisualCuboid(
                prim_path="/World/MarkerB",
                name="marker_b",
                position=np.array([point_b[0], point_b[1], 0.01]),
                scale=np.array([marker_size, marker_size, 0.02]),
                color=np.array([1.0, 0.0, 0.0]),
            )
        )
        log(f"Cube at A={point_a.tolist()}, marker B={point_b.tolist()}")

        # ── 5. Reset & play; then finalize articulation-dependent state
        await world.reset_async()
        await world.play_async()
        state.manipulator.reset()  # open gripper (needs live articulation)
        log("World reset & playing.")

        # ── 6. Chase-cam behind the AMR ──────────────────────
        start_pos = state.navigator.get_pose()[0]
        set_camera_view(
            eye=start_pos + np.array([-2.5, 0.0, 1.8]),
            target=start_pos + np.array([1.0, 0.0, 0.6]),
        )
        log("Chase camera set.")

        # ── 7. Let physics settle a bit ──────────────────────
        for _ in range(60):
            await omni.kit.app.get_app().next_update_async()
        await world.pause_async()

        log(state.summary())
        log("Bootstrap complete. apps/run_*.py can now import from core.state.")

except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())
    raise
