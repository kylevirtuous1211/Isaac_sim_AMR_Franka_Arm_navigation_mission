# ============================================================
# apps/bootstrap.py — load hospital + spawn robots ONCE
#
# Subsequent scripts (apps/run_cortex.py, tests/scripts/*) import
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
sys.modules.pop("apps._common", None)
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

    # ──────────────────────────────────────────────────────────
    # _apply_franka_chassis_collision_filter:
    #   At low mount offsets (≤0.4m) the Franka panda_link0 collision
    #   shape intersects the AMR chassis_link, and PhysX resolves the
    #   contact by pushing the AMR's wheels into the floor. The clean
    #   fix is a USD FilteredPairsAPI on the Franka prim listing the
    #   chassis as a filtered pair: PhysX never generates contacts for
    #   filtered pairs, so the two articulations can geometrically
    #   intersect without coupling.
    #   Idempotent — uses SetTargets so re-applying on fast-reset
    #   doesn't accumulate duplicate targets.
    # ──────────────────────────────────────────────────────────
    def _author_surface_gripper(cfg, _log) -> None:
        """Idempotently re-author the SurfaceGripper prim + cube D6 joint.

        Authoring was previously disabled because the original mount=0.5
        left an 8+ cm gap between the gripper anchor and the cube,
        outside max_grip_distance. With mount=0.35 + collision filter
        + joint reset, the gap is now 4-7 cm — comfortably inside
        max_grip_distance=0.15 — so the D6 joint engages cleanly and
        replaces cube_carry_sync's kinematic teleport with a real
        physics attachment.
        """
        try:
            import omni.usd
            from core.surface_gripper_setup import author_surface_gripper
            stage = omni.usd.get_context().get_stage()
            path = author_surface_gripper(stage, cfg)
            _log(f"SurfaceGripper authored at {path}")
        except Exception as e:
            _log(f"WARN: SurfaceGripper authoring failed: {e}")

    def _author_fixed_joint_mount(cfg, _log) -> None:
        """Author the chassis↔panda_link0 FixedJoint when mount_mode is set.

        No-op when mount_mode != "fixed_joint", so it's safe to call
        unconditionally on both the full-bootstrap and fast-reset paths.
        Idempotent: re-authoring overwrites the same prim instead of
        accumulating duplicates.
        """
        manip_cfg = cfg.get("manipulator") or {}
        if manip_cfg.get("mount_mode", "pose_sync") != "fixed_joint":
            return
        try:
            import omni.usd
            from core.franka_mount_joint import author_franka_mount_joint
            from core.articulation_tuning import bump_solver_iterations
            stage = omni.usd.get_context().get_stage()
            joint_path = author_franka_mount_joint(stage, cfg)
            _log(f"FixedJoint mount authored at {joint_path}")
            tuned = bump_solver_iterations(
                stage,
                [
                    cfg["navigator"]["robot"].get("prim_path", "/World/NovaCarter"),
                    manip_cfg.get("prim_path", "/World/Franka"),
                ],
            )
            _log(f"Solver iterations bumped (32/8) on: {tuned}")
        except Exception as e:
            _log(f"WARN: FixedJoint mount authoring failed: {e}")

    def _apply_franka_chassis_collision_filter(cfg, _log) -> None:
        try:
            from pxr import UsdPhysics, Sdf
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            franka_path = cfg["manipulator"].get("prim_path", "/World/Franka")
            chassis_path = cfg["manipulator"].get(
                "mount_to", "/World/NovaCarter/chassis_link"
            )
            franka_prim = stage.GetPrimAtPath(franka_path)
            if not franka_prim or not franka_prim.IsValid():
                _log(f"WARN: Franka prim {franka_path} not found — skipping collision filter")
                return
            fp_api = UsdPhysics.FilteredPairsAPI.Apply(franka_prim)
            rel = fp_api.CreateFilteredPairsRel()
            rel.SetTargets([Sdf.Path(chassis_path)])  # idempotent
            _log(f"Filtered collision pair: {franka_path} ↔ {chassis_path}")
        except Exception as e:
            _log(f"WARN: collision filter setup failed: {e}")

    # ──────────────────────────────────────────────────────────
    # _reset_to_start_poses:
    #   world.reset_async() only returns articulations to the defaults
    #   captured at spawn time. If any object's reset path didn't fully
    #   restore its pose, the scene ends up in a mixed state.
    #
    #   We explicitly teleport the AMR, the Franka, and the cube to their
    #   configured starts, zero all velocities, and clear navigator /
    #   manipulator internal state. After this helper runs, the scene is
    #   in the exact "stacked start" configuration regardless of whatever
    #   prior script did.
    # ──────────────────────────────────────────────────────────
    def _reset_to_start_poses(world, cfg) -> None:
        nav_cfg = cfg["navigator"]["robot"]
        amr_pos = np.array(nav_cfg.get("start_position", [0.0, 0.0, 0.0]),
                           dtype=float)
        amr_ori = np.array(nav_cfg.get("start_orientation", [1.0, 0.0, 0.0, 0.0]),
                           dtype=float)

        # AMR: root pose + zero velocities + clear navigator bookkeeping
        nav = state.navigator
        if nav is not None and nav.robot is not None:
            nav.robot.set_world_pose(position=amr_pos, orientation=amr_ori)
            try:
                nav.robot.set_linear_velocity(np.zeros(3))
                nav.robot.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
            try:
                nav.robot.set_joint_velocities(
                    np.zeros(len(nav.robot.dof_names))
                )
            except Exception:
                pass
            nav._reached_latch = False
            nav._idx = 0
            nav._waypoints = []
            nav._goal = None
            nav._stuck_counter = 0
            nav._last_pos = None
            nav._replans_used = 0

        # Franka: stacked on AMR for mobile manip, or at configured
        # station position otherwise. Orientation matches AMR.
        manip = state.manipulator
        manip_cfg = cfg["manipulator"]
        if manip is not None and getattr(manip, "franka", None) is not None:
            mount_to = manip_cfg.get("mount_to")
            if mount_to:
                offset = np.array(
                    manip_cfg.get("mount_local_offset", [0.0, 0.0, 0.50]),
                    dtype=float,
                )
                franka_pos = amr_pos + offset
            else:
                franka_pos = np.array(
                    manip_cfg.get("position", [0.0, 0.0, 0.0]), dtype=float,
                )
            manip.franka.set_world_pose(position=franka_pos, orientation=amr_ori)
            try:
                manip.franka.set_linear_velocity(np.zeros(3))
                manip.franka.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

        # Cube: drop back at point_a, resting on the floor. Zero velocities
        # so physics doesn't punt it on the next physics tick.
        cube = world.scene.get_object("target_cube")
        task_cfg = cfg["task"]
        if cube is not None:
            cube_half = float(task_cfg["cube_size"]) / 2.0
            cube_pos = np.array(
                [task_cfg["point_a"][0], task_cfg["point_a"][1], cube_half],
                dtype=float,
            )
            cube.set_world_pose(
                position=cube_pos,
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
            try:
                cube.set_linear_velocity(np.zeros(3))
                cube.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

        # Markers: re-position the green (A) and red (B) pads to match
        # the CURRENT config. Without this, edits to point_a/point_b only
        # take effect on a full reload — fast-reset would leave the visual
        # markers at their original spawn pose.
        marker_a = world.scene.get_object("marker_a")
        if marker_a is not None:
            marker_a.set_world_pose(
                position=np.array(
                    [task_cfg["point_a"][0], task_cfg["point_a"][1], 0.01],
                    dtype=float,
                ),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        marker_b = world.scene.get_object("marker_b")
        if marker_b is not None:
            marker_b.set_world_pose(
                position=np.array(
                    [task_cfg["point_b"][0], task_cfg["point_b"][1], 0.01],
                    dtype=float,
                ),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )

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

        # Abort any stale run_*.py loop still running in the background
        # (Kit process keeps them alive across our script invocations).
        # Inline instead of state.bump_generation() because core.state is
        # preserved across script runs, so new functions added there don't
        # show up in the already-imported module.
        state.nav_generation = getattr(state, "nav_generation", 0) + 1

        world = state.world
        # Re-apply USD-level mount/articulation attributes BEFORE
        # world.reset_async(). PhysX caches physxArticulation:fixedBase
        # and joint excludeFromArticulation when it parses the
        # articulation; modifying them after reset_async() does not
        # propagate. Authoring before reset guarantees the next
        # initialize_simulation reads the current values.
        _apply_franka_chassis_collision_filter(CFG, log)
        _author_surface_gripper(CFG, log)
        _author_fixed_joint_mount(CFG, log)

        await world.reset_async()
        # Pause physics around the multi-body teleport. With mount_mode=
        # fixed_joint and rootJoint disabled, the FixedJoint between
        # chassis_link and panda_link0 is the ONLY constraint holding
        # them together. Any PhysX tick where Carter and Franka aren't
        # at the correct relative offset triggers an infinite-stiffness
        # impulse from the FixedJoint and the broadphase NaNs out.
        # Pausing guarantees both set_world_pose calls land before the
        # next tick.
        await world.pause_async()
        state.manipulator.reset()
        _reset_to_start_poses(world, CFG)
        # Re-install the mount pose-sync callback if the manipulator is
        # configured for mobile manip. Safe to call unconditionally — it's
        # a no-op when mount_to isn't set or when mount_mode=fixed_joint,
        # and idempotent when it does install.
        state.manipulator.ensure_mount_sync(world)
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
    else:
        # Full-bootstrap path also invalidates stale loops.
        state.nav_generation = getattr(state, "nav_generation", 0) + 1

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
        _apply_franka_chassis_collision_filter(CFG, log)
        # FixedJoint mount must be authored AFTER both articulations exist
        # (Carter from build_navigator, Franka from build_manipulator) and
        # BEFORE world.reset_async() so PhysX picks up the constraint at
        # simulation start. No-op when mount_mode != "fixed_joint".
        _author_fixed_joint_mount(CFG, log)

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

        # SurfaceGripper authoring — must happen AFTER the cube exists
        # and BEFORE world.reset_async() so PhysX picks up the joint.
        _author_surface_gripper(CFG, log)

        # ── 5. Reset & play; then finalize articulation-dependent state
        # Order matters on first boot: play before manipulator.reset (the
        # gripper joint handles aren't live until the articulation is
        # playing). Then PAUSE around the teleport — with mount_mode=
        # fixed_joint and rootJoint disabled, any tick where Carter and
        # Franka aren't aligned causes the FixedJoint to apply a huge
        # impulse and the broadphase NaNs out.
        await world.reset_async()
        await world.play_async()
        state.manipulator.reset()                  # open gripper (needs live articulation)
        await world.pause_async()
        _reset_to_start_poses(world, CFG)
        await world.play_async()
        state.manipulator.ensure_mount_sync(world)
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
