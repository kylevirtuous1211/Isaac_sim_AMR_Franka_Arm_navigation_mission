# Codebase Trace — midterm_project

A file-by-file map: what each module does, who imports it, and which
modules it reaches into. Read this before any non-trivial change so you
don't break a load-bearing import.

Generated 2026-05-04, post-physical-grasp refactor (mount_mode=fixed_joint,
SurfaceGripper as source of truth, no cube_carry_sync).

---

## The canonical path

```
run_in_isaac.py midterm_project/apps/bootstrap.py     # spawn world (once)
run_in_isaac.py midterm_project/apps/run_cortex.py    # multi-episode pick-and-place
```

`tune_cortex.py` (top level) is a host-side wrapper that drives many
run_cortex iterations with adjusted config knobs.

Everything else in `apps/` is a one-off diagnostic.

---

## Dependency graph (canonical edges only)

```
tune_cortex.py
    ├─→ apps/bootstrap.py             (run_bootstrap subprocess)
    └─→ apps/run_cortex.py            (run_pipeline subprocess — name is legacy,
                                       script targeted is run_cortex)

apps/bootstrap.py                     populates core.state once per Kit process
    ├─→ apps/_common.py
    ├─→ core/state.py                 set state.world, .navigator, .manipulator, .planner
    ├─→ core/factory.py               build_planner / _navigator / _manipulator
    │       ├─→ core/planner.py
    │       ├─→ core/navigator.py
    │       └─→ core/manipulator.py
    │               └─→ core/surface_gripper_setup.py  (SurfaceGripperWrapper)
    ├─→ core/franka_mount_joint.py    author chassis↔panda_link0 FixedJoint
    │       (only when manipulator.mount_mode == "fixed_joint")
    ├─→ core/articulation_tuning.py   bump PhysX solver iterations on FJ mount
    ├─→ core/surface_gripper_setup.py author SurfaceGripper prim + D6 joint
    └─→ scenes/hospital.py            load_hospital()

apps/run_cortex.py                    reads from core.state — NEVER spawns
    ├─→ apps/_common.py
    ├─→ core/state.py
    ├─→ core/episode.py               apply_episode_to_cfg, reset_world_for_episode
    │       └─→ core/randomizer.py    (Episode dataclass)
    ├─→ core/randomizer.py            Randomizer.sample_episode()
    │       └─→ core/planner.py       Planner.is_valid() for free-space checks
    └─→ cortex/network.py             make_decider_network(ctx)
            ├─→ cortex/context.py     MobileManipContext (subclass of DfRobotApiContext)
            └─→ cortex/states.py      leaf DfStates
                    ├─→ core/navigator.py     (NavStatus enum)
                    └─→ core/manipulator.py   (ManipStatus enum)
```

Every directed arrow is "module A imports something from module B."
Read top-to-bottom for execution order.

---

## `apps/` — entry points

| File | Status | Purpose |
|------|--------|---------|
| `_common.py` | canonical | sys.path + module-reload helpers, logger factory, `load_config()`. Imported by every script in `apps/` and every test script. |
| `bootstrap.py` | canonical | Idempotent world setup. First call: 60–180s (load hospital). Subsequent calls: ~5s fast-reset that re-pins poses. Authors FixedJoint mount + SurfaceGripper. |
| `run_cortex.py` | canonical | Domain-randomized multi-episode loop. Each episode runs the cortex decider until terminal block_state. |
| `diag_surface_gripper.py` | diagnostic | One-off dump of SurfaceGripper + D6 joint state to `sg_diag.log`. Useful when grasp validation fails. Not imported by anything. |

Removed in cleanup (May 2026):
- `run_pipeline.py` — legacy 5-phase imperative orchestrator, replaced by `run_cortex.py` + cortex decider network
- `run_nav.py` — standalone "drive to point_a" diagnostic; functionality covered by run_cortex
- `run_manip.py` — standalone arm exerciser; functionality covered by run_cortex
- `force_reboot.py` — one-off mount-mode-switch helper

---

## `core/` — behavior modules (no Isaac entry points)

Pure library code. None of these are run as scripts; they're imported
by `apps/*.py` and by each other.

| File | Imported by | Imports from core | Purpose |
|------|-------------|-------------------|---------|
| `state.py` | bootstrap, run_cortex, cortex/states (transitively) | — | Process-wide singleton cache. Holds `world`, `navigator`, `manipulator`, `planner`, `config`, `nav_generation`. PRESERVED across script reloads (see `_common.py:_PRESERVE`). |
| `factory.py` | bootstrap | planner, navigator, manipulator | Type-string → class registry. Lets config.yaml flip implementations without code changes. |
| `planner.py` | factory, navigator (arg), randomizer | — | `Planner` ABC + `RRTStarPlanner`, `StraightLinePlanner`, `OccupancyGrid`. |
| `navigator.py` | factory, cortex/states (NavStatus) | planner | `Navigator` ABC + `WaypointNavigator`. The decider network reads `NavStatus` enum values to know when nav has finished. |
| `manipulator.py` | factory, cortex/states (ManipStatus) | surface_gripper_setup | `Manipulator` ABC + `FrankaRMPflowManipulator`. Phase machine: above_pick → at_pick → grasp → lift → carry → above_place → at_place → release → retract → done. The `carry` phase actively holds a stow pose during transit. |
| `surface_gripper_setup.py` | bootstrap, manipulator (lazy), diag_surface_gripper | — | Authors the SurfaceGripper prim + D6 joint between panda_hand and the cube. `SurfaceGripperWrapper` is the runtime API used by manipulator.py. |
| `franka_mount_joint.py` | bootstrap (when mount_mode=fixed_joint) | — | Authors a `UsdPhysics.FixedJoint` between chassis_link and panda_link0 (`excludeFromArticulation=True` to side-step a wheel-jamming bug). |
| `articulation_tuning.py` | bootstrap (when mount_mode=fixed_joint) | — | Bumps PhysX solver iterations on coupled articulations so the FixedJoint doesn't visibly compliance-jitter. |
| `randomizer.py` | run_cortex, episode | planner | Samples (start, cube, place) XY honoring config bounds + planner free-space + min separation. |
| `episode.py` | run_cortex | randomizer | `apply_episode_to_cfg` mutates the dict; `reset_world_for_episode` teleports robots/cube/markers to match. Mirrors bootstrap's `_reset_to_start_poses` for steady-state per-episode resets. |

---

## `cortex/` — decider network for mobile pick-and-place

Built on `isaacsim.cortex.framework.df`. Three modules; the only public
entry point is `make_decider_network(ctx)`.

| File | Purpose |
|------|---------|
| `context.py` | `MobileManipContext` extends `DfRobotApiContext`. Each tick `update_block_state()` reads cube/AMR/EE poses + gripper closed state and classifies into `BlockState` (NEED_NAV_TO_BLOCK, NEED_PICK, NEED_TRANSIT, NEED_PLACE, DONE, FAILED). Provides geometry helpers `cube_park_xy`, `b_standoff_xy`, `place_target_b`. |
| `states.py` | Leaf `DfState` / `DfAction` classes that wrap the navigator/manipulator API: `NavSetGoalState`, `NavStopState`, `ManipResetState`, `ManipPickState`, `ManipPlaceState`, `ManipWaitDoneState`, `BumpRetryState`, `OpenGripperState`, `DoneAction`, `FailAction`. `ManipWaitDoneState` validates pick success via `surface_gripper.gripped()` — empty means the closure was unphysical and we route to `BumpRetryState`. |
| `network.py` | `Dispatch` decider switches on `ctx.block_state` to one of six sub-deciders (`_build_nav_to_block`, `_build_pick_rlds`, `_build_transit`, `_build_place_rlds`, `fail`, `go_home`). Each sub-decider is a `_AutoRestartingStateMachineDecider(DfStateSequence([...]))` so that retries re-arm without forcing a Dispatch detour. |

Reactivity: when the cube moves mid-execution, the next tick's
classifier flips `block_state`, Dispatch returns a different child name,
and `df_descend` exits the in-flight sub-sequence cleanly.

---

## `scenes/`

| File | Purpose |
|------|---------|
| `hospital.py` | `load_hospital(force=False)` — idempotent hospital.usd loader. Caches the loaded path so a second call is a no-op. |

---

## `tests/` — pytest in-sim integration tests

Tests run on the host but assert on state inside the running Isaac Sim
container. They ship tiny scripts to `tests/scripts/`, send them via
`run_in_isaac.py`, and read JSON results back from the bind-mounted log
directory.

| File | Tests |
|------|-------|
| `conftest.py` | `run_in_isaac`, `sync_to_container`, `read_result`, `run_and_read` helpers. Session-scoped `ensure_bootstrap` fixture runs `apps/bootstrap.py` once per session. |
| `test_arm_mount.py` | After bootstrap, Franka XY ≈ AMR XY and Franka Z ≈ AMR Z + offset. After driving, Franka follows the AMR (FixedJoint contract). |
| `test_navigation_reset.py` | Bootstrap clears `_reached_latch` and the AMR can drive again to a fresh goal. |
| `test_scene_reset.py` | Bootstrap fast-reset returns the AMR to start, bumps `nav_generation`, and completes in <30s (didn't reload the stage). |
| `scripts/_testutil.py` | `write_result(name, dict)` — JSON serializer the in-sim scripts use. |
| `scripts/probe.py` | Dumps AMR/Franka pose, mount config, nav_generation, latch state. |
| `scripts/drive_briefly.py` | Sets a goal and runs nav up to 800 ticks. |
| `scripts/perturb_scene.py` | Pushes the AMR off origin and sets the latch — gives bootstrap something to reset. |

These tests verify the **infrastructure** the cortex pipeline depends
on (mount, bootstrap, navigation reset). They do not run a full cortex
episode end-to-end — the run_cortex.py log + `cortex_positions.stream.log`
are the operational signal for that.

---

## Top-level files

| File | Purpose |
|------|---------|
| `config.yaml` | Single source of truth for tunables: planner, navigator, manipulator (mount_mode, surface_gripper params, carry_height), task points, randomization bounds, simulation budgets. |
| `tune_cortex.py` | Closed-loop tuner. Sends bootstrap + run_cortex over many iterations, parses the log, applies rule-based config edits (clearance_height, pick_z_offset, mount offset, etc.) until convergence. |
| `Midterm_Assignment_Digital_Twin_Robotics.pdf` | Class assignment spec. |

---

## Removed in May 2026 cleanup

These files existed but were stale or redundant after the
physical-grasp refactor:

- `apps/run_pipeline.py` — superseded by `apps/run_cortex.py`. The
  cortex decider replaces the imperative 5-phase orchestrator.
- `apps/run_nav.py`, `apps/run_manip.py` — diagnostic single-purpose
  runners. Coverage subsumed by `run_cortex.py` (which exercises both).
- `apps/force_reboot.py` — one-off mount-mode switch helper, no longer
  needed once mount_mode is stable.
- `overnight_loop.py` — wrapper around tune_cortex for batch runs;
  reproducible via `tune_cortex.py --max-iters N` directly.
- `todo.md` — converted to GitHub issues / commit history.
- `cortex/states.py` `cube_carry_sync` machinery — kinematic substitute
  for actual grasping; replaced by SurfaceGripper D6 joint.
- `cortex/states.py` `DisablePoseSyncState` / `EnablePoseSyncState` /
  `RebaseFrankaToAmrState` / `SettleState` — only existed to manage the
  per-tick base teleport in `mount_mode=pose_sync`. With
  `mount_mode=fixed_joint` the base is held by a PhysX constraint and
  these states would actively conflict with it.
- `ManipResetState` `set_joint_positions(HOME_ARM_JOINTS)` — kinematic
  joint snap; replaced by RMPflow planning from whatever pose the arm
  is in.

The `pose_sync` code path in `core/manipulator.py` is preserved for a
future config flip (e.g. fast iteration with no mount mass loading);
the `fixed_joint` path is the canonical default.
