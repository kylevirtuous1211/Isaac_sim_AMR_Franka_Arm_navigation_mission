# Isaac Sim AMR + Franka Arm — Hospital Navigation Mission

Digital-twin midterm project. A Nova Carter AMR navigates `hospital.usd`
and a Franka Panda arm picks up and places a cube. Domain-randomized
start/goal/cube/place positions per run.

- **Scene:** Isaac Sim built-in `Isaac/Environments/Hospital/hospital.usd`
- **AMR:** Nova Carter (differential drive, 2 drive wheels + 4 casters)
- **Arm:** Franka Panda with RMPflow obstacle avoidance, mounted on the
  AMR via a `UsdPhysics.FixedJoint` (mount_mode=fixed_joint)
- **Planner:** RRT\* (sampling-based, asymptotically optimal, shortcut-smoothed)
- **Decider:** Cortex DfStateMachineDecider over a 6-state block_state
  machine (nav → pick → transit → place)

## Architecture

```
          ┌──────────────┐
          │  run_cortex  │  multi-episode loop, randomizer driven
          └──────┬───────┘
                 ▼
          ┌──────────────┐
          │  cortex/     │  Dispatch → {nav, pick, transit, place}
          └──────┬───────┘
                 │
     ┌───────────┴───────────┐
     ▼                       ▼
┌─────────┐           ┌─────────────┐
│Navigator│ ── uses ─▶│   Planner   │  (RRTStar / StraightLine)
└─────────┘           └─────────────┘
     │
     ▼
┌──────────────┐
│ Manipulator  │  (FrankaRMPflow + SurfaceGripper D6 joint)
└──────────────┘
```

Each tier is an abstract base class in `core/` with concrete
implementations registered in `core/factory.py`. Swapping robots or
planners is a one-line `config.yaml` change. The cortex decider in
`cortex/` is independent of which navigator/manipulator is plugged in
— it talks to the abstract API.

## Layout

```
midterm_project/
├── README.md
├── config.yaml                  # all tunables (planner / nav / arm / rand)
├── tune_cortex.py               # closed-loop tuner: drives run_cortex iterations
├── core/                        # behavior modules (no Isaac entry points)
│   ├── planner.py               # Planner ABC + RRTStarPlanner / StraightLinePlanner / OccupancyGrid
│   ├── navigator.py             # Navigator ABC + WaypointNavigator (NavStatus)
│   ├── manipulator.py           # Manipulator ABC + FrankaRMPflow (ManipStatus, phases incl. carry)
│   ├── randomizer.py            # Episode sampler honoring planner.is_valid()
│   ├── episode.py               # apply_episode_to_cfg + reset_world_for_episode
│   ├── factory.py               # build_planner / build_navigator / build_manipulator
│   ├── state.py                 # process-wide cache of live handles (shared across apps)
│   ├── surface_gripper_setup.py # author SurfaceGripper + D6 joint, SG wrapper
│   ├── franka_mount_joint.py    # author FixedJoint chassis ↔ panda_link0
│   └── articulation_tuning.py   # bump PhysX solver iterations on FixedJoint mounts
├── cortex/                      # decider network for the mobile pick-and-place
│   ├── context.py               # MobileManipContext + BlockState classifier
│   ├── states.py                # leaf DfStates (nav / pick / place wrappers)
│   └── network.py               # Dispatch decider + make_decider_network()
├── scenes/
│   └── hospital.py              # idempotent hospital.usd loader
├── apps/                        # entry points (sent via run_in_isaac.py)
│   ├── _common.py               # import/path/logger helpers
│   ├── bootstrap.py             # ONCE: load stage, spawn robots, populate core.state
│   ├── run_cortex.py            # canonical orchestrator: domain-randomized loop
│   └── diag_surface_gripper.py  # one-off diagnostic for SG + D6 joint state
├── tests/                       # pytest, runs in-sim probe scripts via run_in_isaac
│   ├── conftest.py
│   ├── test_arm_mount.py
│   ├── test_navigation_reset.py
│   ├── test_scene_reset.py
│   └── scripts/                 # tiny scripts shipped into the container per test
└── docs/
    └── codebase_trace.md        # this map of who-uses-what
```

## Why `apps/` + `core/state.py`?

Isaac Sim's `isaacsim.code_editor.vscode` extension runs every script
we send in the **same long-running Kit Python process**. Python modules,
the `World` singleton, and the loaded USD stage all persist across script
invocations.

We exploit that:

- `apps/bootstrap.py` is idempotent. It loads `hospital.usd` (60–180s the
  first time), spawns the robots, and stores handles in `core.state`.
  Re-running it is a no-op (fast-reset path: ~5s).
- `apps/run_cortex.py` skips the stage reload — it `import core.state`
  and reuses the live world. Iteration cost: seconds, not minutes.

To start over: set `FORCE_REBOOT=1` or call `core.state.teardown()`.

## Quick start

Prereqs: Isaac Sim container running with `run_in_isaac.py` socket
(see the parent repo's `docker-compose.yml`).

```bash
# 1) Bootstrap once — loads hospital + spawns Nova Carter + Franka,
#    authors the FixedJoint mount + SurfaceGripper.
python3 run_in_isaac.py midterm_project/apps/bootstrap.py

# 2) Run the multi-episode pick-and-place loop — reuses the live stage.
python3 run_in_isaac.py midterm_project/apps/run_cortex.py
```

Logs land in `cache/isaac-sim/logs/<script>.log` on the host. Per-tick
position telemetry streams to `cortex_positions.stream.log`
(`tail -f` to watch).

For closed-loop parameter tuning (over many run_cortex iterations):

```bash
python3 midterm_project/tune_cortex.py --max-iters 20
```

## Swapping implementations

Edit `config.yaml`:

```yaml
planner:
  type: straight_line          # was: rrt_star — skip RRT* for open-space tests
manipulator:
  mount_mode: pose_sync        # alternate: kinematic per-tick teleport
                               # (default: fixed_joint — physical PhysX constraint)
  type: franka_rmpflow         # only registered manipulator type
```

No code changes required — `core/factory.py` picks the registered class
by `type` string.

## Project status

- [x] Scene Setup
- [x] Navigation (RRT\* + WaypointNavigator)
- [x] Arm Pick & Place (Cortex decider + RMPflow + SurfaceGripper D6 joint)
- [x] Domain Randomization (multi-episode loop in run_cortex)
- [x] Physical mount + grasp (FixedJoint, SurfaceGripper instead of kinematic teleport)
- [ ] Report + video
- [ ] Bonus: 3DGS scene + object reconstruction

See [docs/codebase_trace.md](docs/codebase_trace.md) for the
file-by-file responsibility map.

## Running inside the parent repo

This repo is meant to live at `isaac-sim-quickstart/midterm_project/`
in the main [isaac-sim-quickstart](../) project. That parent repo owns
the Isaac Sim Docker setup, `run_in_isaac.py`, and the `configs/rmpflow/`
tuning files that the manipulator consumes.
