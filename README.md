# Isaac Sim AMR + Franka Arm — Hospital Navigation Mission

Digital-twin midterm project. A Nova Carter AMR navigates `hospital.usd`
and a Franka Panda arm picks up and places a cube. Domain-randomized
start/goal/cube/place positions per run.

- **Scene:** Isaac Sim built-in `Isaac/Environments/Hospital/hospital.usd`
- **AMR:** Nova Carter (differential drive, 2 drive wheels + 4 casters)
- **Arm:** Franka Panda with RMPflow obstacle avoidance
- **Planner:** RRT\* (sampling-based, asymptotically optimal, shortcut-smoothed)

## Architecture

Three orthogonal roles, all selectable via `config.yaml` → `type`:

```
          ┌──────────────┐
          │ Orchestrator │  (async FSM / Cortex decider)
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
│ Manipulator  │  (FrankaRMPflow)
└──────────────┘
```

Each tier is an abstract base class in `core/` with one or more concrete
implementations registered in `core/factory.py`. Swapping robots or
planners is a one-line `config.yaml` change.

## Layout

```
midterm_project/
├── README.md
├── config.yaml                  # all tunables (planner / nav / arm / rand)
├── todo.md                      # midterm rubric + tasks
├── core/
│   ├── planner.py               # Planner ABC + RRTStarPlanner + StraightLinePlanner + OccupancyGrid
│   ├── navigator.py             # Navigator ABC + WaypointNavigator
│   ├── manipulator.py           # Manipulator ABC + FrankaRMPflow
│   ├── randomizer.py            # Randomizer for Part 4 (domain randomization)
│   ├── factory.py               # registries + build_planner / build_navigator / build_manipulator
│   └── state.py                 # process-wide cache of live handles (shared across apps)
├── scenes/
│   └── hospital.py              # idempotent hospital.usd loader
└── apps/                        # thin orchestrator scripts (sent via run_in_isaac.py)
    ├── _common.py               # import/path/logger helpers
    ├── bootstrap.py             # ONCE: load stage, spawn robots, populate core.state
    └── run_nav.py               # AMR navigation using cached state
```

## Why `apps/` + `core/state.py`?

Isaac Sim's `isaacsim.code_editor.vscode` extension runs every script
we send in the **same long-running Kit Python process**. Python modules,
the `World` singleton, and the loaded USD stage all persist across script
invocations.

We exploit that:

- `apps/bootstrap.py` is idempotent. It loads `hospital.usd` (60–180s the
  first time), spawns the robots, and stores handles in `core.state`.
  Re-running it is a no-op.
- `apps/run_*.py` scripts **skip the stage reload** — they `import state`
  and reuse the live world. Iteration cost: seconds, not minutes.

To start over: set `FORCE_REBOOT=1` or call `core.state.teardown()`.

## Quick start

Prereqs: Isaac Sim container running with `run_in_isaac.py` socket.

```bash
# 1) Bootstrap once — loads hospital + spawns Nova Carter + Franka
python3 run_in_isaac.py midterm_project/apps/bootstrap.py

# 2) Run navigation — reuses the live stage (fast)
python3 run_in_isaac.py midterm_project/apps/run_nav.py
```

Logs land in `cache/isaac-sim/logs/<script>.log` on the host.

## Swapping implementations

Edit `config.yaml`:

```yaml
planner:
  type: straight_line          # was: rrt_star — skip RRT* for open-space tests
navigator:
  robot:
    usd_path: "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"   # swap to JetBot
manipulator:
  type: franka_rmpflow         # only registered manipulator type
```

No code changes required — `core/factory.py` picks the registered class
by `type` string.

## Project status

See `todo.md` for the rubric breakdown.

- [x] Scene Setup
- [x] Navigation (RRT\* + WaypointNavigator — verified on Point A)
- [ ] Arm Pick & Place (architecture built; end-to-end pending)
- [ ] Domain Randomization (Randomizer implemented; pipeline pending)
- [ ] Report + video
- [ ] Bonus: 3DGS scene + object reconstruction

## Running inside the parent repo

This repo is meant to live at `isaac-sim-quickstart/midterm_project/`
in the main [isaac-sim-quickstart](../) project. That parent repo owns
the Isaac Sim Docker setup, `run_in_isaac.py`, and the `configs/rmpflow/`
tuning files that the manipulator consumes.
