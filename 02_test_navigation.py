# ============================================================
# 02: Test Navigation (uses the pluggable architecture)
# Loads hospital.usd, spawns Nova Carter via WaypointNavigator,
# plans A → B with RRT*, drives the AMR along the path.
#
# Run via: python3 run_in_isaac.py midterm_project/02_test_navigation.py
# Log:     cache/isaac-sim/logs/midterm_02.log
# ============================================================
import sys
import traceback

import numpy as np
import yaml
import omni.kit.app
import omni.usd

# Ensure we can import the core/ package
sys.path.insert(0, "/workspace/midterm_project")

# Force-reload core.* modules — Isaac Sim's Python keeps modules in
# sys.modules across script runs, so edits on disk wouldn't take effect
# without this.
for _m in list(sys.modules):
    if _m == "core" or _m.startswith("core."):
        del sys.modules[_m]

from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.viewports import set_camera_view

from core.factory import build_planner, build_navigator
from core.navigator import NavStatus

LOG_PATH = "/root/.nvidia-omniverse/logs/midterm_02.log"
_lines: list[str] = []


def log(msg: str):
    print(msg, flush=True)
    _lines.append(str(msg))
    try:
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(_lines))
    except Exception:
        pass


try:
    with open("/workspace/midterm_project/config.yaml") as f:
        CFG = yaml.safe_load(f)

    ASSETS_ROOT = get_assets_root_path()
    log(f"ASSETS_ROOT = {ASSETS_ROOT}")

    # ── 1. Open hospital stage ──────────────────────────────
    hospital_usd = ASSETS_ROOT + CFG["scene"]["usd_path"]
    log(f"Opening hospital: {hospital_usd}")
    await omni.usd.get_context().open_stage_async(hospital_usd)
    for _ in range(60):
        await omni.kit.app.get_app().next_update_async()
    log("Hospital stage opened.")

    # ── 2. World + planner + navigator ──────────────────────
    world = World(stage_units_in_meters=1.0)
    await world.initialize_simulation_context_async()
    log("World initialized.")

    # Use the config-selected planner (rrt_star by default). Flip to
    # StraightLine with FORCE_STRAIGHT_LINE=True if RRT* misbehaves.
    FORCE_STRAIGHT_LINE = False
    planner_cfg = dict(CFG["planner"])
    if FORCE_STRAIGHT_LINE:
        planner_cfg["type"] = "straight_line"
    planner = build_planner(planner_cfg)
    log(f"Planner: {type(planner).__name__} (forced={FORCE_STRAIGHT_LINE})")

    # Build planner's scene representation — needs the stage loaded.
    planner.build(world)
    log("Planner built.")

    navigator = build_navigator(CFG["navigator"], planner, world)
    log(f"Navigator: {type(navigator).__name__}")

    # Sanity-check: is the start position we configured actually free?
    start_cfg = np.array(
        CFG["navigator"]["robot"]["start_position"], dtype=float
    )[:2]
    log(f"Start valid? {planner.is_valid(tuple(start_cfg))}")

    # ── 3. Goal marker ──────────────────────────────────────
    goal_xy = np.array(CFG["task"]["point_a"], dtype=float)[:2]
    world.scene.add(
        VisualCuboid(
            prim_path="/World/GoalMarker",
            name="goal_marker",
            position=np.array([goal_xy[0], goal_xy[1], 0.05]),
            scale=np.array([0.20, 0.20, 0.02]),
            color=np.array([0.0, 1.0, 0.0]),
        )
    )
    log(f"Goal: Point A = {goal_xy.tolist()}")
    log(f"Goal valid?  {planner.is_valid(tuple(goal_xy))}")

    # ── 4. Reset + play ─────────────────────────────────────
    await world.reset_async()
    await world.play_async()

    # Chase-cam
    start_pos = navigator.get_pose()[0]
    set_camera_view(
        eye=start_pos + np.array([-2.5, 0.0, 1.8]),
        target=start_pos + np.array([1.0, 0.0, 0.6]),
    )

    # ── 5. Command goal; nav handles planning internally ────
    navigator.set_goal(goal_xy)

    # ── 6. Physics loop ─────────────────────────────────────
    # Drive navigator.step() each simulation tick via a physics callback —
    # this is the recommended pattern for applying robot actions in Isaac Sim.
    max_ticks = int(CFG["simulation"]["max_ticks"])
    state = {"status": NavStatus.RUNNING, "tick": 0}

    def on_physics(_step_size):
        state["status"] = navigator.step()
        state["tick"] += 1

    world.add_physics_callback("nav_step", callback_fn=on_physics)

    last_log_tick = 0
    while state["tick"] < max_ticks:
        await omni.kit.app.get_app().next_update_async()
        if state["tick"] - last_log_tick >= 500:
            last_log_tick = state["tick"]
            pos = navigator.get_pose()[0][:2]
            dist = float(np.linalg.norm(pos - goal_xy))
            log(f"  tick {state['tick']}: pos={pos.tolist()}, "
                f"dist={dist:.2f} m, status={state['status'].value}")
        if state["status"] in (NavStatus.REACHED, NavStatus.FAILED):
            break

    tick = state["tick"]
    status = state["status"]

    if status == NavStatus.REACHED:
        log(f"REACHED Point A at tick {tick}")
    elif status == NavStatus.FAILED:
        log("FAILED to reach Point A (no path / out of replans)")
    else:
        log(f"TIMEOUT at {max_ticks} ticks (still {status.value})")

    navigator.stop()
    await world.pause_async()
    log("Navigation test complete.")

except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())
    raise
