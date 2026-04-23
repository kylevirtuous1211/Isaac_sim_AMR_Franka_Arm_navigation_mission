# ============================================================
# 01: Scene Setup
# Load hospital.usd, spawn Nova Carter + Franka (via the core/
# factory), add a cube at Point A and visual markers at A & B,
# and run a brief verification.
#
# Run via: python3 run_in_isaac.py midterm_project/01_scene_setup.py
# Log:     cache/isaac-sim/logs/midterm_01.log
# ============================================================
import sys
import traceback

import numpy as np
import yaml
import omni.kit.app
import omni.usd

# core/ package lives here
sys.path.insert(0, "/workspace/midterm_project")
# Force-reload core.* so edits between runs take effect
for _m in list(sys.modules):
    if _m == "core" or _m.startswith("core."):
        del sys.modules[_m]

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.viewports import set_camera_view

from core.factory import build_planner, build_navigator, build_manipulator


LOG_PATH = "/root/.nvidia-omniverse/logs/midterm_01.log"
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

    # ── 1. Open hospital.usd as the base stage ──────────────
    hospital_usd = ASSETS_ROOT + CFG["scene"]["usd_path"]
    log(f"Opening hospital stage: {hospital_usd}")
    await omni.usd.get_context().open_stage_async(hospital_usd)
    for _ in range(60):
        await omni.kit.app.get_app().next_update_async()
    log("Hospital stage opened.")

    # ── 2. Create World + factory components ────────────────
    world = World(stage_units_in_meters=1.0)
    await world.initialize_simulation_context_async()
    log("World initialized.")

    planner = build_planner(CFG["planner"])
    planner.build(world)
    log(f"Planner: {type(planner).__name__}")

    navigator = build_navigator(CFG["navigator"], planner, world)
    log(f"Navigator: {type(navigator).__name__}")

    manipulator = build_manipulator(CFG["manipulator"], world)
    log(f"Manipulator: {type(manipulator).__name__}")

    # ── 3. Cube at Point A + visual markers ─────────────────
    task_cfg = CFG["task"]
    point_a = np.array(task_cfg["point_a"], dtype=float)
    point_b = np.array(task_cfg["point_b"], dtype=float)
    cube_size = float(task_cfg["cube_size"])
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
    log(f"Cube placed at Point A = {point_a.tolist()}")

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
    log(f"Markers: A={point_a.tolist()} (green), B={point_b.tolist()} (red)")

    # ── 4. Reset + play (articulations get initialized here) ─
    await world.reset_async()
    await world.play_async()
    log("World reset & playing.")

    # Post-reset gripper / controller init (requires live articulation).
    manipulator.reset()

    # ── 5. Position viewport camera AFTER reset ─────────────
    # open_stage_async and reset_async can both reset the camera; setting it
    # after ensures it sticks.
    start_pos = navigator.get_pose()[0]
    cam_eye = start_pos + np.array([-2.5, 0.0, 1.8])
    cam_target = start_pos + np.array([1.0, 0.0, 0.6])
    set_camera_view(eye=cam_eye, target=cam_target)
    log(f"Camera set (chase view): eye={cam_eye.tolist()}, target={cam_target.tolist()}")

    # ── 6. Let physics settle ───────────────────────────────
    log("Running 60 ticks for verification...")
    for _ in range(60):
        await omni.kit.app.get_app().next_update_async()

    carter_pos, _ = navigator.get_pose()
    franka_pos, _ = manipulator.franka.get_world_pose()
    log(f"Final Nova Carter pos: {carter_pos.tolist()}")
    log(f"Final Franka pos:      {franka_pos.tolist()}")

    await world.pause_async()
    log("Scene setup complete. Viewport should show hospital + Nova Carter + Franka + cube.")

except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())
    raise
