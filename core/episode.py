"""Per-episode helpers for the domain-randomized run loop.

`apply_episode_to_cfg` mutates the parsed `config.yaml` dict so the AMR
start, cube spawn (point_a), and drop target (point_b) reflect a sampled
`Episode`. `reset_world_for_episode` then teleports the AMR / Franka /
cube / markers to those positions and clears stale callbacks. It mirrors
`apps/bootstrap.py:_reset_to_start_poses` so the per-episode reset and
the bootstrap fast-reset converge on the same end state.

Notes:
  - The occupancy grid is built from static hospital geometry only, so
    moving the cube / AMR start does not invalidate planner.is_valid().
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .randomizer import Episode


def apply_episode_to_cfg(cfg: dict, ep: Episode) -> None:
    """Mutate cfg in place so subsequent reads see the sampled episode.

    - Updates navigator.robot.start_position so reset reads the new AMR start.
    - Updates task.point_a (cube spawn) and task.point_b (drop target).
    - Leaves cube z to be derived from cube_size at reset time, so the
      cube always rests on the floor regardless of the sampled point.
    """
    nav_cfg = cfg.setdefault("navigator", {}).setdefault("robot", {})
    nav_cfg["start_position"] = [
        float(ep.start_xy[0]),
        float(ep.start_xy[1]),
        0.0,
    ]
    task_cfg = cfg.setdefault("task", {})
    task_cfg["point_a"] = [float(ep.cube_xyz[0]), float(ep.cube_xyz[1]), 0.0]
    task_cfg["point_b"] = [float(ep.place_xyz[0]), float(ep.place_xyz[1]), 0.0]


def reset_world_for_episode(world, cfg: dict, manipulator, navigator) -> None:
    """Teleport AMR/Franka/cube/markers to match the current cfg.

    Call this between episodes — after `apply_episode_to_cfg` has
    rewritten the start/point_a/point_b, and before `ctx.reset()`.

    Mirrors apps/bootstrap.py:_reset_to_start_poses but is module-level
    so the run-loop can invoke it without re-running bootstrap. The
    bootstrap version still exists for the cold-start / fast-reset
    paths; this version is for steady-state per-episode resets.
    """
    nav_cfg = cfg["navigator"]["robot"]
    amr_pos = np.array(nav_cfg.get("start_position", [0.0, 0.0, 0.0]),
                       dtype=float)
    amr_ori = np.array(nav_cfg.get("start_orientation", [1.0, 0.0, 0.0, 0.0]),
                       dtype=float)

    # AMR root pose + zero velocities + clear navigator bookkeeping.
    if navigator is not None and getattr(navigator, "robot", None) is not None:
        navigator.robot.set_world_pose(position=amr_pos, orientation=amr_ori)
        try:
            navigator.robot.set_linear_velocity(np.zeros(3))
            navigator.robot.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        try:
            navigator.robot.set_joint_velocities(
                np.zeros(len(navigator.robot.dof_names))
            )
        except Exception:
            pass
        navigator._reached_latch = False
        navigator._idx = 0
        navigator._waypoints = []
        navigator._goal = None
        navigator._stuck_counter = 0
        navigator._last_pos = None
        navigator._replans_used = 0

    # Franka stacked on AMR (mobile-manip) or at the standalone station
    # position. Orientation matches AMR so the arm faces forward.
    manip_cfg = cfg["manipulator"]
    if manipulator is not None and getattr(manipulator, "franka", None) is not None:
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
        manipulator.franka.set_world_pose(position=franka_pos, orientation=amr_ori)
        try:
            manipulator.franka.set_linear_velocity(np.zeros(3))
            manipulator.franka.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        manipulator.reset()  # opens gripper, resets phase

    # Cube — spawned at point_a, resting on the floor (z = cube_half).
    task_cfg = cfg["task"]
    cube = world.scene.get_object("target_cube")
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

    # Visual markers reflect the active points so the demo video shows
    # the AMR's intended targets, not the bootstrap defaults.
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

    # Re-install pose-sync. ensure_mount_sync is idempotent and a no-op
    # when mount_to isn't set. Safe to call unconditionally.
    if manipulator is not None:
        manipulator.ensure_mount_sync(world)
