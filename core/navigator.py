"""AMR base execution + reactive avoidance.

Navigator (ABC)
└── WaypointNavigator — consumes a Planner's waypoints, drives a wheeled
    robot via Isaac Sim's WheelBasePoseController, re-plans on blockage.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.controllers.wheel_base_pose_controller import (
    WheelBasePoseController,
)
from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
    DifferentialController,
)
from isaacsim.storage.native import get_assets_root_path

from .planner import Planner


class NavStatus(Enum):
    RUNNING = "running"
    REACHED = "reached"
    BLOCKED = "blocked"     # stuck / obstacle — triggers re-plan internally
    FAILED = "failed"        # no path exists from current pose to goal


# ────────────────────────────────────────────────────────────────
class Navigator(ABC):
    @abstractmethod
    def setup(self, world, cfg: dict, planner: Planner) -> None: ...
    @abstractmethod
    def set_goal(self, xy) -> None: ...
    @abstractmethod
    def step(self) -> NavStatus: ...
    @abstractmethod
    def get_pose(self) -> tuple[np.ndarray, np.ndarray]: ...
    @abstractmethod
    def stop(self) -> None: ...


# ────────────────────────────────────────────────────────────────
class WaypointNavigator(Navigator):
    """Drives a WheeledRobot through a list of 2D waypoints.

    Replan triggers:
      - Stuck: position unchanged for `stuck_threshold_ticks`
      - (Future) Lidar obstacle closer than `lidar_obstacle_distance`
    """

    def __init__(self):
        self.cfg: dict = {}
        self.planner: Planner | None = None
        self.robot: WheeledRobot | None = None
        self.ctrl: WheelBasePoseController | None = None
        self._wheel_indices: list[int] | None = None

        self._waypoints: list[np.ndarray] = []
        self._idx: int = 0
        self._goal: np.ndarray | None = None

        # Blockage detection state
        self._last_pos: np.ndarray | None = None
        self._stuck_counter: int = 0
        self._max_replans: int = 3
        self._replans_used: int = 0

    # ── lifecycle ────────────────────────────────────────────
    def setup(self, world, cfg: dict, planner: Planner) -> None:
        self.cfg = cfg
        self.planner = planner

        robot_cfg = cfg["robot"]
        assets_root = get_assets_root_path()
        usd = assets_root + robot_cfg["usd_path"]
        start_pos = np.array(robot_cfg.get("start_position", [0.0, 0.0, 0.0]),
                             dtype=float)
        start_ori = np.array(robot_cfg.get("start_orientation", [1.0, 0.0, 0.0, 0.0]),
                             dtype=float)

        self.robot = WheeledRobot(
            prim_path=robot_cfg.get("prim_path", "/World/AMR"),
            name=robot_cfg.get("name", "amr"),
            wheel_dof_names=robot_cfg["wheel_dof_names"],
            create_robot=True,
            usd_path=usd,
            position=start_pos,
            orientation=start_ori,
        )
        world.scene.add(self.robot)

        self.ctrl = WheelBasePoseController(
            name="nav_controller",
            open_loop_wheel_controller=DifferentialController(
                name="diff_ctrl",
                wheel_radius=float(robot_cfg["wheel_radius"]),
                wheel_base=float(robot_cfg["wheel_base"]),
                max_linear_speed=float(cfg.get("max_linear_speed", 0.5)),
                max_angular_speed=float(cfg.get("max_angular_speed", 1.0)),
            ),
            is_holonomic=False,
        )

        # wheel_indices are resolved lazily in _ensure_wheel_indices() because
        # robot.dof_names / _wheel_dof_indices are only populated AFTER
        # world.reset_async() initializes the articulation.
        self._wheel_indices = None
        self._wheel_dof_names = list(robot_cfg["wheel_dof_names"])

        self._stuck_limit = int(cfg.get("stuck_threshold_ticks", 240))
        self._stuck_tol = float(cfg.get("stuck_position_tolerance", 0.01))
        self._reach_tol = float(cfg.get("waypoint_reach_threshold", 0.15))
        self._max_replans = int(cfg.get("max_replans", 3))

    # ── goal / plan ──────────────────────────────────────────
    def set_goal(self, xy) -> None:
        if self.planner is None or self.robot is None:
            raise RuntimeError("Navigator.setup() not called")

        self._goal = np.asarray(xy, dtype=float)[:2]
        start = self.get_pose()[0][:2]
        self._waypoints = self.planner.plan(start, self._goal)
        self._idx = 0
        self._stuck_counter = 0
        self._replans_used = 0
        self.ctrl.reset()

        if not self._waypoints:
            print(f"[Navigator] no path from {start.tolist()} to {self._goal.tolist()}")
        else:
            print(f"[Navigator] new plan: {len(self._waypoints)} waypoints")

    # ── per-tick step ────────────────────────────────────────
    def step(self) -> NavStatus:
        if self._goal is None:
            return NavStatus.RUNNING
        if not self._waypoints:
            return NavStatus.FAILED

        pos, ori = self.get_pose()
        pos_xy = pos[:2]

        # Are we done?
        if self._idx >= len(self._waypoints):
            if float(np.linalg.norm(pos_xy - self._goal)) < self._reach_tol:
                return NavStatus.REACHED
            self._idx = len(self._waypoints) - 1  # re-aim at final

        target = self._waypoints[self._idx]
        dist = float(np.linalg.norm(pos_xy - target))

        if dist < self._reach_tol:
            self._idx += 1
            self.ctrl.reset()
            if self._idx >= len(self._waypoints):
                return NavStatus.RUNNING  # next tick will trigger reached check
            target = self._waypoints[self._idx]

        # Blockage — stuck detection
        if self._last_pos is not None:
            if float(np.linalg.norm(pos_xy - self._last_pos[:2])) < self._stuck_tol:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0
        self._last_pos = pos.copy()

        if self._stuck_counter >= self._stuck_limit:
            return self._handle_blockage()

        # Drive — the DifferentialController returns an ArticulationAction with
        # joint_velocities of length 2 but no joint_indices. We attach the
        # resolved wheel indices so apply_action targets only those 2 DOFs.
        action = self.ctrl.forward(
            start_position=pos,
            start_orientation=ori,
            goal_position=np.array([target[0], target[1], 0.0]),
        )
        self._apply_wheel_action(action)
        return NavStatus.RUNNING

    def _ensure_wheel_indices(self) -> None:
        """Look up wheel DOF indices lazily — they're only populated after the
        articulation is initialized (which happens on world.reset())."""
        if self._wheel_indices is not None:
            return
        idx = getattr(self.robot, "_wheel_dof_indices", None)
        if idx:
            self._wheel_indices = list(idx)
        else:
            all_names = list(self.robot.dof_names or [])
            self._wheel_indices = [all_names.index(n) for n in self._wheel_dof_names]
        print(f"[Navigator] wheel DOF indices = {self._wheel_indices}")

    def _apply_wheel_action(self, action) -> None:
        """Safely apply a DifferentialController action to just the wheel DOFs."""
        if action is None or getattr(action, "joint_velocities", None) is None:
            return
        self._ensure_wheel_indices()
        action.joint_indices = np.array(self._wheel_indices, dtype=np.int32)
        self.robot.apply_action(action)

    def _handle_blockage(self) -> NavStatus:
        if self._replans_used >= self._max_replans:
            print("[Navigator] stuck and out of replans — FAILED")
            return NavStatus.FAILED
        self._replans_used += 1
        pos_xy = self.get_pose()[0][:2]
        print(f"[Navigator] stuck → replan #{self._replans_used} "
              f"from {pos_xy.tolist()} to {self._goal.tolist()}")
        self._waypoints = self.planner.plan(pos_xy, self._goal)
        self._idx = 0
        self._stuck_counter = 0
        self.ctrl.reset()
        return NavStatus.BLOCKED if self._waypoints else NavStatus.FAILED

    # ── helpers ──────────────────────────────────────────────
    def get_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.robot.get_world_pose()

    def stop(self) -> None:
        """Zero out wheel velocities."""
        if self.robot is None:
            return
        try:
            self._ensure_wheel_indices()
            from isaacsim.core.utils.types import ArticulationAction
            zero_vel = np.zeros(len(self._wheel_indices))
            action = ArticulationAction(
                joint_velocities=zero_vel,
                joint_indices=np.array(self._wheel_indices, dtype=np.int32),
            )
            self.robot.apply_action(action)
        except Exception as e:
            print(f"[Navigator] stop() failed: {e}")
