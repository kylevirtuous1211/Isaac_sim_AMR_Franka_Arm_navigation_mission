"""Arm control with obstacle avoidance.

Manipulator (ABC)
├── FrankaPickPlaceManipulator — simple time-based PickPlaceController
└── FrankaRMPflowManipulator   — RMPflow with obstacle field (reactive avoidance)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from isaacsim.robot.manipulators.examples.franka import Franka


class ManipStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Action(Enum):
    NONE = "none"
    PICK = "pick"
    PLACE = "place"


# ────────────────────────────────────────────────────────────────
class Manipulator(ABC):
    @abstractmethod
    def setup(self, world, cfg: dict) -> None: ...
    @abstractmethod
    def pick(self, xyz) -> None: ...
    @abstractmethod
    def place(self, xyz) -> None: ...
    @abstractmethod
    def add_obstacle(self, prim_path: str) -> None: ...
    @abstractmethod
    def step(self) -> ManipStatus: ...
    @abstractmethod
    def reset(self) -> None: ...


# ────────────────────────────────────────────────────────────────
# Shared helper — spawns a Franka via the Isaac Sim example class
# ────────────────────────────────────────────────────────────────
def _spawn_franka(world, cfg: dict) -> Franka:
    position = np.array(cfg.get("position", [0.0, 0.0, 0.0]), dtype=float)
    franka = Franka(
        prim_path=cfg.get("prim_path", "/World/Franka"),
        name=cfg.get("name", "franka"),
        position=position,
    )
    world.scene.add(franka)
    return franka


# ────────────────────────────────────────────────────────────────
# Simple time-based controller (from hand_on_2_franka.py)
# ────────────────────────────────────────────────────────────────
class FrankaPickPlaceManipulator(Manipulator):
    """Uses PickPlaceController — a state-machine with hardcoded timing.
    Works well when the cube and placement are stationary and reachable.
    No obstacle avoidance."""

    def __init__(self):
        self.cfg: dict = {}
        self.franka: Franka | None = None
        self.ctrl = None
        self._target_pick = None
        self._target_place = None
        self._action = Action.NONE

    def setup(self, world, cfg: dict) -> None:
        from isaacsim.robot.manipulators.examples.franka.controllers import (
            PickPlaceController,
        )
        self.cfg = cfg
        self.franka = _spawn_franka(world, cfg)
        self.ctrl = PickPlaceController(
            name="pick_place_controller",
            gripper=self.franka.gripper,
            robot_articulation=self.franka,
            events_dt=cfg.get("events_dt"),
        )
        # Gripper init is deferred — see note in FrankaRMPflowManipulator.setup().

    def pick(self, xyz) -> None:
        self._target_pick = np.asarray(xyz, dtype=float)
        self._target_place = None
        self._action = Action.PICK
        self.ctrl.reset()

    def place(self, xyz) -> None:
        # PickPlaceController is a single fused pick→place cycle.
        # To use place() alone, call pick(current_cube_pos) + place(target).
        self._target_place = np.asarray(xyz, dtype=float)
        self._action = Action.PLACE

    def add_obstacle(self, prim_path: str) -> None:
        print(f"[FrankaPickPlace] add_obstacle ignored (no avoidance): {prim_path}")

    def step(self) -> ManipStatus:
        if self._action == Action.NONE:
            return ManipStatus.IDLE
        if self._target_pick is None or self._target_place is None:
            return ManipStatus.RUNNING
        joints = self.franka.get_joint_positions()
        actions = self.ctrl.forward(
            picking_position=self._target_pick,
            placing_position=self._target_place,
            current_joint_positions=joints,
        )
        self.franka.apply_action(actions)
        if self.ctrl.is_done():
            self._action = Action.NONE
            return ManipStatus.DONE
        return ManipStatus.RUNNING

    def reset(self) -> None:
        if self.ctrl is not None:
            self.ctrl.reset()
        if self.franka is not None:
            self.franka.gripper.set_joint_positions(
                self.franka.gripper.joint_opened_positions
            )
        self._action = Action.NONE
        self._target_pick = None
        self._target_place = None


# ────────────────────────────────────────────────────────────────
# RMPflow — reactive, obstacle-aware motion
# ────────────────────────────────────────────────────────────────
class FrankaRMPflowManipulator(Manipulator):
    """RMPflow with registered obstacles.

    Uses a small phase FSM (approach → descend → grasp → lift → transit →
    lower → release → retract), adapted from examples/hand_on_3_rmpflow.py.
    Each phase drives RMPflow toward a target XYZ; obstacle field is shared.
    """

    _PHASES = ["idle", "above_pick", "at_pick", "grasp", "lift",
               "above_place", "at_place", "release", "retract", "done"]

    def __init__(self):
        self.cfg: dict = {}
        self.franka: Franka | None = None
        self.rmp = None
        self.articulation_ctrl = None

        self._phase = "idle"
        self._target_pick: np.ndarray | None = None
        self._target_place: np.ndarray | None = None
        self._tick_in_phase = 0

        # Tunables (loaded from cfg in setup)
        self._clearance_height = 0.15
        self._approach_tol = 0.03
        self._grasp_hold_ticks = 40
        self._release_hold_ticks = 30

    def setup(self, world, cfg: dict) -> None:
        from isaacsim.robot_motion.motion_generation import (
            ArticulationMotionPolicy,
            RmpFlow,
        )
        self.cfg = cfg
        self.franka = _spawn_franka(world, cfg)

        # RMPflow config — use the local files mounted at /workspace/configs/rmpflow.
        rmp_cfg = cfg.get("rmpflow", {})
        rmp_dir = "/workspace/configs/rmpflow"
        self.rmp = RmpFlow(
            robot_description_path=rmp_cfg.get(
                "robot_description_path", f"{rmp_dir}/robot_descriptor.yaml"
            ),
            urdf_path=rmp_cfg.get(
                "urdf_path", f"{rmp_dir}/lula_franka_gen.urdf"
            ),
            rmpflow_config_path=rmp_cfg.get(
                "rmpflow_config_path",
                cfg.get("rmpflow_config", f"{rmp_dir}/franka_rmpflow_common.yaml"),
            ),
            end_effector_frame_name=rmp_cfg.get("end_effector_frame_name", "right_gripper"),
            maximum_substep_size=float(rmp_cfg.get("maximum_substep_size", 0.00334)),
        )
        self.articulation_ctrl = ArticulationMotionPolicy(self.franka, self.rmp)

        self._clearance_height = float(cfg.get("clearance_height", 0.15))
        self._approach_tol = float(cfg.get("approach_tolerance", 0.03))
        self._grasp_hold_ticks = int(cfg.get("grasp_hold_ticks", 40))
        self._release_hold_ticks = int(cfg.get("release_hold_ticks", 30))

        # NOTE: cannot touch gripper here — its joint-positions function is
        # only populated after world.reset_async() initializes the articulation.
        # Call manipulator.reset() AFTER world.reset_async() to open the gripper.

    # ── API ──────────────────────────────────────────────────
    def pick(self, xyz) -> None:
        self._target_pick = np.asarray(xyz, dtype=float)
        self._phase = "above_pick"
        self._tick_in_phase = 0

    def place(self, xyz) -> None:
        self._target_place = np.asarray(xyz, dtype=float)
        if self._phase == "done" or self._phase == "idle":
            # Allow place() standalone only after a successful pick
            self._phase = "above_place"
            self._tick_in_phase = 0

    def add_obstacle(self, prim_path: str) -> None:
        """Register a static prim as an obstacle for RMPflow."""
        if self.rmp is None:
            raise RuntimeError("Manipulator.setup() not called")
        from isaacsim.core.utils.prims import get_prim_at_path
        prim = get_prim_at_path(prim_path)
        if prim is None or not prim.IsValid():
            print(f"[RMPflow] obstacle prim not found: {prim_path}")
            return
        try:
            self.rmp.add_obstacle(prim)
            print(f"[RMPflow] obstacle added: {prim_path}")
        except Exception as e:
            print(f"[RMPflow] add_obstacle failed for {prim_path}: {e}")

    # ── per-tick ─────────────────────────────────────────────
    def step(self) -> ManipStatus:
        if self._phase in ("idle", "done"):
            return ManipStatus.IDLE if self._phase == "idle" else ManipStatus.DONE

        target = self._phase_target()
        if target is None:
            return ManipStatus.FAILED

        self.rmp.set_end_effector_target(target_position=target)
        self.rmp.update_world()
        action = self.articulation_ctrl.get_next_articulation_action()
        self.franka.apply_action(action)

        # Phase transitions
        ee_pos = self._ee_position()
        dist = float(np.linalg.norm(ee_pos - target))
        self._tick_in_phase += 1
        self._advance_phase_if_done(dist)

        return ManipStatus.DONE if self._phase == "done" else ManipStatus.RUNNING

    def reset(self) -> None:
        self._phase = "idle"
        self._tick_in_phase = 0
        self._target_pick = None
        self._target_place = None
        if self.franka is not None:
            try:
                self.franka.gripper.set_joint_positions(
                    self.franka.gripper.joint_opened_positions
                )
            except Exception as e:
                print(f"[FrankaRMPflow] gripper open failed: {e}")

    # ── phase machinery ──────────────────────────────────────
    def _phase_target(self) -> np.ndarray | None:
        up = np.array([0.0, 0.0, self._clearance_height])
        if self._phase in ("above_pick",):
            return self._target_pick + up if self._target_pick is not None else None
        if self._phase in ("at_pick", "grasp"):
            return self._target_pick
        if self._phase == "lift":
            return self._target_pick + up if self._target_pick is not None else None
        if self._phase == "above_place":
            return self._target_place + up if self._target_place is not None else None
        if self._phase in ("at_place", "release"):
            return self._target_place
        if self._phase == "retract":
            return self._target_place + up if self._target_place is not None else None
        return None

    def _advance_phase_if_done(self, dist: float) -> None:
        if self._phase == "above_pick" and dist < self._approach_tol:
            self._phase, self._tick_in_phase = "at_pick", 0
        elif self._phase == "at_pick" and dist < self._approach_tol:
            self.franka.gripper.set_joint_positions(
                self.franka.gripper.joint_closed_positions
            )
            self._phase, self._tick_in_phase = "grasp", 0
        elif self._phase == "grasp" and self._tick_in_phase >= self._grasp_hold_ticks:
            self._phase, self._tick_in_phase = "lift", 0
        elif self._phase == "lift" and dist < self._approach_tol:
            if self._target_place is not None:
                self._phase, self._tick_in_phase = "above_place", 0
            else:
                self._phase, self._tick_in_phase = "done", 0
        elif self._phase == "above_place" and dist < self._approach_tol:
            self._phase, self._tick_in_phase = "at_place", 0
        elif self._phase == "at_place" and dist < self._approach_tol:
            self.franka.gripper.set_joint_positions(
                self.franka.gripper.joint_opened_positions
            )
            self._phase, self._tick_in_phase = "release", 0
        elif self._phase == "release" and self._tick_in_phase >= self._release_hold_ticks:
            self._phase, self._tick_in_phase = "retract", 0
        elif self._phase == "retract" and dist < self._approach_tol:
            self._phase, self._tick_in_phase = "done", 0

    def _ee_position(self) -> np.ndarray:
        """Return end-effector world position from the RMPflow frame lookup."""
        if self.rmp is None:
            return np.zeros(3)
        try:
            pose = self.rmp.get_end_effector_pose()
            # Different Isaac Sim versions return different shapes; take the first 3
            return np.asarray(pose[0] if isinstance(pose, tuple) else pose)[:3]
        except Exception:
            return np.zeros(3)
