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
# Shared helpers — spawn Franka, optionally carry it on an AMR chassis
# ────────────────────────────────────────────────────────────────
def _spawn_franka(world, cfg: dict) -> Franka:
    """Spawn Franka at a fixed world position.

    If cfg["mount_to"] is set, we spawn the Franka above that prim and
    install a per-tick pose-sync so the base follows it.

    Why not a PhysX FixedJoint: linking two separate articulations
    (Nova Carter + Franka) via FixedJoint causes PhysX to treat one of
    them as rooted to world, jamming the wheels. The kinematic-sync
    pattern avoids that by keeping the articulations independent and
    just teleporting Franka's base every physics tick.
    """
    mount_to = cfg.get("mount_to")
    if mount_to:
        mount_pos = _world_pos_of(mount_to)
        local_offset = np.array(cfg.get("mount_local_offset", [0.0, 0.0, 0.30]),
                                dtype=float)
        position = mount_pos + local_offset
    else:
        position = np.array(cfg.get("position", [0.0, 0.0, 0.0]), dtype=float)

    franka = Franka(
        prim_path=cfg.get("prim_path", "/World/Franka"),
        name=cfg.get("name", "franka"),
        position=position,
    )
    world.scene.add(franka)
    return franka


def _world_pos_of(prim_path: str) -> np.ndarray:
    """Return the world-space translation of a prim."""
    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"[manipulator] mount target not found: {prim_path} — using origin")
        return np.zeros(3)
    xform = UsdGeom.Xformable(prim)
    mat = xform.ComputeLocalToWorldTransform(0)
    t = mat.ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=float)


def _world_pose_of(prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    """World-space (position, quaternion wxyz) of a prim."""
    import omni.usd
    from pxr import UsdGeom, Gf
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    t = mat.ExtractTranslation()
    rot = mat.ExtractRotationQuat()
    w = rot.GetReal()
    x, y, z = rot.GetImaginary()
    return (
        np.array([t[0], t[1], t[2]], dtype=float),
        np.array([w, x, y, z], dtype=float),
    )


def _make_base_kinematic(franka_prim_path: str, base_link: str = "panda_link0") -> bool:
    """Mark Franka's base rigid body as kinematic.

    Kinematic bodies:
      - Are NOT simulated dynamically (gravity / forces don't affect them)
      - CAN be teleported via set_world_pose without destabilizing physics
      - Still provide collision and constraint anchors for dynamic children

    This is what makes per-tick pose-sync stable: the base just obeys our
    teleport; the arm's dynamic links (panda_link1..7 + fingers) remain
    dynamic, so RMPflow torque control still drives them normally.
    """
    import omni.usd
    from pxr import UsdPhysics
    stage = omni.usd.get_context().get_stage()
    base_path = f"{franka_prim_path}/{base_link}"
    prim = stage.GetPrimAtPath(base_path)
    if not prim or not prim.IsValid():
        print(f"[manipulator] base link not found at {base_path}")
        return False
    # UsdPhysics.RigidBodyAPI is typically already applied by the Franka USD;
    # we just flip kinematicEnabled=True.
    rb = UsdPhysics.RigidBodyAPI(prim)
    if not rb:
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateKinematicEnabledAttr(True).Set(True)
    print(f"[manipulator] {base_path}: kinematicEnabled = True")
    return True


def _install_pose_sync(world, franka: Franka, mount_prim: str,
                      local_offset: np.ndarray, cb_name: str) -> None:
    """Install a physics callback that teleports Franka's base to
    mount_prim.world_pose() + local_offset every tick.

    Prereq: Franka's base must be kinematic (see _make_base_kinematic).
    Otherwise teleporting a dynamic body creates spurious forces and
    destabilizes both robots.
    """
    # Remove any prior callback with this name (re-setup idempotency)
    try:
        world.remove_physics_callback(cb_name)
    except Exception:
        pass

    offset = np.asarray(local_offset, dtype=float)

    def _sync(_step_size):
        try:
            mount_pos, mount_rot = _world_pose_of(mount_prim)
            rotated = _rotate_by_quat(offset, mount_rot)
            franka.set_world_pose(
                position=mount_pos + rotated,
                orientation=mount_rot,
            )
        except Exception as e:
            print(f"[manipulator] pose-sync failed: {e}")

    world.add_physics_callback(cb_name, callback_fn=_sync)
    print(f"[manipulator] pose-sync installed: {mount_prim} + {offset.tolist()} → Franka base")


def _rotate_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by a quaternion (wxyz)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    # q * v * q^-1, expanded
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return np.array([rx, ry, rz], dtype=float)


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
        self._done_latch = False

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
        self._done_latch = False
        self.ctrl.reset()

    def rebase(self, world_position, world_orientation=None) -> None:
        """Move the Franka articulation to a new base pose and rebuild the
        PickPlaceController so its IK solver uses the new root frame.

        Use when the Franka is riding on an AMR that has just arrived at
        a pickup/drop-off point: teleport Franka to AMR's current pose,
        then `rebase(new_pos)` so the controller plans from the new frame.
        """
        if self.franka is None or self.ctrl is None:
            return
        ori = world_orientation
        if ori is None:
            ori = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        # Update the visual/physx xform
        self.franka.set_world_pose(position=np.asarray(world_position, dtype=float),
                                   orientation=np.asarray(ori, dtype=float))
        # Update the articulation's default (post_reset) pose so the
        # controller's kinematic reference moves with it
        try:
            self.franka.set_default_state(
                position=np.asarray(world_position, dtype=float),
                orientation=np.asarray(ori, dtype=float),
            )
        except Exception as e:
            print(f"[FrankaPickPlace] set_default_state failed: {e}")

        # Rebuild the controller so its internal reference pose is fresh
        from isaacsim.robot.manipulators.examples.franka.controllers import (
            PickPlaceController,
        )
        self.ctrl = PickPlaceController(
            name="pick_place_controller",
            gripper=self.franka.gripper,
            robot_articulation=self.franka,
            events_dt=self.cfg.get("events_dt"),
        )
        self._done_latch = False
        print(f"[FrankaPickPlace] rebased to {list(world_position)}")

    def place(self, xyz) -> None:
        # PickPlaceController is a single fused pick→place cycle.
        # To use place() alone, call pick(current_cube_pos) + place(target).
        self._target_place = np.asarray(xyz, dtype=float)
        self._action = Action.PLACE
        self._done_latch = False

    def add_obstacle(self, prim_path: str) -> None:
        print(f"[FrankaPickPlace] add_obstacle ignored (no avoidance): {prim_path}")

    def step(self) -> ManipStatus:
        if self._action == Action.NONE:
            return ManipStatus.IDLE
        if self._done_latch:
            return ManipStatus.DONE
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
            self._done_latch = True
            return ManipStatus.DONE
        return ManipStatus.RUNNING

    def get_phase(self) -> str:
        """For symmetry with RMPflow. PickPlaceController is event-driven
        internally; return a coarse state indicator."""
        if self._done_latch:
            return "done"
        if self._action == Action.NONE:
            return "idle"
        return "running"

    def is_done(self) -> bool:
        return self._done_latch

    def reset(self) -> None:
        if self.ctrl is not None:
            self.ctrl.reset()
        if self.franka is not None:
            try:
                self.franka.gripper.set_joint_positions(
                    self.franka.gripper.joint_opened_positions
                )
            except Exception as e:
                print(f"[FrankaPickPlace] gripper open failed: {e}")
        self._action = Action.NONE
        self._target_pick = None
        self._target_place = None
        self._done_latch = False


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
        self._phase_timeout_ticks = int(cfg.get("phase_timeout_ticks", 800))

        # Mobile-manipulation mount — Franka is fixed-base by default,
        # so set_world_pose() cleanly teleports the whole articulation.
        # The pose-sync callback drives that every tick so Franka rides
        # on top of the moving chassis. Collision with the chassis is
        # avoided by picking an offset taller than the chassis height.
        mount_to = cfg.get("mount_to")
        if mount_to:
            _install_pose_sync(
                world,
                self.franka,
                mount_prim=mount_to,
                local_offset=np.array(cfg.get("mount_local_offset", [0.0, 0.0, 0.50]),
                                      dtype=float),
                cb_name=cfg.get("mount_sync_name", "franka_base_sync"),
            )

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
        if self._phase in ("idle", "done", "failed"):
            if self._phase == "done":
                return ManipStatus.DONE
            if self._phase == "failed":
                return ManipStatus.FAILED
            return ManipStatus.IDLE

        target = self._phase_target()
        if target is None:
            return ManipStatus.FAILED

        # Keep RMPflow's internal base frame in sync with the ACTUAL Franka
        # base pose. Required when the arm is mounted on a moving AMR (the
        # pose-sync callback teleports panda_link0 every tick; if RMPflow
        # still thinks the base is at (0, 0, 0) it plans into the floor).
        try:
            base_pos, base_ori = self.franka.get_world_pose()
            self.rmp.set_robot_base_pose(
                robot_position=np.asarray(base_pos, dtype=float),
                robot_orientation=np.asarray(base_ori, dtype=float),
            )
        except Exception as e:
            print(f"[FrankaRMPflow] set_robot_base_pose failed: {e}")

        self.rmp.set_end_effector_target(target_position=target)
        self.rmp.update_world()
        action = self.articulation_ctrl.get_next_articulation_action()
        self.franka.apply_action(action)

        # Phase transitions
        ee_pos = self._ee_position()
        dist = float(np.linalg.norm(ee_pos - target))
        self._tick_in_phase += 1

        # Phase-level timeout: if we've spent more than N ticks in the same
        # phase without advancing, something is stuck (RMPflow target
        # unreachable, base pose stale, gripper fight, etc.) — bail out.
        if self._tick_in_phase > self._phase_timeout_ticks:
            print(f"[FrankaRMPflow] phase '{self._phase}' timed out after "
                  f"{self._tick_in_phase} ticks (dist_to_target={dist:.3f})")
            self._phase = "failed"
            return ManipStatus.FAILED

        self._advance_phase_if_done(dist)
        return ManipStatus.DONE if self._phase == "done" else ManipStatus.RUNNING

    def get_phase(self) -> str:
        return self._phase

    def is_done(self) -> bool:
        return self._phase == "done"

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
