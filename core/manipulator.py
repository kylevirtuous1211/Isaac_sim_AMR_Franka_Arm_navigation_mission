"""Arm control with obstacle avoidance.

Manipulator (ABC)
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
    def ensure_mount_sync(self, world) -> None:
        """Re-install the pose-sync physics callback if this manipulator is
        configured with a mount. Default: no-op. See
        FrankaRMPflowManipulator.ensure_mount_sync for the mobile-manip case.
        """
        return None


# ────────────────────────────────────────────────────────────────
# Shared helpers — spawn Franka, optionally carry it on an AMR chassis
# ────────────────────────────────────────────────────────────────
def _spawn_franka(world, cfg: dict) -> Franka:
    """Spawn Franka at a fixed world position.

    If cfg["mount_to"] is set, we spawn the Franka above that prim. The
    actual mount enforcement depends on cfg["mount_mode"]:
      - "pose_sync" (default): a per-tick callback teleports the Franka
        base to chassis_link.world_pose + offset. Fast to iterate; the
        Franka mass doesn't load the wheels.
      - "fixed_joint": a UsdPhysics.FixedJoint with excludeFromArticulation
        is authored at bootstrap (see core/franka_mount_joint.py). PhysX
        treats it as a maximal-coordinate constraint between two
        independent articulations — Franka mass loads the AMR correctly
        without the wheel-jamming bug a naive intra-articulation join
        would trigger.
    """
    mount_to = cfg.get("mount_to")
    if mount_to:
        mount_pos, _ = _amr_mount_pose(mount_to)
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


def _amr_mount_pose(prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Live world-space (position, quaternion wxyz) of the AMR mount.

    Prefers the navigator's articulation handle (`state.navigator.robot`),
    which returns the physics-driven pose every tick. USD's
    ComputeLocalToWorldTransform is a fallback for sub-link prims; for
    articulation links it returns the AUTHORED xform, not the current
    physics state, so the Franka would stay frozen near spawn even as
    the AMR drives. The articulation API is the source of truth.
    """
    try:
        from core import state as _state
        nav = getattr(_state, "navigator", None)
        if nav is not None and getattr(nav, "robot", None) is not None:
            pos, ori = nav.robot.get_world_pose()
            return (
                np.asarray(pos, dtype=float),
                np.asarray(ori, dtype=float),
            )
    except Exception as e:
        print(f"[manipulator] navigator pose lookup failed, falling back to USD: {e}")

    # USD fallback (used pre-bootstrap or if the navigator isn't built yet)
    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"[manipulator] mount target not found: {prim_path} — using origin")
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


def _install_pose_sync(world, franka: Franka, mount_prim: str,
                      local_offset: np.ndarray, cb_name: str) -> None:
    """Install a physics callback that teleports Franka's base to
    mount_prim.world_pose() + local_offset every tick.

    Franka is fixed-base by default, so set_world_pose() cleanly
    teleports the whole articulation without destabilizing physics.
    """
    # Remove any prior callback with this name (re-setup idempotency)
    try:
        world.remove_physics_callback(cb_name)
    except Exception:
        pass

    offset = np.asarray(local_offset, dtype=float)

    def _sync(_step_size):
        try:
            mount_pos, mount_rot = _amr_mount_pose(mount_prim)
            rotated = _rotate_by_quat(offset, mount_rot)
            franka.set_world_pose(
                position=mount_pos + rotated,
                orientation=mount_rot,
            )
        except Exception as e:
            print(f"[manipulator] pose-sync failed: {e}")

    world.add_physics_callback(cb_name, callback_fn=_sync)


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

        # SurfaceGripper handle. Authored at bootstrap (see
        # apps/bootstrap.py), but the runtime interface needs the physics
        # context live, so we instantiate the wrapper lazily on first
        # reset() (after world.reset_async + world.play_async).
        self.surface_gripper = None

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

        # Mobile-manipulation mount — see _spawn_franka docstring for the
        # two strategies. In fixed_joint mode the joint is authored from
        # apps/bootstrap.py (after both articulations spawn but before
        # world.reset_async); we only install the pose-sync callback in
        # the legacy pose_sync mode.
        mount_to = cfg.get("mount_to")
        mount_mode = cfg.get("mount_mode", "pose_sync")
        if mount_to and mount_mode == "pose_sync":
            _install_pose_sync(
                world,
                self.franka,
                mount_prim=mount_to,
                local_offset=np.array(cfg.get("mount_local_offset", [0.0, 0.0, 0.50]),
                                      dtype=float),
                cb_name=cfg.get("mount_sync_name", "franka_base_sync"),
            )
        elif mount_to and mount_mode == "fixed_joint":
            # Defensive: if a stale pose_sync callback survives a config
            # flip mid-session, remove it so it doesn't fight the joint.
            try:
                world.remove_physics_callback(
                    cfg.get("mount_sync_name", "franka_base_sync")
                )
            except Exception:
                pass

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

        # Periodic in-phase diagnostic — surfaces whether RMPflow is
        # converging (dist trending down) vs. stuck (dist plateaued).
        if self._tick_in_phase > 0 and self._tick_in_phase % 100 == 0:
            print(f"[step] phase={self._phase} tick={self._tick_in_phase} "
                  f"dist={dist:.3f} tol={self._approach_tol:.3f} "
                  f"ee={ee_pos.tolist()} target={target.tolist()}")

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

        # SurfaceGripper — lazy-init on first reset() call after bootstrap
        # has authored the prim and world.play_async() has initialized the
        # PhysX surface_gripper interface. The wrapper holds a C++ handle
        # so it must be built post-play, not during setup().
        if self.surface_gripper is None:
            try:
                from .surface_gripper_setup import SurfaceGripperWrapper
                self.surface_gripper = SurfaceGripperWrapper()
            except Exception as e:
                print(f"[FrankaRMPflow] SurfaceGripperWrapper init failed: {e}")
                self.surface_gripper = None
        # Open the SG on reset so a re-run starts with no cube attached.
        if self.surface_gripper is not None:
            try:
                self.surface_gripper.open()
            except Exception:
                pass

    def ensure_mount_sync(self, world) -> None:
        """Re-install the pose-sync callback if `mount_to` is configured
        AND `mount_mode == "pose_sync"`.

        `setup()` installs it once on first bootstrap, but scripts like
        `run_manip.py` may remove it mid-session, and `bootstrap.py`'s
        fast-reset path doesn't re-run `setup()`. Calling this after each
        reset makes the mount state idempotent regardless of workflow.

        In `mount_mode == "fixed_joint"` this is a no-op (the FixedJoint
        is authored from bootstrap and persists in USD across resets).
        Any stale callback from a prior pose_sync session is also cleared
        so it can't fight the joint.
        """
        mount_to = self.cfg.get("mount_to")
        mount_mode = self.cfg.get("mount_mode", "pose_sync")
        if not mount_to or self.franka is None:
            return
        cb_name = self.cfg.get("mount_sync_name", "franka_base_sync")
        if mount_mode == "fixed_joint":
            try:
                world.remove_physics_callback(cb_name)
            except Exception:
                pass
            return
        _install_pose_sync(
            world,
            self.franka,
            mount_prim=mount_to,
            local_offset=np.array(
                self.cfg.get("mount_local_offset", [0.0, 0.0, 0.50]),
                dtype=float,
            ),
            cb_name=cb_name,
        )

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
            # Fire BOTH grippers at the above_pick→at_pick edge: the
            # ParallelGripper closes its fingers (visual clamp), and
            # the SurfaceGripper requests a D6-joint engagement. The
            # SG manager polls within retry_interval seconds and wires
            # up the joint as soon as the cube enters max_grip_distance
            # during the at_pick descent.
            self.franka.gripper.close()
            if self.surface_gripper is not None:
                try:
                    self.surface_gripper.close()
                except Exception as e:
                    print(f"[FrankaRMPflow] SG close failed: {e}")
            self._phase, self._tick_in_phase = "at_pick", 0
        elif self._phase == "at_pick" and self._tick_in_phase >= self._grasp_hold_ticks:
            # Time-based transition — `at_pick` is now "hold while
            # descending and let the SurfaceGripper engage". The actual
            # close fired at the previous transition.
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
            # Open BOTH grippers. SurfaceGripper.open() releases the D6
            # joint so the cube starts physics-simulating again; the
            # ParallelGripper.open() spreads the fingers visually.
            self.franka.gripper.open()
            if self.surface_gripper is not None:
                try:
                    self.surface_gripper.open()
                except Exception as e:
                    print(f"[FrankaRMPflow] SG open failed: {e}")
            self._phase, self._tick_in_phase = "release", 0
        elif self._phase == "release" and self._tick_in_phase >= self._release_hold_ticks:
            self._phase, self._tick_in_phase = "retract", 0
        elif self._phase == "retract" and dist < self._approach_tol:
            self._phase, self._tick_in_phase = "done", 0

    def _ee_position(self) -> np.ndarray:
        """Return end-effector world position via the Franka's `end_effector`
        rigid-prim handle (the `right_gripper` frame).

        Don't use `rmp.get_end_effector_pose()` here: that method takes a
        required `active_joint_positions` argument and (in older code) was
        called bare-arg, raising TypeError silently — which made `dist`
        compute against (0,0,0) and pinned the FSM in `above_pick` forever.
        `franka.end_effector.get_world_pose()` is the canonical path and
        matches how examples/hand_on_3_rmpflow.py reads it.
        """
        if self.franka is None:
            return np.zeros(3)
        try:
            pos, _ = self.franka.end_effector.get_world_pose()
            return np.asarray(pos, dtype=float)
        except Exception as e:
            if not getattr(self, "_ee_pose_warned", False):
                print(f"[FrankaRMPflow] _ee_position failed: {type(e).__name__}: {e}")
                self._ee_pose_warned = True
            return np.zeros(3)
