"""MobileManipContext — DfRobotApiContext subclass driving the
mobile pick-and-place pipeline via the existing navigator + manipulator.

Responsibilities:
  - Hold direct references to the navigator, manipulator, cube prim,
    and the configured task points.
  - Compute a `block_state` enum each tick that the Dispatch decider
    routes on (NEED_NAV_TO_BLOCK / NEED_PICK / NEED_TRANSIT /
    NEED_PLACE / DONE / FAILED).
  - Track `pick_attempts` and `failed`. Reset the counter on
    transit-time slips (the spec asks for separate retry budgets per
    pick session — a slipped cube doesn't burn the retry budget).
  - Provide standoff math helpers (cube_park_xy, b_standoff_xy,
    place_target_b) that mirror run_pipeline.py exactly so behavior
    is identical when the happy path runs.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from isaacsim.cortex.framework.dfb import DfRobotApiContext


class BlockState(Enum):
    NEED_NAV_TO_BLOCK = "need_nav_to_block"
    NEED_PICK = "need_pick"
    NEED_TRANSIT = "need_transit"
    NEED_PLACE = "need_place"
    DONE = "done"
    FAILED = "failed"


class _AdapterRobot:
    """Stub object satisfying DfRobotApiContext's `robot` slot.

    DfRobotApiContext stores `robot` and never reads it itself. We don't
    use any leaf states from the stock `peck_decider_network` (which DO
    expect `robot.arm` / `robot.gripper`), so a no-op stub is enough.
    """
    pass


class MobileManipContext(DfRobotApiContext):
    """Context for the mobile pick-and-place decider network.

    Args:
        navigator: live WaypointNavigator (`core.navigator`).
        manipulator: live FrankaRMPflowManipulator (`core.manipulator`).
        cube: the target_cube scene object (DynamicCuboid).
        point_a: task pickup XY (np.ndarray of length 2).
        point_b: task drop XY (np.ndarray of length 2).
        cfg: parsed config.yaml dict.
    """

    MAX_PICK_ATTEMPTS = 3

    def __init__(self, navigator, manipulator, cube, point_a, point_b, cfg):
        super().__init__(robot=_AdapterRobot())

        self.navigator = navigator
        self.manipulator = manipulator
        self.cube = cube
        self.point_a = np.asarray(point_a, dtype=float)[:2]
        self.point_b = np.asarray(point_b, dtype=float)[:2]
        self.cfg = cfg

        # ── Tunables (loaded with defaults if missing from config.yaml) ──
        task_cfg = cfg.get("task", {})
        nav_cfg = cfg.get("navigator", {})
        manip_cfg = cfg.get("manipulator", {})

        self.cube_size = float(task_cfg.get("cube_size", 0.0515))
        self.cube_half = self.cube_size / 2.0
        self.place_tolerance = float(task_cfg.get("place_tolerance", 0.20))
        self.in_gripper_tolerance = float(
            task_cfg.get("in_gripper_tolerance", 0.06)
        )
        self.standoff = float(task_cfg.get("place_standoff", 0.40))
        self.park_tolerance = (
            float(nav_cfg.get("waypoint_reach_threshold", 0.25)) + 0.05
        )
        self.place_z_offset = float(manip_cfg.get("place_z_offset", 0.10))

        # Mount/sync names — reused by states.py to disable/enable pose-sync.
        self.mount_to = manip_cfg.get("mount_to")
        self.mount_sync_name = manip_cfg.get("mount_sync_name", "franka_base_sync")
        self.mount_local_offset = np.array(
            manip_cfg.get("mount_local_offset", [0.0, 0.0, 0.50]),
            dtype=float,
        )

        self.reset()

    # ── DfRobotApiContext API ────────────────────────────────
    def reset(self) -> None:
        """Called on construction and by DfNetwork.reset(). Returns the
        context to a clean baseline so re-runs start fresh."""
        self.pick_attempts = 0
        self.failed = False
        self.block_state = BlockState.NEED_NAV_TO_BLOCK
        self.prev_block_state = None
        self.last_manip_status = None
        self.last_nav_status = None
        # Pinned cube pose at pick-time — used by `cube_park_xy()` so the
        # AMR keeps driving toward the originally-spotted location even
        # if the cube wobbles a bit while the robot is on its way.
        self._cube_pos_at_dispatch = None

    # ── Per-tick classifier ──────────────────────────────────
    def update_block_state(self) -> None:
        """Reads live world poses + gripper state and sets self.block_state.

        Called by the run_cortex main loop BEFORE network.step() each
        tick. We don't rely on DfNetwork's monitor mechanism because it
        only invokes monitors passed to its constructor — context
        monitors added via add_monitors() are documentation-only in this
        framework version (see df.py:1049-1059).
        """
        self.prev_block_state = self.block_state

        # ── Gather observations
        cube_pos, _ = self.cube.get_world_pose()
        cube_pos = np.asarray(cube_pos, dtype=float)
        amr_pos, _ = self.navigator.get_pose()
        amr_xy = np.asarray(amr_pos, dtype=float)[:2]
        ee_pos = self.manipulator._ee_position()  # uses the robust helper

        # Gripper-closed test: a successfully-grasped cube blocks the
        # fingers from reaching `joint_closed_positions` — they sit
        # ~cube_half away. A 0.005 m threshold against `closed` reads
        # False *exactly when the cube is in the gripper*, which makes
        # `in_gripper` permanently False during a successful grasp.
        # Use cube_half + slack so close() reads as "closed" whether
        # the gripper is empty or holding the cube.
        gripper_closed = False
        try:
            cur = np.asarray(
                self.manipulator.franka.gripper.get_joint_positions(),
                dtype=float,
            )
            closed = np.asarray(
                self.manipulator.franka.gripper.joint_closed_positions,
                dtype=float,
            )
            gripper_closed = bool(
                np.mean(np.abs(cur - closed)) < self.cube_half + 0.005
            )
        except Exception:
            gripper_closed = False

        cube_xy = cube_pos[:2]
        in_gripper = (
            gripper_closed
            and float(np.linalg.norm(ee_pos - cube_pos)) < self.in_gripper_tolerance
        )
        cube_at_b = (
            float(np.linalg.norm(cube_xy - self.point_b)) < self.place_tolerance
            and not in_gripper
        )

        # ── Decision tree (priority order)
        if self.failed:
            self.block_state = BlockState.FAILED
        elif cube_at_b:
            self.block_state = BlockState.DONE
        elif in_gripper:
            standoff_xy = self.b_standoff_xy(amr_xy)
            if float(np.linalg.norm(amr_xy - standoff_xy)) > self.park_tolerance:
                self.block_state = BlockState.NEED_TRANSIT
            else:
                self.block_state = BlockState.NEED_PLACE
        else:
            # AMR drives directly to the cube (cube_park_xy returns
            # cube_xy). Once parked within reach_tol of the goal, the
            # navigator latches REACHED — so on a successful park,
            # AMR-to-cube < reach_tol. PARK_TOL is reach_tol + 5cm,
            # which is the right threshold here.
            if float(np.linalg.norm(amr_xy - cube_xy)) > self.park_tolerance:
                self.block_state = BlockState.NEED_NAV_TO_BLOCK
            else:
                self.block_state = BlockState.NEED_PICK

        # ── Slip detection: transit/place → nav/pick means we lost the
        # cube post-grasp. Reset the pick counter so the recovery pick
        # gets a fresh 3-attempt budget.
        carrying = {BlockState.NEED_TRANSIT, BlockState.NEED_PLACE}
        regrasp = {BlockState.NEED_NAV_TO_BLOCK, BlockState.NEED_PICK}
        if self.prev_block_state in carrying and self.block_state in regrasp:
            if self.pick_attempts > 0:
                print(
                    f"[cortex] transit-time slip detected "
                    f"({self.prev_block_state.value} -> {self.block_state.value}); "
                    f"resetting pick_attempts (was {self.pick_attempts})"
                )
            self.pick_attempts = 0
            self._cube_pos_at_dispatch = None

        # Pin the cube's current XY when we first decide to drive toward
        # it — so the navigator's goal stays stable while the AMR is en
        # route (cube physics jitter is in the millimeters; we don't want
        # to retarget every tick).
        if (
            self.block_state == BlockState.NEED_NAV_TO_BLOCK
            and self._cube_pos_at_dispatch is None
        ):
            self._cube_pos_at_dispatch = cube_xy.copy()

    # ── Geometry helpers ────────────────────────────────────
    def cube_park_xy(self) -> np.ndarray:
        """Navigator goal for the pick approach.

        Returns the cube's XY directly — the navigator's reach_tol
        (0.25 m) handles the parking offset. We deliberately do NOT
        apply place_standoff here: that's a place-side adjustment so
        the arm has reach over the marker B without driving onto it.
        For pick, the cube is the goal, and run_pipeline.py confirms
        this works (AMR parks ~0.22 m from cube and the Franka can
        descend to z≈0.07 — close enough to grasp). With a 0.40 m
        standoff the AMR ends up ~0.47 m from the cube and the Franka
        plateaus 12 cm above the cube due to its reach envelope."""
        target_xy = (
            self._cube_pos_at_dispatch
            if self._cube_pos_at_dispatch is not None
            else np.asarray(self.cube.get_world_pose()[0], dtype=float)[:2]
        )
        return target_xy.copy()

    def b_standoff_xy(self, current_xy: np.ndarray) -> np.ndarray:
        """Navigator goal for the place approach.

        Returns point_b directly — same reasoning as cube_park_xy(): the
        navigator's reach_tol does the parking, and a 0.40 m standoff
        leaves the Franka with ~0.6 m of horizontal extension at place
        time, which exceeds the comfortable kinematic reach when
        descending to a low z target. Marker B has no collision, so
        driving AMR to point_b is safe. Method name kept for backward
        compatibility with NavSetGoalState's lookup."""
        return self.point_b.copy()

    def place_target_b(self) -> np.ndarray:
        """Drop pose at point_b — XY at B, Z = cube_half + place_z_offset.
        Mirrors run_pipeline.py:356."""
        return np.array(
            [self.point_b[0], self.point_b[1],
             self.cube_half + self.place_z_offset],
            dtype=float,
        )

    def live_cube_pos(self) -> np.ndarray:
        """Pick target — read the cube's pose RIGHT NOW (used at the
        moment the manipulator FSM is armed via .pick(...))."""
        cube_pos, _ = self.cube.get_world_pose()
        return np.asarray(cube_pos, dtype=float)
