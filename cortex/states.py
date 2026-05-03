"""Leaf DfState / DfAction implementations for the mobile-manip
decider network. Each one wraps a tiny piece of behavior on the
existing navigator / manipulator / pose-sync helpers.

States transition by returning the next state from `step()`:
  - return None        — terminal (used inside DfStateSequence to
                          advance to the next state)
  - return self        — keep stepping (still working)
  - return another     — explicit transition (used to route into
                          BumpRetryAction on pick failure)

`enter()` runs once when the state is entered (used for one-shot
side effects like sending a nav goal or arming the manipulator FSM).
"""
from __future__ import annotations

import numpy as np

from isaacsim.cortex.framework.df import DfAction, DfState

from core import state as core_state
from core.manipulator import ManipStatus
from core.navigator import NavStatus


# ────────────────────────────────────────────────────────────────
# Pose-sync transitions
# ────────────────────────────────────────────────────────────────
class DisablePoseSyncState(DfState):
    """One-shot: remove the franka_base_sync physics callback so
    Phase 2/4 can teleport Franka without it being overwritten.
    Mirrors run_pipeline.py `_disable_pose_sync()`."""

    def enter(self):
        ctx = self.context
        if not ctx.mount_to:
            return
        try:
            core_state.world.remove_physics_callback(ctx.mount_sync_name)
        except Exception:
            pass

    def step(self):
        return None  # terminal


class EnablePoseSyncState(DfState):
    """One-shot: re-install the franka_base_sync callback. Mirrors
    run_pipeline.py `_enable_pose_sync()`."""

    def enter(self):
        ctx = self.context
        if not ctx.mount_to:
            return
        ctx.manipulator.ensure_mount_sync(core_state.world)

    def step(self):
        return None


class SettleState(DfState):
    """Wait N physics ticks before continuing. Used after teleporting
    Franka to give physics time to propagate the new pose before
    RMPflow starts driving from it.

    Mirrors the `for _ in range(30): await next_update_async()` blocks
    in run_pipeline.py:229-230 and 248-251 — without this delay,
    RMPflow runs from a freshly-teleported (and thus dynamically
    inconsistent) state and converges to a different (worse) local
    solution. Empirically this is the difference between EE plateau
    z=0.069 (run_pipeline, settled) and z=0.131 (run_cortex, unsettled).
    """

    def __init__(self, ticks: int = 30):
        super().__init__()
        self._ticks_remaining = ticks
        self._ticks_initial = ticks

    def enter(self):
        self._ticks_remaining = self._ticks_initial

    def step(self):
        self._ticks_remaining -= 1
        if self._ticks_remaining <= 0:
            return None
        return self


class RebaseFrankaToAmrState(DfState):
    """One-shot: teleport Franka to AMR_root + mount_offset so the arm
    is positioned relative to the AMR's current spot before pick/place.
    Mirrors run_pipeline.py `_rebase_franka_to_amr()`."""

    def enter(self):
        ctx = self.context
        if not ctx.mount_to:
            return
        amr_pos, amr_ori = ctx.navigator.get_pose()
        ctx.manipulator.franka.set_world_pose(
            position=np.asarray(amr_pos, dtype=float) + ctx.mount_local_offset,
            orientation=amr_ori,
        )

    def step(self):
        return None


# ────────────────────────────────────────────────────────────────
# Navigator wrappers
# ────────────────────────────────────────────────────────────────
class NavSetGoalState(DfState):
    """Arm the navigator with a goal computed at entry-time from a
    callable on the context. Holds while the navigator is RUNNING and
    terminates on REACHED/FAILED/BLOCKED.

    Args:
        goal_fn_name: name of a no-arg method on the context that
            returns the (x, y) goal. Indirection by name keeps the
            target live: `cube_park_xy()` reads from
            ctx._cube_pos_at_dispatch which is pinned at dispatch-time,
            and `b_standoff_xy(amr_xy)` reads the AMR's current pose.
    """

    def __init__(self, goal_fn_name: str):
        super().__init__()
        self.goal_fn_name = goal_fn_name

    def enter(self):
        ctx = self.context
        goal_fn = getattr(ctx, self.goal_fn_name)
        # b_standoff_xy needs the current AMR xy.
        try:
            goal = goal_fn()
        except TypeError:
            amr_pos, _ = ctx.navigator.get_pose()
            goal = goal_fn(np.asarray(amr_pos, dtype=float)[:2])
        print(f"[cortex] NavSetGoal({self.goal_fn_name}) -> {goal.tolist()}")
        ctx.navigator.set_goal(goal)

    def step(self):
        ctx = self.context
        # The main loop already calls navigator.step() each tick — we
        # just observe the cached status here.
        s = ctx.last_nav_status
        if s is None:
            return self
        if s == NavStatus.REACHED:
            return None
        if s in (NavStatus.FAILED,):
            print(f"[cortex] NavSetGoal({self.goal_fn_name}) FAILED")
            return None  # let Dispatch reclassify on next tick
        return self


class NavStopState(DfState):
    """One-shot: latch navigator REACHED + zero wheels."""

    def enter(self):
        self.context.navigator.stop()

    def step(self):
        return None


# ────────────────────────────────────────────────────────────────
# Manipulator wrappers
# ────────────────────────────────────────────────────────────────
class ManipResetState(DfState):
    """One-shot: reset the manipulator FSM (idle phase, gripper open)
    AND set Franka arm joints to a deterministic home pose.

    Without the joint reset, RMPflow convergence varies between runs
    based on whatever joint config the previous run left behind. The
    home pose chosen here matches Franka's standard "ready" position
    (Isaac Sim Franka.dof_default_state from examples), which gives
    RMPflow a stable starting point for top-down picks.
    """

    # Standard Franka home pose — matches isaacsim Franka examples.
    # 7 arm joints + 2 gripper fingers = 9 DOFs.
    HOME_ARM_JOINTS = np.array(
        [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        dtype=float,
    )

    def enter(self):
        ctx = self.context
        ctx.manipulator.reset()  # opens gripper, sets phase=idle
        # Reset arm joints (first 7 DOFs only — leave gripper alone;
        # manipulator.reset() already opened the gripper).
        try:
            franka = ctx.manipulator.franka
            cur = np.asarray(franka.get_joint_positions(), dtype=float)
            new = cur.copy()
            n = min(7, len(cur))
            new[:n] = self.HOME_ARM_JOINTS[:n]
            franka.set_joint_positions(new)
            franka.set_joint_velocities(np.zeros_like(cur))
        except Exception as e:
            print(f"[cortex] ManipResetState joint reset failed: {e}")

    def step(self):
        return None


class ManipPickState(DfState):
    """Arm the manipulator FSM for pick at the cube's CURRENT pose.
    Reads the cube's live world pose at entry-time so the target
    reflects whatever the cube has been moved to."""

    def enter(self):
        ctx = self.context
        target = ctx.live_cube_pos()
        print(f"[cortex] ManipPick target={target.tolist()} (attempt {ctx.pick_attempts + 1}/{ctx.MAX_PICK_ATTEMPTS})")
        ctx.manipulator.pick(target)

    def step(self):
        return None  # terminal — ManipWaitDone handles the polling


class ManipPlaceState(DfState):
    """Arm the manipulator FSM for place at point_b.

    Force the FSM phase to "idle" before calling place(): the pick
    sub-decider can be torn down mid-flight (e.g. when the classifier
    flips to NEED_TRANSIT as soon as in_gripper=True), leaving the FSM
    stuck at "grasp" or "lift". manipulator.place() is a no-op unless
    phase is "idle" or "done" (see core/manipulator.py:259-264), so
    without the reset the place phase silently fails to start and the
    FSM eventually times out.
    """

    def enter(self):
        ctx = self.context
        # Don't call manipulator.reset() — that would open the gripper,
        # dropping the cube. Just zero the FSM's phase tracking.
        ctx.manipulator._phase = "idle"
        ctx.manipulator._tick_in_phase = 0
        target = ctx.place_target_b()
        print(f"[cortex] ManipPlace target={target.tolist()}")
        ctx.manipulator.place(target)

    def step(self):
        return None


class ManipWaitDoneState(DfState):
    """Poll the manipulator FSM until it reports DONE or FAILED.

    On FAILED, transitions to a configurable on_failure state (used by
    the pick sub-sequence to route into BumpRetryAction). If
    on_failure is None, treats FAILED as terminal (place uses this — a
    place timeout fails immediately per spec).

    `on_phase_entry` is an optional dict mapping phase-name → callable
    invoked once the FIRST tick we observe the manipulator entering
    that phase. Used to install/remove cube_carry_sync at exactly the
    grasp/release transitions without polluting the manipulator FSM
    itself.
    """

    def __init__(self, on_failure: "DfState | None" = None,
                 fail_label: str = "manip",
                 on_phase_entry: "dict | None" = None):
        super().__init__()
        self._on_failure = on_failure
        self._fail_label = fail_label
        self._on_phase_entry = on_phase_entry or {}
        self._fired_phases: set = set()

    def enter(self):
        # New invocation of the wait state — reset phase-tracking so
        # callbacks can fire again on this entry.
        self._fired_phases = set()

    def step(self):
        ctx = self.context
        s = ctx.last_manip_status
        phase = ctx.manipulator.get_phase()

        # Fire phase-entry callbacks once per unique phase observed.
        if phase in self._on_phase_entry and phase not in self._fired_phases:
            self._fired_phases.add(phase)
            try:
                self._on_phase_entry[phase](ctx)
            except Exception as e:
                print(f"[cortex] phase-entry callback for '{phase}' failed: {e}")

        if phase == "done":
            # FSM completed normally. With carry_sync as the holding
            # mechanism, the cube IS at the EE by the time we reach
            # phase=done (carry_sync was installed at phase=lift via
            # on_phase_entry), so any inline grasp-distance check
            # would always pass. SurfaceGripper-based validation is
            # disabled here because the D6 joint engagement is
            # currently unreliable; we'd false-fail on every pick if
            # we required sg.gripped() to be non-empty. Real failures
            # come through phase==failed below.
            return None
        if phase == "failed" or s == ManipStatus.FAILED:
            print(f"[cortex] {self._fail_label} reported FAILED (phase={phase})")
            if self._on_failure is not None:
                # The on_failure state isn't in the parent
                # DfStateSequence's sequence list, so its context isn't
                # bound by the sequence's bind() pass. Bind it here
                # before returning so its step() can read self.context.
                self._on_failure.bind(self.context, self.params)
                return self._on_failure
            # No retry path — terminal failure for this branch.
            ctx.failed = True
            return None
        return self


def install_cube_carry_sync(ctx) -> None:
    """Install a per-tick physics callback that teleports the cube to
    the EE pose. Mirrors run_pipeline.py's `_install_cube_carry_sync`
    (lines 117-148) — a "belt-and-braces" mechanism that keeps the
    cube glued to the gripper through transit + place even when the
    parallel-gripper finger friction isn't enough to hold it.

    Also freezes the arm joint positions captured at install time, so
    gravity doesn't flail the arm during transit (the original
    workaround uncovered that without freezing, the arm would shove
    the AMR via collision and send it on chaotic trajectories).
    """
    franka = ctx.manipulator.franka
    cube = ctx.cube
    try:
        frozen_joints = np.asarray(franka.get_joint_positions(), dtype=float)
    except Exception:
        frozen_joints = None

    def _carry(_step_size):
        try:
            if frozen_joints is not None:
                franka.set_joint_positions(frozen_joints)
                franka.set_joint_velocities(np.zeros_like(frozen_joints))
            ee_pos, ee_ori = franka.end_effector.get_world_pose()
            cube.set_world_pose(position=ee_pos, orientation=ee_ori)
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
        except Exception as _e:
            print(f"[cortex carry_sync] failed: {_e}")

    try:
        core_state.world.remove_physics_callback("cube_carry_sync")
    except Exception:
        pass
    core_state.world.add_physics_callback("cube_carry_sync", callback_fn=_carry)
    n_joints = len(frozen_joints) if frozen_joints is not None else 0
    print(f"[cortex] cube_carry_sync installed (frozen joints={n_joints})")


def remove_cube_carry_sync(ctx) -> None:
    try:
        core_state.world.remove_physics_callback("cube_carry_sync")
        print("[cortex] cube_carry_sync removed")
    except Exception:
        pass


class InstallCubeCarrySyncState(DfState):
    """One-shot: install cube_carry_sync as a physics callback.

    Used as the success terminal of pick_rlds, AFTER ManipWaitDoneState
    confirms the FSM reached phase=done. By then:
      - The arm is in lift-final pose (cube + clearance height) — a
        good "carrying" config worth freezing for transit.
      - The gripper has had grasp_hold_ticks * 2 ticks closed.
    Capturing frozen_joints here means transit's joint-pinning
    preserves the lift-final pose, not a mid-trajectory snapshot.
    """

    def enter(self):
        install_cube_carry_sync(self.context)

    def step(self):
        return None


class ReregisterCarrySyncState(DfState):
    """Removes and re-adds cube_carry_sync so it sits LATER in
    PhysX's callback registration order than pose-sync.

    PhysX fires physics callbacks in registration order. carry_sync
    teleports the cube to the EE; pose-sync teleports the Franka base.
    If carry_sync runs first, it sees the OLD EE position (before
    pose-sync moves the Franka) and the cube ends up 1 frame behind.

    By re-registering carry_sync AFTER EnablePoseSyncState has
    re-installed pose-sync, we ensure carry_sync fires last and reads
    the post-pose-sync EE pose.

    Runs only if carry_sync already exists; otherwise it's a no-op
    (e.g. if pick failed and we're somehow re-entering transit).
    """

    def enter(self):
        ctx = self.context
        # Check if carry_sync exists; if so, re-install (which removes + adds)
        try:
            install_cube_carry_sync(ctx)
        except Exception as e:
            print(f"[cortex] ReregisterCarrySync failed: {e}")

    def step(self):
        return None


class BumpRetryState(DfState):
    """Increments the pick-attempt counter on a real grasp failure
    then terminates the pick sub-sequence. Sets ctx.failed when the
    budget is exhausted.

    Implemented as a DfState so it can be returned as the next state
    from `ManipWaitDoneState.step()` — DfStateSequence will switch to
    it, then terminate when this state's step() returns None.
    """

    def step(self):
        ctx = self.context
        ctx.pick_attempts += 1
        # Clear the manipulator's FAILED phase so a subsequent pick()
        # call can set _phase=above_pick.
        ctx.manipulator.reset()
        if ctx.pick_attempts >= ctx.MAX_PICK_ATTEMPTS:
            ctx.failed = True
            print(f"[cortex] {ctx.pick_attempts}/{ctx.MAX_PICK_ATTEMPTS} "
                  f"pick failures — FAILED")
        else:
            print(f"[cortex] pick attempt {ctx.pick_attempts}/"
                  f"{ctx.MAX_PICK_ATTEMPTS} failed — will retry")
        return None  # terminal — DfStateSequence will exit the pick branch


# ────────────────────────────────────────────────────────────────
# Terminal actions
# ────────────────────────────────────────────────────────────────
class RemoveCubeCarrySyncState(DfState):
    """One-shot: remove cube_carry_sync. Use at place entry — the
    cube is currently glued to the EE at lift-final height; removing
    the callback lets it fall under gravity to the floor below."""

    def enter(self):
        remove_cube_carry_sync(self.context)

    def step(self):
        return None


class OpenGripperState(DfState):
    """One-shot: open the parallel gripper. Visually completes the
    place (cube has already fallen by the time we open)."""

    def enter(self):
        try:
            self.context.manipulator.franka.gripper.open()
        except Exception as e:
            print(f"[cortex] OpenGripper failed: {e}")

    def step(self):
        return None


class DoneAction(DfAction):
    """Cube reached B — log once and idle. The main loop reads
    ctx.block_state == DONE and breaks the outer loop."""

    def __init__(self):
        super().__init__()
        self._announced = False

    def step(self):
        if not self._announced:
            print("[cortex] Dispatch -> go_home (cube delivered)")
            self._announced = True


class FailAction(DfAction):
    """Pick budget exhausted — log once and idle. Outer loop reads
    ctx.failed (== block_state FAILED) and breaks."""

    def __init__(self):
        super().__init__()
        self._announced = False

    def step(self):
        if not self._announced:
            print("[cortex] Dispatch -> fail (pick budget exhausted)")
            self._announced = True
