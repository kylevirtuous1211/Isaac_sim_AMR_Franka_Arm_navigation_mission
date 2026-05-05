"""Leaf DfState / DfAction implementations for the mobile-manip
decider network. Each one wraps a tiny piece of behavior on the
existing navigator / manipulator helpers.

States transition by returning the next state from `step()`:
  - return None        — terminal (used inside DfStateSequence to
                          advance to the next state)
  - return self        — keep stepping (still working)
  - return another     — explicit transition (used to route into
                          BumpRetryState on pick failure)

`enter()` runs once when the state is entered (used for one-shot
side effects like sending a nav goal or arming the manipulator FSM).
"""
from __future__ import annotations

from isaacsim.cortex.framework.df import DfAction, DfState

from core.manipulator import ManipStatus
from core.navigator import NavStatus

try:
    from core.diag import diag
except Exception:  # core.diag absent in older bootstraps — fall back silently
    def diag(_msg):  # noqa: D401
        pass


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
        import numpy as np
        diag(f"[NavSetGoalState.enter] FIRED for goal_fn_name={self.goal_fn_name!r}")
        ctx = self.context
        diag(f"[NavSetGoalState.enter] ctx={type(ctx).__name__} "
             f"has_navigator={hasattr(ctx, 'navigator')}")
        try:
            goal_fn = getattr(ctx, self.goal_fn_name)
        except Exception as e:
            diag(f"[NavSetGoalState.enter] getattr FAILED: {type(e).__name__}: {e}")
            raise
        # b_standoff_xy needs the current AMR xy.
        try:
            try:
                goal = goal_fn()
            except TypeError:
                amr_pos, _ = ctx.navigator.get_pose()
                goal = goal_fn(np.asarray(amr_pos, dtype=float)[:2])
        except Exception as e:
            diag(f"[NavSetGoalState.enter] goal_fn() FAILED: {type(e).__name__}: {e}")
            raise
        diag(f"[NavSetGoalState.enter] computed goal={goal.tolist()}")
        print(f"[cortex] NavSetGoal({self.goal_fn_name}) -> {goal.tolist()}")
        try:
            ctx.navigator.set_goal(goal)
        except Exception as e:
            diag(f"[NavSetGoalState.enter] set_goal FAILED: {type(e).__name__}: {e}")
            raise
        diag(f"[NavSetGoalState.enter] set_goal returned OK")

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
        diag("[NavStopState.enter] FIRED — calling navigator.stop()")
        self.context.navigator.stop()

    def step(self):
        return None


# ────────────────────────────────────────────────────────────────
# Manipulator wrappers
# ────────────────────────────────────────────────────────────────
class ManipResetState(DfState):
    """One-shot: reset the manipulator FSM (idle phase, gripper open).

    No kinematic joint reset — RMPflow plans from whatever pose the arm
    is in. Snapping joints with set_joint_positions() was a patch over
    the lack of a smooth motion path; with the carry-phase hold and
    fixed-joint mount in place, the arm always lives in a sane pose.
    """

    def enter(self):
        ctx = self.context
        ctx.manipulator.reset()  # opens gripper, sets phase=idle

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
    stuck at "grasp", "lift", or "carry". manipulator.place() is a
    no-op unless phase is "idle" or "done" (see core/manipulator.py),
    so without the reset the place phase silently fails to start.
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
    the pick sub-sequence to route into BumpRetryState). If
    on_failure is None, treats FAILED as terminal (place uses this — a
    place timeout fails immediately per spec).
    """

    def __init__(self, on_failure: "DfState | None" = None,
                 fail_label: str = "manip",
                 treat_failure_as_done: bool = False):
        super().__init__()
        self._on_failure = on_failure
        self._fail_label = fail_label
        # When True, a phase=failed (e.g. RMPflow phase_timeout while
        # descending the last cm to at_place) is logged but doesn't
        # propagate to ctx.failed. Used by the place subdecider — by
        # the time the FSM hits this state, the cube is already
        # within place_tolerance of point_b in the vast majority of
        # cases.
        self._treat_failure_as_done = treat_failure_as_done

    def step(self):
        ctx = self.context
        s = ctx.last_manip_status
        phase = ctx.manipulator.get_phase()

        # Pick success: lift transitions to "carry" (not "done"). The
        # carry phase actively holds the stow pose during transit; the
        # pick FSM is logically complete here. Validate the grasp via
        # the SurfaceGripper — if the D6 joint didn't engage, the
        # closure was unphysical (closed above the block, etc.) and
        # routes to retry.
        if self._fail_label == "pick" and phase == "carry":
            sg = getattr(ctx.manipulator, "surface_gripper", None)
            if sg is not None and not sg.gripped():
                print("[cortex] pick reached carry but SG.gripped() empty — failing")
                if self._on_failure is not None:
                    self._on_failure.bind(self.context, self.params)
                    return self._on_failure
                ctx.failed = True
                return None
            return None  # success
        if phase == "done":
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
            if self._treat_failure_as_done:
                # The classifier (cube_at_b vs place_tolerance) decides
                # success. RMPflow phase_timeout during the at_place
                # final-cm descent is common but the cube is already
                # placed by then.
                return None
            # No retry path — terminal failure for this branch.
            ctx.failed = True
            return None
        return self


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
class OpenGripperState(DfState):
    """One-shot: open the parallel gripper. Belt-and-braces in case
    the manipulator FSM's at_place->release transition didn't fire."""

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
