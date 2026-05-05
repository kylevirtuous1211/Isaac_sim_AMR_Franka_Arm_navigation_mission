"""Dispatch decider + `make_decider_network()` factory.

The Dispatch reads `ctx.block_state` (set each tick by
`MobileManipContext.update_block_state()`) and routes to one of:
  - go_home      cube at B → terminal success
  - fail         pick budget exhausted → terminal failure
  - nav_to_block AMR not yet at the cube → drive there
  - pick_rlds    AMR parked at cube → run pick FSM (retry on failure)
  - transit      cube in gripper, AMR not at B-standoff → drive
  - place_rlds   cube in gripper, AMR at B-standoff → run place FSM

Sub-trees are DfStateMachineDecider(DfStateSequence([...])) — exactly
the pattern in peck_decider_network.py. Reactivity is automatic: when
the cube moves mid-execution, the next tick's classifier flips
block_state, Dispatch returns a different child name, and df_descend
exits the in-flight sub-sequence cleanly.
"""
from __future__ import annotations

from isaacsim.cortex.framework.df import (
    DfDecider,
    DfDecision,
    DfNetwork,
    DfStateMachineDecider,
    DfStateSequence,
)

try:
    from core.diag import diag, diag_throttled
except Exception:
    def diag(_msg):
        pass

    def diag_throttled(_k, _m, every=100):
        pass

from .context import BlockState, MobileManipContext
from .states import (
    BumpRetryState,
    DoneAction,
    FailAction,
    ManipPickState,
    ManipPlaceState,
    ManipResetState,
    ManipWaitDoneState,
    NavSetGoalState,
    NavStopState,
    OpenGripperState,
)


class _AutoRestartingStateMachineDecider(DfStateMachineDecider):
    """DfStateMachineDecider that re-arms its state machine when its
    internal state becomes None.

    Stock DfStateMachineDecider only resets self.state via enter(),
    which df_descend only fires on a Dispatch path change. That makes
    retry awkward: when pick_rlds completes (state -> None) and
    Dispatch immediately routes back to pick_rlds (path unchanged
    because block_state is still NEED_PICK), decide() bails out.

    With this subclass, decide() detects the None state and re-enters
    the init_state in-place, so retries work without forcing a
    transient detour through another Dispatch branch.
    """

    def decide(self):
        if self.state is None and self.init_state is not None:
            diag(f"[AutoRestart] re-entering init_state for {type(self).__name__}")
            self.state = self.init_state
            self._bind_state()
            self.state.enter()
        return super().decide()


# ────────────────────────────────────────────────────────────────
# Sub-decider builders. Each is a sequential state machine wrapped
# in DfStateMachineDecider so it acts as a child of Dispatch.
# ────────────────────────────────────────────────────────────────
def _build_nav_to_block() -> DfStateMachineDecider:
    return _AutoRestartingStateMachineDecider(DfStateSequence([
        NavSetGoalState(goal_fn_name="cube_park_xy"),
        NavStopState(),
    ]))


def _build_pick_rlds() -> DfStateMachineDecider:
    """Pick RLDS — open gripper, send RMPflow to the cube, poll until
    the FSM reports done or failed.

    On failure (RMPflow timeout OR SurfaceGripper.gripped() empty at
    phase=done), ManipWaitDoneState transitions into BumpRetryState
    which increments ctx.pick_attempts. The sequence terminates;
    Dispatch reroutes on the next tick (back to pick_rlds for a retry,
    or to fail if the budget is exhausted).
    """
    bump_retry = BumpRetryState()
    return _AutoRestartingStateMachineDecider(DfStateSequence([
        ManipResetState(),
        ManipPickState(),
        ManipWaitDoneState(on_failure=bump_retry, fail_label="pick"),
    ]))


def _build_transit() -> DfStateMachineDecider:
    """Transit — drive the AMR to point_b standoff. The Franka is
    held by the FixedJoint mount (mount_mode=fixed_joint), and the
    arm sits in the manipulator FSM's `carry` phase actively tracking
    a stow pose above the AMR via RMPflow."""
    return _AutoRestartingStateMachineDecider(DfStateSequence([
        NavSetGoalState(goal_fn_name="b_standoff_xy"),
        NavStopState(),
    ]))


def _build_place_rlds() -> DfStateMachineDecider:
    """Place RLDS — descend to point_b, open gripper, cube falls.

    With the SurfaceGripper holding the cube physically (no
    cube_carry_sync teleport), opening the SG releases the D6 joint
    and the cube falls under gravity. The parallel gripper opens at
    the same time, spreading the fingers.

    OpenGripperState is belt-and-braces: ManipPlace's at_place→release
    transition already calls gripper.open(), but calling it again here
    is idempotent and ensures release even if the FSM bailed early.
    """
    return _AutoRestartingStateMachineDecider(DfStateSequence([
        ManipPlaceState(),
        ManipWaitDoneState(fail_label="place", treat_failure_as_done=True),
        OpenGripperState(),
    ]))


# ────────────────────────────────────────────────────────────────
# Top-level dispatch
# ────────────────────────────────────────────────────────────────
class Dispatch(DfDecider):
    """Routes on ctx.block_state. Mirrors peck_decider_network.Dispatch
    structurally; the body is just a flat switch."""

    def __init__(self):
        super().__init__()
        self.add_child("fail", FailAction())
        self.add_child("go_home", DoneAction())
        self.add_child("nav_to_block", _build_nav_to_block())
        self.add_child("pick_rlds", _build_pick_rlds())
        self.add_child("transit", _build_transit())
        self.add_child("place_rlds", _build_place_rlds())

    def decide(self) -> DfDecision:
        s = self.context.block_state
        diag_throttled("dispatch:decide", f"Dispatch.decide block_state={s.value if s else 'None'}")
        if s == BlockState.FAILED:
            return DfDecision("fail")
        if s == BlockState.DONE:
            return DfDecision("go_home")
        if s == BlockState.NEED_NAV_TO_BLOCK:
            return DfDecision("nav_to_block")
        if s == BlockState.NEED_PICK:
            return DfDecision("pick_rlds")
        if s == BlockState.NEED_TRANSIT:
            return DfDecision("transit")
        if s == BlockState.NEED_PLACE:
            return DfDecision("place_rlds")
        # Fallback — shouldn't be reached. Send to nav_to_block which
        # is benign (the navigator just drives toward the cube).
        return DfDecision("nav_to_block")


def make_decider_network(ctx: MobileManipContext) -> DfNetwork:
    """Build the DfNetwork. Mirrors `peck_decider_network.make_decider_network`."""
    return DfNetwork(Dispatch(), context=ctx)
