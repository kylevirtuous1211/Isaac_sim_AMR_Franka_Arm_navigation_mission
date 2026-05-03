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

from .context import BlockState, MobileManipContext
from .states import (
    BumpRetryState,
    DisablePoseSyncState,
    DoneAction,
    EnablePoseSyncState,
    FailAction,
    ManipPickState,
    ManipResetState,
    ManipWaitDoneState,
    NavSetGoalState,
    NavStopState,
    OpenGripperState,
    RebaseFrankaToAmrState,
    RemoveCubeCarrySyncState,
    ReregisterCarrySyncState,
    SettleState,
    install_cube_carry_sync,
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
    """Pick RLDS — analogous to make_pick_rlds() in
    block_stacking_behavior. We open the gripper (via ManipReset),
    teleport Franka above the AMR, arm the FSM with the cube's live
    pose, and poll for done/failed.

    On failure, ManipWaitDoneState transitions into BumpRetryAction
    which increments ctx.pick_attempts. The sequence terminates;
    Dispatch reroutes on the next tick (back to pick_rlds for a retry,
    or to fail if the budget is exhausted)."""
    bump_retry = BumpRetryState()
    return _AutoRestartingStateMachineDecider(DfStateSequence([
        ManipResetState(),
        DisablePoseSyncState(),
        RebaseFrankaToAmrState(),
        SettleState(ticks=30),
        ManipPickState(),
        # Install cube_carry_sync at phase=lift — by then the gripper
        # has finished closing (40 ticks of grasp_hold) and the EE has
        # plateaued near the cube (~0.082 m vs cube_z=0.026, ~5 cm
        # away). The teleport jump is therefore ~5 cm rather than
        # the ~15 cm we'd see if installed at phase=done after the
        # arm lifts. SurfaceGripper.close() also fires at the FSM
        # level (see core/manipulator.py); if its D6 joint engages,
        # the kinematic teleport is redundant but harmless.
        ManipWaitDoneState(
            on_failure=bump_retry,
            fail_label="pick",
            on_phase_entry={"lift": install_cube_carry_sync},
        ),
    ]))


def _build_transit() -> DfStateMachineDecider:
    return _AutoRestartingStateMachineDecider(DfStateSequence([
        EnablePoseSyncState(),
        # Re-register carry_sync after pose-sync so PhysX fires it
        # last each tick (otherwise carry_sync teleports cube to OLD
        # EE pose, then pose-sync moves the Franka, leaving cube 1
        # frame behind).
        ReregisterCarrySyncState(),
        NavSetGoalState(goal_fn_name="b_standoff_xy"),
        NavStopState(),
    ]))


def _build_place_rlds() -> DfStateMachineDecider:
    """Place RLDS — drop-style place.

    The cube is currently teleported to the EE by carry_sync at lift-
    final height (~0.176 m). Removing carry_sync at place entry lets
    gravity deposit it on point_b. SurfaceGripper.open() in the
    OpenGripperState path is a belt-and-braces release in case the
    SG D6 joint engaged.
    """
    return _AutoRestartingStateMachineDecider(DfStateSequence([
        DisablePoseSyncState(),
        RebaseFrankaToAmrState(),
        SettleState(ticks=10),
        RemoveCubeCarrySyncState(),
        SettleState(ticks=60),
        OpenGripperState(),
        SettleState(ticks=20),
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
