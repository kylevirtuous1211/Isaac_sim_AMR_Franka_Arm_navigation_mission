"""PhysX articulation solver-iteration bumps for tightly-coupled mounts.

A `UsdPhysics.FixedJoint` between two articulations is solved as a
maximal-coordinate constraint (we set excludeFromArticulation=True; see
core/franka_mount_joint.py). Default solver iteration counts (8 position,
1 velocity) leave that constraint visibly compliant under arm motion —
the Franka base lags the chassis by a few millimetres each tick. Bumping
to 32/8 tightens it to the point where the lag is unobservable.

This is the FixedJoint analogue of "tuning joint stiffness" — we make the
constraint *solve harder*, not stiffer (it's already infinitely stiff in
principle).
"""
from __future__ import annotations

from typing import Iterable


def bump_solver_iterations(
    stage,
    prim_paths: Iterable[str],
    position: int = 32,
    velocity: int = 8,
) -> list[str]:
    """Apply PhysxArticulationAPI with raised iteration counts to each prim.

    Returns the list of prim paths actually tuned (skips invalid ones with
    a printed warning rather than raising).

    Idempotent: re-applying the API and re-setting the attributes produces
    the same USD state.
    """
    from pxr import PhysxSchema

    tuned: list[str] = []
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            print(f"[articulation_tuning] prim not found, skipping: {path}")
            continue
        api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
        api.CreateSolverPositionIterationCountAttr().Set(int(position))
        api.CreateSolverVelocityIterationCountAttr().Set(int(velocity))
        tuned.append(path)
    return tuned
