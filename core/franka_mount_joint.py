"""Rigid FixedJoint mount between Carter chassis and Franka base.

Why this exists
---------------
The default mount strategy is `_install_pose_sync` in manipulator.py — a
per-tick physics callback that teleports panda_link0 to chassis_link.world_pose
plus an offset. That works for slow nav-and-pick missions but the Franka's
mass/inertia never loads the wheels, and reaction torques from arm motion
don't propagate to the base.

A `UsdPhysics.FixedJoint` makes Carter and Franka one coupled physics system
in PhysX. The crucial flag is `excludeFromArticulation=True`: a previous
attempt without it caused PhysX to treat one of the two articulations as
rooted to world, jamming Carter's wheels (see core/manipulator.py for the
historical comment). Marking the joint as a maximal-coordinate constraint
side-steps that reduced-coordinate trap — the joint becomes a regular
constraint between two independent articulations rather than an attempt to
splice their kinematic trees.

Companion knob: `bump_solver_iterations` in core/articulation_tuning.py
tightens the MC constraint so the FixedJoint doesn't visibly compliance-jitter.
"""
from __future__ import annotations

from pxr import Gf, Sdf, UsdPhysics


MOUNT_JOINT_PATH_DEFAULT = "/World/Franka/franka_mount"


def _set_articulation_fixed_base(stage, root_path: str, fixed: bool) -> None:
    """Toggle fixed-base on an articulation root via
    `physxArticulation:fixedBase`. PhysX honors this attribute even though
    it isn't part of the typed PhysxArticulationAPI schema; it's the only
    way to defeat the implicit world anchor on a Franka articulation root
    that has no parent joint.

    fixed=False is required when an external FixedJoint will be the only
    constraint holding panda_link0 in place — otherwise the implicit world
    anchor and our joint fight each other and PhysX can't reconcile.

    Idempotent.
    """
    prim = stage.GetPrimAtPath(root_path)
    if not prim or not prim.IsValid():
        return
    attr = prim.GetAttribute("physxArticulation:fixedBase")
    if not attr or not attr.IsValid():
        attr = prim.CreateAttribute(
            "physxArticulation:fixedBase", Sdf.ValueTypeNames.Bool
        )
    attr.Set(bool(fixed))


def author_franka_mount_joint(stage, cfg: dict) -> str:
    """Idempotently author a FixedJoint between chassis_link and panda_link0.

    Must be called AFTER both articulations spawn and BEFORE
    world.reset_async() so PhysX picks the joint up at simulation start.
    Returns the joint prim path.

    Local frames are set so body0 (chassis_link) + mount_local_offset
    coincides with body1 (panda_link0) origin, matching how `_spawn_franka`
    placed the Franka on first bootstrap.

    Idempotent via `SetTargets` and `Set` on attributes — re-calling this on
    fast-reset overwrites the same prim instead of accumulating duplicates.
    """
    manip_cfg = cfg.get("manipulator") or {}
    chassis_path = manip_cfg.get("mount_to", "/World/NovaCarter/chassis_link")
    franka_root = manip_cfg.get("prim_path", "/World/Franka")
    panda_link0 = f"{franka_root}/panda_link0"
    joint_path = manip_cfg.get("mount_joint_path", MOUNT_JOINT_PATH_DEFAULT)
    offset = manip_cfg.get("mount_local_offset", [0.0, 0.0, 0.35])

    chassis_prim = stage.GetPrimAtPath(chassis_path)
    if not chassis_prim or not chassis_prim.IsValid():
        raise RuntimeError(f"FixedJoint mount: chassis prim not found: {chassis_path}")
    panda_prim = stage.GetPrimAtPath(panda_link0)
    if not panda_prim or not panda_prim.IsValid():
        raise RuntimeError(f"FixedJoint mount: panda_link0 not found: {panda_link0}")

    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(chassis_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(panda_link0)])

    # excludeFromArticulation=True is the workaround for the wheel-jamming
    # bug — see module docstring.
    joint.CreateExcludeFromArticulationAttr().Set(True)

    # Local frames: body0 (chassis) at +offset, body1 (panda_link0) at origin.
    # PhysX will enforce body0_world * localPos0 == body1_world * localPos1.
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(offset[0]), float(offset[1]), float(offset[2])))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # Don't let arm reaction torques snap the mount. inf == effectively unbreakable.
    joint.CreateBreakForceAttr().Set(3.4028235e38)
    joint.CreateBreakTorqueAttr().Set(3.4028235e38)

    # Defeat the Franka's implicit world anchor — without this, panda_link0
    # is pinned to its spawn pose by the articulation root and our FixedJoint
    # to chassis_link cannot reconcile, producing the "arm flung off the AMR"
    # symptom observed in the first smoke test.
    _set_articulation_fixed_base(stage, franka_root, fixed=False)

    return joint_path


def remove_franka_mount_joint(stage, cfg: dict) -> bool:
    """Remove the FixedJoint prim if it exists and restore the Franka to
    fixed-base. Idempotent — returns True when a prim was actually removed,
    False when there was nothing to do (the fixed-base restore still runs
    in both cases so the manipulator is safe to use in pose_sync mode).

    Used when switching mount_mode back to pose_sync at runtime so a stale
    FixedJoint doesn't fight the per-tick teleport AND so a floating Franka
    doesn't fall under gravity once the joint is gone.
    """
    manip_cfg = cfg.get("manipulator") or {}
    joint_path = manip_cfg.get("mount_joint_path", MOUNT_JOINT_PATH_DEFAULT)
    franka_root = manip_cfg.get("prim_path", "/World/Franka")

    # Restore fixed-base FIRST so when the joint is removed the Franka
    # doesn't free-fall during the teardown tick.
    _set_articulation_fixed_base(stage, franka_root, fixed=True)

    prim = stage.GetPrimAtPath(joint_path)
    if not prim or not prim.IsValid():
        return False
    stage.RemovePrim(joint_path)
    return True
