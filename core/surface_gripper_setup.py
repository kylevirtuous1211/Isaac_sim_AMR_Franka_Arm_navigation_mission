"""SurfaceGripper authoring + lightweight control wrapper.

Why this exists
---------------
Isaac Sim 5.1's `SurfaceGripper` enables a real PhysX D6 joint between a
gripper prim and a "candidate" rigid body when the action is closed. That
joint kinematically tracks parent-prim teleports (which our pose-sync
callback does every physics tick), so the cube travels with Franka cleanly
when the AMR drives — something a `DynamicCuboid` held only by gripper
friction cannot do.

This module:
  - authors `/World/Franka/panda_hand/SurfaceGripper` plus one D6 joint
    targeting `/World/TargetCube` once at scene-setup time
  - exposes a tiny `SurfaceGripperWrapper` that wraps the C++ interface
    so manipulator.py can call `.close() / .open() / .status() / .gripped()`
"""
from __future__ import annotations

import math
from typing import Any

from pxr import Gf, Sdf, UsdPhysics


# Conventional locations — kept in module-level constants so other code
# can cross-reference without typo'ing the prim paths.
GRIPPER_PATH = "/World/Franka/panda_hand/SurfaceGripper"
JOINTS_ROOT = "/World/Surface_Gripper_Joints"
CUBE_JOINT_PATH = f"{JOINTS_ROOT}/D6Joint_Cube"
CUBE_PATH = "/World/TargetCube"
HAND_PATH = "/World/Franka/panda_hand"


def _author_d6_attachment(
    stage,
    joint_path: str,
    body0: str,
    body1: str,
    local_pos0: tuple[float, float, float],
    local_pos1: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Any:
    """Author a D6 joint shaped like the official SurfaceGripper sample.

    Locks transX / transY (limit-low > limit-high), allows transZ travel
    of [0, 0.01] m for the closure stroke, and ±3 rad on rotational DOFs.
    Also stamps the `IsaacAttachmentPointAPI` schema and the `isaac:*`
    attributes the SurfaceGripperManager looks at.
    """
    joint = UsdPhysics.Joint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateExcludeFromArticulationAttr().Set(True)
    # Start DISABLED — body0 and body1 are far apart at scene start (cube
    # sits at point_a, panda_hand sits on the AMR ~2 m away). With the
    # joint enabled and a transZ drive of stiffness 5000 N/m, PhysX would
    # try to pull them together each tick, dragging the AMR around.
    # The SurfaceGripper manager re-enables this joint when close_gripper()
    # is called and the cube enters max_grip_distance.
    joint.CreateJointEnabledAttr().Set(False)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateBreakForceAttr().Set(3.4028235e38)
    joint.CreateBreakTorqueAttr().Set(3.4028235e38)

    prim = joint.GetPrim()

    # Lock translation in X / Y by inverting the limits — the convention
    # the official D6 attachment USD uses ("low > high" means locked).
    for axis in ("transX", "transY"):
        api = UsdPhysics.LimitAPI.Apply(prim, axis)
        api.CreateHighAttr().Set(-1.0)
        api.CreateLowAttr().Set(1.0)

    # transZ: allow up to 15 cm of separation so the joint can engage
    # even when RMPflow's reach plateau leaves a ~8 cm gap between the
    # gripper anchor and the cube. The official sample uses [0, 0.01]
    # because its gripper actually touches the candidate; ours doesn't.
    api_z = UsdPhysics.LimitAPI.Apply(prim, "transZ")
    api_z.CreateHighAttr().Set(0.15)
    api_z.CreateLowAttr().Set(0.0)

    # Rotational limits — allow small rotation as the cube settles in the gripper
    for axis in ("rotX", "rotY", "rotZ"):
        api = UsdPhysics.LimitAPI.Apply(prim, axis)
        api.CreateHighAttr().Set(3.0)
        api.CreateLowAttr().Set(-3.0)

    # Drives — the transZ drive is what pulls the cube into the gripper.
    drive_z = UsdPhysics.DriveAPI.Apply(prim, "transZ")
    drive_z.CreateStiffnessAttr().Set(5000.0)
    drive_z.CreateDampingAttr().Set(100.0)
    for axis, stiffness in (("rotX", 100.0), ("rotY", 100.0), ("rotZ", 10000.0)):
        d = UsdPhysics.DriveAPI.Apply(prim, axis)
        d.CreateStiffnessAttr().Set(stiffness)

    # IsaacAttachmentPointAPI — the schema that marks this as a gripper joint.
    # ApplyAPI accepts a string name in Isaac Sim's USD bindings.
    try:
        prim.ApplyAPI("IsaacAttachmentPointAPI")
    except Exception:
        # Some Isaac Sim versions need the full identifier
        prim.ApplyAPI(Sdf.Path("IsaacAttachmentPointAPI"))

    # Isaac-specific scalar attributes the manager reads
    prim.CreateAttribute("isaac:forwardAxis", Sdf.ValueTypeNames.Token).Set("Z")
    prim.CreateAttribute("isaac:clearanceOffset", Sdf.ValueTypeNames.Float).Set(0.008)

    return joint


def author_surface_gripper(stage, cfg: dict) -> str:
    """Idempotently author the SurfaceGripper prim + cube D6 joint.

    Must be called AFTER the Franka and the cube exist on the stage and
    BEFORE `world.reset_async()` so PhysX picks them up.

    Returns the gripper prim path (also available as `GRIPPER_PATH`).
    """
    import usd.schema.isaac.robot_schema as robot_schema

    sg_cfg = (cfg.get("manipulator") or {}).get("surface_gripper") or {}
    max_grip = float(sg_cfg.get("max_grip_distance", 0.08))
    coax = float(sg_cfg.get("coaxial_force_limit", 50.0))
    shear = float(sg_cfg.get("shear_force_limit", 50.0))
    retry = float(sg_cfg.get("retry_interval", 0.1))
    grip_z = float(sg_cfg.get("grip_offset_z", 0.10))

    # Container Xform for the joints
    if not stage.GetPrimAtPath(JOINTS_ROOT).IsValid():
        stage.DefinePrim(JOINTS_ROOT, "Xform")

    # The D6 joint between panda_hand and the cube
    _author_d6_attachment(
        stage,
        CUBE_JOINT_PATH,
        body0=HAND_PATH,
        body1=CUBE_PATH,
        local_pos0=(0.0, 0.0, grip_z),
    )

    # The SurfaceGripper prim itself, parented under panda_hand so it
    # teleports cleanly with the AMR pose-sync.
    robot_schema.CreateSurfaceGripper(stage, GRIPPER_PATH)
    gripper_prim = stage.GetPrimAtPath(GRIPPER_PATH)

    # Hook the gripper to its candidate joints (just our cube joint)
    rel = gripper_prim.GetRelationship(robot_schema.Relations.ATTACHMENT_POINTS.name)
    rel.SetTargets([CUBE_JOINT_PATH])

    # Properties
    gripper_prim.GetAttribute(robot_schema.Attributes.MAX_GRIP_DISTANCE.name).Set(max_grip)
    gripper_prim.GetAttribute(robot_schema.Attributes.COAXIAL_FORCE_LIMIT.name).Set(coax)
    gripper_prim.GetAttribute(robot_schema.Attributes.SHEAR_FORCE_LIMIT.name).Set(shear)
    gripper_prim.GetAttribute(robot_schema.Attributes.RETRY_INTERVAL.name).Set(retry)

    return GRIPPER_PATH


class SurfaceGripperWrapper:
    """Thin wrapper around the C++ surface_gripper interface.

    Reasons we don't use `GripperView`:
      - Single gripper, not batched. The interface API is simpler.
      - GripperView re-derives transforms each call which is overkill here.
    """

    def __init__(self, prim_path: str = GRIPPER_PATH):
        import isaacsim.robot.surface_gripper._surface_gripper as _sg
        self._sg_module = _sg
        self.prim_path = prim_path
        self._iface = _sg.acquire_surface_gripper_interface()
        # Mirror state to USD so other tools (e.g. the property panel) reflect it
        try:
            self._iface.set_write_to_usd(True)
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._iface.close_gripper(self.prim_path)
        except Exception as e:
            print(f"[SurfaceGripper] close failed: {e}")

    def open(self) -> None:
        try:
            self._iface.open_gripper(self.prim_path)
        except Exception as e:
            print(f"[SurfaceGripper] open failed: {e}")

    def status(self) -> str:
        """Return one of "Open", "Closing", "Closed", or "Unknown"."""
        try:
            st = self._iface.get_gripper_status(self.prim_path)
        except Exception:
            return "Unknown"
        names = {
            self._sg_module.GripperStatus.Open: "Open",
            self._sg_module.GripperStatus.Closing: "Closing",
            self._sg_module.GripperStatus.Closed: "Closed",
        }
        return names.get(st, "Unknown")

    def gripped(self) -> list[str]:
        try:
            return list(self._iface.get_gripped_objects(self.prim_path))
        except Exception:
            return []
