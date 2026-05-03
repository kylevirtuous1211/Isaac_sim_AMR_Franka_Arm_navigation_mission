# ============================================================
# diag_surface_gripper.py — dump SurfaceGripper + D6 joint state.
# Output goes to /root/.nvidia-omniverse/logs/sg_diag.log
# (== cache/isaac-sim/logs/sg_diag.log on the host).
# ============================================================
import sys

sys.path.insert(0, "/workspace/midterm_project")
sys.modules.pop("apps._common", None)
from apps._common import bootstrap_imports, make_logger  # noqa: E402

bootstrap_imports(reload_packages=("core", "scenes"))
log, _ = make_logger("sg_diag")

try:
    import omni.usd
    from pxr import UsdPhysics, Sdf
    from core.surface_gripper_setup import (
        GRIPPER_PATH, JOINTS_ROOT, CUBE_JOINT_PATH, HAND_PATH, CUBE_PATH
    )

    stage = omni.usd.get_context().get_stage()
    log(f"\n=== SurfaceGripper diagnostic ===")
    for path in (HAND_PATH, CUBE_PATH, JOINTS_ROOT, CUBE_JOINT_PATH, GRIPPER_PATH):
        prim = stage.GetPrimAtPath(path)
        valid = prim.IsValid() if prim else False
        type_name = prim.GetTypeName() if valid else "(invalid)"
        applied = list(prim.GetAppliedSchemas()) if valid else []
        log(f"\n[{path}]")
        log(f"  valid={valid} type={type_name}")
        log(f"  applied schemas: {applied}")
        if valid and "Joint" in type_name:
            j = UsdPhysics.Joint(prim)
            b0 = j.GetBody0Rel().GetTargets()
            b1 = j.GetBody1Rel().GetTargets()
            enabled = prim.GetAttribute("physics:jointEnabled").Get()
            log(f"  body0={b0}")
            log(f"  body1={b1}")
            log(f"  jointEnabled={enabled}")
            for attr_name in ("isaac:forwardAxis", "isaac:clearanceOffset"):
                a = prim.GetAttribute(attr_name)
                log(f"  {attr_name}={a.Get() if a else '(missing)'}")
        if valid and path == GRIPPER_PATH:
            for rel_name in ("attachmentPoints", "isaac:attachmentPoints"):
                rel = prim.GetRelationship(rel_name)
                if rel:
                    log(f"  rel[{rel_name}].targets={list(rel.GetTargets())}")
            for attr_name in ("isaac:maxGripDistance", "isaac:coaxialForceLimit",
                              "isaac:shearForceLimit", "isaac:retryInterval",
                              "maxGripDistance", "coaxialForceLimit"):
                a = prim.GetAttribute(attr_name)
                if a:
                    log(f"  {attr_name}={a.Get()}")

    # Also try the runtime SG interface to see if it sees the gripper
    log(f"\n=== Runtime SG interface ===")
    try:
        import isaacsim.robot.surface_gripper._surface_gripper as _sg
        iface = _sg.acquire_surface_gripper_interface()
        try:
            st = iface.get_gripper_status(GRIPPER_PATH)
            log(f"  status({GRIPPER_PATH}) = {st}")
        except Exception as e:
            log(f"  get_gripper_status failed: {e}")
        try:
            gripped = iface.get_gripped_objects(GRIPPER_PATH)
            log(f"  gripped({GRIPPER_PATH}) = {list(gripped)}")
        except Exception as e:
            log(f"  get_gripped_objects failed: {e}")

        # Try close() and observe state change.
        log(f"\n=== Attempting close_gripper() ===")
        try:
            iface.close_gripper(GRIPPER_PATH)
            log(f"  close_gripper called, no exception")
        except Exception as e:
            log(f"  close_gripper failed: {e}")
        # Sleep a few ticks via app update so the manager has a chance to
        # process the close request.
        import omni.kit.app
        for _ in range(120):  # ~2s at 60Hz
            await omni.kit.app.get_app().next_update_async()
        try:
            st = iface.get_gripper_status(GRIPPER_PATH)
            log(f"  status after close + 2s = {st}")
            gripped = iface.get_gripped_objects(GRIPPER_PATH)
            log(f"  gripped after close + 2s = {list(gripped)}")
        except Exception as e:
            log(f"  status read failed: {e}")
    except Exception as e:
        log(f"  interface acquire failed: {e}")

except Exception as e:
    import traceback
    log(f"diag ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())
