"""Dump current scene state to probe.json.

Reports positions of AMR + Franka, goal config, nav_generation, and
whether the state cache is populated. Used both as a bootstrap check
and by other tests to compare before/after values.
"""
import sys
sys.path.insert(0, "/workspace/midterm_project")
from apps._common import bootstrap_imports  # noqa: E402

bootstrap_imports()

import traceback  # noqa: E402
import numpy as np  # noqa: E402

from tests.scripts._testutil import write_result, capture_exception  # noqa: E402
from core import state  # noqa: E402


try:
    result = {"ready": state.is_ready(), "ok": True}

    if state.is_ready():
        amr_pos, amr_ori = state.navigator.get_pose()
        franka_pos, _ = state.manipulator.franka.get_world_pose()

        cfg = state.config or {}
        mount_to = cfg.get("manipulator", {}).get("mount_to")
        offset = cfg.get("manipulator", {}).get("mount_local_offset", [0, 0, 0])

        result.update({
            "amr": {
                "pos": [float(v) for v in amr_pos],
                "ori": [float(v) for v in amr_ori],
            },
            "franka": {
                "pos": [float(v) for v in franka_pos],
            },
            "config": {
                "mount_to": mount_to,
                "mount_local_offset": list(offset),
                "start_position": cfg.get("navigator", {})
                                      .get("robot", {})
                                      .get("start_position"),
                "goal_A": cfg.get("task", {}).get("point_a"),
                "reach_tol": cfg.get("navigator", {})
                                 .get("waypoint_reach_threshold"),
            },
            "nav_generation": getattr(state, "nav_generation", 0),
            "nav_latched": bool(getattr(state.navigator, "_reached_latch", False))
                if state.navigator else None,
        })

    write_result("probe", result)
except Exception as e:
    write_result("probe", capture_exception(e))
