"""Push the AMR off its origin so the next bootstrap has something to
reset. Writes perturb.json with the new AMR pose.
"""
import sys
sys.path.insert(0, "/workspace/midterm_project")
from apps._common import bootstrap_imports  # noqa: E402

bootstrap_imports()

import numpy as np  # noqa: E402
from tests.scripts._testutil import write_result, capture_exception  # noqa: E402
from core import state  # noqa: E402


try:
    state.require_ready()
    # Teleport the AMR to an obviously-not-origin spot
    new_pos = np.array([3.5, 2.0, 0.0])
    state.navigator.robot.set_world_pose(position=new_pos)
    # Also latch nav as if we had reached something
    state.navigator._reached_latch = True

    amr_pos, _ = state.navigator.get_pose()
    write_result("perturb", {
        "ok": True,
        "amr_pos": [float(v) for v in amr_pos],
        "nav_latched": bool(state.navigator._reached_latch),
        "nav_generation": getattr(state, "nav_generation", 0),
    })
except Exception as e:
    write_result("perturb", capture_exception(e))
