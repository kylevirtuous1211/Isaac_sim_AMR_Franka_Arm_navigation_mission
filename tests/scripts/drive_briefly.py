"""Run navigation for a short burst so the AMR actually moves away
from origin. Writes drive.json with the final pose.

Used by navigation-reset tests: we want to 'mess up' the nav state
(move AMR + potentially latch) so bootstrap has something meaningful
to reset.
"""
import sys
import traceback

sys.path.insert(0, "/workspace/midterm_project")
from apps._common import bootstrap_imports  # noqa: E402

bootstrap_imports()

import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402

from tests.scripts._testutil import write_result, capture_exception  # noqa: E402
from core import state  # noqa: E402
from core.navigator import NavStatus  # noqa: E402


# Progress markers go to a SEPARATE file (drive.progress). The test
# harness only polls drive.json, which is written exactly once at
# end-of-script with the final result.
def _mark(stage):
    with open("/root/.nvidia-omniverse/logs/drive.progress", "w") as _f:
        _f.write(stage)

_mark("imports_done")


try:
    _mark("starting")
    state.require_ready()
    _mark("state_ready")
    world = state.world
    nav = state.navigator

    cfg = state.config or {}
    goal = np.array(cfg.get("task", {}).get("point_a", [2.0, 1.0, 0.0]),
                    dtype=float)[:2]

    my_gen = getattr(state, "nav_generation", 0)
    nav.set_goal(goal)
    _mark("goal_set")

    # Non-async play to avoid hangs when the world was already paused/playing
    # from a previous script's state.
    try:
        world.play()
    except Exception:
        pass
    _mark("playing")

    final_status = NavStatus.RUNNING
    for _tick in range(800):
        await omni.kit.app.get_app().next_update_async()
        if getattr(state, "nav_generation", 0) != my_gen:
            break
        final_status = nav.step()
        if final_status in (NavStatus.REACHED, NavStatus.FAILED):
            break

    _mark("loop_done")

    pos, _ = nav.get_pose()
    write_result("drive", {
        "ok": True,
        "final_pos": [float(v) for v in pos],
        "final_status": final_status.value,
        "nav_latched": bool(getattr(nav, "_reached_latch", False)),
        "ticks_run": _tick + 1,
    })
except Exception as e:
    write_result("drive", {
        "ok": False,
        "error": f"{type(e).__name__}: {e}",
        "traceback": traceback.format_exc(),
    })
