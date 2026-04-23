"""Emergency cleanup: remove lingering physics callbacks and force teardown."""
import sys
sys.path.insert(0, "/workspace/midterm_project")
from apps._common import bootstrap_imports, make_logger
bootstrap_imports()
log, _ = make_logger("clean_callbacks")

from core import state

if state.world is not None:
    for name in ("franka_base_sync", "run_nav_step", "nav_step"):
        try:
            state.world.remove_physics_callback(name)
            log(f"removed {name}")
        except Exception as e:
            log(f"skip {name}: {e}")

state.nav_generation = getattr(state, "nav_generation", 0) + 1
log(f"nav_generation -> {state.nav_generation}")

if state.navigator is not None:
    state.navigator.stop()
    log("navigator.stop()")

state.teardown()
log("state teardown complete")
