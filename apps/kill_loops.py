"""Force-abort all currently-running run_*.py loops by bumping nav_generation."""
import sys
sys.path.insert(0, "/workspace/midterm_project")
from apps._common import bootstrap_imports, make_logger
bootstrap_imports()
log, _ = make_logger("kill_loops")

from core import state
before = getattr(state, "nav_generation", 0)
state.nav_generation = before + 1
log(f"nav_generation: {before} -> {state.nav_generation}")

# Also zero the wheels immediately
if state.navigator is not None:
    try:
        state.navigator.stop()
        log("navigator.stop() called")
    except Exception as e:
        log(f"stop failed: {e}")
