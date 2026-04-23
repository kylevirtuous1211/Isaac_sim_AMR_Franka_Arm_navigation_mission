# ============================================================
# apps/reset_state.py — tear down the cached state so the next
# bootstrap.py starts from a clean slate. Useful after config.yaml
# edits or if the scene got into a weird state.
# ============================================================
import sys
sys.path.insert(0, "/workspace/midterm_project")
from apps._common import bootstrap_imports, make_logger  # noqa: E402

bootstrap_imports()
log, _ = make_logger("reset_state")

from core import state  # noqa: E402

log(state.summary())
state.teardown()
log("state.teardown() done — next bootstrap will fully reload.")
