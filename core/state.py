"""Process-wide cache of live Isaac Sim handles.

Why this exists
---------------
The Isaac Sim `code_editor.vscode` socket extension executes every script
we send in the SAME long-running Kit Python process. Python module state
(including everything attached to this module) persists between scripts.

We exploit that by caching the heavy objects — the `World`, the loaded
hospital stage, the navigator, the manipulator — in this module. A
bootstrap script populates the cache once, and every subsequent script
imports from here instead of reloading the hospital (which costs 60–180s
over the network).

Usage
-----
    # In apps/bootstrap.py:
    from core import state
    if not state.is_ready():
        state.world = build_world()
        state.navigator = build_navigator(...)
        ...

    # In apps/run_nav.py:
    from core import state
    state.require_ready()
    navigator = state.navigator
    ...

Call `teardown()` only if you genuinely want to start over.
"""
from __future__ import annotations

from typing import Any, Optional


# ── Cached handles ───────────────────────────────────────────
# All None until bootstrap.py populates them.
world: Optional[Any] = None
planner: Optional[Any] = None
navigator: Optional[Any] = None
manipulator: Optional[Any] = None

# Bookkeeping
scene_loaded_path: Optional[str] = None   # URL of whichever stage is open
config: Optional[dict] = None             # parsed config.yaml
extras: dict[str, Any] = {}               # any other per-run handles


def is_ready() -> bool:
    """True once bootstrap has populated the core handles."""
    return (
        world is not None
        and planner is not None
        and navigator is not None
        and manipulator is not None
    )


def require_ready() -> None:
    """Raise if bootstrap hasn't run yet — gives downstream scripts a
    clear error instead of a `NoneType has no attribute` crash."""
    if not is_ready():
        raise RuntimeError(
            "core.state is not populated. "
            "Run `python3 run_in_isaac.py midterm_project/apps/bootstrap.py` "
            "first (it loads hospital.usd and spawns the robots)."
        )


def teardown() -> None:
    """Clear the cache. Next bootstrap will start fresh."""
    global world, planner, navigator, manipulator, scene_loaded_path, config
    world = None
    planner = None
    navigator = None
    manipulator = None
    scene_loaded_path = None
    config = None
    extras.clear()


def summary() -> str:
    """One-line human-readable status — useful in logs."""
    if not is_ready():
        return "state: EMPTY (bootstrap not run)"
    return (
        f"state: ready — "
        f"scene={scene_loaded_path}, "
        f"planner={type(planner).__name__}, "
        f"navigator={type(navigator).__name__}, "
        f"manipulator={type(manipulator).__name__}"
    )
