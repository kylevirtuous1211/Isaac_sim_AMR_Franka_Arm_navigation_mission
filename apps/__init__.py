"""Thin orchestrator scripts sent to Isaac Sim via run_in_isaac.py.

Each app is a small top-level script that:
  1. Ensures `core` is importable + freshly reloaded (see `_common.py`).
  2. Loads the config.
  3. Runs one specific workflow (bootstrap, navigate, manipulate, full pipeline).

Apps expect `apps/bootstrap.py` to have populated `core.state` — they will
raise a clear error if not.
"""
