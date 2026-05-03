"""Shared boilerplate for all apps/*.py scripts.

Responsibilities:
  - Put /workspace/midterm_project on sys.path
  - Force-reload core.* / scenes.* so edits between runs take effect
    (Isaac Sim keeps modules in sys.modules across script runs)
  - Expose a simple `make_logger(name)` that writes to the Isaac Sim
    log directory AND prints to stdout (visible via `docker compose logs`)
  - Load config.yaml
"""
from __future__ import annotations

import importlib
import os
import sys


_REPO_ROOT = "/workspace/midterm_project"
_LOG_DIR = "/root/.nvidia-omniverse/logs"

# Modules that hold LIVE state across script runs — never purge.
_PRESERVE = {"core.state"}


def bootstrap_imports(
    reload_packages: tuple[str, ...] = ("core", "scenes"),
) -> None:
    """Put the repo root on sys.path and purge cached submodules so edits reload.

    Notes:
      - We purge `core.*` and `scenes.*` so Python re-reads edited .py files
      - We do NOT purge `core.state` (keeps the live world / navigator cache)
      - We do NOT purge `apps` — that's the package the caller is running from;
        purging it mid-execution breaks relative imports
      - Parent packages are PRESERVED; purging them forces a rescan of the
        filesystem that can race with `importlib.invalidate_caches()`.
        Clearing only SUBMODULES (e.g. `core.planner`) is enough to pick up
        code edits.
    """
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    to_delete = []
    for mod in list(sys.modules):
        if mod in _PRESERVE:
            continue
        # Only submodules — leave top-level package stubs intact
        for pkg in reload_packages:
            if mod.startswith(pkg + "."):
                to_delete.append(mod)
                break
    for mod in to_delete:
        del sys.modules[mod]

    # Drop finder caches so freshly-purged modules can re-import cleanly.
    importlib.invalidate_caches()


def make_logger(name: str):
    """Return a `log(msg)` callable that writes to disk + stdout.

    Log file: /root/.nvidia-omniverse/logs/<name>.log
    """
    path = os.path.join(_LOG_DIR, f"{name}.log")
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(str(msg))
        try:
            with open(path, "w") as f:
                f.write("\n".join(lines))
        except Exception:
            pass

    return log, path


def make_stream_logger(name: str):
    """High-frequency append-mode logger for per-tick telemetry.

    Unlike make_logger (which rewrites the whole buffer every call),
    this opens with "a" so each emit costs one fsync. Truncates once
    at start so old runs don't bleed in. `tail -f` from the host shows
    a live stream — drop-in equivalent of a ROS topic for our use.

    Log file: /root/.nvidia-omniverse/logs/<name>.stream.log
    """
    path = os.path.join(_LOG_DIR, f"{name}.stream.log")
    try:
        open(path, "w").close()  # truncate previous run
    except Exception:
        pass

    def emit(msg: str) -> None:
        try:
            with open(path, "a") as f:
                f.write(str(msg) + "\n")
        except Exception:
            pass

    return emit, path


def load_config() -> dict:
    """Load and return the parsed config.yaml."""
    import yaml
    with open(os.path.join(_REPO_ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)
