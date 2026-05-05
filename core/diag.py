"""Disk-based diagnostic logger for the cortex/navigator pipeline.

The Isaac Sim VS Code socket executor doesn't reliably surface print()
output (kit log only catches a subset, run_in_isaac response only shows
up after the script ends). When debugging "why didn't this fire" it's
much easier to write directly to disk and tail -f from the host.

Writes append-mode to /root/.nvidia-omniverse/logs/nav_diag.stream.log.
Call truncate() once at the start of a run to clear the previous file.
"""
from __future__ import annotations

import os

PATH = "/root/.nvidia-omniverse/logs/nav_diag.stream.log"
_counters: dict[str, int] = {}


def truncate() -> None:
    try:
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        open(PATH, "w").close()
    except Exception:
        pass


def diag(msg: str) -> None:
    try:
        with open(PATH, "a") as f:
            f.write(str(msg) + "\n")
    except Exception:
        pass


def diag_throttled(key: str, msg: str, every: int = 100) -> None:
    """Logs first hit and every `every`-th hit thereafter for the given key.
    Counters reset on module reload (i.e. on each run_cortex script invocation
    via bootstrap_imports), so each run starts fresh."""
    n = _counters.get(key, 0) + 1
    _counters[key] = n
    if n == 1 or n % every == 0:
        diag(f"[n={n} key={key}] {msg}")
