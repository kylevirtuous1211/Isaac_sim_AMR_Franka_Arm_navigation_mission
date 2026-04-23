"""Shared helpers for in-sim test scripts.

Each test script ends by calling `write_result(name, dict)` — the host-side
pytest reads that JSON back via the log bind-mount and asserts on it.
"""
from __future__ import annotations

import json
import os
import traceback
from typing import Any


_LOG_DIR = "/root/.nvidia-omniverse/logs"


def write_result(name: str, data: dict[str, Any]) -> str:
    """Serialize `data` to /root/.nvidia-omniverse/logs/<name>.json."""
    path = os.path.join(_LOG_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_fallback)
    return path


def _fallback(x):
    # numpy arrays, ndarrays, Gf types, enums, etc.
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "value"):
        return x.value
    return str(x)


def capture_exception(e: BaseException) -> dict:
    return {
        "ok": False,
        "error": f"{type(e).__name__}: {e}",
        "traceback": traceback.format_exc(),
    }
