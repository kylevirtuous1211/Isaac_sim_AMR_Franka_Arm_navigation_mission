"""Pytest fixtures / helpers for integration tests.

Architecture:
    Tests run on the host in a regular Python + pytest environment,
    but the stable state they assert on lives inside the running
    Isaac Sim container. We bridge by sending small in-sim scripts
    via `run_in_isaac.py` (TCP socket to port 8226); each script
    writes a JSON result file to /root/.nvidia-omniverse/logs/,
    which is bind-mounted to cache/isaac-sim/logs/ on the host.

Prereqs:
    - Isaac Sim container running with isaacsim.code_editor.vscode
    - `apps/bootstrap.py` has been run at least once this session
      (tests assume state is populated). A session fixture below
      verifies this and bootstraps if needed.

Run with:
    cd /home/kyle/Desktop/isaac-sim-quickstart/midterm_project
    pytest tests/ -v
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import time

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
HOST_LOG_DIR = REPO_ROOT / "cache" / "isaac-sim" / "logs"
CONTAINER_NAME = "isaac-sim-quickstart-isaac-sim-1"


# ────────────────────────────────────────────────────────────────
def run_in_isaac(script_rel_path: str, timeout: float = 60.0) -> None:
    """Send a script already in the container at
    /workspace/midterm_project/<script_rel_path> to Isaac Sim.
    Returns when the TCP socket closes. Blocking.
    """
    full_path = f"midterm_project/{script_rel_path}"
    result = subprocess.run(
        ["python3", "run_in_isaac.py", full_path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"run_in_isaac.py exited {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


def sync_to_container(src_rel_path: str) -> None:
    """Copy a file from the host repo into the container at the same
    relative path under /workspace/midterm_project/."""
    src = REPO_ROOT / "midterm_project" / src_rel_path
    dst = f"{CONTAINER_NAME}:/workspace/midterm_project/{src_rel_path}"
    # Ensure parent dir exists in container
    parent = "/workspace/midterm_project/" + str(pathlib.Path(src_rel_path).parent)
    subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "mkdir", "-p", parent], check=False
    )
    subprocess.run(["docker", "cp", str(src), dst], check=True)


def read_result(name: str, timeout: float = 10.0) -> dict:
    """Wait for /root/.nvidia-omniverse/logs/<name>.json to appear on the
    host, then return its parsed contents."""
    path = HOST_LOG_DIR / f"{name}.json"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass  # still being written, retry
        time.sleep(0.5)
    raise TimeoutError(f"{path} didn't appear within {timeout}s")


def clear_result(name: str) -> None:
    """Remove any prior result file so a stale one can't fool us."""
    subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "rm", "-f",
         f"/root/.nvidia-omniverse/logs/{name}.json"],
        check=False,
    )


def run_and_read(
    script_rel_path: str, result_name: str,
    exec_timeout: float = 60.0, read_timeout: float = 15.0,
) -> dict:
    """Full cycle: clear old result, sync script, run it, read JSON result."""
    clear_result(result_name)
    sync_to_container(script_rel_path)
    # Also sync the _testutil helper every time — cheap, avoids staleness
    try:
        sync_to_container("tests/scripts/_testutil.py")
        sync_to_container("tests/scripts/__init__.py")
        sync_to_container("tests/__init__.py")
    except subprocess.CalledProcessError:
        pass
    run_in_isaac(script_rel_path, timeout=exec_timeout)
    return read_result(result_name, timeout=read_timeout)


# ────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def ensure_bootstrap():
    """Run bootstrap.py once per test session. Idempotent — fast-reset
    when state is already populated, full-reload only on first run."""
    sync_to_container("apps/bootstrap.py")
    sync_to_container("apps/_common.py")
    sync_to_container("core/state.py")
    # Generous timeout — first bootstrap can take 60-180s downloading
    # hospital.usd over the network.
    run_in_isaac("apps/bootstrap.py", timeout=240.0)
    # Verify state is ready by probing
    result = run_and_read("tests/scripts/probe.py", "probe")
    assert result.get("ready") is True, \
        f"Bootstrap failed to populate state: {result}"
    return result
