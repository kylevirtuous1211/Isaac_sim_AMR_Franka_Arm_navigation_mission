#!/usr/bin/env python3
"""Closed-loop parameter tuner for the Cortex pipeline.

What it does
------------
Iteratively:
  1. Run apps/run_cortex.py via run_in_isaac.py (blocks until completion,
     ~30-90s per iteration depending on how far it gets).
  2. Parse cache/isaac-sim/logs/run_cortex.log + cortex_positions.stream.log
     into a fixed set of metrics describing what happened.
  3. Apply rule-based config.yaml updates for whichever failure mode the
     metrics indicate.
  4. Repeat until success or the rule book runs dry.

Why this exists
---------------
The Cortex pipeline has ~10 interacting parameters across navigator,
manipulator, mount, and classifier thresholds. Tuning by eye + manual
re-runs is slow because you have to remember which knob fixed which
symptom. This script encodes the diagnosis-fix mapping so the loop
converges instead of oscillating.

Diagnosis → Fix table
---------------------
  Symptom (parsed from logs)              | Rule fires           | Param touched
  ----------------------------------------+----------------------+------------------------
  EE plateau > cube_z + 5cm during pick   | rule_pick_z_offset   | pick_z_offset (more neg)
  AMR-to-cube > reach_tol+5cm at need_pick | rule_park_too_far   | waypoint_reach_threshold
  manip phase timeout (FAILED status)     | rule_phase_timeout   | phase_timeout_ticks
  Place error > place_tolerance           | rule_place_z_offset  | place_z_offset
  Wheels sunk (AMR z < -0.005 mid-run)    | rule_mount_offset    | mount_local_offset z
  Run hits max_ticks, no terminal state   | rule_max_ticks       | simulation.max_ticks

Convergence
-----------
The loop exits as soon as outcome=success OR no rule fires (no more
known-bad signals to fix). The latter case usually means you need to
add a rule (or that the failure is deterministic and outside this
parameter space).

Usage
-----
  # Prereq: bootstrap.py has run once and Isaac Sim is up.
  python3 midterm_project/tune_cortex.py
  python3 midterm_project/tune_cortex.py --max-iter 20 --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import yaml


REPO = Path(__file__).resolve().parent.parent
CFG_PATH = REPO / "midterm_project" / "config.yaml"
LOG_PATH = REPO / "cache" / "isaac-sim" / "logs" / "run_cortex.log"
BOOTSTRAP_LOG = REPO / "cache" / "isaac-sim" / "logs" / "bootstrap.log"
STREAM_PATH = REPO / "cache" / "isaac-sim" / "logs" / "cortex_positions.stream.log"
RUN_IN_ISAAC = REPO / "run_in_isaac.py"
SCRIPT = "midterm_project/apps/run_cortex.py"
BOOTSTRAP_SCRIPT = "midterm_project/apps/bootstrap.py"
# History goes under midterm_project/ (user-owned) — the cache/ dir is
# owned by root because the container writes it.
HISTORY_DIR = REPO / "midterm_project" / ".tune_history"

# Marker the tuner waits for in run_cortex.log before parsing. run_cortex.py
# emits this as its final log() line — see midterm_project/apps/run_cortex.py.
COMPLETE_MARKER = "run_cortex complete."
ERROR_MARKER = "ERROR:"
# Bootstrap completion marker — bootstrap.py emits this on the fast-reset path.
BOOTSTRAP_DONE_MARKER = "Scene reset to defaults"
BOOTSTRAP_FULL_MARKER = "Bootstrap complete."


# ────────────────────────────────────────────────────────────
# Metric extraction
# ────────────────────────────────────────────────────────────
_STREAM_RE = re.compile(
    r"t=(?P<t>\d+) "
    r"state=(?P<state>\S+) "
    r"pick_attempts=(?P<pa>\d+) "
    r"manip_phase=(?P<phase>\S+) "
    r"cube=(?P<cx>[-\d.]+),(?P<cy>[-\d.]+),(?P<cz>[-\d.]+) "
    r"ee=(?P<ex>[-\d.]+),(?P<ey>[-\d.]+),(?P<ez>[-\d.]+) "
    r"amr=(?P<ax>[-\d.]+),(?P<ay>[-\d.]+)"
)


_EP_RE = re.compile(
    r"ep=(?P<ep>\d+):\s+(?P<outcome>\S+)\s+err=(?P<err>[\d.]+)\s+m\s+ticks=(?P<ticks>\d+)\s+picks=(?P<picks>\d+)"
)
_TOTAL_RE = re.compile(r"TOTAL:\s+(?P<n_ok>\d+)/(?P<n_tot>\d+)\s+episodes succeeded")


def parse_outcome() -> dict:
    """Returns a dict of metrics for the LATEST run_cortex.log.

    Tuner now drives a multi-episode randomized loop. The aggregate
    summary at the end of run_cortex.log is the source of truth:

        TOTAL: K/N episodes succeeded

    Per-episode lines are also parsed so rules can target whichever
    episode failed first. Stream-based metrics (EE plateau, park
    distance, manip phase) are scoped to the FIRST FAILED episode in
    the stream (ep=K_first_fail).
    """
    metrics = {
        "outcome": "no_log",         # success | partial | all_failed | error | unknown
        "n_success": 0,
        "n_total": 0,
        "episodes": [],              # [{ep, outcome, err, ticks, picks}, ...]
        "first_fail_ep": None,
        "first_fail_outcome": None,
        "first_fail_err": None,
        "first_fail_ticks": None,
        # Stream-derived metrics for the first failing episode (or any
        # episode if none failed) — these drive parameter rules.
        "ee_plateau_z": None,
        "park_dist": None,
        "phase_timed_out": False,
        "place_err_xy": None,
        "max_ticks_hit": False,
    }

    if not LOG_PATH.exists():
        return metrics
    log = LOG_PATH.read_text()

    if "ERROR:" in log:
        metrics["outcome"] = "error"

    # Episode rows from the summary block.
    eps = []
    for em in _EP_RE.finditer(log):
        d = em.groupdict()
        eps.append({
            "ep": int(d["ep"]),
            "outcome": d["outcome"],
            "err": float(d["err"]),
            "ticks": int(d["ticks"]),
            "picks": int(d["picks"]),
        })
    metrics["episodes"] = eps

    tm = _TOTAL_RE.search(log)
    if tm:
        n_ok = int(tm.group("n_ok"))
        n_tot = int(tm.group("n_tot"))
        metrics["n_success"] = n_ok
        metrics["n_total"] = n_tot
        if n_ok == n_tot and n_tot > 0:
            metrics["outcome"] = "success"
        elif n_ok > 0:
            metrics["outcome"] = "partial"
        elif n_tot > 0:
            metrics["outcome"] = "all_failed"
    elif eps:
        # Per-episode rows present but TOTAL: line missing — run mid-flight.
        metrics["outcome"] = "in_progress"

    # First failing episode anchors the per-episode rule firing.
    fail_eps = [e for e in eps if e["outcome"] != "success"]
    target = fail_eps[0] if fail_eps else (eps[0] if eps else None)
    if target is not None:
        metrics["first_fail_ep"] = target["ep"]
        metrics["first_fail_outcome"] = target["outcome"]
        metrics["first_fail_err"] = target["err"]
        metrics["first_fail_ticks"] = target["ticks"]
        metrics["place_err_xy"] = target["err"]
        if target["outcome"] == "tick_budget_exhausted":
            metrics["max_ticks_hit"] = True

    # Phase timeout indicator (anywhere in log) — RMPflow stuck somewhere.
    if "phase '" in log and "timed out" in log:
        metrics["phase_timed_out"] = True

    # Stream — scoped to first failing episode (or all if none failed).
    if STREAM_PATH.exists():
        target_ep = metrics["first_fail_ep"] if metrics["first_fail_ep"] is not None else 0
        ep_prefix = f"ep={target_ep} "
        pick_zs = []
        park_dist_first = None
        for line in STREAM_PATH.read_text().splitlines():
            if not line.startswith(ep_prefix):
                continue
            sm = _STREAM_RE.search(line)
            if not sm:
                continue
            d = sm.groupdict()
            if d["phase"] in ("at_pick", "grasp", "lift"):
                pick_zs.append(float(d["ez"]))
            if d["state"] == "need_pick" and park_dist_first is None:
                ax, ay = float(d["ax"]), float(d["ay"])
                cx, cy = float(d["cx"]), float(d["cy"])
                park_dist_first = ((ax - cx) ** 2 + (ay - cy) ** 2) ** 0.5

        if pick_zs:
            metrics["ee_plateau_z"] = min(pick_zs)
        if park_dist_first is not None:
            metrics["park_dist"] = park_dist_first

    return metrics


# ────────────────────────────────────────────────────────────
# Rule-based tuner
# ────────────────────────────────────────────────────────────
def _round(v: float, n: int = 3) -> float:
    return float(round(v, n))


def tune(cfg: dict, m: dict, prev_metrics: Optional[dict] = None) -> tuple[dict, list[str]]:
    """Apply zero or more rules to cfg in-place. Returns (cfg, actions_log).

    `prev_metrics` is the metrics dict from the previous iteration (or
    None on the first iteration). Used by no-progress guards: if a
    rule fired last iter and produced no measurable improvement,
    don't fire it again on the same lever.
    """
    actions: list[str] = []

    if m["outcome"] == "success":
        return cfg, ["no-op (already success)"]

    # ── Rule 1: EE plateau too high during pick → drop pick target.
    # RMPflow plateaus a fixed gap above its target at near-extreme reach;
    # pushing the target below the cube center makes the actual EE land
    # near cube center where the gripper can close around it.
    #
    # No-progress guard: if last iter ALSO had ee_plateau ~ same value,
    # the bottleneck isn't the target — it's a kinematic limit. Stop
    # firing this rule (caller will see "no rule fired" and bail).
    cube_z = float(cfg["task"]["cube_size"]) / 2.0  # cube sitting on floor
    plateau_unchanged = (
        prev_metrics is not None
        and prev_metrics.get("ee_plateau_z") is not None
        and m.get("ee_plateau_z") is not None
        and abs(m["ee_plateau_z"] - prev_metrics["ee_plateau_z"]) < 0.01
    )
    if m["ee_plateau_z"] is not None and not plateau_unchanged:
        gap = m["ee_plateau_z"] - cube_z
        # Multi-episode mode: any failed-or-partial run with a high EE
        # plateau is a candidate. Specifically include "tick_budget_exhausted"
        # (the new dominant failure mode for stuck pick / transit).
        if gap > 0.05 and m["outcome"] in ("partial", "all_failed"):
            cur = float(cfg["manipulator"].get("pick_z_offset", 0.0))
            # Push offset by ~60% of the gap, capped at -0.10 m total
            # (deeper than that puts the target below the floor, which
            # RMPflow ignores because it can't physically descend there).
            delta = -min(0.05, gap * 0.6)
            new = _round(max(-0.10, cur + delta), 3)
            if new != cur:
                cfg["manipulator"]["pick_z_offset"] = new
                actions.append(
                    f"pick_z_offset {cur:+.3f} → {new:+.3f} "
                    f"(EE plateau {m['ee_plateau_z']:.3f} m, gap {gap:.3f} m above cube)"
                )
    elif plateau_unchanged and m["outcome"] != "success":
        # ── Rule 1b: pick_z_offset didn't move the EE — kinematic limit.
        # Lower the mount so the Franka base sits closer to the cube,
        # giving the arm more vertical reach. Mount ~0.35 is the empirical
        # sweet spot (EE descends to z≈0.07). Below that the chassis
        # collision starts to interfere.
        cur = float(cfg["manipulator"].get("mount_local_offset", [0, 0, 0.5])[2])
        # Step down by 0.05 per iteration, floor at 0.35.
        new = _round(max(0.35, cur - 0.05), 3)
        if new != cur:
            offset = list(cfg["manipulator"].get("mount_local_offset", [0, 0, 0.5]))
            offset[2] = new
            cfg["manipulator"]["mount_local_offset"] = offset
            actions.append(
                f"mount_local_offset[z] {cur:.3f} → {new:.3f} "
                f"(EE plateau unchanged at {m['ee_plateau_z']:.3f} m — "
                f"kinematic limit; lowering Franka base for more reach)"
            )
            # Also reset pick_z_offset since the prior negative value
            # was a wrong-lever fix that didn't help.
            old_pzo = float(cfg["manipulator"].get("pick_z_offset", 0.0))
            if old_pzo != 0.0:
                cfg["manipulator"]["pick_z_offset"] = 0.0
                actions.append(
                    f"pick_z_offset {old_pzo:+.3f} → +0.000 "
                    f"(reverted; was no-op against kinematic limit)"
                )
        else:
            actions.append(
                f"DIAG: EE plateau unchanged ({m['ee_plateau_z']:.3f} m) "
                f"and mount already at floor (0.35 m). Kinematic limit "
                f"reached — needs longer-reach arm or closer parking."
            )

    # ── Rule 2: AMR parked too far at the start of pick → tighten reach_tol.
    # waypoint_reach_threshold of 0.25 lets the AMR park ~0.25 m off the
    # cube. Beyond ~0.30 m, the Franka can't reach down far enough.
    if m["park_dist"] is not None and m["park_dist"] > 0.30 and m["outcome"] != "success":
        cur = float(cfg["navigator"].get("waypoint_reach_threshold", 0.25))
        new = _round(max(0.15, cur * 0.85), 3)
        if new != cur:
            cfg["navigator"]["waypoint_reach_threshold"] = new
            actions.append(
                f"waypoint_reach_threshold {cur:.3f} → {new:.3f} "
                f"(park_dist {m['park_dist']:.3f} m > 0.30)"
            )

    # ── Rule 3: phase-timeout → bump the budget (RMPflow occasionally needs
    # more ticks at near-extreme reach to settle).
    if m["phase_timed_out"]:
        cur = int(cfg["manipulator"].get("phase_timeout_ticks", 800))
        new = min(2000, int(cur * 1.5))
        if new != cur:
            cfg["manipulator"]["phase_timeout_ticks"] = new
            actions.append(
                f"phase_timeout_ticks {cur} → {new} (FSM phase timed out)"
            )

    # ── Rule 4: place error too high → bump place_z_offset DOWN so the
    # cube is released closer to the marker.
    if (
        m["outcome"] in ("partial", "all_failed")
        and m["place_err_xy"] is not None
        and m["place_err_xy"] > float(cfg["task"].get("place_tolerance", 0.20))
    ):
        cur = float(cfg["manipulator"].get("place_z_offset", 0.10))
        new = _round(max(0.02, cur - 0.02), 3)
        if new != cur:
            cfg["manipulator"]["place_z_offset"] = new
            actions.append(
                f"place_z_offset {cur:.3f} → {new:.3f} "
                f"(place_err_xy {m['place_err_xy']:.3f} m)"
            )

    return cfg, actions


# ────────────────────────────────────────────────────────────
# Iteration driver
# ────────────────────────────────────────────────────────────
def _log_mtime() -> float:
    return LOG_PATH.stat().st_mtime if LOG_PATH.exists() else 0.0


def _wait_for_completion(prev_mtime: float, timeout_s: int) -> bool:
    """Poll run_cortex.log until COMPLETE_MARKER or ERROR_MARKER appears.

    These markers are the canonical end-of-run signals from
    run_cortex.py — `"run_cortex complete."` is log()'d as the last
    line of the normal try-block and the finally cleanup, and `ERROR:`
    is log()'d in the except path. Polling for them is deterministic.

    Isaac Sim's TCP executor returns ~0.1s after queuing the script
    (not when the script finishes), so we cannot rely on
    run_in_isaac.py's exit code as a completion signal.

    No "log quiet" fallback — any heuristic that infers completion
    from log inactivity will fire DURING long phases (RMPflow stuck
    in at_pick for 40+ ticks, multi-second nav segments) and parse a
    partial log. The 180s timeout is the only crash-detection
    backstop.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if LOG_PATH.exists() and LOG_PATH.stat().st_mtime > prev_mtime:
            try:
                content = LOG_PATH.read_text()
            except Exception:
                content = ""
            if COMPLETE_MARKER in content or ERROR_MARKER in content:
                return True
        time.sleep(0.5)
    return False


def run_cortex_subprocess(timeout_s: int = 300) -> tuple[bool, str]:
    """Send run_cortex.py to Isaac Sim and wait for the log's completion
    marker. Returns (ok, stdout_tail)."""
    prev_mtime = _log_mtime()
    proc = subprocess.run(
        [sys.executable, str(RUN_IN_ISAAC), SCRIPT],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    out = (proc.stdout + proc.stderr)[-1000:]
    if proc.returncode != 0:
        return False, out
    # Wait for the actual pipeline run to finish (log marker).
    finished = _wait_for_completion(prev_mtime, timeout_s)
    if not finished:
        return False, out + "\n[tune] timed out waiting for log completion marker"
    return True, out


def run_bootstrap(timeout_s: int = 180) -> bool:
    """Send bootstrap.py to Isaac Sim. On the fast-reset path this
    resets cube + AMR + Franka to canonical start poses without
    reloading the hospital stage (~5s), guaranteeing each tuner
    iteration starts from the same world state.

    Without this between-iter reset, prior runs' physics drift
    accumulates: failed pick attempts knock the cube off-target, the
    next iter parks AMR on top of the cube, and the Franka can't reach
    back laterally to grasp it.
    """
    prev_mtime = BOOTSTRAP_LOG.stat().st_mtime if BOOTSTRAP_LOG.exists() else 0.0
    proc = subprocess.run(
        [sys.executable, str(RUN_IN_ISAAC), BOOTSTRAP_SCRIPT],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        print(f"[tune] bootstrap subprocess failed: {(proc.stdout + proc.stderr)[-400:]}")
        return False
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if BOOTSTRAP_LOG.exists() and BOOTSTRAP_LOG.stat().st_mtime > prev_mtime:
            content = BOOTSTRAP_LOG.read_text()
            if BOOTSTRAP_DONE_MARKER in content or BOOTSTRAP_FULL_MARKER in content:
                return True
            if "ERROR:" in content:
                print(f"[tune] bootstrap reported ERROR — see {BOOTSTRAP_LOG}")
                return False
        time.sleep(0.5)
    return False


def archive_iteration(i: int) -> None:
    """Snapshot logs + config so you can diff what changed across iters."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    base = HISTORY_DIR / f"iter_{i:02d}"
    if LOG_PATH.exists():
        shutil.copy(LOG_PATH, str(base) + ".run_cortex.log")
    if STREAM_PATH.exists():
        shutil.copy(STREAM_PATH, str(base) + ".stream.log")
    if CFG_PATH.exists():
        shutil.copy(CFG_PATH, str(base) + ".config.yaml")


def save_cfg(cfg: dict) -> None:
    # YAML dump preserving comments is nontrivial; ruamel would be better
    # but we use stdlib yaml. Existing comments will be lost when this
    # rewrites the file — which is acceptable since the file is in git.
    CFG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute rule actions but don't apply or run.")
    ap.add_argument("--timeout", type=int, default=300,
                    help="Per-iteration pipeline timeout (seconds).")
    ap.add_argument("--no-reset", action="store_true",
                    help="Skip the bootstrap fast-reset between iterations. "
                         "Default is to reset, ensuring each iter starts from "
                         "canonical poses (cube at point_a, AMR at origin, etc.).")
    args = ap.parse_args()

    print(f"[tune] config: {CFG_PATH}")
    print(f"[tune] script: {SCRIPT}")
    print(f"[tune] history: {HISTORY_DIR}")

    if args.dry_run:
        m = parse_outcome()
        print(f"[tune] (dry-run) metrics: {json.dumps(m, indent=2)}")
        cfg = yaml.safe_load(CFG_PATH.read_text())
        _, actions = tune(cfg, m)
        for a in actions:
            print(f"[tune] (dry-run) {a}")
        return 0

    prev_metrics: Optional[dict] = None
    for i in range(1, args.max_iter + 1):
        print(f"\n=== Iteration {i}/{args.max_iter} ===")
        if not args.no_reset:
            t_reset = time.time()
            ok_reset = run_bootstrap()
            print(f"[tune] bootstrap reset: ok={ok_reset} ({time.time()-t_reset:.1f}s)")
            if not ok_reset:
                print("[tune] bootstrap reset failed — bailing")
                return 1
        t0 = time.time()
        ok, out = run_cortex_subprocess(timeout_s=args.timeout)
        dt = time.time() - t0
        print(f"[tune] pipeline returned ok={ok} in {dt:.1f}s")
        if not ok:
            print(f"[tune] run_in_isaac stderr tail:\n{out}")

        m = parse_outcome()
        print(f"[tune] metrics: outcome={m['outcome']} "
              f"({m['n_success']}/{m['n_total']} eps) "
              f"first_fail_ep={m['first_fail_ep']} "
              f"first_fail_outcome={m['first_fail_outcome']} "
              f"ee_plateau_z={m['ee_plateau_z']} "
              f"park_dist={m['park_dist']} "
              f"phase_timed_out={m['phase_timed_out']} "
              f"place_err_xy={m['place_err_xy']}")
        for e in m["episodes"]:
            print(f"[tune]   ep={e['ep']} {e['outcome']:<22} "
                  f"err={e['err']:.3f} ticks={e['ticks']} picks={e['picks']}")

        archive_iteration(i)

        if m["outcome"] == "success":
            print(f"[tune] ✓ SUCCESS at iter {i}")
            return 0

        cfg = yaml.safe_load(CFG_PATH.read_text())
        cfg, actions = tune(cfg, m, prev_metrics)
        # Filter out diagnostic-only actions (DIAG: prefix) from the
        # "real action" check — diagnostics don't constitute progress.
        real_actions = [a for a in actions if not a.startswith("DIAG:")]
        for a in actions:
            print(f"[tune] {a}")
        if not real_actions or real_actions == ["no-op (already success)"]:
            print(f"[tune] ✗ no rule fired — bailing (outcome={m['outcome']})")
            return 1

        save_cfg(cfg)
        print(f"[tune] config updated, retrying...")
        prev_metrics = m

    print(f"[tune] ✗ max iterations reached without success")
    return 1


if __name__ == "__main__":
    sys.exit(main())
