"""Episode randomization for Section 4 (Domain Randomization, 20%).

Samples (start, goal, cube, place) from the config-provided bounds,
enforcing:
  - Each XY lies in free space (Planner.is_valid)
  - Minimum pairwise separation between sampled points
Falls back to defaults after N failed attempts.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .planner import Planner


@dataclass
class Episode:
    start_xy: np.ndarray
    goal_xy: np.ndarray
    cube_xyz: np.ndarray
    place_xyz: np.ndarray


class Randomizer:
    def __init__(self, planner: Planner, cfg: dict):
        self.planner = planner
        self.cfg = cfg
        self._rng = random.Random(cfg.get("seed"))
        self._max_attempts = int(cfg.get("max_attempts", 100))
        self._min_sep = float(cfg.get("min_separation", 0.5))
        self._cube_z = float(cfg.get("cube_z", 0.025))

    def _sample_xy(self, bounds) -> np.ndarray:
        x_min, x_max, y_min, y_max = bounds
        return np.array([
            self._rng.uniform(x_min, x_max),
            self._rng.uniform(y_min, y_max),
        ])

    def _sample_valid(self, bounds) -> np.ndarray | None:
        for _ in range(self._max_attempts):
            p = self._sample_xy(bounds)
            if self.planner.is_valid(tuple(p)):
                return p
        return None

    def sample_episode(self) -> Episode:
        start = self._sample_valid(self.cfg["start_bounds"])
        goal = self._sample_valid(self.cfg["goal_bounds"])
        cube = self._sample_valid(self.cfg["cube_bounds"])
        place = self._sample_valid(self.cfg["place_bounds"])

        # Fallback defaults if sampling failed repeatedly
        if start is None: start = np.array([0.0, 0.0])
        if goal  is None: goal  = np.array([2.0, 0.0])
        if cube  is None: cube  = np.array([1.0, 0.0])
        if place is None: place = np.array([2.5, 0.5])

        # Separation check — nudge if too close
        points = {"start": start, "goal": goal, "cube": cube, "place": place}
        for a_name, a in points.items():
            for b_name, b in points.items():
                if a_name >= b_name:
                    continue
                if float(np.linalg.norm(a - b)) < self._min_sep:
                    print(f"[Randomizer] {a_name}/{b_name} too close "
                          f"({np.linalg.norm(a - b):.2f} < {self._min_sep:.2f})")

        return Episode(
            start_xy=start,
            goal_xy=goal,
            cube_xyz=np.array([cube[0], cube[1], self._cube_z]),
            place_xyz=np.array([place[0], place[1], self._cube_z]),
        )
