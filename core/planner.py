"""Global path planning.

Planner (ABC)
├── StraightLinePlanner — trivial goal-as-waypoint (baseline / sanity test)
└── RRTStarPlanner      — sampling-based, asymptotically optimal

Both depend on an OccupancyGrid for collision checking. The grid is built
once per scene by raycasting the static USD geometry.
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod

import numpy as np


# ────────────────────────────────────────────────────────────────
# Occupancy grid — fast 2D collision oracle built from the scene
# ────────────────────────────────────────────────────────────────
class OccupancyGrid:
    """2D boolean grid: True = occupied (wall / furniture), False = free.

    Built once per scene by raycasting down from above — cells whose first
    hit is above `floor_z + obstacle_threshold` count as occupied.
    """

    def __init__(self, bounds, cell_size: float):
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.cell_size = cell_size
        self.nx = max(1, int(math.ceil((self.x_max - self.x_min) / cell_size)))
        self.ny = max(1, int(math.ceil((self.y_max - self.y_min) / cell_size)))
        self.grid = np.zeros((self.nx, self.ny), dtype=bool)
        self._built = False

    def world_to_cell(self, xy) -> tuple[int, int]:
        ix = int((xy[0] - self.x_min) / self.cell_size)
        iy = int((xy[1] - self.y_min) / self.cell_size)
        return ix, iy

    def in_bounds(self, xy) -> bool:
        return (self.x_min <= xy[0] < self.x_max and
                self.y_min <= xy[1] < self.y_max)

    def is_free(self, xy) -> bool:
        if not self.in_bounds(xy):
            return False
        ix, iy = self.world_to_cell(xy)
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            return False
        return not self.grid[ix, iy]

    def segment_is_free(self, a, b, step=None) -> bool:
        """Check if the straight line between two XY points is collision-free."""
        if step is None:
            step = self.cell_size * 0.5
        dist = float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
        n = max(2, int(math.ceil(dist / step)))
        for i in range(n + 1):
            t = i / n
            p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if not self.is_free(p):
                return False
        return True

    def build_from_raycasts(
        self,
        raycast_z: float = 5.0,
        obstacle_threshold: float = 0.15,
        max_distance: float = 20.0,
    ):
        """Raycast downward from high above each cell. Mark cells whose first
        hit is TALLER than `obstacle_threshold` as occupied.

        Why high origin: raycasting from inside a wall gives unreliable results.
        Starting from z=5m (well above any indoor obstacle) guarantees the ray
        only enters solid geometry from outside, so the first-hit point is the
        top surface of whatever is below.

        Cells with no hit at all are treated as FREE (open sky / scene edge).
        """
        from omni.physx import get_physx_scene_query_interface
        query = get_physx_scene_query_interface()

        occupied = 0
        no_hit = 0
        for ix in range(self.nx):
            for iy in range(self.ny):
                cx = self.x_min + (ix + 0.5) * self.cell_size
                cy = self.y_min + (iy + 0.5) * self.cell_size
                origin = (float(cx), float(cy), float(raycast_z))
                direction = (0.0, 0.0, -1.0)
                hit = query.raycast_closest(origin, direction, float(max_distance))
                if hit and hit.get("hit"):
                    hit_z = hit["position"][2] if "position" in hit else 0.0
                    if hit_z > obstacle_threshold:
                        self.grid[ix, iy] = True
                        occupied += 1
                else:
                    no_hit += 1
        self._built = True
        total = self.nx * self.ny
        print(f"[OccupancyGrid] built {self.nx}x{self.ny} "
              f"({occupied}/{total} occupied = {100*occupied/total:.1f}%, "
              f"{no_hit} cells had no hit — treated as free)")

    def build_empty(self):
        """Fallback: mark everything free. Useful for debugging the planner
        without a scene, or when the scene has no static obstacles."""
        self.grid[:] = False
        self._built = True
        print(f"[OccupancyGrid] built empty {self.nx}x{self.ny} (all free)")


# ────────────────────────────────────────────────────────────────
# Planner abstract base class
# ────────────────────────────────────────────────────────────────
class Planner(ABC):
    @abstractmethod
    def build(self, world) -> None:
        """Build scene-dependent data (occupancy, etc.). Call once per scene."""

    @abstractmethod
    def plan(self, start_xy, goal_xy) -> list[np.ndarray]:
        """Return list of XY waypoints from start to goal. [] = no path."""

    @abstractmethod
    def is_valid(self, xy) -> bool:
        """Is this point reachable / not inside an obstacle?"""


# ────────────────────────────────────────────────────────────────
# Trivial baseline — goal as single waypoint
# ────────────────────────────────────────────────────────────────
class StraightLinePlanner(Planner):
    """Returns [goal] — useful for open spaces or sanity testing the nav layer."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.grid: OccupancyGrid | None = None

    def build(self, world) -> None:
        occ = self.cfg.get("occupancy", {})
        bounds = occ.get("bounds", [-10, 10, -10, 10])
        cell_size = float(occ.get("cell_size", 0.1))
        self.grid = OccupancyGrid(bounds, cell_size)
        raycast_z = float(occ.get("raycast_z", 5.0))
        obstacle_threshold = float(occ.get("obstacle_threshold", 0.15))
        try:
            self.grid.build_from_raycasts(
                raycast_z=raycast_z, obstacle_threshold=obstacle_threshold
            )
        except Exception as e:
            print(f"[StraightLinePlanner] raycast build failed ({e}); using empty grid")
            self.grid.build_empty()

    def plan(self, start_xy, goal_xy) -> list[np.ndarray]:
        return [np.array([goal_xy[0], goal_xy[1]], dtype=float)]

    def is_valid(self, xy) -> bool:
        return self.grid is None or self.grid.is_free(xy)


# ────────────────────────────────────────────────────────────────
# RRT* — sampling-based, asymptotically optimal
# ────────────────────────────────────────────────────────────────
class _Node:
    __slots__ = ("xy", "parent", "cost")

    def __init__(self, xy, parent=None, cost=0.0):
        self.xy = np.asarray(xy, dtype=float)
        self.parent = parent
        self.cost = cost


class RRTStarPlanner(Planner):
    """RRT* in continuous 2D.

    Pipeline per plan() call:
      1. Sample — uniform in world bounds, with `goal_bias` prob. of sampling goal
      2. Nearest — find closest tree node to sample
      3. Steer  — move from nearest toward sample by `step_size`
      4. Collision check — reject if edge crosses occupied cells
      5. Connect — add as child of nearest
      6. Rewire — for near neighbors in `rewire_radius`, reparent if going
                  through the new node gives lower cost
    Terminates early when a node reaches within `goal_tolerance` of goal.
    Optionally applies shortcut smoothing as a post-process.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        rrt = cfg.get("rrt_star", {})
        self.max_iter       = int(rrt.get("max_iter", 3000))
        self.step_size      = float(rrt.get("step_size", 0.5))
        self.goal_bias      = float(rrt.get("goal_bias", 0.1))
        self.rewire_radius  = float(rrt.get("rewire_radius", 1.2))
        self.goal_tolerance = float(rrt.get("goal_tolerance", 0.2))
        self.smooth         = bool(rrt.get("smooth", True))

        self.grid: OccupancyGrid | None = None
        self._rng = random.Random(rrt.get("seed"))

    # ── lifecycle ────────────────────────────────────────────
    def build(self, world) -> None:
        occ = self.cfg.get("occupancy", {})
        bounds = occ.get("bounds", [-10, 10, -10, 10])
        cell_size = float(occ.get("cell_size", 0.1))
        raycast_z = float(occ.get("raycast_z", 5.0))
        obstacle_threshold = float(occ.get("obstacle_threshold", 0.15))
        self.grid = OccupancyGrid(bounds, cell_size)
        try:
            self.grid.build_from_raycasts(
                raycast_z=raycast_z, obstacle_threshold=obstacle_threshold
            )
        except Exception as e:
            print(f"[RRTStarPlanner] raycast build failed ({e}); using empty grid")
            self.grid.build_empty()

    def is_valid(self, xy) -> bool:
        return self.grid is not None and self.grid.is_free(xy)

    # ── main planning ────────────────────────────────────────
    def plan(self, start_xy, goal_xy) -> list[np.ndarray]:
        if self.grid is None:
            raise RuntimeError("RRTStarPlanner.plan() called before build()")

        start = np.asarray(start_xy, dtype=float)[:2]
        goal = np.asarray(goal_xy, dtype=float)[:2]

        if not self.grid.is_free(tuple(start)):
            print(f"[RRT*] start {start.tolist()} is in an obstacle")
            return []
        if not self.grid.is_free(tuple(goal)):
            print(f"[RRT*] goal {goal.tolist()} is in an obstacle")
            return []

        # Early out — straight line is clear
        if self.grid.segment_is_free(tuple(start), tuple(goal)):
            return [goal]

        root = _Node(start, parent=None, cost=0.0)
        nodes: list[_Node] = [root]
        best_goal_node: _Node | None = None

        x_min, x_max, y_min, y_max = (self.grid.x_min, self.grid.x_max,
                                      self.grid.y_min, self.grid.y_max)

        for _ in range(self.max_iter):
            # 1. sample
            if self._rng.random() < self.goal_bias:
                sample = goal
            else:
                sample = np.array([self._rng.uniform(x_min, x_max),
                                   self._rng.uniform(y_min, y_max)])

            # 2. nearest
            nearest = min(nodes, key=lambda n: float(np.linalg.norm(n.xy - sample)))

            # 3. steer
            direction = sample - nearest.xy
            dist = float(np.linalg.norm(direction))
            if dist < 1e-6:
                continue
            step = min(self.step_size, dist)
            new_xy = nearest.xy + direction * (step / dist)

            # 4. collision check
            if not self.grid.segment_is_free(tuple(nearest.xy), tuple(new_xy)):
                continue

            # 5. connect — pick best parent among near neighbors
            near = [n for n in nodes
                    if float(np.linalg.norm(n.xy - new_xy)) <= self.rewire_radius]
            best_parent = nearest
            best_cost = nearest.cost + float(np.linalg.norm(new_xy - nearest.xy))
            for n in near:
                if not self.grid.segment_is_free(tuple(n.xy), tuple(new_xy)):
                    continue
                c = n.cost + float(np.linalg.norm(new_xy - n.xy))
                if c < best_cost:
                    best_parent = n
                    best_cost = c

            new_node = _Node(new_xy, parent=best_parent, cost=best_cost)
            nodes.append(new_node)

            # 6. rewire
            for n in near:
                if n is best_parent:
                    continue
                if not self.grid.segment_is_free(tuple(new_node.xy), tuple(n.xy)):
                    continue
                c = new_node.cost + float(np.linalg.norm(n.xy - new_node.xy))
                if c < n.cost:
                    n.parent = new_node
                    n.cost = c

            # Check goal
            if float(np.linalg.norm(new_node.xy - goal)) <= self.goal_tolerance:
                if best_goal_node is None or new_node.cost < best_goal_node.cost:
                    best_goal_node = new_node

        if best_goal_node is None:
            print(f"[RRT*] no path found after {self.max_iter} iterations")
            return []

        # Trace path from goal to root, then reverse
        path = []
        node: _Node | None = best_goal_node
        while node is not None:
            path.append(node.xy)
            node = node.parent
        path.reverse()

        # Append the exact goal if the last node wasn't it
        if float(np.linalg.norm(path[-1] - goal)) > 1e-3:
            if self.grid.segment_is_free(tuple(path[-1]), tuple(goal)):
                path.append(goal)

        if self.smooth:
            path = self._shortcut_smooth(path)

        print(f"[RRT*] path: {len(path)} waypoints, "
              f"cost={best_goal_node.cost:.2f}, tree_size={len(nodes)}")
        return path

    # ── post-processing ──────────────────────────────────────
    def _shortcut_smooth(self, path: list[np.ndarray]) -> list[np.ndarray]:
        """Greedy shortcut: repeatedly try to drop intermediate waypoints
        whenever the direct line between two non-adjacent points is clear."""
        if len(path) <= 2:
            return path
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            # Find the furthest j we can connect to directly
            j = len(path) - 1
            while j > i + 1:
                if self.grid.segment_is_free(tuple(path[i]), tuple(path[j])):
                    break
                j -= 1
            out.append(path[j])
            i = j
        return out
