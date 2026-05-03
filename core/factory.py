"""Wire concrete classes to config `type` strings.

Example config:
    planner:      {type: rrt_star, rrt_star: {...}, occupancy: {...}}
    navigator:    {type: waypoint, robot: {...}}
    manipulator:  {type: franka_rmpflow, ...}

Usage:
    planner = build_planner(cfg["planner"])
    nav     = build_navigator(cfg["navigator"], planner, world)
    arm     = build_manipulator(cfg["manipulator"], world)
"""
from __future__ import annotations

from typing import Type

from .planner import Planner, RRTStarPlanner, StraightLinePlanner
from .navigator import Navigator, WaypointNavigator
from .manipulator import (
    Manipulator,
    FrankaRMPflowManipulator,
)


PLANNERS: dict[str, Type[Planner]] = {
    "rrt_star": RRTStarPlanner,
    "straight_line": StraightLinePlanner,
}

NAVIGATORS: dict[str, Type[Navigator]] = {
    "waypoint": WaypointNavigator,
}

MANIPULATORS: dict[str, Type[Manipulator]] = {
    "franka_rmpflow": FrankaRMPflowManipulator,
}


def _lookup(registry: dict, name: str, kind: str):
    if name not in registry:
        raise KeyError(
            f"Unknown {kind} type '{name}'. "
            f"Available: {sorted(registry.keys())}"
        )
    return registry[name]


def build_planner(cfg: dict) -> Planner:
    cls = _lookup(PLANNERS, cfg["type"], "planner")
    return cls(cfg)


def build_navigator(cfg: dict, planner: Planner, world) -> Navigator:
    cls = _lookup(NAVIGATORS, cfg["type"], "navigator")
    nav = cls()
    nav.setup(world, cfg, planner)
    return nav


def build_manipulator(cfg: dict, world) -> Manipulator:
    cls = _lookup(MANIPULATORS, cfg["type"], "manipulator")
    arm = cls()
    arm.setup(world, cfg)
    return arm
