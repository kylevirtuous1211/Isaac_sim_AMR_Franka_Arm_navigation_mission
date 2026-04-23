"""Pluggable components for the midterm project.

Layers:
- planner.py      — global path planning (RRT*, straight-line)
- navigator.py    — AMR base execution + reactive avoidance
- manipulator.py  — arm control with obstacle avoidance
- orchestrator.py — composes Navigator + Manipulator for A→pick→B→place
- randomizer.py   — samples valid episodes for domain randomization
- factory.py      — registries + build_from_config(cfg, world)
"""
