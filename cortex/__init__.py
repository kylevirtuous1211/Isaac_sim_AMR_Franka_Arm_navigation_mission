"""Cortex-style decider network for the mobile-manip pipeline.

See `apps/run_cortex.py` for the entry point. This package wraps the
existing `core.navigator` + `core.manipulator` APIs in a Cortex
`DfNetwork` so the pipeline becomes reactive to the cube's actual
world position (drag the cube → robot replans), and so pick failures
are retried up to MAX_PICK_ATTEMPTS before stopping.
"""
