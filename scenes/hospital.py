"""Idempotent loader for Isaac Sim's built-in hospital.usd environment."""
from __future__ import annotations

import omni.kit.app
import omni.usd

from isaacsim.storage.native import get_assets_root_path


def hospital_usd_url() -> str:
    """Full URL of Isaac Sim's built-in hospital scene."""
    return get_assets_root_path() + "/Isaac/Environments/Hospital/hospital.usd"


async def load_hospital(force: bool = False, settle_ticks: int = 60) -> str:
    """Open hospital.usd as the root stage (if not already open).

    Checks the current stage's root layer and skips the reload if it's
    already the hospital — this is what avoids the 60–180s network load
    on every script run.

    Args:
        force: reload even if the hospital is already open.
        settle_ticks: app ticks to wait after opening for geometry to settle.

    Returns:
        The full URL of the hospital USD that is now open.
    """
    target = hospital_usd_url()
    ctx = omni.usd.get_context()

    stage = ctx.get_stage()
    current_root = None
    if stage is not None:
        root_layer = stage.GetRootLayer()
        if root_layer is not None:
            current_root = root_layer.realPath or root_layer.identifier

    if not force and current_root and target.split("/")[-1] in (current_root or ""):
        print(f"[scenes.hospital] already loaded ({current_root}); skipping")
        return target

    print(f"[scenes.hospital] opening {target}")
    await ctx.open_stage_async(target)
    for _ in range(settle_ticks):
        await omni.kit.app.get_app().next_update_async()
    print("[scenes.hospital] stage ready")
    return target
