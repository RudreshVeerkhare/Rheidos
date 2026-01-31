"""
Runtime session cache for Houdini nodes.
"""

from __future__ import annotations

import os


def _resolve_multiprocessing_executable() -> str | None:
    override = os.environ.get("RHEIDOS_MULTIPROC_EXECUTABLE")
    if override:
        return os.path.expandvars(os.path.expanduser(override))

    hfs = os.environ.get("HFS")
    if not hfs:
        return None
    exe = os.path.join(hfs, "bin", "hython")
    if os.name == "nt":
        exe += ".exe"
    if not os.path.isfile(exe):
        return None
    return exe


def _configure_multiprocessing_executable() -> None:
    exe = _resolve_multiprocessing_executable()
    if not exe:
        return
    try:
        import multiprocessing as mp
    except Exception:
        return
    try:
        mp.set_executable(exe)
        return
    except Exception:
        pass
    try:
        import multiprocessing.spawn as mp_spawn
    except Exception:
        return
    try:
        mp_spawn.set_executable(exe)
    except Exception:
        return


_configure_multiprocessing_executable()

from .cook_context import CookContext, build_cook_context
from .driver import run_cook, run_solver
from .publish import publish_geometry_minimal, publish_group, publish_point_attrib, publish_prim_attrib
from .reset_pipeline import reset_and_reload, reset_and_reload_with_ui
from .resource_keys import (
    GEO_P,
    GEO_TRIANGLES,
    geo_P,
    geo_triangles,
    SIM_DT,
    SIM_FRAME,
    SIM_SUBSTEP,
    SIM_TIME,
    point_attrib,
    point_group_indices,
    point_group_mask,
    prim_attrib,
)
from .session import (
    AccessMode,
    ComputeRuntime,
    SessionAccess,
    SessionKey,
    WorldSession,
    get_runtime,
    get_sim_context,
    make_session_key,
    make_session_key_for_path,
    set_sim_context,
)

__all__ = [
    "CookContext",
    "ComputeRuntime",
    "AccessMode",
    "GEO_P",
    "GEO_TRIANGLES",
    "geo_P",
    "geo_triangles",
    "SessionAccess",
    "SIM_DT",
    "SIM_FRAME",
    "SIM_SUBSTEP",
    "SIM_TIME",
    "SessionKey",
    "WorldSession",
    "build_cook_context",
    "get_sim_context",
    "get_runtime",
    "make_session_key",
    "make_session_key_for_path",
    "point_attrib",
    "point_group_indices",
    "point_group_mask",
    "prim_attrib",
    "publish_geometry_minimal",
    "publish_group",
    "publish_point_attrib",
    "publish_prim_attrib",
    "reset_and_reload",
    "reset_and_reload_with_ui",
    "run_cook",
    "run_solver",
    "set_sim_context",
]
