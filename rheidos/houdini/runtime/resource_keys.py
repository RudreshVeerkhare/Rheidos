"""Standard resource key schema for Houdini integration."""

from __future__ import annotations

GEO_P = "geo.P"
GEO_TRIANGLES = "geo.triangles"

SIM_TIME = "sim.time"
SIM_DT = "sim.dt"
SIM_FRAME = "sim.frame"
SIM_SUBSTEP = "sim.substep"


def point_attrib(name: str) -> str:
    return f"geo.point_attrib.{name}"


def prim_attrib(name: str) -> str:
    return f"geo.prim_attrib.{name}"


def point_group_mask(name: str) -> str:
    return f"geo.point_group.{name}.mask"


def point_group_indices(name: str) -> str:
    return f"geo.point_group.{name}.indices"


__all__ = [
    "GEO_P",
    "GEO_TRIANGLES",
    "SIM_TIME",
    "SIM_DT",
    "SIM_FRAME",
    "SIM_SUBSTEP",
    "point_attrib",
    "prim_attrib",
    "point_group_mask",
    "point_group_indices",
]
