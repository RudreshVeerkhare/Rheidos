"""Standard resource key schema for Houdini integration."""

from __future__ import annotations

GEO_P = "geo.P"
GEO_TRIANGLES = "geo.triangles"

SIM_TIME = "sim.time"
SIM_DT = "sim.dt"
SIM_FRAME = "sim.frame"
SIM_SUBSTEP = "sim.substep"


def geo_P(index: int = 0) -> str:
    if index == 0:
        return GEO_P
    return f"geo.input{index}.P"


def geo_triangles(index: int = 0) -> str:
    if index == 0:
        return GEO_TRIANGLES
    return f"geo.input{index}.triangles"


def point_attrib(name: str, *, index: int = 0) -> str:
    if index == 0:
        return f"geo.point_attrib.{name}"
    return f"geo.input{index}.point_attrib.{name}"


def prim_attrib(name: str, *, index: int = 0) -> str:
    if index == 0:
        return f"geo.prim_attrib.{name}"
    return f"geo.input{index}.prim_attrib.{name}"


def point_group_mask(name: str, *, index: int = 0) -> str:
    if index == 0:
        return f"geo.point_group.{name}.mask"
    return f"geo.input{index}.point_group.{name}.mask"


def point_group_indices(name: str, *, index: int = 0) -> str:
    if index == 0:
        return f"geo.point_group.{name}.indices"
    return f"geo.input{index}.point_group.{name}.indices"


__all__ = [
    "GEO_P",
    "GEO_TRIANGLES",
    "geo_P",
    "geo_triangles",
    "SIM_TIME",
    "SIM_DT",
    "SIM_FRAME",
    "SIM_SUBSTEP",
    "point_attrib",
    "prim_attrib",
    "point_group_mask",
    "point_group_indices",
]
