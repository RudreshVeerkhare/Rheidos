"""Geometry adapters for Houdini."""

from .adapter import GeometryIO
from .schema import (
    AttribDesc,
    GeometrySchema,
    OWNER_DETAIL,
    OWNER_POINT,
    OWNER_PRIM,
    OWNER_VERTEX,
    OWNERS,
)

__all__ = [
    "AttribDesc",
    "GeometryIO",
    "GeometrySchema",
    "OWNER_DETAIL",
    "OWNER_POINT",
    "OWNER_PRIM",
    "OWNER_VERTEX",
    "OWNERS",
]
