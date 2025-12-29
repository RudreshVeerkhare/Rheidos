"""Geometry schema helpers for Houdini adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

OWNER_POINT = "point"
OWNER_PRIM = "prim"
OWNER_VERTEX = "vertex"
OWNER_DETAIL = "detail"

OWNERS: Tuple[str, ...] = (OWNER_POINT, OWNER_PRIM, OWNER_VERTEX, OWNER_DETAIL)


@dataclass(frozen=True)
class AttribDesc:
    name: str
    owner: str
    storage_type: str
    tuple_size: int


@dataclass(frozen=True)
class GeometrySchema:
    point: Tuple[AttribDesc, ...] = ()
    prim: Tuple[AttribDesc, ...] = ()
    vertex: Tuple[AttribDesc, ...] = ()
    detail: Tuple[AttribDesc, ...] = ()

    def by_owner(self, owner: str) -> Tuple[AttribDesc, ...]:
        if owner == OWNER_POINT:
            return self.point
        if owner == OWNER_PRIM:
            return self.prim
        if owner == OWNER_VERTEX:
            return self.vertex
        if owner == OWNER_DETAIL:
            return self.detail
        raise ValueError(f"Unknown owner '{owner}'")
