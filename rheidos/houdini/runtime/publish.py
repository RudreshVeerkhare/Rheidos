"""Publishing helpers for standard Houdini compute resources."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from ..geo.schema import OWNER_POINT, OWNER_PRIM
from .cook_context import CookContext
from .resource_keys import (
    GEO_P,
    GEO_TRIANGLES,
    point_attrib,
    point_group_indices,
    point_group_mask,
    prim_attrib,
)


def _intrinsic_count(geo: Any, name: str, fallback: int) -> int:
    getter = getattr(geo, "intrinsicValue", None)
    if callable(getter):
        try:
            return int(getter(name))
        except Exception:
            return int(fallback)
    return int(fallback)


def _geometry_change_id(geo: Any) -> Optional[Any]:
    for attr in ("dataId", "geometryHash", "hash"):
        getter = getattr(geo, attr, None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                continue
    return None


def _quick_topology_key(geo: Any) -> Optional[Tuple[int, int, Any]]:
    change_id = _geometry_change_id(geo)
    if change_id is None:
        return None
    point_count = _intrinsic_count(geo, "pointcount", len(geo.points()))
    prim_count = _intrinsic_count(geo, "primitivecount", len(geo.prims()))
    return (point_count, prim_count, change_id)


def _triangle_checksum(triangles: np.ndarray) -> int:
    if triangles.size == 0:
        return 0
    total = np.sum(triangles, dtype=np.int64)
    return int(total & 0xFFFFFFFF)


def _topology_signature(point_count: int, triangles: np.ndarray) -> Tuple[int, int, int]:
    return (int(point_count), int(triangles.shape[0]), _triangle_checksum(triangles))


def _read_triangles_cached(ctx: CookContext) -> np.ndarray:
    session = ctx.session
    quick_key = _quick_topology_key(ctx.geo_in)
    if (
        quick_key is not None
        and session.last_topology_key == quick_key
        and session.last_triangles is not None
    ):
        return session.last_triangles

    triangles = ctx.triangles()
    point_count = _intrinsic_count(ctx.geo_in, "pointcount", len(ctx.geo_in.points()))
    session.last_triangles = triangles
    session.last_topology_sig = _topology_signature(point_count, triangles)
    session.last_topology_key = quick_key
    return triangles


def publish_geometry_minimal(ctx: CookContext) -> None:
    points = ctx.P()
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"geo.P expected (N, 3), got {points.shape}")
    triangles = _read_triangles_cached(ctx)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"geo.triangles expected (M, 3), got {triangles.shape}")
    ctx.publish_many({GEO_P: points, GEO_TRIANGLES: triangles})


def publish_group(ctx: CookContext, group_name: str, *, as_mask: bool = True) -> None:
    values = ctx.read_group(OWNER_POINT, group_name, as_mask=as_mask)
    if as_mask:
        if values.ndim != 1 or values.dtype != np.bool_:
            raise ValueError(f"Group mask expected 1D bool array, got {values.dtype} {values.shape}")
        key = point_group_mask(group_name)
    else:
        if values.ndim != 1:
            raise ValueError(f"Group indices expected 1D array, got {values.shape}")
        key = point_group_indices(group_name)
    ctx.publish(key, values)


def publish_point_attrib(ctx: CookContext, name: str) -> None:
    values = ctx.read(OWNER_POINT, name)
    _validate_count(ctx, OWNER_POINT, values, name)
    ctx.publish(point_attrib(name), values)


def publish_prim_attrib(ctx: CookContext, name: str) -> None:
    values = ctx.read(OWNER_PRIM, name)
    _validate_count(ctx, OWNER_PRIM, values, name)
    ctx.publish(prim_attrib(name), values)


def _validate_count(ctx: CookContext, owner: str, values: np.ndarray, name: str) -> None:
    if owner == OWNER_POINT:
        count = len(ctx.geo_in.points())
    elif owner == OWNER_PRIM:
        count = len(ctx.geo_in.prims())
    else:
        raise ValueError(f"Unsupported owner '{owner}'")

    if values.shape[0] != count:
        raise ValueError(
            f"{owner} attrib '{name}' expected {count} elements, got {values.shape[0]}"
        )
