"""Publishing helpers for standard Houdini compute resources."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from ..geo.schema import OWNER_POINT, OWNER_PRIM
from .cook_context import CookContext
from .resource_keys import (
    GEO_P,
    GEO_TRIANGLES,
    geo_P,
    geo_triangles,
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
    # TODO: Prefer topology-only ids (e.g. geo.topologyDataId/primitiveIntrinsicsDataId) when available.
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


def _read_triangles_cached(ctx: CookContext, *, input_index: Optional[int] = None) -> np.ndarray:
    session = ctx.session
    if input_index is None:
        geo = ctx.geo_in
        quick_key = _quick_topology_key(geo)
        if (
            quick_key is not None
            and session.last_topology_key == quick_key
            and session.last_triangles is not None
        ):
            return session.last_triangles

        # TODO: Add a topology-only key path so we can skip reading triangles when only point positions change.
        triangles = ctx.io.read_prims(arity=3)
        point_count = _intrinsic_count(geo, "pointcount", len(geo.points()))
        session.last_triangles = triangles
        session.last_topology_sig = _topology_signature(point_count, triangles)
        session.last_topology_key = quick_key
        return triangles

    geo = ctx.input_geo(input_index)
    quick_key = _quick_topology_key(geo)
    cached_triangles = session.last_triangles_by_input.get(input_index)
    cached_key = session.last_topology_key_by_input.get(input_index)
    if (
        quick_key is not None
        and cached_key == quick_key
        and cached_triangles is not None
    ):
        return cached_triangles

    # TODO: Add a topology-only key path so we can skip reading triangles when only point positions change.
    io = ctx.input_io(input_index)
    if io is None:
        raise RuntimeError(f"Input geometry {input_index} is not connected.")
    triangles = io.read_prims(arity=3)
    point_count = _intrinsic_count(geo, "pointcount", len(geo.points()))
    session.last_triangles_by_input[input_index] = triangles
    session.last_topology_sig_by_input[input_index] = _topology_signature(point_count, triangles)
    session.last_topology_key_by_input[input_index] = quick_key
    if input_index == 0:
        session.last_triangles = triangles
        session.last_topology_sig = session.last_topology_sig_by_input[input_index]
        session.last_topology_key = quick_key
    return triangles


def publish_geometry_minimal(ctx: CookContext, *, input_index: Optional[int] = None) -> None:
    # TODO: Cache P reads (e.g. via P attrib dataId + pointcount) to avoid per-cook numpy extraction.
    if input_index is None:
        io = ctx.io
        points_key = GEO_P
        triangles_key = GEO_TRIANGLES
    else:
        io = ctx.input_io(input_index)
        if io is None:
            raise RuntimeError(f"Input geometry {input_index} is not connected.")
        points_key = geo_P(input_index)
        triangles_key = geo_triangles(input_index)

    points = io.read(OWNER_POINT, "P", components=3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"geo.P expected (N, 3), got {points.shape}")
    triangles = _read_triangles_cached(ctx, input_index=input_index)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"geo.triangles expected (M, 3), got {triangles.shape}")
    ctx.publish_many({points_key: points, triangles_key: triangles})


def publish_group(
    ctx: CookContext,
    group_name: str,
    *,
    as_mask: bool = True,
    input_index: Optional[int] = None,
) -> None:
    if input_index is None:
        io = ctx.io
        key_index = 0
    else:
        io = ctx.input_io(input_index)
        if io is None:
            raise RuntimeError(f"Input geometry {input_index} is not connected.")
        key_index = input_index
    values = io.read_group(OWNER_POINT, group_name, as_mask=as_mask)
    if as_mask:
        if values.ndim != 1 or values.dtype != np.bool_:
            raise ValueError(f"Group mask expected 1D bool array, got {values.dtype} {values.shape}")
        key = point_group_mask(group_name, index=key_index)
    else:
        if values.ndim != 1:
            raise ValueError(f"Group indices expected 1D array, got {values.shape}")
        key = point_group_indices(group_name, index=key_index)
    ctx.publish(key, values)


def publish_point_attrib(
    ctx: CookContext,
    name: str,
    *,
    input_index: Optional[int] = None,
) -> None:
    if input_index is None:
        io = ctx.io
        key_index = 0
    else:
        io = ctx.input_io(input_index)
        if io is None:
            raise RuntimeError(f"Input geometry {input_index} is not connected.")
        key_index = input_index
    values = io.read(OWNER_POINT, name)
    _validate_count(ctx, OWNER_POINT, values, name, input_index=input_index)
    ctx.publish(point_attrib(name, index=key_index), values)


def publish_prim_attrib(
    ctx: CookContext,
    name: str,
    *,
    input_index: Optional[int] = None,
) -> None:
    if input_index is None:
        io = ctx.io
        key_index = 0
    else:
        io = ctx.input_io(input_index)
        if io is None:
            raise RuntimeError(f"Input geometry {input_index} is not connected.")
        key_index = input_index
    values = io.read(OWNER_PRIM, name)
    _validate_count(ctx, OWNER_PRIM, values, name, input_index=input_index)
    ctx.publish(prim_attrib(name, index=key_index), values)


def _validate_count(
    ctx: CookContext,
    owner: str,
    values: np.ndarray,
    name: str,
    *,
    input_index: Optional[int] = None,
) -> None:
    geo = ctx.geo_in if input_index is None else ctx.input_geo(input_index)
    if owner == OWNER_POINT:
        count = len(geo.points())
    elif owner == OWNER_PRIM:
        count = len(geo.prims())
    else:
        raise ValueError(f"Unsupported owner '{owner}'")

    if values.shape[0] != count:
        raise ValueError(
            f"{owner} attrib '{name}' expected {count} elements, got {values.shape[0]}"
        )
