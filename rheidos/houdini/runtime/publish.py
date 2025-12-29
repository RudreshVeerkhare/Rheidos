"""Publishing helpers for standard Houdini compute resources."""

from __future__ import annotations

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


def publish_geometry_minimal(ctx: CookContext) -> None:
    points = ctx.P()
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"geo.P expected (N, 3), got {points.shape}")
    triangles = ctx.triangles()
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"geo.triangles expected (M, 3), got {triangles.shape}")
    ctx.publish(GEO_P, points)
    ctx.publish(GEO_TRIANGLES, triangles)


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
