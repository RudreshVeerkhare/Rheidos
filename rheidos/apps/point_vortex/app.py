"""Poisson DEC cook script for Houdini nodes."""

from __future__ import annotations

from typing import Optional

import numpy as np

import taichi as ti

from rheidos.houdini.geo import OWNER_POINT, OWNER_PRIM
from rheidos.houdini.runtime import GEO_P, GEO_TRIANGLES

from .modules.surface_mesh import SurfaceMeshModule
from .modules.dec_operator import SurfaceDECModule
from .modules.poisson_solver import PoissonSolverModule
from .modules.point_vortex import PointVortexModule

from rheidos.houdini.runtime.cook_context import CookContext


def _taichi_initialized() -> bool:
    checker = getattr(ti, "is_initialized", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False
    core = getattr(ti, "core", None)
    if core is not None and hasattr(core, "is_initialized"):
        try:
            return bool(core.is_initialized())
        except Exception:
            return False
    return False


def _ensure_taichi_init(session) -> None:
    if session.stats.get("taichi_initialized"):
        return
    if _taichi_initialized():
        session.stats["taichi_initialized"] = True
        return
    ti.init()
    session.stats["taichi_initialized"] = True


def _ensure_vector_field(ref, count: int, *, lanes: int, dtype) -> "ti.Field":

    if ref is None:
        return ti.Vector.field(lanes, dtype=dtype, shape=(count,))

    field = ref.peek()
    if (
        field is None
        or tuple(field.shape) != (count,)
        or getattr(field, "n", lanes) != lanes
    ):
        field = ti.Vector.field(lanes, dtype=dtype, shape=(count,))
        ref.set_buffer(field, bump=False)
    return field


def _ensure_scalar_field(ref, count: int, *, dtype) -> "ti.Field":

    if ref is None:
        return ti.field(dtype=dtype, shape=(count,))

    field = ref.peek()
    if field is None or tuple(field.shape) != (count,):
        field = ti.field(dtype=dtype, shape=(count,))
        ref.set_buffer(field, bump=False)

    return field


def _try_group_mask(ctx, name: str, count: int) -> Optional[np.ndarray]:
    try:
        mask = ctx.read_group(OWNER_POINT, name, as_mask=True)
    except KeyError:
        return None
    if mask.shape[0] != count:
        raise ValueError(f"Group '{name}' expected {count} points, got {mask.shape[0]}")
    return mask


def cook2(ctx: CookContext) -> None:
    _ensure_taichi_init(ctx.session)

    world = ctx.world()


def cook(ctx: CookContext) -> None:
    """Run the Poisson DEC solve and write the solution as a point attribute."""
    _ensure_taichi_init(ctx.session)

    points = ctx.fetch(GEO_P)
    triangles = ctx.fetch(GEO_TRIANGLES)
    if points is None:
        raise RuntimeError("Geometry is not set")

    points = np.asarray(points, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int32)
    nV = int(points.shape[0])
    nF = int(triangles.shape[0])

    if nV == 0:
        return
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"Expected triangles (M, 3), got {triangles.shape}")

    world = ctx.world()
    mesh = world.require(SurfaceMeshModule)
    dec = world.require(SurfaceDECModule)
    poisson = world.require(PoissonSolverModule)
    point_vortex = world.require(PointVortexModule)

    V = _ensure_vector_field(mesh.V_pos, nV, lanes=3, dtype=ti.f32)
    V.from_numpy(points)
    mesh.V_pos.commit()

    F = _ensure_vector_field(mesh.F_verts, nF, lanes=3, dtype=ti.i32)
    F.from_numpy(triangles)
    mesh.F_verts.commit()

    # xV = _ensure_scalar_field(None, nV, dtype=ti.f32)
    # yV = _ensure_scalar_field(None, nV, dtype=ti.f32)
    # dec.apply_laplacian0(xV, yV)

    mask = _ensure_scalar_field(poisson.constraint_mask, nV, dtype=ti.i32)
    value = _ensure_scalar_field(poisson.constraint_value, nV, dtype=ti.f32)

    pos_charges = ctx.read_group(OWNER_POINT, "pos_charges")
    if len(pos_charges) > 0:
        for id in pos_charges:
            mask[id] = 1
            value[id] = 1.0

    neg_charges = ctx.read_group(OWNER_POINT, "neg_charges")
    if len(neg_charges) > 0:
        for id in neg_charges:
            mask[id] = 1
            value[id] = -1.0
    poisson.constraint_mask.bump()
    poisson.constraint_value.bump()

    point_vortices = ctx.read_group(OWNER_PRIM, "point_vortices")
    for face_id in point_vortices:
        point_vortex.add_vortex(face_id, (0.33, 0.33, 0.34))

    # ctx.write(OWNER_PRIM, "bary", )

    u = poisson.u.get().to_numpy().astype(np.float32, copy=False)
    ctx.write(OWNER_POINT, "u", u, create=True)

    F_normals = mesh.F_normal.get()
    face_normal = F_normals.to_numpy().astype(np.float32, copy=False)
    ctx.write(OWNER_PRIM, "face_normal", face_normal, create=True)
