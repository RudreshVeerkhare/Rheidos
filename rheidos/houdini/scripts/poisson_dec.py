"""Poisson DEC cook script for Houdini nodes."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import taichi as ti
except Exception as exc:  # pragma: no cover - only runs in Houdini
    raise RuntimeError("Taichi is required for the Poisson DEC cook script") from exc

from rheidos.houdini.geo import OWNER_POINT
from rheidos.houdini.runtime import GEO_P, GEO_TRIANGLES

from rheidos.apps.poisson_dec.compute.mesh import MeshModule
from rheidos.apps.poisson_dec.compute.poisson import PoissonSolverModule


POS_GROUP = "poisson_pos"
NEG_GROUP = "poisson_neg"
CHARGE_ATTR = "poisson_charge"
OUTPUT_ATTR = "poisson_u"


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
    field = ref.peek()
    if field is None or tuple(field.shape) != (count,) or getattr(field, "n", lanes) != lanes:
        field = ti.Vector.field(lanes, dtype=dtype, shape=(count,))
        ref.set_buffer(field, bump=False)
    return field


def _ensure_scalar_field(ref, count: int, *, dtype) -> "ti.Field":
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


def _read_charges(ctx, count: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        values = ctx.read(OWNER_POINT, CHARGE_ATTR)
    except KeyError:
        values = None

    if values is not None:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.shape[0] != count:
            raise ValueError(f"Attribute '{CHARGE_ATTR}' expected {count} points, got {values.shape[0]}")
        mask = (values != 0).astype(np.int32)
        return mask, values

    mask = np.zeros((count,), dtype=np.int32)
    values = np.zeros((count,), dtype=np.float32)
    pos_mask = _try_group_mask(ctx, POS_GROUP, count)
    neg_mask = _try_group_mask(ctx, NEG_GROUP, count)

    if pos_mask is not None:
        values[pos_mask] = 1.0
        mask[pos_mask] = 1
    if neg_mask is not None:
        values[neg_mask] = -1.0
        mask[neg_mask] = 1

    return mask, values


def cook(ctx) -> None:
    """Run the Poisson DEC solve and write the solution as a point attribute."""
    _ensure_taichi_init(ctx.session)

    points = ctx.fetch(GEO_P)
    triangles = ctx.fetch(GEO_TRIANGLES)
    if points is None:
        return

    points = np.asarray(points, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int32)
    nV = int(points.shape[0])
    nF = int(triangles.shape[0])

    if nV == 0:
        return
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"Expected triangles (M, 3), got {triangles.shape}")
    if nF == 0:
        ctx.write(OWNER_POINT, OUTPUT_ATTR, np.zeros((nV,), dtype=np.float32), create=True)
        return

    world = ctx.world()
    mesh = world.require(MeshModule)
    poisson = world.require(PoissonSolverModule)

    V = _ensure_vector_field(mesh.V_pos, nV, lanes=3, dtype=ti.f32)
    V.from_numpy(points)
    mesh.V_pos.commit()

    F = _ensure_vector_field(mesh.F_verts, nF, lanes=3, dtype=ti.i32)
    F.from_numpy(triangles)
    mesh.F_verts.commit()

    mask_np, values_np = _read_charges(ctx, nV)
    mask = _ensure_scalar_field(poisson.constraint_mask, nV, dtype=ti.i32)
    value = _ensure_scalar_field(poisson.constraint_value, nV, dtype=ti.f32)
    mask.from_numpy(mask_np)
    value.from_numpy(values_np)
    poisson.constraint_mask.commit()
    poisson.constraint_value.commit()

    u_field = poisson.u.get()
    u = u_field.to_numpy().astype(np.float32, copy=False)
    ctx.write(OWNER_POINT, OUTPUT_ATTR, u, create=True)
