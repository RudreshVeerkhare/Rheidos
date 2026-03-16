"""point_vortex_p2 app: P2 FEEC point-vortex simulation pipeline."""

from __future__ import annotations

import numpy as np

from rheidos.houdini.runtime.cook_context import CookContext
from .modules.midpoint_advection import (
    advect_single_field_batch,
    advect_stage_b_from_midpoint_batch,
)


from .modules import (
    FaceGeometryModule,
    MidpointAdvectionModule,
    P2PoissonModule,
    P2ScalarSpaceModule,
    P2VelocityModule,
    PointVortexModule,
    SurfaceMeshModule,
)

ATTR_FACE = "p2_face"
ATTR_BARY = "p2_bary"
ATTR_GAMMA = "p2_gamma"

ATTR_STREAM = "p2_stream"
ATTR_VEL_FACE = "p2_vel_face"
ATTR_VEL_CORNER = "p2_vel_corner"


class P2Modules:
    def __init__(self, ctx: CookContext):
        world = ctx.world()
        self.mesh = world.require(SurfaceMeshModule)
        self.vort = world.require(PointVortexModule)
        self.space = world.require(P2ScalarSpaceModule)
        self.geom = world.require(FaceGeometryModule)
        self.poisson = world.require(P2PoissonModule)
        self.velocity = world.require(P2VelocityModule)
        self.midpoint = world.require(MidpointAdvectionModule)


def _require_point_attr(io, name: str, *, components: int | None = None) -> np.ndarray:
    try:
        return np.asarray(io.read_point(name, components=components))
    except Exception as exc:
        raise RuntimeError(
            f"Missing required attribute '{name}' for point_vortex_p2 input."
        ) from exc


def _load_mesh(mods: P2Modules, points: np.ndarray, triangles: np.ndarray) -> None:
    mods.mesh.set_mesh(points, triangles)
    boundary_edges = int(mods.mesh.boundary_edge_count.get())
    if boundary_edges > 0:
        raise RuntimeError(
            f"point_vortex_p2 requires closed meshes; found {boundary_edges} boundary edges"
        )


def _sequence_count(obj: object, attr: str) -> int:
    getter = getattr(obj, attr, None)
    if not callable(getter):
        return 0
    try:
        return int(len(getter()))
    except Exception:
        return 0


def _intrinsic_count(geo: object, name: str, fallback: int) -> int:
    getter = getattr(geo, "intrinsicValue", None)
    if callable(getter):
        try:
            return int(getter(name))
        except Exception:
            pass
    return int(fallback)


def _geometry_change_id(geo: object) -> object | None:
    for attr in ("dataId", "geometryHash", "hash"):
        getter = getattr(geo, attr, None)
        if not callable(getter):
            continue
        try:
            return getter()
        except Exception:
            continue
    return None


def _mesh_input_key(mesh_io: object) -> tuple[int, int, object] | None:
    geo = getattr(mesh_io, "geo_in", None)
    if geo is None:
        return None
    change_id = _geometry_change_id(geo)
    if change_id is None:
        return None
    point_count = _intrinsic_count(geo, "pointcount", _sequence_count(geo, "points"))
    prim_count = _intrinsic_count(geo, "primitivecount", _sequence_count(geo, "prims"))
    return (point_count, prim_count, change_id)


def _ensure_mesh_current(ctx: CookContext, mods: P2Modules, mesh_io: object) -> bool:
    session = ctx.session
    key = _mesh_input_key(mesh_io)
    if key is not None and getattr(session, "_p2_mesh_input_key", None) == key:
        return False

    points = np.asarray(mesh_io.read_point("P", components=3), dtype=np.float64)
    triangles = np.asarray(mesh_io.read_prims(arity=3), dtype=np.int32)
    _load_mesh(mods, points, triangles)
    setattr(session, "_p2_mesh_input_key", key)
    return True


def _resolve_step_dt(ctx: CookContext) -> float:
    session = ctx.session
    cur_time = float(getattr(ctx, "time", 0.0) or 0.0)
    last_time = getattr(session, "_p2_last_time", None)

    dt = 0.0
    if last_time is not None:
        try:
            dt = float(cur_time - float(last_time))
        except Exception:
            dt = 0.0

    if not np.isfinite(dt) or dt <= 0.0:
        dt = float(getattr(ctx, "dt", 0.0) or 0.0)
    if not np.isfinite(dt) or dt <= 0.0:
        dt = 0.01

    setattr(session, "_p2_last_time", cur_time)
    return float(dt)


def _load_vortices(mods: P2Modules, face_ids: np.ndarray, bary: np.ndarray, gamma: np.ndarray) -> None:
    mods.vort.set_state(face_ids, bary, gamma)


def _write_detail_scalar(ctx: CookContext, name: str, value: float) -> None:
    ctx.write_detail(name, np.array([value], dtype=np.float32), create=True)


def _publish_diagnostics(
    ctx: CookContext,
    mods: P2Modules,
    bary: np.ndarray,
    *,
    include_advection: bool,
) -> None:
    residual_l2 = float(mods.poisson.residual_l2.get())
    rhs_circ = float(mods.poisson.rhs_circulation.get())
    k_ones_inf = float(mods.poisson.k_ones_inf.get())
    if include_advection:
        hops_total = float(mods.midpoint.hops_total.get())
        hops_max = float(mods.midpoint.hops_max.get())
    else:
        hops_total = 0.0
        hops_max = 0.0

    _write_detail_scalar(ctx, "p2_diag_residual_l2", residual_l2)
    _write_detail_scalar(ctx, "p2_diag_rhs_circulation", rhs_circ)
    _write_detail_scalar(ctx, "p2_diag_k_ones_inf", k_ones_inf)
    _write_detail_scalar(ctx, "p2_diag_hops_total", hops_total)
    _write_detail_scalar(ctx, "p2_diag_hops_max", hops_max)

    if bary.size > 0:
        bary = np.asarray(bary, dtype=np.float64)
        bary_sum = bary.sum(axis=1)
        _write_detail_scalar(ctx, "p2_diag_bary_min", float(bary.min()))
        _write_detail_scalar(ctx, "p2_diag_bary_max", float(bary.max()))
        _write_detail_scalar(ctx, "p2_diag_bary_sum_min", float(bary_sum.min()))
        _write_detail_scalar(ctx, "p2_diag_bary_sum_max", float(bary_sum.max()))


def solve_fields(mods: P2Modules) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trigger stream solve + velocity build and return outputs."""
    stream_vertex = np.asarray(mods.velocity.stream_vertex.get(), dtype=np.float64)
    vel_face = np.asarray(mods.velocity.vel_face.get(), dtype=np.float64)
    vel_corner = np.asarray(mods.velocity.vel_corner.get(), dtype=np.float64)
    return stream_vertex, vel_face, vel_corner


def setup(ctx: CookContext) -> None:
    mods = P2Modules(ctx)
    mesh_io = ctx.input_io(1)
    if mesh_io is None:
        raise RuntimeError("Solver setup expects triangle mesh on input 1")

    _ensure_mesh_current(ctx, mods, mesh_io)


def step(ctx: CookContext) -> None:
    mods = P2Modules(ctx)

    mesh_io = ctx.input_io(1)
    if mesh_io is None:
        raise RuntimeError("Solver step expects triangle mesh on input 1")

    _ensure_mesh_current(ctx, mods, mesh_io)

    io = ctx.io
    face_ids = _require_point_attr(io, ATTR_FACE).astype(np.int32, copy=False)
    bary = _require_point_attr(io, ATTR_BARY, components=3).astype(np.float64, copy=False)
    gamma = _require_point_attr(io, ATTR_GAMMA).astype(np.float64, copy=False)
    _load_vortices(mods, face_ids, bary, gamma)

    # Solve at start state.
    _, _, vel_corner_start = solve_fields(mods)
    dt = _resolve_step_dt(ctx)

    vertices = np.asarray(mods.mesh.V_pos.get(), dtype=np.float64)
    faces = np.asarray(mods.mesh.F_verts.get(), dtype=np.int32)
    f_adj = np.asarray(mods.mesh.F_adj.get(), dtype=np.int32)

    # Stage A: half-step to midpoint using velocity from start-state solve.
    face_mid, bary_mid, _, hops_a = advect_single_field_batch(
        vertices,
        faces,
        f_adj,
        vel_corner_start,
        face_ids,
        bary,
        0.5 * dt,
    )

    # Midpoint solve: recompute field from midpoint particle state.
    _load_vortices(mods, face_mid, bary_mid, gamma)
    _, _, vel_corner_mid = solve_fields(mods)

    # Stage B: full step from original start using midpoint velocity samples.
    face_out, bary_out, pos_out, hops_b = advect_stage_b_from_midpoint_batch(
        vertices,
        faces,
        f_adj,
        vel_corner_mid,
        face_ids,
        bary,
        face_mid,
        bary_mid,
        dt,
    )

    hops_per_particle = hops_a.astype(np.int64) + hops_b.astype(np.int64)
    hops_total = int(np.sum(hops_per_particle, dtype=np.int64))
    hops_max = int(np.max(hops_per_particle)) if hops_per_particle.size > 0 else 0
    mods.midpoint.hops_total.set(hops_total)
    mods.midpoint.hops_max.set(hops_max)

    ctx.write_point(ATTR_FACE, face_out)
    ctx.write_point(ATTR_BARY, bary_out)
    ctx.write_point("P", pos_out)

    _publish_diagnostics(ctx, mods, bary_out, include_advection=True)


def cook(ctx: CookContext) -> None:
    """Single-cook field solve/visualization path on mesh output."""
    mods = P2Modules(ctx)

    mesh_io = ctx.io
    points = np.asarray(mesh_io.read_point("P", components=3), dtype=np.float64)
    triangles = np.asarray(mesh_io.read_prims(arity=3), dtype=np.int32)
    _load_mesh(mods, points, triangles)

    vort_io = ctx.input_io(1)
    if vort_io is None:
        raise RuntimeError("cook() expects point-vortex input on input 1")

    face_ids = _require_point_attr(vort_io, ATTR_FACE).astype(np.int32, copy=False)
    bary = _require_point_attr(vort_io, ATTR_BARY, components=3).astype(np.float64, copy=False)
    gamma = _require_point_attr(vort_io, ATTR_GAMMA).astype(np.float64, copy=False)
    _load_vortices(mods, face_ids, bary, gamma)

    stream_vertex, vel_face, vel_corner = solve_fields(mods)

    # Mesh visualization outputs.
    ctx.write_point(ATTR_STREAM, stream_vertex.astype(np.float32), create=True)
    ctx.write_prim(ATTR_VEL_FACE, vel_face.astype(np.float32), create=True)
    ctx.write_prim(
        ATTR_VEL_CORNER,
        vel_corner.reshape(vel_corner.shape[0], 9).astype(np.float32),
        create=True,
    )

    # Diagnostics in detail attrs.
    _publish_diagnostics(ctx, mods, bary, include_advection=False)
