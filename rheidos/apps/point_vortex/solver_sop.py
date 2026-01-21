# Houdini Solver SOP template for Rheidos apps
#
# Wire it like a normal Solver:
#   input 0: prev frame feedback (from the solver itself)
#   input 1: external/current geometry (optional)
#
# This template:
# - Seeds output from prev-frame geometry (preferred) or current input
# - Builds a solver CookContext (is_solver=True, substep support)
# - Publishes geometry + sim timing keys
# - Calls your app's setup(ctx) once (optional) and step(ctx) every cook

import hou
from types import SimpleNamespace

from rheidos.houdini.debug import (
    consume_break_next_button,
    debug_config_from_node,
    ensure_debug_server,
    maybe_break_now,
    request_break_next,
)
from rheidos.houdini.runtime import (
    build_cook_context,
    get_runtime,
    publish_geometry_minimal,
    CookContext,
)
from rheidos.houdini.runtime import driver as runtime_driver
from rheidos.houdini.runtime.resource_keys import (
    SIM_DT,
    SIM_FRAME,
    SIM_SUBSTEP,
    SIM_TIME,
)

from rheidos.houdini.geo import OWNER_POINT, OWNER_PRIM

from rheidos.apps.point_vortex.modules.surface_mesh import SurfaceMeshModule
from rheidos.apps.point_vortex.modules.point_vortex import PointVortexModule

from rheidos.apps.point_vortex.modules.rk4_advection import RK4AdvectionModule

# from rheidos.apps.point_vortex.modules.rt0_rk4_advection import RK4AdvectionModule
from rheidos.apps.point_vortex.modules.velocity_field import VelocityFieldModule
from rheidos.apps.point_vortex.modules.stream_func import StreamFunctionModule


import taichi as ti
import numpy as np
from time import perf_counter_ns
from rheidos.compute.profiler.core import profiled
from rheidos.compute.profiler.runtime import reset_current_profiler, set_current_profiler

# === IMPORT: change this to your app ===
# You can have only `step(ctx)` if you want; `setup(ctx)` is optional.
# from rheidos.apps.point_vortex.app import setup, step  # <-- your app entrypoints


def _eval_parm_optional(node: hou.Node, name: str):
    parm = node.parm(name)
    if parm is None:
        return None
    try:
        return parm.eval()
    except Exception:
        return None


def _eval_parm_optional_bool(
    node: hou.Node, name: str, default: bool = False
) -> bool:
    value = _eval_parm_optional(node, name)
    if value is None:
        return default
    return bool(value)


def _eval_parm_optional_int(node: hou.Node, name: str, default: int) -> int:
    value = _eval_parm_optional(node, name)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _eval_parm_optional_float(node: hou.Node, name: str, default: float) -> float:
    value = _eval_parm_optional(node, name)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _eval_parm_optional_str(node: hou.Node, name: str, default: str = "") -> str:
    value = _eval_parm_optional(node, name)
    if value is None:
        return default
    return "" if value is None else str(value)


def _profiler_cfg_from_node(node: hou.Node) -> SimpleNamespace:
    return SimpleNamespace(
        profile=_eval_parm_optional_bool(node, "profile", False),
        profile_logdir=_eval_parm_optional_str(node, "profile_logdir") or None,
        profile_export_hz=_eval_parm_optional_float(node, "profile_export_hz", 5.0),
        profile_mode=_eval_parm_optional_str(node, "profile_mode") or None,
        profile_trace_cooks=_eval_parm_optional_int(node, "profile_trace_cooks", 64),
        profile_trace_edges=_eval_parm_optional_int(node, "profile_trace_edges", 20000),
        profile_overhead=_eval_parm_optional_bool(node, "profile_overhead", False),
        profile_taichi=_eval_parm_optional_bool(node, "profile_taichi", True),
        profile_taichi_every=_eval_parm_optional_int(node, "profile_taichi_every", 30),
        profile_taichi_sync=_eval_parm_optional_bool(node, "profile_taichi_sync", True),
        profile_taichi_scoped_once=_eval_parm_optional_bool(
            node, "profile_taichi_scoped_once", False
        ),
    )


def _enter_profiler(node, session, cfg):
    runtime_driver._configure_profiler(session, cfg, node)
    token = set_current_profiler(session.profiler)
    cook_index = session.profiler.next_cook_index()
    sample_every = max(1, session.profiler.cfg.taichi_sample_every_n_cooks)
    is_sample = (
        session.profiler.cfg.enabled
        and session.profiler.cfg.taichi_enabled
        and (cook_index % sample_every == 0)
    )
    session.profiler.set_taichi_sample(is_sample)
    probe = session.taichi_probe if is_sample else None
    return token, probe


def _exit_profiler(session, token) -> None:
    if token is not None:
        reset_current_profiler(token)
    session.profiler.set_taichi_sample(False)


def setup(ctx: CookContext) -> None:

    _ensure_taichi_init(ctx.session)
    ## Read mesh input (index 0) explicitly via the primary IO.
    mesh_io = ctx.input_io(1)
    if mesh_io is None:
        raise RuntimeError(f"Expected triangle mesh on input 1, received {mesh_io}")
    mesh_points = mesh_io.read(OWNER_POINT, "P", components=3)
    mesh_triangles = mesh_io.read_prims(arity=3)
    nV = int(mesh_points.shape[0])
    nF = int(mesh_triangles.shape[0])

    world = ctx.world()
    mesh = world.require(SurfaceMeshModule)
    V = _ensure_vector_field(mesh.V_pos, nV, lanes=3, dtype=ti.f32)
    V.from_numpy(mesh_points.astype(np.float32))
    mesh.V_pos.commit()

    F = _ensure_vector_field(mesh.F_verts, nF, lanes=3, dtype=ti.i32)
    F.from_numpy(mesh_triangles.astype(np.int32))
    mesh.F_verts.commit()
    print("Setup Done!")
    # Add DEC Mesh setup here


@profiled("step", cat="solver")
def step(ctx: CookContext) -> None:
    print(ctx.frame)
    # For now read all point vortices and and their state (CPU -> GPU -> CPU)

    prof = ctx.prof

    with prof.span("io_read_inputs", cat="solver"):
        # Read scatter points input (index 1) via the input IO.
        points_io = ctx.io
        scatter_points = points_io.read(OWNER_POINT, "P", components=3)
        bary = points_io.read(OWNER_POINT, "bary", components=3)
        face_ids = points_io.read(OWNER_POINT, "faceid")
        gammas = points_io.read(OWNER_POINT, "gamma")

    with prof.span("compute_setup", cat="solver"):
        world = ctx.world()
        point_vortices = world.require(PointVortexModule)
        point_vortices.set_frame(int(ctx.frame))
        point_vortices.set_n_vortices(len(scatter_points))
        point_vortices.set_bary(bary)
        point_vortices.set_face_ids(face_ids)
        point_vortices.set_gammas(gammas)

    with prof.span("compute_advect", cat="solver"):
        # Advection
        dt = 0.01  # use `ctx.dt` for real-time
        rk4_intergrator = world.require(RK4AdvectionModule)
        rk4_intergrator.advect(ctx.dt)

    with prof.span("io_read_outputs", cat="solver"):
        nVortices = len(scatter_points)
        new_barys = point_vortices.bary.get().to_numpy()[:nVortices]
        new_faceids = point_vortices.face_ids.get().to_numpy()[:nVortices]
        new_pos = point_vortices.pos_world.get().to_numpy()[:nVortices]
    # edge_hop_advection = world.require(EdgeHopPtVortexAdvectionModule)
    # edge_hop_advection.dt.set_buffer(dt)
    # new_face_ids = edge_hop_advection.new_face_ids.get()
    # new_bary = edge_hop_advection.new_bary.get()
    # new_pos = edge_hop_advection.new_pos.get()

    # # Write new positions to sim state
    # point_vortices.set_bary(new_bary)
    # point_vortices.set
    # vel_module = world.require(VelocityFieldModule)
    # per_face_vel = vel_module.F_velocity.get()
    # ctx.write(OWNER_PRIM, "velocity", per_face_vel.to_numpy(), create=True)
    with prof.span("io_write_outputs", cat="solver"):
        ctx.write(OWNER_POINT, "faceid", new_faceids)
        ctx.write(OWNER_POINT, "bary", new_barys)
        ctx.write(OWNER_POINT, "P", new_pos)
    # # ctx.write(OWNER_POINT, "faceid", )

    # _velocity = world.require(VelocityFieldModule)
    # f_vel = _velocity.F_velocity.get()
    # v_vel = _velocity.V_velocity.get()

    print("Hello")

    ## Read stream function and set the output
    # pt_vortex_sim = world.require(StreamFunctionModule)
    # psi = pt_vortex_sim.psi.get().to_numpy()
    # mesh_io = ctx.input_io(1)
    # if mesh_io is None:
    #     raise RuntimeError(f"Expected triangle mesh on input 1, received {mesh_io}")
    # mesh_io.write(OWNER_POINT, "stream_func", psi, create=True)

    # ## Calculate facewise constant velocity field from stream function
    # vel_module = world.require(VelocityFieldModule)
    # F_velocity = vel_module.F_velocity.get().to_numpy()
    # ctx.write(OWNER_PRIM, "velocity", F_velocity, create=True)


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


def _ensure_vector_field(ref, count: int, *, lanes: int, dtype) -> "ti.Field":

    shape = (count,)
    if count == 0:
        shape = ()

    if ref is None:
        return ti.Vector.field(lanes, dtype=dtype, shape=shape)

    field = ref.peek()
    if (
        field is None
        or tuple(field.shape) != shape
        or getattr(field, "n", lanes) != lanes
    ):
        field = ti.Vector.field(lanes, dtype=dtype, shape=shape)
        ref.set_buffer(field, bump=False)
    return field


def _ensure_scalar_field(ref, count: int, *, dtype) -> "ti.Field":

    shape = (count,)
    if count == 0:
        shape = ()

    if ref is None:
        return ti.field(dtype=dtype, shape=shape)

    field = ref.peek()
    if field is None or tuple(field.shape) != shape:
        field = ti.field(dtype=dtype, shape=shape)
        ref.set_buffer(field, bump=False)

    return field


def _kernel_profiler_enabled(session) -> bool:
    profiler = getattr(session, "profiler", None)
    cfg = getattr(profiler, "cfg", None)
    return bool(getattr(cfg, "enabled", False) and getattr(cfg, "taichi_enabled", False))


def _ensure_taichi_init(session) -> None:
    if session.stats.get("taichi_initialized"):
        return
    if _taichi_initialized():
        session.stats["taichi_initialized"] = True
        return
    ti.init(arch=ti.metal, kernel_profiler=_kernel_profiler_enabled(session))
    session.stats["taichi_initialized"] = True


def run_solver_new() -> None:
    node = hou.pwd()
    geo_out = node.geometry()

    t0 = perf_counter_ns()
    session = get_runtime().get_or_create_session(node)
    t1 = perf_counter_ns()
    prof_cfg = _profiler_cfg_from_node(node)
    prof_token, probe = _enter_profiler(node, session, prof_cfg)

    try:
        session.profiler.record_value(
            "solver", "get_session", None, (t1 - t0) / 1e6
        )

        with session.profiler.span("run_solver_new", cat="houdini"):
            with session.profiler.span("solver_total", cat="cook"):
                with session.profiler.span("debug_config", cat="solver"):
                    dbg_cfg = debug_config_from_node(node)
                    ensure_debug_server(dbg_cfg, node=node)
                    if consume_break_next_button(node):
                        request_break_next(node=node)
                    maybe_break_now(node=node)

                with session.profiler.span("fetch_inputs", cat="solver"):
                    # Input mapping
                    # 0 -> point vortices (prev frame)
                    # 1 -> Triangle Mesh
                    mesh_geo = _get_input_geo(node, index=0)
                    points_geo = _get_input_geo(node, index=1)

                with session.profiler.span("seed_output", cat="solver"):
                    # Pass mesh through to output so downstream SOPs see the surface.
                    _seed_output(geo_out, mesh_geo)

                with session.profiler.span("build_context", cat="solver"):
                    ctx = build_cook_context(
                        node, mesh_geo, geo_out, session, geo_inputs=[mesh_geo, points_geo]
                    )
                    ctx.world()

                if probe is not None:
                    with session.profiler.span("taichi_probe_clear", cat="solver"):
                        probe.clear()

                # 6) Run setup once, then step every time
                with session.profiler.span("setup_once", cat="solver"):
                    if ctx.frame == 1 or not getattr(session, "_solver_did_setup", None):
                        if callable(setup):
                            setup(ctx)
                        session._solver_did_setup = True

                # Guard against Houdini double-cooks of same frame/substep (common in Solvers)
                step_key = (ctx.frame, ctx.substep)
                if step_key == getattr(session, "_solver_last_step_key", None):
                    return

                with session.profiler.span("step_call", cat="solver"):
                    step(ctx)
                session._solver_last_step_key = step_key

                if probe is not None:
                    with session.profiler.span("taichi_probe_sync", cat="solver"):
                        probe.sync()
                        k_ms = probe.kernel_total_ms()
                        session.profiler.record_value("taichi", "kernel_total", None, k_ms)

        runtime_driver._maybe_log_taichi_scoped(session, prof_cfg)
    finally:
        _exit_profiler(session, prof_token)


def _get_input_geo(node: hou.Node, index: int = 0) -> hou.Geometry:
    inputs = node.inputs()
    if not inputs:
        raise RuntimeError("Connect input geometry to the Python SOP.")
    geo_in = inputs[index].geometry()
    if geo_in is None:
        raise RuntimeError("Input geometry is None.")
    return geo_in


def probe():
    node = hou.pwd()
    geo_out = node.geometry()

    # Interactive debugging setup
    cfg = debug_config_from_node(node)
    ensure_debug_server(cfg, node=node)
    if consume_break_next_button(node):
        request_break_next(node=node)
    maybe_break_now(node=node)

    # Input mapping
    # 0 -> point vortices (prev frame)
    # 1 -> Triangle Mesh
    mesh_geo = _get_input_geo(node, index=0)

    # Pass mesh through to output so downstream SOPs see the surface.
    _seed_output(geo_out, mesh_geo)

    session = get_runtime().get_or_create_session(node)

    ctx = build_cook_context(node, mesh_geo, geo_out, session)

    with ctx.session_access("/obj/grid_object1/solver1/d/s/python1") as other:
        _world = other.session.world
        _velocity = _world.require(VelocityFieldModule)
        pt_vortex = _world.require(PointVortexModule)
        frame = pt_vortex.frame.get()[None]
        f_vel = _velocity.F_velocity.get()
        v_vel = _velocity.V_velocity.get()
        ctx.write(OWNER_PRIM, "velocity", f_vel.to_numpy(), create=True)
        ctx.write(OWNER_POINT, "velocity", v_vel.to_numpy(), create=True)

        stream_func = _world.require(StreamFunctionModule)
        psi = stream_func.psi.get()
        ctx.write(OWNER_POINT, "stream_func", psi.to_numpy(), create=True)


def _seed_output(geo_out: hou.Geometry, source: hou.Geometry) -> None:
    geo_out.clear()
    geo_out.merge(source)


def _ensure_state(session) -> None:
    # Store per-node persistent flags/caches on the session object if possible.
    # If your WorldSession already has fields like did_setup/last_step_key, use them.
    if not hasattr(session, "_solver_did_setup"):
        session._solver_did_setup = False
    if not hasattr(session, "_solver_last_step_key"):
        session._solver_last_step_key = None


def _publish_sim_keys(ctx) -> None:
    # Your CookContext likely already has these; publish so producers can depend on them.
    ctx.publish(SIM_FRAME, ctx.frame)
    ctx.publish(SIM_TIME, ctx.time)
    ctx.publish(SIM_DT, ctx.dt)
    ctx.publish(SIM_SUBSTEP, ctx.substep)


def run_solver() -> None:
    node = hou.pwd()
    geo_out = node.geometry()

    cfg = debug_config_from_node(node)
    ensure_debug_server(cfg, node=node)
    if consume_break_next_button(node):
        request_break_next(node=node)
    maybe_break_now(node=node)

    print(cfg)

    geo_prev = _geo_from_input(node, 0)  # solver feedback
    geo_in = _geo_from_input(node, 1)  # optional external input this frame

    source = geo_prev or geo_in
    if source is None:
        raise RuntimeError(
            "Solver SOP needs at least input 0 (prev) or input 1 (current) geometry."
        )

    # 1) Pass-through seed: start from previous frame (preferred), else current input
    _seed_output(geo_out, source)

    # 2) Persistent session for this node
    session = get_runtime().get_or_create_session(node)
    _ensure_state(session)
    prof_cfg = _profiler_cfg_from_node(node)
    prof_token, probe = _enter_profiler(node, session, prof_cfg)

    # 3) Optional: substep parameter (create an int parm 'substep' if you want)
    substep = int(node.evalParm("substep")) if node.parm("substep") else 0

    # 4) Build a solver context
    input_geos = _collect_input_geos(node)
    if input_geos:
        input_geos[0] = geo_prev
        if len(input_geos) > 1:
            input_geos[1] = geo_in
    else:
        input_geos = [geo_prev, geo_in]
    try:
        ctx = build_cook_context(
            node,
            source,  # input snapshot for reading
            geo_out,  # output geometry to write
            session,
            geo_inputs=input_geos,
            substep=substep,
            is_solver=True,
        )

        with session.profiler.span("run_solver", cat="houdini"):
            with session.profiler.span("solver_total", cat="cook"):
                if probe is not None:
                    probe.clear()

                # 5) Publish geometry + sim timing keys
                publish_geometry_minimal(ctx)
                _publish_sim_keys(ctx)

                # 6) Run setup once, then step every time
                if not session._solver_did_setup:
                    if callable(setup):
                        setup(ctx)
                    session._solver_did_setup = True

                # Guard against Houdini double-cooks of same frame/substep (common in Solvers)
                step_key = (ctx.frame, ctx.substep)
                if step_key == session._solver_last_step_key:
                    return

                step(ctx)
                session._solver_last_step_key = step_key

                if probe is not None:
                    probe.sync()
                    k_ms = probe.kernel_total_ms()
                    session.profiler.record_value("taichi", "kernel_total", None, k_ms)

        runtime_driver._maybe_log_taichi_scoped(session, prof_cfg)
    finally:
        _exit_profiler(session, prof_token)

    # === OPTIONAL: apply outputs back to Houdini if your app writes resources ===
    # if ctx.exists("out.P"):
    #     ctx.set_P(ctx.fetch("out.P"))
