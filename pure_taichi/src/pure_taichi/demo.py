from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import SimulationConfig
from .scenarios import vortex_positions_from_face_bary
from .sim import P2PointVortexSim, run_headless
from .taichi_compat import ensure_taichi_initialized


def _stream_colors(stream_vertex: np.ndarray) -> np.ndarray:
    s = np.asarray(stream_vertex, dtype=np.float64)
    if s.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    lo = float(np.min(s))
    hi = float(np.max(s))
    if hi - lo < 1e-12:
        t = np.zeros_like(s)
    else:
        t = (s - lo) / (hi - lo)

    r = 0.2 + 0.8 * t
    g = 0.4 + 0.4 * (1.0 - np.abs(t - 0.5) * 2.0)
    b = 1.0 - 0.8 * t
    return np.column_stack([r, g, b]).astype(np.float32)


def _glyph_segments(vertices: np.ndarray, faces: np.ndarray, vel_face: np.ndarray, scale: float) -> np.ndarray:
    tri = vertices[faces]
    centroids = tri.mean(axis=1)
    tips = centroids + np.asarray(vel_face, dtype=np.float64) * float(scale)

    seg = np.zeros((centroids.shape[0] * 2, 3), dtype=np.float32)
    seg[0::2] = centroids.astype(np.float32)
    seg[1::2] = tips.astype(np.float32)
    return seg


def run_demo(
    config: SimulationConfig,
    *,
    steps: int | None = None,
    seed: int | None = None,
    no_gui: bool = False,
) -> dict[str, object] | None:
    if no_gui:
        return run_headless(config, steps=int(steps or 100), seed=seed)

    if seed is not None:
        config = replace(config, seed=int(seed))

    ti = ensure_taichi_initialized("cpu")
    sim = P2PointVortexSim(config)

    state = sim.solve_fields()

    vertices = np.asarray(sim.mesh.vertices, dtype=np.float32)
    faces = np.asarray(sim.mesh.faces, dtype=np.int32)

    nV = int(vertices.shape[0])
    nF = int(faces.shape[0])

    v_field = ti.Vector.field(3, dtype=ti.f32, shape=nV)
    idx_field = ti.field(dtype=ti.i32, shape=nF * 3)
    color_field = ti.Vector.field(3, dtype=ti.f32, shape=nV)

    v_field.from_numpy(vertices)
    idx_field.from_numpy(faces.reshape(-1).astype(np.int32, copy=False))

    n_vort = int(state.state.face_ids.shape[0])
    vort_field = ti.Vector.field(3, dtype=ti.f32, shape=n_vort)

    glyph_field = ti.Vector.field(3, dtype=ti.f32, shape=max(1, 2 * nF))

    window = ti.ui.Window(
        "Pure Taichi P2 Point Vortex",
        (int(config.render.width), int(config.render.height)),
        vsync=bool(config.render.vsync),
    )
    canvas = window.get_canvas()
    camera = ti.ui.Camera()

    camera.position(2.5, 1.8, 2.5)
    camera.lookat(0.0, 0.0, 0.0)

    paused = False
    single_step = False
    show_glyphs = bool(config.render.show_glyphs)
    show_stream_tint = bool(config.render.show_stream_tint)
    glyph_scale = float(config.render.glyph_scale)
    dt = float(config.time.dt)
    substeps = int(config.time.substeps)

    frame = 0

    while window.running:
        for ev in window.get_events(ti.ui.PRESS):
            key = ev.key
            if key == ti.ui.SPACE:
                paused = not paused
            elif key in ("n", "N", "s", "S"):
                single_step = True
            elif key in ("r", "R"):
                sim.reset(seed=config.seed)
                state = sim.solve_fields()

        if not paused or single_step:
            state = sim.step(dt=dt, substeps=substeps)
            single_step = False

        pos = vortex_positions_from_face_bary(
            sim.mesh.vertices,
            sim.mesh.faces,
            state.state.face_ids,
            state.state.bary,
        ).astype(np.float32)
        vort_field.from_numpy(pos)

        if show_stream_tint:
            color_field.from_numpy(_stream_colors(state.stream_vertex))

        scene = window.get_scene()
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.ambient_light((0.45, 0.45, 0.45))
        scene.point_light(pos=(2.0, 2.0, 2.0), color=(0.95, 0.95, 0.95))

        if show_stream_tint:
            scene.mesh(v_field, indices=idx_field, per_vertex_color=color_field, two_sided=True)
        else:
            scene.mesh(v_field, indices=idx_field, color=(0.75, 0.75, 0.8), two_sided=True)

        scene.particles(vort_field, radius=float(config.render.vortex_radius), color=(1.0, 0.2, 0.2))

        if show_glyphs:
            seg = _glyph_segments(sim.mesh.vertices, sim.mesh.faces, state.vel_face, glyph_scale)
            if seg.shape[0] > 0:
                glyph_field.from_numpy(seg)
                drew = False
                if hasattr(scene, "lines"):
                    try:
                        scene.lines(glyph_field, width=1.0, color=(0.1, 0.9, 0.95))
                        drew = True
                    except Exception:
                        drew = False
                if not drew:
                    scene.particles(glyph_field, radius=0.0025, color=(0.1, 0.9, 0.95))

        canvas.scene(scene)

        gui = window.get_gui()
        gui.begin("Controls", 0.02, 0.02, 0.34, 0.36)
        gui.text(f"frame: {frame}")
        gui.text(f"solver: {state.diagnostics.solver_backend}")
        gui.text(f"residual_l2: {state.diagnostics.residual_l2:.3e}")
        gui.text(f"rhs_circ: {state.diagnostics.rhs_circulation:.3e}")
        gui.text(f"hops total/max: {state.diagnostics.hops_total}/{state.diagnostics.hops_max}")

        dt = float(gui.slider_float("dt", dt, 1e-4, 0.05))
        substeps = int(gui.slider_int("substeps", substeps, 1, 8))
        glyph_scale = float(gui.slider_float("glyph_scale", glyph_scale, 0.01, 0.5))
        show_stream_tint = bool(gui.checkbox("stream_tint", show_stream_tint))
        show_glyphs = bool(gui.checkbox("show_glyphs", show_glyphs))

        if gui.button("reset"):
            sim.reset(seed=config.seed)
            state = sim.solve_fields()
        if gui.button("step"):
            single_step = True
        gui.end()

        window.show()

        frame += 1
        if steps is not None and frame >= int(steps):
            break

    return None
