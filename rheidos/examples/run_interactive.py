from __future__ import annotations

from pathlib import Path

from rheidos.engine import Engine
from rheidos.rendering import Renderer
from rheidos.ui.panels.render_settings import renderer_panel_factory
from rheidos.views import AxesView, StudioView, MeshSurfaceView
from rheidos.resources import cube, load_mesh
from rheidos.controllers import (
    FpvCameraController,
    PauseController,
    ScreenshotController,
)


def main() -> None:
    eng = Engine(window_title="Rheidos — Interactive", interactive=True, auto_start=False)
    renderer = Renderer(eng.session)

    # Load a richer mesh (falls back to cube if assets or deps are missing).
    primitive = _load_demo_mesh()
    bounds = primitive.bounds
    scale = _fit_bounds_scale(bounds, target_span=2.4)
    primitive.mesh.node_path.setScale(scale)
    mins, maxs = bounds
    scaled_bounds = (mins * scale, maxs * scale)

    try:
        from panda3d.core import Material, Vec4

        mat = Material("BrushedMetal")
        mat.setDiffuse(Vec4(0.78, 0.66, 0.62, 1.0))
        mat.setAmbient(Vec4(0.2, 0.18, 0.18, 1.0))
        mat.setSpecular(Vec4(1.0, 0.94, 0.88, 1.0))
        mat.setEmission(Vec4(0.03, 0.03, 0.04, 1.0))
        mat.setShininess(96.0)
    except Exception:
        mat = None

    eng.add_view(StudioView(ground_from_bounds=scaled_bounds, ground_margin=0.05, sort=-20))
    eng.add_view(MeshSurfaceView(primitive.mesh, material=mat, sort=-10))
    eng.add_view(AxesView(axis_length=1.5, sort=0))
    eng.add_imgui_panel_factory(renderer.panel_factory())
    eng.add_imgui_panel_factory(renderer.render_pipeline_panel_factory())

    eng.add_controller(PauseController(eng, key="space"))
    eng.add_controller(FpvCameraController(speed=8.0, speed_fast=14.0, mouse_sensitivity=0.12))
    eng.add_controller(ScreenshotController(eng, key="s", filename="interactive_shot.png"))

    # Start with a sensible camera pose before FPV takes over.
    try:
        cam = eng.session.base.camera
        cam.setPos(0.0, -3.5, 1.6)
        cam.lookAt(0.0, 0.0, 0.6)
    except Exception:
        pass

    print("Interactive loop started. Press Ctrl+C to stop.")
    try:
        eng.start()
    except KeyboardInterrupt:
        pass
    finally:
        eng.stop()


def _fit_bounds_scale(bounds: tuple, target_span: float = 2.0) -> float:
    mins, maxs = bounds
    span = float((maxs - mins).max())
    if span <= 1e-6:
        return 1.0
    return target_span / span


def _load_demo_mesh():
    base = Path(__file__).resolve().parents[2] / "models"
    candidates = ["spot.obj", "armadillo.obj", "double_torus.obj", "bunny.obj"]
    for name in candidates:
        path = base / name
        if not path.exists():
            continue
        try:
            return load_mesh(path)
        except Exception:
            continue
    print("Falling back to cube() demo mesh.")
    return cube(1.6)


if __name__ == "__main__":
    main()
