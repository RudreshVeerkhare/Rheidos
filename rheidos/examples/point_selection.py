from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from panda3d.core import AmbientLight, DirectionalLight, Material, Vec3, Vec4, BitMask32

from rheidos.engine import Engine
from rheidos.resources import cube, load_mesh
from rheidos.views import (
    AxesView,
    MeshSurfaceView,
    MeshWireframeView,
    PointSelectionView,
)
from rheidos.controllers import (
    ExitController,
    FpvCameraController,
    SceneSurfacePointSelector,
    SceneVertexPointSelector,
    ScreenshotController,
    ToggleViewController,
)


def setup_lighting(session) -> None:
    render = session.render
    render.clearLight()

    ambient = AmbientLight("ambient")
    ambient.setColor(Vec4(0.18, 0.18, 0.22, 1.0))
    ambient_np = render.attachNewNode(ambient)
    render.setLight(ambient_np)

    key = DirectionalLight("key")
    key.setColor(Vec4(0.85, 0.85, 0.9, 1.0))
    key_np = render.attachNewNode(key)
    key_np.setHpr(-35, -45, 0)
    render.setLight(key_np)

    fill = DirectionalLight("fill")
    fill.setColor(Vec4(0.35, 0.35, 0.45, 1.0))
    fill_np = render.attachNewNode(fill)
    fill_np.setHpr(60, -20, 0)
    render.setLight(fill_np)


def compute_camera_pose(bounds: tuple[np.ndarray, np.ndarray]) -> tuple[Vec3, Vec3]:
    mins, maxs = bounds
    center = (mins + maxs) * 0.5
    extent = (maxs - mins)
    radius = float(np.linalg.norm(extent) * 0.5)
    radius = max(radius, 1.0)
    direction = Vec3(1.4, -2.6, 1.2)
    direction.normalize()
    distance = radius * 3.0
    pos = Vec3(center[0], center[1], center[2]) + direction * distance
    look = Vec3(center[0], center[1], center[2])
    return pos, look


def build_material() -> Material:
    material = Material("Glossy")
    material.setDiffuse((0.8, 0.82, 0.9, 1.0))
    material.setSpecular((1.0, 1.0, 1.0, 1.0))
    material.setShininess(64.0)
    return material


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick points on a mesh surface")
    parser.add_argument("mesh", nargs="?", help="Path to mesh file (OBJ, PLY, STL, ...)")
    parser.add_argument("--no-center", action="store_true", help="Do not recenter mesh to origin")
    parser.add_argument("--surface", action="store_true", help="Use surface hits (no vertex snap)")
    args = parser.parse_args()

    mesh_path = Path(args.mesh).expanduser() if args.mesh else None

    eng = Engine(window_title="Mesh Point Selector", interactive=False)

    if mesh_path:
        try:
            primitive = load_mesh(mesh_path, center=not args.no_center)
        except Exception as exc:
            print(f"Failed to load mesh '{mesh_path}': {exc}")
            return
    else:
        primitive = cube(size=2.0, name="preview_cube")

    material = build_material()
    surface = MeshSurfaceView(primitive.mesh, name="surface", sort=0, material=material)
    wireframe = MeshWireframeView(primitive.mesh, name="wireframe", sort=1)
    markers = PointSelectionView(name="selected_points", sort=5)

    eng.add_view(AxesView(axis_length=1.5, sort=-10))
    eng.add_view(surface)
    eng.add_view(wireframe)
    eng.add_view(markers)

    setup_lighting(eng.session)

    cam_pos, cam_look = compute_camera_pose(primitive.bounds)
    cam = eng.session.base.camera
    cam.setPos(cam_pos)
    cam.lookAt(cam_look)

    # Make mesh nodes pickable (match controller pick_mask bit 4)
    surface._node.setCollideMask(BitMask32.bit(4))
    wireframe._node.setCollideMask(BitMask32.bit(4))

    def on_selection_changed(points):
        print(f"Selected {len(points)} points ({'vertex' if not args.surface else 'surface'} mode)")
        for p in points:
            print(f"  node={p.node_name} index={p.index} world={p.world} normal={p.normal} snapped={p.snapped_to_vertex}")

    selector_cls = SceneSurfacePointSelector if args.surface else SceneVertexPointSelector
    selector = selector_cls(
        engine=eng,
        markers_view=markers,
        store_key="surface_points" if args.surface else "vertex_points",
        on_change=on_selection_changed,
    )

    eng.add_controller(selector)
    eng.add_controller(FpvCameraController(speed=6.0, speed_fast=12.0))
    eng.add_controller(ToggleViewController(eng, groups=[["surface"], ["wireframe"]], key="v", name="WireframeToggle"))
    eng.add_controller(ScreenshotController(eng, key="p", filename="mesh_points.png"))
    eng.add_controller(ExitController(eng, key="escape"))

    eng.start()


if __name__ == "__main__":
    main()
