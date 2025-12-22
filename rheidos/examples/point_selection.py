from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from panda3d.core import AmbientLight, DirectionalLight, Material, Vec3, Vec4, BitMask32

from rheidos.engine import Engine
from rheidos.resources import cube, load_mesh
from rheidos.scene_config import load_scene_from_config
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
    parser.add_argument("--scene-config", type=Path, help="Load meshes/lights/camera from a JSON/YAML scene file")
    parser.add_argument("--no-center", action="store_true", help="Do not recenter mesh to origin")
    parser.add_argument("--surface", action="store_true", help="Use surface hits (no vertex snap)")
    args = parser.parse_args()

    mesh_path = Path(args.mesh).expanduser() if args.mesh else None
    scene_cfg = Path(args.scene_config).expanduser() if args.scene_config else None
    pick_mask = BitMask32.bit(4)

    eng = Engine(window_title="Mesh Point Selector", interactive=False)
    eng.add_view(AxesView(axis_length=1.5, sort=-10))

    surface_names: list[str] = []
    wireframe_names: list[str] = []
    bounds: tuple[np.ndarray, np.ndarray] | None = None

    if scene_cfg:
        try:
            scene = load_scene_from_config(eng, scene_cfg, default_pick_mask=pick_mask)
        except Exception as exc:
            print(f"Failed to load scene config '{scene_cfg}': {exc}")
            return

        if not scene.objects:
            print(f"Scene config '{scene_cfg}' did not define any meshes.")
            return

        bounds = scene.bounds
        surface_names = [obj.surface.name for obj in scene.objects if obj.surface]
        wireframe_names = [obj.wireframe.name for obj in scene.objects if obj.wireframe]

        if not scene.lights_applied:
            setup_lighting(eng.session)
        if bounds is not None and not scene.camera_applied:
            cam_pos, cam_look = compute_camera_pose(bounds)
            cam = eng.session.base.camera
            cam.setPos(cam_pos)
            cam.lookAt(cam_look)
    else:
        if mesh_path:
            try:
                primitive = load_mesh(mesh_path, center=not args.no_center)
            except Exception as exc:
                print(f"Failed to load mesh '{mesh_path}': {exc}")
                return
        else:
            primitive = cube(size=2.0, name="preview_cube")

        material = build_material()
        surface = MeshSurfaceView(
            primitive.mesh,
            name="surface",
            sort=0,
            material=material,
            collide_mask=pick_mask,
        )
        wireframe = MeshWireframeView(
            primitive.mesh,
            name="wireframe",
            sort=1,
            collide_mask=pick_mask,
        )
        eng.add_view(surface)
        eng.add_view(wireframe)

        bounds = primitive.bounds
        surface_names = [surface.name]
        wireframe_names = [wireframe.name]

        setup_lighting(eng.session)

        cam_pos, cam_look = compute_camera_pose(primitive.bounds)
        cam = eng.session.base.camera
        cam.setPos(cam_pos)
        cam.lookAt(cam_look)

    markers = PointSelectionView(name="selected_points", sort=5)
    eng.add_view(markers)

    selector_cls = SceneSurfacePointSelector if args.surface else SceneVertexPointSelector
    selector = selector_cls(
        engine=eng,
        markers_view=markers,
        store_key="surface_points" if args.surface else "vertex_points",
        on_change=None,
    )

    eng.add_controller(selector)
    eng.add_controller(FpvCameraController(speed=6.0, speed_fast=12.0))
    groups: list[list[str]] = []
    if surface_names:
        groups.append(surface_names)
    if wireframe_names:
        groups.append(wireframe_names)
    if len(groups) >= 2:
        eng.add_controller(
            ToggleViewController(eng, groups=groups, key="v", name="WireframeToggle")
        )
    eng.add_controller(ScreenshotController(eng, key="p", filename="mesh_points.png"))
    eng.add_controller(ExitController(eng, key="escape"))

    eng.start()


if __name__ == "__main__":
    main()
