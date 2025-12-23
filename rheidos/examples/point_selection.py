from __future__ import annotations

import argparse
from pathlib import Path

from panda3d.core import BitMask32

from rheidos.engine import Engine
from rheidos.scene_config import load_scene_from_config
from rheidos.views import AxesView, PointSelectionView
from rheidos.controllers import (
    SceneSurfacePointSelector,
    SceneVertexPointSelector,
    ToggleViewController,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Pick points on a mesh surface")
    parser.add_argument("--scene-config", type=Path, help="Load meshes/lights/camera from a JSON/YAML scene file")
    parser.add_argument("--surface", action="store_true", help="Use surface hits (no vertex snap)")
    args = parser.parse_args()

    if not args.scene_config:
        print("Please provide --scene-config pointing to a YAML/JSON scene file.")
        return

    scene_cfg = Path(args.scene_config).expanduser()
    pick_mask = BitMask32.bit(4)
    eng = Engine(window_title="Mesh Point Selector", interactive=False)
    eng.add_view(AxesView(axis_length=1.5, sort=-10))

    surface_names: list[str] = []
    wireframe_names: list[str] = []
    try:
        scene = load_scene_from_config(eng, scene_cfg, default_pick_mask=pick_mask)
    except Exception as exc:
        print(f"Failed to load scene config '{scene_cfg}': {exc}")
        return

    if not scene.objects:
        print(f"Scene config '{scene_cfg}' did not define any meshes.")
        return

    surface_names = [obj.surface.name for obj in scene.objects if obj.surface]
    wireframe_names = [obj.wireframe.name for obj in scene.objects if obj.wireframe]

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
    groups: list[list[str]] = []
    if surface_names:
        groups.append(surface_names)
    if wireframe_names:
        groups.append(wireframe_names)
    if len(groups) >= 2:
        eng.add_controller(
            ToggleViewController(eng, groups=groups, key="v", name="WireframeToggle")
        )

    eng.start()


if __name__ == "__main__":
    main()
