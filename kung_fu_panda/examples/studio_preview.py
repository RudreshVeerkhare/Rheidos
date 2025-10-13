from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from panda3d.core import Material

from kung_fu_panda.engine import Engine
from kung_fu_panda.resources import cube, load_mesh
from kung_fu_panda.views import (
    StudioView,
    MeshSurfaceView,
    MeshWireframeView,
    MeshPositionLabelsView,
    OrientationGizmoView,
)
from kung_fu_panda.controllers import (
    FpvCameraController,
    ToggleViewController,
    ScreenshotController,
    ExitController,
)


def compute_camera_pose(bounds: tuple[np.ndarray, np.ndarray]):
    mins, maxs = bounds
    center = (mins + maxs) * 0.5
    extent = maxs - mins
    radius = float(np.linalg.norm(extent) * 0.5)
    radius = max(radius, 1.0)
    pos = center + np.array([1.4, -2.6, 1.2]) * (radius * 0.8)
    look = center
    return pos, look


def main() -> None:
    parser = argparse.ArgumentParser(description="Studio environment preview")
    parser.add_argument(
        "mesh", nargs="?", help="Path to mesh file (OBJ, PLY, STL, ...)"
    )
    parser.add_argument(
        "--no-center", action="store_true", help="Do not recenter mesh to origin"
    )
    args = parser.parse_args()

    eng = Engine(window_title="Studio Preview", interactive=False)

    if args.mesh:
        primitive = load_mesh(Path(args.mesh).expanduser(), center=not args.no_center)
    else:
        primitive = cube(size=2.0, name="preview_cube")

    # Studio with ground snapped to mesh bounds so it doesn't intersect
    eng.add_view(
        StudioView(
            ground_from_bounds=primitive.bounds, ground_margin=0.02, ground_tiles=10
        )
    )

    mat = Material("Glossy")
    mat.setDiffuse((0.8, 0.82, 0.9, 1.0))
    mat.setSpecular((1, 1, 1, 1))
    mat.setShininess(64)
    surface = MeshSurfaceView(primitive.mesh, name="surface", material=mat)
    wire = MeshWireframeView(primitive.mesh, name="wire")
    labels = MeshPositionLabelsView(
        primitive.mesh, name="labels", sort=5, scale_factor=0.02, offset_factor=0.03
    )

    eng.add_view(surface)
    eng.add_view(wire)
    eng.add_view(labels)
    eng.add_view(OrientationGizmoView(size=0.09, margin=0.01, anchor="top-right"))
    eng.enable_view("labels", False)

    pos, look = compute_camera_pose(primitive.bounds)
    cam = eng.session.base.camera
    cam.setPos(*pos)
    cam.lookAt(*look)

    eng.add_controller(FpvCameraController())
    eng.add_controller(
        ToggleViewController(
            eng, groups=[["surface"], ["wire"]], key="space", name="WireframeToggle"
        )
    )
    eng.add_controller(
        ToggleViewController(eng, groups=[["labels"], []], key="l", name="LabelToggle")
    )
    eng.add_controller(
        ScreenshotController(eng, key="p", filename="studio_preview.png")
    )
    eng.add_controller(ExitController(eng, key="escape"))

    eng.start()


if __name__ == "__main__":
    main()
