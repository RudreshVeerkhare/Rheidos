# Point Selector Controller

Pick points on a mesh surface with a simple controller + marker view. Uses Panda3D collision rays to hit the mesh, then lets you choose whether to snap to the nearest vertex or keep the exact surface hit point.

## What it provides
- Click-to-toggle selections (left mouse by default).
- UI actions: enable/disable picking, clear selection, toggle “Snap To Nearest Vertex”.
- Optional markers via `PointSelectionView` for hover/selected dots.
- Selected points pushed into `engine.store` (default key `selected_points`) and exposed via the controller’s `selected_points()` plus an `on_change` callback.

`SelectedPoint` fields:
- `index`: nearest vertex index (or `None` for surface mode)
- `world` / `local`: coordinates in world space and mesh-local space
- `normal`: surface normal at the hit
- `snapped_to_vertex`: whether the hit was snapped

## Quick usage
```python
from rheidos.engine import Engine
from rheidos.resources import cube
from rheidos.views import MeshSurfaceView, PointSelectionView
from rheidos.controllers import PointSelectorController

eng = Engine(window_title="Picker", interactive=False)
prim = cube(size=2.0)
eng.add_view(MeshSurfaceView(prim.mesh, name="surface"))
markers = PointSelectionView(name="selected_points")
eng.add_view(markers)

selector = PointSelectorController(
    engine=eng,
    mesh=prim.mesh,
    target_view="surface",      # align ray space to this view if it moves
    markers_view=markers,       # show hover/selected dots
    snap_to_vertex=True,        # False keeps exact surface hit
    store_key="selected_points",
    on_change=lambda pts: print(f"Selected {len(pts)} pts"),
)
eng.add_controller(selector)
eng.start()
```

## CLI example
Run the bundled demo (snapping on by default, pass `--surface` to keep exact hits):
```
python -m rheidos.examples.point_selection            # snap to vertex
python -m rheidos.examples.point_selection --surface  # keep surface points
```

Controls:
- Left mouse: toggle selection at cursor
- Panel actions: enable/disable picking, clear selection, toggle snapping
