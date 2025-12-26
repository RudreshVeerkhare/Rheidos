# Point Selector Controllers

Scene-wide point picking powered by Panda3D collision rays. Two flavors:
- `SceneSurfacePointSelector` — fast surface hits (no vertex snap).
- `SceneVertexPointSelector` — snap to the nearest vertex of the hit geometry.

Both share:
- Click-to-toggle selections (left mouse by default).
- UI actions: enable/disable picking, clear selection.
- Optional markers via `PointSelectionView` for hover/selected dots.
- Selected points pushed into `engine.store` (`surface_points` / `vertex_points` by default) and exposed via `selected_points()` plus an `on_change` callback.
- Controller names are unique; pass `name="..."` if you want explicit IDs when adding multiple selectors.
- `select_button` can be a string or a list/tuple (e.g., `("mouse3", "mouse2")`) for trackpad/right-click compatibility.

`SelectedPoint` fields:
- `index`: nearest vertex index (None for surface selector)
- `world` / `local`: coordinates in world space and hit-node local space
- `normal`: surface normal at the hit
- `snapped_to_vertex`: True for vertex selector when a vertex was found
- `node_name`: name of the hit node

> Picking relies on Panda3D collide masks. Ensure pickable nodes have the `pick_mask` bit set (default `BitMask32.bit(4)`). The ray uses that bit; non-matching nodes are skipped.
> When loading scenes from YAML, set `pickable: true` on meshes (defaults to true) so they inherit the mask; custom nodes should call `setCollideMask(BitMask32.bit(4))` (or your chosen mask) before attaching selectors.

## Quick usage (vertex snapping)
```python
from panda3d.core import BitMask32
from rheidos.engine import Engine
from rheidos.resources import cube
from rheidos.views import MeshSurfaceView, PointSelectionView
from rheidos.controllers import SceneVertexPointSelector

eng = Engine(window_title="Picker", interactive=False)
prim = cube(size=2.0)
surface = MeshSurfaceView(prim.mesh, name="surface")
eng.add_view(surface)

# Make the surface pickable (matches pick_mask bit 4)
surface._node.setCollideMask(BitMask32.bit(4))

markers = PointSelectionView(name="selected_points")
eng.add_view(markers)

selector = SceneVertexPointSelector(
    engine=eng,
    markers_view=markers,     # show hover/selected dots
    store_key="vertex_points",
    on_change=lambda pts: print(f"Selected {len(pts)} pts"),
)
eng.add_controller(selector)
eng.start()
```

## Variant: surface-only (no snapping)
```python
from rheidos.controllers import SceneSurfacePointSelector
selector = SceneSurfacePointSelector(engine=eng, markers_view=markers, store_key="surface_points")
eng.add_controller(selector)
```

## Controls
- Left mouse: toggle selection at cursor
- Panel actions: enable/disable picking, clear selection
