# Rheidos User Guide

A practical walkthrough of how to use Rheidos as it exists in this repo today. It leans on runnable examples, calls out required/optional dependencies, and spells out caveats so you know what to expect.

## Install and Dependencies

- Core and extras are bundled in the default install: `panda3d`, `taichi`, `trimesh`, `pyyaml`, `p3dimgui`, `imgui-bundle`, and `numpy` (Python 3.9+).
- Without ImGui deps, hotkeys still work but the tools window is skipped; they are installed by default to avoid confusion.

Install everything:

```bash
pip install -e .
```

## Fast Start (script mode)

```python
from rheidos.engine import Engine
from rheidos.resources import cube
from rheidos.views import MeshSurfaceView, MeshWireframeView, StudioView, OrientationGizmoView
from rheidos.controllers import FpvCameraController, ToggleViewController, ScreenshotController, ExitController

eng = Engine(window_title="Rheidos Demo", interactive=False)
prim = cube(size=2.0)

eng.add_view(StudioView(ground_from_bounds=prim.bounds, ground_margin=0.02))
eng.add_view(MeshSurfaceView(prim.mesh, name="surface"))
eng.add_view(MeshWireframeView(prim.mesh, name="wireframe"))
eng.add_view(OrientationGizmoView(size=0.14))

eng.add_controller(FpvCameraController(speed=6.0, speed_fast=12.0))
eng.add_controller(ToggleViewController(eng, groups=[["surface"], ["wireframe"]], key="space"))
eng.add_controller(ScreenshotController(eng, key="p", filename="shot.png"))
eng.add_controller(ExitController(eng, key="escape"))

eng.start()  # blocks until window close/ESC
```

Controls: hold left mouse to look, WASD to move, Q/E vertical, Shift to sprint, Z/C roll, Space toggles surface/wireframe, P screenshot, ESC exit.

## Interactive / Notebook Loop

Use `interactive=True` with `auto_start=False` if you don’t want `__init__` to block in a script:

```python
from rheidos.engine import Engine
from rheidos.views import AxesView

eng = Engine(window_title="Rheidos — Interactive", interactive=True, auto_start=False)
eng.add_view(AxesView(axis_length=1.5))

# In an async cell:
await eng.start_async()
# ...add/remove views/controllers live...
# await eng.stop_async() when done
```

## Resources and Views

- `rheidos.resources.Mesh` expects float32 positions/normals/texcoords and uint8/float32 colors; indices must be triangles (len % 3 == 0) and non-negative.
- Helpers:
  - `cube(size)` -> `Primitive(mesh, bounds)`
  - `load_mesh(path, center=True)` -> `Primitive` (requires `trimesh`)
- Ready-made views:
  - `MeshSurfaceView` / `MeshWireframeView`
  - `MeshPositionLabelsView` (labels nearest vertex under mouse)
  - `StudioView` (ground plane, sky tint, optional material application)
  - `OrientationGizmoView` (screen-corner axes)
  - `PointSelectionView` (shows hover/selected dots)
  - `PointVortexStreamFunctionView` (2D stream function textured quad)

Point selection overlay example:

```python
from panda3d.core import BitMask32
from rheidos.views import MeshSurfaceView, PointSelectionView
from rheidos.controllers import SceneVertexPointSelector
from rheidos.resources import load_mesh

prim = load_mesh("models/bunny.obj")
surface = MeshSurfaceView(prim.mesh, collide_mask=BitMask32.bit(4))  # make pickable
eng.add_view(surface)

markers = PointSelectionView()
eng.add_view(markers)
eng.add_controller(SceneVertexPointSelector(engine=eng, markers_view=markers))
```

Stream function view example (CPU-side colormap):

```python
from rheidos.views import PointVortexStreamFunctionView

eng.add_view(
    PointVortexStreamFunctionView(
        positions=[(-0.6, 0.0), (0.6, 0.0)],
        strengths=[1.0, -1.0],
        bounds=((-1.5, -1.0), (1.5, 1.0)),
        resolution=(512, 512),
        plane_z=0.0,
    )
)
```

## Controllers and Actions

- `FpvCameraController`: fly cam with roll, sprint; ignores hotkeys when ImGui wants the mouse.
- `ToggleViewController`: cycles groups of view names.
- `PauseController`, `ScreenshotController`, `ExitController`.
- `SceneSurfacePointSelector` / `SceneVertexPointSelector`: click-to-toggle selections; store results in `engine.store` (`surface_points`/`vertex_points`) and expose actions to enable/clear.

Actions automatically show in the ImGui tools window (if `p3dimgui` + `imgui-bundle` are installed); shortcuts still work without UI.

## Scene Configs and Live Editing

Load meshes/lights/camera/studio from YAML/JSON:

```python
from pathlib import Path
from rheidos.scene_config import load_scene_from_config

scene = load_scene_from_config(eng, Path("rheidos/examples/scene_configs/point_selection.yaml"))
# scene.objects -> list of SceneObject with .primitive/.surface/.wireframe
```

Requirements: `pyyaml` for YAML, `trimesh` for mesh loading. `ui.scene_config_panel: true` in the config enables the live-edit ImGui panel (needs `p3dimgui` + `imgui-bundle`). The panel diffs by mesh name and rebuilds meshes/studio/lights/camera/custom components when those sections change; “Force Reload” tears everything down and rebuilds.

## ImGui Panels

- Default: `StoreStatePanel` is added when imgui is available.
- Add your own panel factories at engine init or later:

```python
from rheidos.ui.panels.store_state import StoreStatePanel

eng = Engine(imgui_panel_factories=(lambda session, store: StoreStatePanel(store=store),))
eng.add_imgui_panel_factory(lambda session, store: MyPanel(engine=eng))
```

Panel factories receive `(session, store)` and should return a panel object with `id`, `title`, `order`, and `draw(imgui)`.

## Caveats and Limitations

- `interactive=True` auto-starts the render loop inside `__init__` unless you pass `auto_start=False`; in scripts this can block unexpectedly.
- Render/update wrappers swallow exceptions to keep the loop alive; add your own logging inside views/controllers/observers to debug.
- Picking requires collide masks: set `collide_mask=BitMask32.bit(4)` on pickable nodes (or choose your own and pass it to selectors).
- Scene-config live edits rebuild or replace views/controllers; external references to old NodePaths will go stale after reloads.
- `PointVortexStreamFunctionView` renders on the CPU; high resolutions can hitch.
- `Texture2D.from_numpy_rgba` only accepts `(H, W, 4) uint8`; changing resolution reallocates the texture.
- ImGui UI depends on `p3dimgui` + `imgui-bundle`; neither is included in the extras.

## Doc Gaps Found (so you know what changed)

- `docs/api.md` missed `PointSelectionView`, `PointVortexStreamFunctionView`, and the Scene* point selectors; this guide covers them.
- `docs/scene_config.md` listed only the `[config]` extra but scene loading also requires `trimesh`; dependency callouts above fix that.
- ImGui dependencies (`p3dimgui` + `imgui-bundle`) were undocumented; they’re now spelled out here.
- The auto-start behavior of `Engine(interactive=True)` and exception-swallowing loop behavior were undocumented; see Caveats.
- Live scene-config caveats (reloads rebuild components, stale references) were not surfaced; they’re noted above.
