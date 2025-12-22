Rendering

Panda3D basics (crash course)
- Scene graph of `NodePath` objects under `render`
- Right‑handed, Y forward, Z up
- Per‑frame tasks update and draw
- Materials, lights, and shaders affect appearance

Meshes
- Class: `rheidos/resources/mesh.py:Mesh`
- Separate vertex arrays for positions, normals, colors, texcoords
- Dynamic updates supported (buffers created with dynamic usage by default)
- Construct and attach quickly:

  ```python
  import numpy as np
  from rheidos.resources import Mesh

  # A single triangle
  V = np.array([[-1,0,0],[1,0,0],[0,0,1]], dtype=np.float32)
  N = np.tile([0,1,0], (3,1)).astype(np.float32)
  C = np.tile([0.8,0.9,1.0,1.0], (3,1)).astype(np.float32)
  I = np.array([0,1,2], dtype=np.int32)

  mesh = Mesh(vertices=V, indices=I, normals=N, colors=C, name="tri")
  mesh.reparent_to(eng.session.render)
  ```

Updating mesh data
- All setters accept contiguous arrays:
  - `set_vertices((N,3) float32)`
  - `set_normals((N,3) float32)`
  - `set_colors((N,4) float32 in 0..1)` or `set_colors_uint8((N,4) uint8)`
  - `set_texcoords((N,2) float32)`
  - `set_indices((M*3,) or (M,3) int32)`

Primitives and loaders
- Procedural cube: `from rheidos.resources import cube`
- Load external mesh with trimesh: `load_mesh(path, center=True)`

  ```python
  from rheidos.resources import load_mesh, cube
  primitive = load_mesh("~/models/bunny.obj")  # or cube(size=2.0)
  eng.session.base.camera.lookAt(0,0,0)
  ```

Ready‑made Views
- `MeshSurfaceView`: glossy shaded surface with `setShaderAuto()` and optional `Material`
- `MeshWireframeView`: wireframe with a nice teal color
- `MeshPositionLabelsView`: shows a label with vertex coordinates near the mouse cursor and highlights the closest vertex

Studio environment (CAD‑like base)
- `StudioView`: ground checker plane, gentle sky background, and a solid default light rig. Optionally applies a mild glossy material to your model nodes.

  ```python
  from rheidos.views import StudioView, MeshSurfaceView, OrientationGizmoView
  from rheidos.resources import load_mesh, cube
  from panda3d.core import Material

  primitive = cube(2.0)
  # Option 1: let MeshSurfaceView set the material
  glossy = Material("Glossy"); glossy.setShininess(64)
  # Snap ground to sit just below the mesh's lowest point (Z)
  eng.add_view(StudioView(ground_from_bounds=primitive.bounds, ground_margin=0.02))
  eng.add_view(MeshSurfaceView(primitive.mesh, material=glossy))
  # Tiny orientation widget in the top-left
  eng.add_view(OrientationGizmoView(size=0.16, margin=0.02))

  # Option 2: let StudioView apply its default glossy material to the mesh node
  # eng.add_view(StudioView(apply_material_to=primitive.mesh.node_path))
  # eng.add_view(MeshSurfaceView(primitive.mesh))
  ```

Studio ground offset
- Control the ground plane height (Panda3D up axis is Z):
  - Fixed height: `StudioView(ground_height=0.0)`
  - Snap to bounds: `StudioView(ground_from_bounds=primitive.bounds, ground_margin=0.02)`
  - After creation: `studio.set_ground_height(z)` or `studio.snap_ground_to_bounds(bounds, margin)`

Orientation gizmo
- `OrientationGizmoView(size=0.18, margin=0.02)`: renders a small RGB axis overlay that rotates with the current camera

  ```python
  from rheidos.views import MeshSurfaceView, MeshWireframeView, MeshPositionLabelsView
  from panda3d.core import Material

  mat = Material("Glossy"); mat.setShininess(64)
  surface = MeshSurfaceView(primitive.mesh, material=mat, two_sided=False)
  wire = MeshWireframeView(primitive.mesh)
  labels = MeshPositionLabelsView(primitive.mesh, scale_factor=0.02)

  eng.add_view(surface)
  eng.add_view(wire)
  eng.add_view(labels)
  eng.enable_view("labels", False)  # start hidden
  ```

Lighting
- Panda3D’s lights are standard nodes; attach to scene and enable per‑render root:

  ```python
  from panda3d.core import AmbientLight, DirectionalLight, Vec4
  r = eng.session.render
  r.clearLight()
  amb = AmbientLight("ambient"); amb.setColor(Vec4(0.18,0.18,0.22,1)); r.setLight(r.attachNewNode(amb))
  key = DirectionalLight("key"); key.setColor(Vec4(0.85,0.85,0.9,1)); n = r.attachNewNode(key); n.setHpr(-35,-45,0); r.setLight(n)
  ```

Axes helper
- `AxesView` draws RGB axes (X red, Y green, Z blue). Simple but handy for orientation.

Two‑sided rendering
- For thin surfaces, call `node.setTwoSided(True)` via `MeshSurfaceView(two_sided=True)` or `mesh.set_two_sided(True)`

Renderer presets & ImGui panel
- Drop-in wrapper that keeps visuals decoupled from the sim: `renderer = Renderer(eng.session)`
- Plug UI controls into the tools window (requires panda3d-imgui):
  - `eng.add_imgui_panel_factory(renderer.panel_factory())` (fast/CommonFilters)
  - `eng.add_imgui_panel_factory(renderer.render_pipeline_panel_factory())` (RenderPipeline-only controls)
- Presets (fast backend): `fast` (minimal), `balanced` (default HDR + bloom), `quality` (HDR, bloom, SSAO, fog, shadowed key)
- Presets (RenderPipeline backend): `fast`, `balanced`, `quality` mapped to RP plugins (AO/bloom/SSR/volumetrics/AA/shadows) and resolution scale
- Toggles/sliders for HDR exposure, bloom size/strength, SSAO radius/strength, sharpen amount, fog density, and default light rig + shadow map size on the fast path. RenderPipeline-specific sliders live in the new “Render Pipeline” panel.

End-to-end usage (with ImGui panel)
```python
from rheidos.engine import Engine
from rheidos.rendering import Renderer

eng = Engine(window_title="Render Demo", interactive=True, auto_start=False)
renderer = Renderer(eng.session)

# Optional: add your views / geometry here
# eng.add_view(...)

# Hook the render settings panel (appears as a separate window; toggle in Rheidos Tools)
eng.add_imgui_panel_factory(renderer.panel_factory())
eng.add_imgui_panel_factory(renderer.render_pipeline_panel_factory())  # RP controls (optional)
eng.start()
```
- Open “Rheidos Tools” → check “Show Render Settings” to pop out the panel.
- Open “Render Pipeline” to switch to RP and tweak its options (if installed).
- Swap presets fast/balanced/quality, then tune: exposure, bloom size/strength, SSAO radius/strength, sharpen, fog density, light rig on/off, light intensity, shadow map size.

Minimal usage (no ImGui)
```python
renderer = Renderer(eng.session)
renderer.apply_preset("quality")
renderer.update_config(light_rig=True, light_intensity=1.3, fog=True, fog_density=0.03)
```

Example scene (interactive demo)
```bash
PYTHONPATH=. python rheidos/examples/run_interactive.py
```
- Loads a detailed mesh if available under `models/` (spot/armadillo/double_torus/bunny) and falls back to a cube.
- Shows Studio ground + axes, FPV camera controller, and the Render Settings panel for live tweaking.
- Add the “Render Pipeline” panel to toggle into the RP backend and adjust RP-only options.

Notes
- Post stack uses Panda3D `CommonFilters`; ensure `panda3d` (and `panda3d-imgui` for UI) are installed.
- SSAO/shadows/SSR are scene-scale sensitive; adjust light intensity and exposure if highlights blow out after enabling bloom/SSAO.
- The renderer stays data-agnostic: it only needs `session` and manages its own light rig and post effects without touching simulation state.

RenderPipeline backend (optional)
- The RenderPipeline backend stays decoupled from sim data and is opt-in. Fast/CommonFilters remain the default.
- Vendor location: clone the latest RenderPipeline into `third_party/RenderPipeline` (or `third_party/render_pipeline`) and run `python setup.py` there. Use flags like `--skip-native --skip-update` if you don’t want to build C++ modules. This writes `data/install.flag` and unpacks assets. The UI will show “unavailable” until this folder exists and is set up.
- Once available, open the “Render Pipeline” panel to switch backends. Presets map to AO/bloom/SSR/volumetrics/AA/shadows settings; a resolution scale slider, exposure, tonemap, AA mode, and light rig controls live in that panel.
- Backend toggle is also available at the top of the regular Render Settings panel; switching back to “fast” returns to the CommonFilters stack instantly.
