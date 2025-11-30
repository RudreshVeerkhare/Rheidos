API Quick Reference

Engine (rheidos/engine.py)
- `Engine(window_title="Rheidos", window_size=(1280,720), fps=60, interactive=False, auto_start=None)`
- `.start()` / `.stop()` — blocking run/stop for scripts
- `.start_async(fps=None)` / `.stop_async()` — notebook‑friendly
- `.is_running()` / `.set_paused(bool)` / `.is_paused()`
- `.session` — PandaSession
- `.store` — StoreState
- Views: `.add_view(view)` / `.remove_view(name)` / `.enable_view(name, enabled=True)`
- Observers: `.add_observer(obs)` / `.remove_observer(name)`
- Controllers: `.add_controller(ctrl)` / `.remove_controller(name)`
- `.screenshot(filename, use_default=False)`
- `.dispatch(fn)` — run small callable on next frame (render thread)

PandaSession (rheidos/session.py)
- `.base`, `.task_mgr`, `.render`, `.win`, `.clock`
- `.accept(event, callback, *args)` / `.ignore(event)` — Panda3D event binding

StoreState (rheidos/store.py)
- `.get(key, default=None)` / `.set(key, value)` / `.update(**kvs)`
- `.subscribe(key, fn) -> unsubscribe()`
- `.as_dict()`

Base classes (rheidos/abc)
- `View(name=None, sort=0)` — override `setup(session)`, `update(dt)`, `teardown()`, `on_enable()`, `on_disable()`
- `Observer(name=None, sort=-10)` — override `setup(session)`, `update(dt)`, `teardown()`
- `Controller(name=None)` — override `attach(session)`, `detach()`

Views (rheidos/views)
- `AxesView(axis_length=1.0, sort=0)`
- `MeshSurfaceView(mesh, material=None, two_sided=False, sort=0)`
- `MeshWireframeView(mesh, sort=0)`
- `MeshPositionLabelsView(mesh, scale_factor=0.015, offset_factor=0.02, text_color=(1,0.9,0.3,1), fmt="({x:.4f}, {y:.4f}, {z:.4f})", include_index=True, sort=0)`
- `StudioView(ground_size=40.0, ground_tiles=40, checker_light=(0.92,0.93,0.96,1), checker_dark=(0.86,0.87,0.90,1), sky_color=(0.92,0.95,1.0,1), add_lights=True, apply_material_to=None, material=None)`
- `OrientationGizmoView(size=0.18, margin=0.02, thickness=3.0, fov_deg=28.0, sort=1000)`

StudioView helpers/params
- `ground_height: Optional[float]` — set ground plane world Z directly
- `ground_from_bounds: Optional[(mins, maxs)]` — snap ground to min Z of bounds
- `ground_margin: float` — subtract margin from min Z before snapping
- Methods: `set_ground_height(z)`, `snap_ground_to_bounds(bounds, margin=0.0)`

Controllers (rheidos/controllers)
- `FpvCameraController(speed=6.0, speed_fast=12.0, mouse_sensitivity=0.15, invert_y=False)`
- `ToggleViewController(engine, groups, key="v")`
- `PauseController(engine, key="space")`
- `ScreenshotController(engine, key="s", filename="screenshot.png")`
- `ExitController(engine, key="escape")`

Resources (rheidos/resources)
- `Mesh(vertices=None, indices=None, normals=None, colors=None, texcoords=None, dynamic=True, name="mesh")`
  - `.set_vertices((N,3) float32)` / `.set_normals((N,3) float32)` / `.set_colors((N,4) float32)` / `.set_colors_uint8((N,4) uint8)` / `.set_texcoords((N,2) float32)` / `.set_indices(int32)`
  - `.reparent_to(nodepath)` / `.set_two_sided(True/False)`
- `cube(size=1.0, name="cube") -> Primitive(mesh, bounds=(mins, maxs))`
- `load_mesh(path, name=None, center=True) -> Primitive(mesh, bounds)`
- `Texture2D.from_numpy_rgba(image_uint8)`

Taichi bridge (rheidos/utils/taichi_bridge.py)
- `field_to_numpy(field) -> np.ndarray`
- `numpy_to_field(np_array, field)`
