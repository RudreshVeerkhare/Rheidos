Scene configs (YAML/JSON)
=========================

The `rheidos.scene_config.load_scene_from_config` helper lets you keep scene setup (meshes, materials, lights, camera, ground) in a YAML/JSON file instead of code.

Prerequisites
-------------
- Install with YAML support: `pip install -e .[config]`
- Example model bundle is in `models/`; example config in `rheidos/examples/scene_configs/point_selection.yaml`.

Running the built‑in demo
-------------------------
```
python -m rheidos.examples.point_selection --scene-config rheidos/examples/scene_configs/point_selection.yaml
```
Flags still work with configs (e.g., `--surface` to pick surface hits).

Config structure (quick reference)
----------------------------------
- `meshes` (list, required): each mesh entry can include:
  - `path` (str, required): file path (relative to the config file or absolute)
  - `name` (str): node name
  - `center` (bool, default true): recenter mesh to origin
  - `two_sided` (bool): render backfaces
  - `pickable` (bool, default true): sets collide mask for picking
  - `surface` / `wireframe` (bool): enable those views
  - `material` (object): `diffuse`, `specular`, `ambient`, `emission` (3 or 4 floats), `shininess`
  - `transform` (object): `position`/`translation` (vec3), `hpr`/`rotation` (vec3 degrees), `scale` (number or vec3)
- `camera` (object):
  - `position`, `look_at` (vec3): explicit camera placement
  - or `auto_frame` (bool, default true), `offset_dir`, `distance_scale`, `min_distance`
- `lights` (object):
  - `ambient` (vec3/vec4) (use `null` to skip)
  - `directionals` (list): each with `name`, `color`, `hpr` (heading, pitch, roll)
- `studio` (object):
  - `enabled` (bool, default true)
  - `ground_size`, `ground_tiles`, `ground_margin`
  - `snap_ground_to_bounds` (bool, default true)
  - `ground_height` (number): overrides snapping
  - `checker_light`, `checker_dark`, `sky_color` (vec4)
  - `add_lights` (bool): set false if you only want custom lights
  - `apply_material_to_meshes` (bool) + optional `material` block
- `background_color` (vec4): overrides window clear color

Minimal config example
----------------------
```yaml
meshes:
  - path: ../../../models/bunny.obj
    name: bunny

camera:
  auto_frame: true
```

Richer example (used by the demo)
---------------------------------
```yaml
meshes:
  - path: ../../../models/bunny.obj
    name: bunny
    center: true
    two_sided: false
    pickable: true
    material:
      diffuse: [0.80, 0.82, 0.90, 1.0]
      specular: [1.0, 1.0, 1.0, 1.0]
      shininess: 64

camera:
  auto_frame: true
  offset_dir: [1.4, -2.6, 1.2]
  distance_scale: 3.0

lights:
  ambient: [0.18, 0.18, 0.22, 1.0]
  directionals:
    - name: key
      color: [0.85, 0.85, 0.90, 1.0]
      hpr: [-35, -45, 0]
    - name: fill
      color: [0.35, 0.35, 0.45, 1.0]
      hpr: [60, -20, 0]

studio:
  enabled: true
  ground_size: 20
  ground_tiles: 40
  snap_ground_to_bounds: true
  ground_margin: 0.02
```

Using it in code
----------------
```python
from pathlib import Path
from rheidos.engine import Engine
from rheidos.scene_config import load_scene_from_config

eng = Engine(window_title="Config-driven scene")
scene = load_scene_from_config(eng, Path("my_scene.yaml"))

# scene.objects gives you per-mesh views and primitives
for obj in scene.objects:
    print(obj.name, obj.primitive.bounds)

# If the config skipped lights/camera, you can still apply your own here
```

Adding your own views/controllers with config values
----------------------------------------------------
You can store extra app-specific settings in your YAML and read them alongside `load_scene_from_config`. Example:

```yaml
meshes:
  - path: ../../../models/trefoil_knot.obj
    name: knot

app:
  grid_spacing: 0.5
  hotspot_color: [1.0, 0.4, 0.2, 1.0]
  flycam:
    speed: 8.0
    speed_fast: 16.0
```

```python
import yaml
from pathlib import Path
from rheidos.engine import Engine
from rheidos.scene_config import load_scene_from_config
from rheidos.controllers import FpvCameraController

cfg_path = Path("my_scene.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

eng = Engine(window_title="Custom scene")
scene = load_scene_from_config(eng, cfg_path)

app_cfg = cfg.get("app", {})
flycam_cfg = app_cfg.get("flycam", {})
eng.add_controller(
    FpvCameraController(
        speed=flycam_cfg.get("speed", 6.0),
        speed_fast=flycam_cfg.get("speed_fast", 12.0),
    )
)

# Custom view that uses config color
from rheidos.views import AxesView
eng.add_view(AxesView(axis_length=app_cfg.get("grid_spacing", 0.5)))
```

Pattern:
- Add your own top-level keys (e.g., `app`, `ui`, `physics`) to the YAML.
- Read the config file yourself (`yaml.safe_load`) in the same script where you call `load_scene_from_config`.
- Pass the extracted values into your custom views/controllers as constructor arguments.

Plug-and-play custom views/controllers from YAML
------------------------------------------------
`load_scene_from_config` can also build custom components declared in the YAML. Add a `views` or `controllers` list where each entry points to a factory callable. The factory is called with any of the following args that it declares: `engine/eng`, `session/sess`, `config/cfg/settings`, `entry` (raw YAML entry). It should return a `View` or `Controller`.

```yaml
meshes:
  - path: ../../../models/bunny.obj

controllers:
  - factory: mypkg.controllers:make_highlight    # module:path or module.func
    config:
      key: h
      color: [1.0, 0.3, 0.2, 1.0]
    enabled: true

views:
  - factory: mypkg.grid:make_grid_view
    config:
      size: 10
      spacing: 0.5
```

Factories (example)
```python
# mypkg/controllers.py
from panda3d.core import Vec4
from rheidos.controllers import Controller

def make_highlight(engine, config):
    key = config.get("key", "h")
    color = Vec4(*config.get("color", (1, 0.3, 0.2, 1)))
    class Highlight(Controller):
        def __init__(self):
            super().__init__("Highlight")
        def attach(self, session):
            super().attach(session)
            session.accept(key, self._toggle)
        def detach(self):
            self._session.ignore(key)
        def _toggle(self):
            print("toggle highlight", color)
    return Highlight()
```

```python
# mypkg/grid.py
from rheidos.abc.view import View

def make_grid_view(engine, cfg):
    size = cfg.get("size", 10)
    spacing = cfg.get("spacing", 0.5)
    class Grid(View):
        def __init__(self):
            super().__init__("Grid")
        def setup(self, session):
            super().setup(session)
            # build grid geometry using size/spacing here
    return Grid()
```

Usage is the same as before: `load_scene_from_config(engine, "my_scene.yaml")` will also attach those custom components automatically. For convenience you can use `factory: pkg.mod:fn` or `factory: pkg.mod.fn`.

Tips
----
- Paths in the config are resolved relative to the config file location, so you can keep configs and assets together.
- Use JSON instead of YAML if you don’t want the extra dependency; the loader accepts both based on file extension.
- `load_scene_from_config` returns what it created (objects, bounds, flags for lights/camera) so you can branch if something wasn’t set by the file.
