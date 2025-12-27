from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable
import importlib
import inspect

import numpy as np
from panda3d.core import AmbientLight, BitMask32, DirectionalLight, Material, Vec3, Vec4

from .resources import Primitive, load_mesh
from .views import MeshSurfaceView, MeshWireframeView, StudioView

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class SceneObject:
    name: str
    primitive: Primitive
    surface: Optional[MeshSurfaceView]
    wireframe: Optional[MeshWireframeView]


@dataclass
class SceneConfigResult:
    objects: List[SceneObject]
    bounds: Optional[Tuple[np.ndarray, np.ndarray]]
    studio_view: Optional[StudioView]
    lights_applied: bool
    camera_applied: bool
    custom_views: List[object]
    custom_controllers: List[object]


def load_scene_from_config(
    engine,
    config_path: str | Path,
    *,
    default_pick_mask: BitMask32 = BitMask32.bit(4),
) -> SceneConfigResult:
    """
    Build a scene (meshes, lights, camera, optional studio ground) from a JSON/YAML config file.
    Returns the created objects so callers can wire controllers/UI on top.
    """
    cfg_path = Path(config_path).expanduser()
    data = _read_config(cfg_path)
    result = build_scene_from_data(engine, data, cfg_path, default_pick_mask=default_pick_mask)
    _maybe_register_scene_config_panel(engine, cfg_path, data, result, default_pick_mask)
    return result


def build_scene_from_data(
    engine,
    data: Dict[str, Any],
    cfg_path: Path,
    *,
    default_pick_mask: BitMask32 = BitMask32.bit(4),
) -> SceneConfigResult:
    """
    Build a scene from an already-parsed config dictionary. Useful for callers that have
    edited the config in-memory (e.g., live reload tools).
    """
    meshes_cfg = data.get("meshes") or data.get("models")
    if not meshes_cfg:
        raise ValueError("Scene config must contain a non-empty 'meshes' list.")

    objects: List[SceneObject] = []
    bounds_list: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx, mesh_cfg in enumerate(meshes_cfg):
        obj = _build_mesh(engine, cfg_path, mesh_cfg, idx, default_pick_mask)
        objects.append(obj)
        bounds_list.append(obj.primitive.bounds)

    bounds = _combine_bounds(bounds_list)

    studio_view = _maybe_add_studio(engine, data.get("studio", {}), bounds, objects, bool(data.get("lights")))
    lights_applied = _apply_lights(engine.session, data.get("lights"), studio_view)
    camera_applied = _apply_camera(engine.session, data.get("camera"), bounds)

    bg = data.get("background_color")
    if bg is not None:
        try:
            engine.session.base.setBackgroundColor(*_vec4(bg))
        except Exception:
            pass

    custom_views = _build_custom_items(engine, data.get("views"), kind="view")
    custom_controllers = _build_custom_items(engine, data.get("controllers"), kind="controller")

    return SceneConfigResult(
        objects=objects,
        bounds=bounds,
        studio_view=studio_view,
        lights_applied=lights_applied,
        camera_applied=camera_applied,
        custom_views=custom_views,
        custom_controllers=custom_controllers,
    )


# --- Internals ---------------------------------------------------------------


def _read_config(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML configs; install 'pyyaml' or use JSON instead.")
        parsed = yaml.safe_load(text)
    else:
        parsed = json.loads(text)
    return parsed or {}


def _maybe_register_scene_config_panel(
    engine,
    cfg_path: Path,
    data: Dict[str, Any],
    result: SceneConfigResult,
    default_pick_mask: BitMask32,
) -> None:
    ui_cfg = data.get("ui")
    if not isinstance(ui_cfg, dict) or not ui_cfg.get("scene_config_panel"):
        return
    try:
        from .ui.panels.scene_config_panel import SceneConfigPanel
    except Exception:
        return
    try:
        engine.add_imgui_panel_factory(
            lambda session, store: SceneConfigPanel(
                engine=engine,
                config_path=cfg_path,
                default_pick_mask=default_pick_mask,
                initial_config=data,
                initial_result=result,
            )
        )
    except Exception:
        pass


def _build_mesh(
    engine,
    cfg_path: Path,
    mesh_cfg: Dict[str, Any],
    idx: int,
    default_pick_mask: Optional[BitMask32],
) -> SceneObject:
    raw_path = mesh_cfg.get("path") or mesh_cfg.get("file")
    if not raw_path:
        raise ValueError("Each mesh entry must include a 'path'.")
    mesh_path = (cfg_path.parent / Path(str(raw_path))).expanduser().resolve()

    name = mesh_cfg.get("name") or mesh_path.stem or f"mesh{idx}"
    primitive = load_mesh(
        mesh_path,
        name=name,
        center=mesh_cfg.get("center", True),
        dynamic=bool(mesh_cfg.get("dynamic", False)),
    )

    material = _material_from_cfg(mesh_cfg.get("material"))
    pick_mask = default_pick_mask if mesh_cfg.get("pickable", True) else None
    two_sided = bool(mesh_cfg.get("two_sided", False))
    transform = _transform_from_cfg(mesh_cfg.get("transform"))

    surface_enabled = mesh_cfg.get("surface", True)
    wireframe_enabled = mesh_cfg.get("wireframe", True)

    surface_name = mesh_cfg.get("surface_name") or f"{name}-surface"
    wire_name = mesh_cfg.get("wireframe_name") or f"{name}-wire"
    surface_sort = int(mesh_cfg.get("surface_sort", mesh_cfg.get("sort", 0)))
    wire_sort = int(mesh_cfg.get("wireframe_sort", surface_sort + 1))

    surface_view = (
        MeshSurfaceView(
            primitive.mesh,
            name=surface_name,
            sort=surface_sort,
            material=material,
            two_sided=two_sided,
            collide_mask=pick_mask,
            transform=transform,
        )
        if surface_enabled
        else None
    )
    wireframe_view = (
        MeshWireframeView(
            primitive.mesh,
            name=wire_name,
            sort=wire_sort,
            collide_mask=pick_mask,
            transform=transform,
        )
        if wireframe_enabled
        else None
    )

    if surface_view:
        engine.add_view(surface_view)
    if wireframe_view:
        engine.add_view(wireframe_view)

    return SceneObject(
        name=name,
        primitive=primitive,
        surface=surface_view,
        wireframe=wireframe_view,
    )


def _maybe_add_studio(
    engine,
    studio_cfg: Dict[str, Any],
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    objects: List[SceneObject],
    has_custom_lights: bool,
) -> Optional[StudioView]:
    if studio_cfg is None:
        studio_cfg = {}
    enabled = studio_cfg.get("enabled", True)
    if not enabled:
        return None

    ground_from_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    snap_to_bounds = studio_cfg.get("snap_ground_to_bounds", True)

    ground_height_val = studio_cfg.get("ground_height")
    try:
        ground_height = float(ground_height_val) if ground_height_val is not None else None
    except Exception:
        ground_height = None

    if snap_to_bounds and bounds is not None and ground_height is None:
        ground_from_bounds = bounds

    apply_mat = studio_cfg.get("apply_material_to_meshes", False)
    targets = [obj.primitive.mesh for obj in objects] if apply_mat else None
    material = _material_from_cfg(studio_cfg.get("material"))

    cam_far_cfg = studio_cfg.get("camera_far")
    try:
        camera_far = float(cam_far_cfg) if cam_far_cfg is not None else None
    except Exception:
        camera_far = None

    studio = StudioView(
        name=studio_cfg.get("name"),
        sort=int(studio_cfg.get("sort", -20)),
        ground_size=float(studio_cfg.get("ground_size", 40.0)),
        ground_tiles=int(studio_cfg.get("ground_tiles", 40)),
        checker_light=tuple(studio_cfg.get("checker_light", (0.92, 0.93, 0.96, 1.0))),
        checker_dark=tuple(studio_cfg.get("checker_dark", (0.86, 0.87, 0.90, 1.0))),
        sky_color=_vec4(studio_cfg.get("sky_color", (0.92, 0.95, 1.0, 1.0))),
        add_lights=bool(studio_cfg.get("add_lights", not has_custom_lights)),
        apply_material_to=targets,
        material=material,
        ground_height=ground_height,
        ground_from_bounds=ground_from_bounds,
        ground_margin=float(studio_cfg.get("ground_margin", 0.0)),
        camera_near=float(studio_cfg.get("camera_near", 0.03)),
        camera_far=camera_far,
    )
    engine.add_view(studio)
    return studio


def _apply_lights(session, lights_cfg: Optional[Dict[str, Any]], studio_view: Optional[StudioView]) -> bool:
    if lights_cfg:
        render = session.render
        render.clearLight()

        if "ambient" in lights_cfg:
            amb = lights_cfg.get("ambient")
            if amb not in (None, False):
                ambient = AmbientLight("ambient")
                ambient.setColor(_vec4(amb))
                render.setLight(render.attachNewNode(ambient))
        else:
            ambient = AmbientLight("ambient")
            ambient.setColor(Vec4(0.18, 0.18, 0.22, 1.0))
            render.setLight(render.attachNewNode(ambient))

        directionals = lights_cfg.get("directionals")
        if directionals is None:
            directionals = [
                {"name": "key", "color": (0.85, 0.85, 0.9, 1.0), "hpr": (-35, -45, 0)},
                {"name": "fill", "color": (0.35, 0.35, 0.45, 1.0), "hpr": (60, -20, 0)},
            ]
        for d in directionals:
            light = DirectionalLight(d.get("name", "dir"))
            light.setColor(_vec4(d.get("color", (0.8, 0.8, 0.85, 1.0))))
            node = render.attachNewNode(light)
            hpr = d.get("hpr", (-35, -45, 0))
            node.setHpr(float(hpr[0]), float(hpr[1]), float(hpr[2]))
            render.setLight(node)
        return True

    if studio_view is not None and studio_view.add_lights:
        # StudioView will attach its own light rig during setup.
        return True

    # Fallback: simple ambient + two directionals similar to the examples.
    render = session.render
    render.clearLight()

    ambient = AmbientLight("ambient")
    ambient.setColor(Vec4(0.18, 0.18, 0.22, 1.0))
    render.setLight(render.attachNewNode(ambient))

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
    return True


def _build_custom_items(engine, entries: Any, kind: str) -> List[object]:
    """
    Build custom views/controllers declared in YAML:
      views:
        - factory: mypkg.myview:make_view
          config: {foo: 1}
          enabled: true
      controllers:
        - factory: mypkg.controllers:make_controller
          config: {speed: 10}
    Factories are called with any of these args if present in their signature:
      engine/eng, session/sess, config/cfg/settings, entry
    """
    items: List[object] = []
    if not entries:
        return items
    if isinstance(entries, dict) and "factory" not in entries:
        # allow mapping of name -> entry
        iterable = entries.values()
    elif isinstance(entries, (list, tuple)):
        iterable = entries
    else:
        iterable = [entries]

    for entry in iterable:
        if entry is None:
            continue
        enabled = entry.get("enabled", True) if isinstance(entry, dict) else True
        if not enabled:
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"Custom {kind} entry must be a mapping, got {type(entry)}")
        target = entry.get("factory") or entry.get("target")
        if not target:
            raise ValueError(f"Custom {kind} entry is missing 'factory'")
        factory = _resolve_callable(target)
        cfg = entry.get("config") or {}
        try:
            obj = _call_factory(factory, engine, cfg, entry)
        except Exception as exc:
            raise RuntimeError(f"Failed to build custom {kind} from '{target}': {exc}") from exc
        if obj is None:
            continue
        try:
            if kind == "view":
                engine.add_view(obj)
            elif kind == "controller":
                engine.add_controller(obj)
        except Exception as exc:
            raise RuntimeError(f"Failed to attach custom {kind} from '{target}': {exc}") from exc
        items.append(obj)
    return items


def _call_factory(factory: Callable[..., object], engine, cfg: dict, entry: dict) -> object:
    sig = inspect.signature(factory)
    kwargs: Dict[str, Any] = {}
    for name in sig.parameters.keys():
        if name in ("engine", "eng"):
            kwargs[name] = engine
        elif name in ("session", "sess"):
            kwargs[name] = engine.session
        elif name in ("config", "cfg", "settings"):
            kwargs[name] = cfg
        elif name == "entry":
            kwargs[name] = entry
    return factory(**kwargs)


def _resolve_callable(target: str) -> Callable[..., object]:
    mod_attr = target.replace(":", ".")
    if ":" in target:
        module_name, attr_name = target.split(":", 1)
    else:
        module_name, attr_name = mod_attr.rsplit(".", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, attr_name)
    if not callable(fn):
        raise TypeError(f"Target '{target}' is not callable")
    return fn


def _apply_camera(
    session,
    camera_cfg: Optional[Dict[str, Any]],
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
) -> bool:
    if camera_cfg is None:
        camera_cfg = {}

    pos_cfg = camera_cfg.get("position")
    look_cfg = camera_cfg.get("look_at") or camera_cfg.get("target")
    if pos_cfg is not None and look_cfg is not None:
        try:
            session.base.camera.setPos(_vec3(pos_cfg))
            session.base.camera.lookAt(_vec3(look_cfg))
            return True
        except Exception:
            pass

    auto_frame = camera_cfg.get("auto_frame", True)
    if not auto_frame or bounds is None:
        return False

    direction = camera_cfg.get("offset_dir", (1.4, -2.6, 1.2))
    distance_scale = float(camera_cfg.get("distance_scale", 3.0))
    min_distance = float(camera_cfg.get("min_distance", 1.0))

    pos, look = _camera_from_bounds(bounds, direction, distance_scale, min_distance)
    try:
        session.base.camera.setPos(pos)
        session.base.camera.lookAt(look)
        return True
    except Exception:
        return False


def _camera_from_bounds(
    bounds: Tuple[np.ndarray, np.ndarray],
    direction: Sequence[float],
    distance_scale: float,
    min_distance: float,
) -> Tuple[Vec3, Vec3]:
    mins, maxs = bounds
    center = (mins + maxs) * 0.5
    extent = (maxs - mins)
    radius = float(np.linalg.norm(extent) * 0.5)
    radius = max(radius, float(min_distance))

    dir_vec = Vec3(float(direction[0]), float(direction[1]), float(direction[2]))
    dir_vec.normalize()
    distance = radius * float(distance_scale)

    pos = Vec3(center[0], center[1], center[2]) + dir_vec * distance
    look = Vec3(center[0], center[1], center[2])
    return pos, look


def _combine_bounds(bounds: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not bounds:
        return None
    mins = np.vstack([b[0] for b in bounds]).min(axis=0)
    maxs = np.vstack([b[1] for b in bounds]).max(axis=0)
    return mins, maxs


def _material_from_cfg(cfg: Optional[Dict[str, Any]]) -> Optional[Material]:
    if not cfg:
        return None
    mat = Material(cfg.get("name", "Material"))
    if "diffuse" in cfg:
        mat.setDiffuse(_vec4(cfg["diffuse"]))
    if "specular" in cfg:
        mat.setSpecular(_vec4(cfg["specular"]))
    if "ambient" in cfg:
        mat.setAmbient(_vec4(cfg["ambient"]))
    if "emission" in cfg:
        mat.setEmission(_vec4(cfg["emission"]))
    if "shininess" in cfg:
        try:
            mat.setShininess(float(cfg["shininess"]))
        except Exception:
            pass
    return mat


def _vec4(values: Sequence[float]) -> Vec4:
    vals = list(values)
    if len(vals) == 3:
        vals.append(1.0)
    if len(vals) != 4:
        raise ValueError(f"Expected 3 or 4 components for color, got {len(vals)}")
    return Vec4(float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))


def _vec3(values: Sequence[float]) -> Vec3:
    vals = list(values)
    if len(vals) != 3:
        raise ValueError(f"Expected 3 components for Vec3, got {len(vals)}")
    return Vec3(float(vals[0]), float(vals[1]), float(vals[2]))


def _transform_from_cfg(cfg: Optional[Dict[str, Any]]) -> Optional[tuple[Optional[Vec3], Optional[Vec3], Optional[tuple[float, float, float]]]]:
    if not cfg:
        return None
    pos = None
    hpr = None
    scale = None
    if "position" in cfg or "translate" in cfg or "translation" in cfg:
        v = cfg.get("position", cfg.get("translate", cfg.get("translation")))
        pos = _vec3(v)
    if "hpr" in cfg or "rotation" in cfg:
        v = cfg.get("hpr", cfg.get("rotation"))
        hpr = _vec3(v)
    if "scale" in cfg:
        s = cfg["scale"]
        if isinstance(s, (int, float)):
            scale = (float(s), float(s), float(s))
        elif isinstance(s, (list, tuple)) and len(s) == 3:
            scale = (float(s[0]), float(s[1]), float(s[2]))
        else:
            raise ValueError(f"Invalid scale value: {s}")
    return (pos, hpr, scale)
