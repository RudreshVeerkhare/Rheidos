from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from panda3d.core import BitMask32

from . import scene_config as sc
from .scene_config import SceneConfigResult, SceneObject


@dataclass
class DiffSummary:
    added_meshes: List[str]
    removed_meshes: List[str]
    updated_meshes: List[str]
    lights_changed: bool
    camera_changed: bool
    studio_changed: bool
    custom_views_changed: bool
    custom_controllers_changed: bool
    background_changed: bool
    forced: bool = False

    def describe(self) -> str:
        parts: List[str] = []
        if self.added_meshes:
            parts.append(f"+{len(self.added_meshes)} meshes")
        if self.removed_meshes:
            parts.append(f"-{len(self.removed_meshes)} meshes")
        if self.updated_meshes:
            parts.append(f"~{len(self.updated_meshes)} meshes")
        if self.studio_changed:
            parts.append("studio")
        if self.lights_changed:
            parts.append("lights")
        if self.camera_changed:
            parts.append("camera")
        if self.background_changed:
            parts.append("background")
        if self.custom_views_changed:
            parts.append("views")
        if self.custom_controllers_changed:
            parts.append("controllers")
        if not parts:
            parts.append("no-op")
        label = ", ".join(parts)
        return f"(forced) {label}" if self.forced else label


class SceneConfigLiveManager:
    """
    Applies scene-config edits incrementally:
      - Meshes are added/removed/rebuilt based on name identity.
      - Studio/lights/camera/background/custom components are reapplied when their config changes.
      - Provides a force-reload path that tears down everything and rebuilds from scratch.
    """

    def __init__(
        self,
        engine,
        config_path: Path,
        *,
        default_pick_mask: BitMask32 = BitMask32.bit(4),
        initial_cfg: Optional[Dict[str, Any]] = None,
        initial_result: Optional[SceneConfigResult] = None,
    ) -> None:
        self.engine = engine
        self.config_path = Path(config_path).expanduser()
        self.default_pick_mask = default_pick_mask
        self._current_cfg: Dict[str, Any] = deepcopy(initial_cfg) if initial_cfg else {}
        self._result: Optional[SceneConfigResult] = initial_result

    # --- Public API -------------------------------------------------

    def apply_text(self, text: str, *, force: bool = False) -> DiffSummary:
        """
        Parse the provided config text (YAML/JSON) and apply it.
        If force=True or no existing scene, do a full rebuild.
        Must be called on the render thread.
        """
        new_cfg = self._parse_text(text)
        if force or self._result is None:
            summary = self._force_reload(new_cfg)
            summary.forced = True
            return summary
        return self._apply_diff(new_cfg)

    def reload_from_disk(self, *, force: bool = True) -> DiffSummary:
        text = self.config_path.read_text()
        return self.apply_text(text, force=force)

    # --- Internals --------------------------------------------------

    def _parse_text(self, text: str) -> Dict[str, Any]:
        suffix = self.config_path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            if sc.yaml is None:
                raise RuntimeError("PyYAML is required for YAML scene configs.")
            data = sc.yaml.safe_load(text)
        else:
            data = json.loads(text)
        return data or {}

    def _force_reload(self, new_cfg: Dict[str, Any]) -> DiffSummary:
        self._teardown_current()
        result = sc.build_scene_from_data(
            self.engine, deepcopy(new_cfg), self.config_path, default_pick_mask=self.default_pick_mask
        )
        self._result = result
        self._current_cfg = deepcopy(new_cfg)
        meshes_cfg = new_cfg.get("meshes") or new_cfg.get("models") or []
        mesh_names = [self._mesh_key(cfg, idx) for idx, cfg in enumerate(meshes_cfg)]
        return DiffSummary(
            added_meshes=mesh_names,
            removed_meshes=[],
            updated_meshes=[],
            lights_changed=True,
            camera_changed=True,
            studio_changed=True,
            custom_views_changed=True,
            custom_controllers_changed=True,
            background_changed=True,
            forced=True,
        )

    def _apply_diff(self, new_cfg: Dict[str, Any]) -> DiffSummary:
        assert self._result is not None
        old_cfg = self._current_cfg
        old_meshes_cfg = old_cfg.get("meshes") or old_cfg.get("models") or []
        new_meshes_cfg = new_cfg.get("meshes") or new_cfg.get("models") or []
        old_map, old_order = self._mesh_map(old_meshes_cfg)
        new_map, new_order = self._mesh_map(new_meshes_cfg)

        removed_names = [n for n in old_order if n not in new_map]
        added_names = [n for n in new_order if n not in old_map]
        updated_names: List[str] = []

        # Detect changed meshes (same name, config differs)
        for name in new_order:
            if name in added_names or name in removed_names or name not in old_map:
                continue
            old_sig = self._mesh_signature(old_map[name][0])
            new_sig = self._mesh_signature(new_map[name][0])
            if old_sig != new_sig:
                updated_names.append(name)

        # Remove deleted/changed meshes
        for name in removed_names + updated_names:
            self._remove_mesh_by_name(name)

        # Build added/changed meshes
        for name in added_names + updated_names:
            cfg, idx = new_map[name]
            obj = sc._build_mesh(
                self.engine,
                self.config_path,
                cfg,
                idx,
                self.default_pick_mask,
            )
            self._result.objects.append(obj)

        # Reorder objects to match new config order
        objects_by_name = {o.name: o for o in self._result.objects}
        new_objects: List[SceneObject] = []
        bounds_list: List[Tuple[Any, Any]] = []
        for name in new_order:
            obj = objects_by_name.get(name)
            if obj is None:
                continue
            new_objects.append(obj)
            bounds_list.append(obj.primitive.bounds)
        self._result.objects = new_objects
        bounds = sc._combine_bounds(bounds_list)
        self._result.bounds = bounds

        # Studio handling
        studio_changed = self._section_changed("studio", old_cfg, new_cfg)
        if studio_changed:
            if self._result.studio_view is not None:
                try:
                    self.engine.remove_view(self._result.studio_view.name)
                except Exception:
                    pass
                self._result.studio_view = None
            self._result.studio_view = sc._maybe_add_studio(
                self.engine, new_cfg.get("studio", {}), bounds, self._result.objects, bool(new_cfg.get("lights"))
            )
        else:
            if (
                self._result.studio_view is not None
                and bounds is not None
                and isinstance(new_cfg.get("studio"), dict)
            ):
                studio_cfg = new_cfg.get("studio", {})
                snap = studio_cfg.get("snap_ground_to_bounds", True)
                ground_height = studio_cfg.get("ground_height")
                if snap and ground_height is None:
                    try:
                        margin = float(studio_cfg.get("ground_margin", 0.0))
                    except Exception:
                        margin = 0.0
                    try:
                        self._result.studio_view.snap_ground_to_bounds(bounds, margin=margin)
                    except Exception:
                        pass

        # Lights
        lights_changed = self._section_changed("lights", old_cfg, new_cfg) or studio_changed
        if lights_changed:
            self._result.lights_applied = sc._apply_lights(
                self.engine.session, new_cfg.get("lights"), self._result.studio_view
            )

        # Camera
        camera_changed = self._section_changed("camera", old_cfg, new_cfg) or (
            bounds is not None and new_cfg.get("camera", {}).get("auto_frame", True)
        )
        if camera_changed:
            self._result.camera_applied = sc._apply_camera(self.engine.session, new_cfg.get("camera"), bounds)

        # Background color
        bg_changed = self._section_changed("background_color", old_cfg, new_cfg)
        if bg_changed:
            bg = new_cfg.get("background_color")
            if bg is not None:
                try:
                    self.engine.session.base.setBackgroundColor(*sc._vec4(bg))
                except Exception:
                    pass

        # Custom views/controllers (replace wholesale on change)
        views_changed = self._section_changed("views", old_cfg, new_cfg)
        ctrls_changed = self._section_changed("controllers", old_cfg, new_cfg)
        if views_changed:
            self._remove_custom_views()
            self._result.custom_views = sc._build_custom_items(self.engine, new_cfg.get("views"), kind="view")
        if ctrls_changed:
            self._remove_custom_controllers()
            self._result.custom_controllers = sc._build_custom_items(
                self.engine, new_cfg.get("controllers"), kind="controller"
            )

        self._current_cfg = deepcopy(new_cfg)
        return DiffSummary(
            added_meshes=added_names,
            removed_meshes=removed_names,
            updated_meshes=updated_names,
            lights_changed=lights_changed,
            camera_changed=camera_changed,
            studio_changed=studio_changed,
            custom_views_changed=views_changed,
            custom_controllers_changed=ctrls_changed,
            background_changed=bg_changed,
            forced=False,
        )

    def _teardown_current(self) -> None:
        if self._result is None:
            return
        for obj in list(self._result.objects):
            self._remove_mesh(obj)
        self._remove_custom_views()
        self._remove_custom_controllers()
        if self._result.studio_view is not None:
            try:
                self.engine.remove_view(self._result.studio_view.name)
            except Exception:
                pass
        self._result = None

    def _remove_mesh_by_name(self, name: str) -> None:
        if self._result is None:
            return
        obj = next((o for o in self._result.objects if o.name == name), None)
        if obj:
            self._remove_mesh(obj)
            self._result.objects = [o for o in self._result.objects if o.name != name]

    def _remove_mesh(self, obj: SceneObject) -> None:
        if obj.surface is not None:
            try:
                self.engine.remove_view(obj.surface.name)
            except Exception:
                pass
        if obj.wireframe is not None:
            try:
                self.engine.remove_view(obj.wireframe.name)
            except Exception:
                pass

    def _remove_custom_views(self) -> None:
        if self._result is None:
            return
        for v in self._result.custom_views:
            name = getattr(v, "name", None)
            if name:
                try:
                    self.engine.remove_view(name)
                except Exception:
                    pass
        self._result.custom_views = []

    def _remove_custom_controllers(self) -> None:
        if self._result is None:
            return
        for c in self._result.custom_controllers:
            name = getattr(c, "name", None)
            if name:
                try:
                    self.engine.remove_controller(name)
                except Exception:
                    pass
        self._result.custom_controllers = []

    # --- Helpers ----------------------------------------------------

    def _mesh_map(self, meshes_cfg: List[Dict[str, Any]]) -> Tuple[Dict[str, Tuple[Dict[str, Any], int]], List[str]]:
        mapping: Dict[str, Tuple[Dict[str, Any], int]] = {}
        order: List[str] = []
        for idx, cfg in enumerate(meshes_cfg):
            name = self._mesh_key(cfg, idx)
            mapping[name] = (cfg, idx)
            order.append(name)
        return mapping, order

    def _mesh_key(self, cfg: Dict[str, Any], idx: int) -> str:
        name = cfg.get("name")
        if name:
            return str(name)
        raw_path = cfg.get("path") or cfg.get("file") or f"mesh{idx}"
        try:
            return Path(str(raw_path)).stem or f"mesh{idx}"
        except Exception:
            return f"mesh{idx}"

    def _mesh_signature(self, cfg: Dict[str, Any]) -> str:
        # Normalize path to absolute to avoid diff noise when relative changes but resolves the same.
        norm = deepcopy(cfg)
        raw_path = norm.get("path") or norm.get("file")
        if raw_path is not None:
            try:
                norm["__abs_path__"] = str((self.config_path.parent / Path(str(raw_path))).expanduser().resolve())
            except Exception:
                norm["__abs_path__"] = str(raw_path)
        norm.pop("name", None)  # identity handled separately
        try:
            return json.dumps(norm, sort_keys=True, default=str)
        except Exception:
            return repr(norm)

    def _section_changed(self, key: str, old_cfg: Dict[str, Any], new_cfg: Dict[str, Any]) -> bool:
        old_val = old_cfg.get(key)
        new_val = new_cfg.get(key)
        try:
            return json.dumps(old_val, sort_keys=True, default=str) != json.dumps(new_val, sort_keys=True, default=str)
        except Exception:
            return old_val != new_val
