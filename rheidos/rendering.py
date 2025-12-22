from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:  # Panda3D may be missing in some environments
    from direct.filter.CommonFilters import CommonFilters
    from panda3d.core import AmbientLight, DirectionalLight, Fog, Vec4
except Exception:  # pragma: no cover
    CommonFilters = None  # type: ignore
    AmbientLight = None  # type: ignore
    DirectionalLight = None  # type: ignore
    Fog = None  # type: ignore
    Vec4 = None  # type: ignore


@dataclass
class RenderConfig:
    """
    User-facing render configuration that stays decoupled from simulation data.
    """

    backend: str = "fast"
    preset: str = "balanced"
    hdr: bool = True
    exposure: float = 1.0
    bloom: bool = True
    bloom_strength: float = 0.65
    bloom_size: str = "medium"
    ssao: bool = False
    ssao_radius: float = 0.8
    ssao_strength: float = 0.9
    sharpen: float = 0.3
    fog: bool = False
    fog_density: float = 0.02
    fog_color: tuple[float, float, float] = (0.92, 0.95, 1.0)
    light_rig: bool = True
    light_intensity: float = 1.0
    enable_shadows: bool = False
    shadow_map_size: int = 2048


DEFAULT_RENDER_PRESETS: Dict[str, RenderConfig] = {
    "fast": RenderConfig(
        backend="fast",
        preset="fast",
        hdr=False,
        exposure=1.0,
        bloom=False,
        ssao=False,
        sharpen=0.0,
        fog=False,
        enable_shadows=False,
    ),
    "balanced": RenderConfig(
        backend="fast",
        preset="balanced",
        hdr=True,
        exposure=1.0,
        bloom=True,
        bloom_strength=0.65,
        bloom_size="medium",
        ssao=False,
        sharpen=0.35,
        fog=False,
        enable_shadows=False,
    ),
    "quality": RenderConfig(
        backend="fast",
        preset="quality",
        hdr=True,
        exposure=1.05,
        bloom=True,
        bloom_strength=0.82,
        bloom_size="large",
        ssao=True,
        ssao_radius=0.9,
        ssao_strength=1.2,
        sharpen=0.55,
        fog=True,
        fog_density=0.025,
        enable_shadows=True,
        shadow_map_size=4096,
    ),
}


@dataclass
class RenderPipelineConfig:
    """RenderPipeline-specific settings (runtime + startup presets)."""

    backend: str = "renderpipeline"
    preset: str = "quality"
    resolution_scale: float = 1.0
    exposure: float = 1.05
    tonemap: str = "uncharted2"
    ao: bool = True
    bloom: bool = True
    ssr: bool = True
    volumetrics: bool = True
    motion_blur: bool = False
    env_probes: bool = True
    aa: str = "smaa"  # options: smaa, fxaa, none
    shadows: bool = True
    scattering: bool = True
    light_rig: bool = True
    light_intensity: float = 1.0
    enable_shadows: bool = True
    shadow_map_size: int = 2048


DEFAULT_RENDER_PIPELINE_PRESETS: Dict[str, RenderPipelineConfig] = {
    "fast": RenderPipelineConfig(
        backend="renderpipeline",
        preset="fast",
        resolution_scale=0.9,
        exposure=1.0,
        tonemap="uncharted2",
        ao=False,
        bloom=False,
        ssr=False,
        volumetrics=False,
        motion_blur=False,
        env_probes=False,
        aa="fxaa",
        shadows=False,
        scattering=False,
        light_intensity=1.0,
        enable_shadows=False,
    ),
    "balanced": RenderPipelineConfig(
        backend="renderpipeline",
        preset="balanced",
        resolution_scale=1.0,
        exposure=1.05,
        tonemap="uncharted2",
        ao=True,
        bloom=True,
        ssr=False,
        volumetrics=False,
        motion_blur=False,
        env_probes=True,
        aa="smaa",
        shadows=True,
        scattering=True,
        light_intensity=1.1,
        enable_shadows=True,
    ),
    "quality": RenderPipelineConfig(
        backend="renderpipeline",
        preset="quality",
        resolution_scale=1.0,
        exposure=1.12,
        tonemap="optimized",
        ao=True,
        bloom=True,
        ssr=True,
        volumetrics=True,
        motion_blur=True,
        env_probes=True,
        aa="smaa",
        shadows=True,
        scattering=True,
        light_intensity=1.2,
        enable_shadows=True,
        shadow_map_size=4096,
    ),
}


class _RendererBackend:
    name: str
    display_name: str

    def apply_preset(self, name: str) -> Any:
        raise NotImplementedError

    def update_config(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def apply_config(self, config: Optional[Any] = None) -> Any:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError

    @property
    def config(self) -> Any:  # pragma: no cover - protocol-like
        raise NotImplementedError

    @property
    def preset_names(self) -> list[str]:
        raise NotImplementedError

    @property
    def filters_available(self) -> bool:
        return False

    def availability(self) -> Tuple[bool, Optional[str]]:
        return True, None


class _FastRendererBackend(_RendererBackend):
    name = "fast"
    display_name = "CommonFilters"

    def __init__(
        self,
        session: Any,
        config: Optional[RenderConfig] = None,
        presets: Optional[Dict[str, RenderConfig]] = None,
    ) -> None:
        self._session = session
        self._config = config or RenderConfig()
        self._presets: Dict[str, RenderConfig] = dict(presets or DEFAULT_RENDER_PRESETS)

        self._filters: Any = None
        self._fog: Any = None
        self._fog_np: Any = None

        self.apply_config(self._config)

    # --- protocol props ---
    @property
    def config(self) -> RenderConfig:
        return self._config

    @property
    def preset_names(self) -> list[str]:
        return list(self._presets.keys())

    @property
    def filters_available(self) -> bool:
        return CommonFilters is not None and hasattr(self._session, "base")

    # --- API ---
    def apply_preset(self, name: str) -> RenderConfig:
        preset = self._presets.get(name)
        if preset is None:
            return self._config
        cfg = replace(preset, preset=name, backend=self.name)
        self.apply_config(cfg)
        return cfg

    def update_config(self, **kwargs: Any) -> RenderConfig:
        cfg = replace(self._config, **kwargs)
        if "preset" not in kwargs:
            cfg = replace(cfg, preset="custom")
        cfg = replace(cfg, backend=self.name)
        self.apply_config(cfg)
        return cfg

    def apply_config(self, config: Optional[RenderConfig] = None) -> RenderConfig:
        if config is not None:
            self._config = config

        cfg = self._config
        self._apply_filters(cfg)
        self._apply_fog(cfg)
        return cfg

    def shutdown(self) -> None:
        if self._filters is not None:
            try:
                self._filters.cleanup()
            except Exception:
                pass
        self._filters = None
        self._disable_fog()

    # --- internals ---
    def _apply_filters(self, cfg: RenderConfig) -> None:
        self._rebuild_filters()
        if self._filters is None:
            return

        if cfg.hdr:
            self._call_filter("setHighDynamicRange")
            self._call_filter("setExposureAdjust", cfg.exposure)
        else:
            self._call_filter("delHighDynamicRange")

        if cfg.bloom:
            self._call_filter(
                "setBloom",
                blend=(0.25, 0.35, 0.3, 0.0),
                desat=0.6,
                intensity=max(0.0, cfg.bloom_strength),
                size=cfg.bloom_size,
            )
        else:
            self._call_filter("delBloom")

        if cfg.ssao:
            self._call_filter(
                "setAmbientOcclusion",
                num_samples=16,
                radius=max(0.05, cfg.ssao_radius),
                strength=max(0.0, cfg.ssao_strength),
            )
        else:
            self._call_filter("delAmbientOcclusion")

        if abs(cfg.sharpen) > 1e-3:
            self._call_filter("setBlurSharpen", cfg.sharpen)
        else:
            self._call_filter("delBlurSharpen")

    def _rebuild_filters(self) -> None:
        if self._filters is not None:
            try:
                self._filters.cleanup()
            except Exception:
                pass
        if CommonFilters is None:
            self._filters = None
            return
        try:
            self._filters = CommonFilters(
                self._session.base.win, getattr(self._session.base, "cam", None)
            )
        except Exception:
            self._filters = None

    def _call_filter(self, name: str, *args: Any, **kwargs: Any) -> None:
        fn = getattr(self._filters, name, None)
        if not callable(fn):
            return
        try:
            fn(*args, **kwargs)
        except Exception:
            pass

    def _apply_fog(self, cfg: RenderConfig) -> None:
        if not cfg.fog or Fog is None:
            self._disable_fog()
            return
        try:
            if self._fog is None:
                self._fog = Fog("rheidos-fog")
            if self._fog_np is None:
                self._fog_np = self._session.render.attachNewNode(self._fog)
            self._fog.setColor(*cfg.fog_color)
            self._fog.setExpDensity(max(0.0, cfg.fog_density))
            self._session.render.setFog(self._fog_np)
        except Exception:
            self._disable_fog()

    def _disable_fog(self) -> None:
        if self._fog is None:
            return
        try:
            self._session.render.clearFog()
            if self._fog_np is not None:
                self._fog_np.removeNode()
        except Exception:
            pass
        self._fog = None
        self._fog_np = None


def _default_rp_path() -> Path:
    base = Path(__file__).resolve().parents[1] / "third_party"
    for name in ("RenderPipeline", "render_pipeline", "renderpipeline"):
        candidate = base / name
        if candidate.exists():
            return candidate
    return base / "RenderPipeline"


class _RenderPipelineBackend(_RendererBackend):
    name = "renderpipeline"
    display_name = "RenderPipeline"

    def __init__(
        self,
        session: Any,
        config: Optional[RenderPipelineConfig] = None,
        presets: Optional[Dict[str, RenderPipelineConfig]] = None,
        render_pipeline_path: Optional[Path] = None,
    ) -> None:
        self._session = session
        self._config = config or RenderPipelineConfig()
        self._presets: Dict[str, RenderPipelineConfig] = dict(
            presets or DEFAULT_RENDER_PIPELINE_PRESETS
        )
        base_path = render_pipeline_path or _default_rp_path()
        self._rp_path = Path(base_path)
        self._runtime_config_dir = self._rp_path / "_rheidos_config"
        self._runtime_temp_dir = self._rp_path / "_rheidos_temp"

        self._pipeline: Any = None
        self._status_error: Optional[str] = None
        self._last_plugin_mask: Optional[Dict[str, bool]] = None
        self._last_resolution_scale: Optional[float] = None

    # --- protocol props ---
    @property
    def config(self) -> RenderPipelineConfig:
        return self._config

    @property
    def preset_names(self) -> list[str]:
        return list(self._presets.keys())

    def availability(self) -> Tuple[bool, Optional[str]]:
        ok, reason = self._compute_availability()
        return ok, reason

    # --- API ---
    def apply_preset(self, name: str) -> RenderPipelineConfig:
        preset = self._presets.get(name)
        if preset is None:
            return self._config
        cfg = replace(preset, preset=name, backend=self.name)
        self.apply_config(cfg)
        return cfg

    def update_config(self, **kwargs: Any) -> RenderPipelineConfig:
        cfg = replace(self._config, **kwargs)
        if "preset" not in kwargs:
            cfg = replace(cfg, preset="custom")
        cfg = replace(cfg, backend=self.name)
        self.apply_config(cfg)
        return cfg

    def apply_config(self, config: Optional[RenderPipelineConfig] = None) -> RenderPipelineConfig:
        if config is not None:
            self._config = config

        ok, reason = self._compute_availability()
        self._status_error = None if ok else reason
        if not ok:
            self.shutdown()
            return self._config

        cfg = self._config
        plugin_mask = self._plugin_mask(cfg)
        needs_rebuild = (
            self._pipeline is None
            or plugin_mask != self._last_plugin_mask
            or (self._last_resolution_scale is None)
            or abs(self._last_resolution_scale - float(cfg.resolution_scale)) > 1e-4
        )

        if needs_rebuild:
            created = self._recreate_pipeline(cfg, plugin_mask)
            if not created:
                return self._config

        self._apply_runtime_settings(cfg)
        return self._config

    def shutdown(self) -> None:
        if self._pipeline is None:
            return
        try:
            self._pipeline.plugin_mgr.unload()
        except Exception:
            pass
        try:
            task_mgr = getattr(self._session, "task_mgr", getattr(self._session, "taskMgr", None))
            if task_mgr is not None:
                for name in (
                    "RP_UpdateManagers",
                    "RP_Plugin_BeforeRender",
                    "RP_Plugin_AfterRender",
                    "RP_UpdateInputsAndStages",
                    "RP_ClearStateCache",
                ):
                    try:
                        task_mgr.remove(name)
                    except Exception:
                        continue
        except Exception:
            pass
        self._pipeline = None

    # --- internals ---
    def _compute_availability(self) -> Tuple[bool, Optional[str]]:
        path = self._rp_path
        if not path.exists():
            return False, f"RenderPipeline path missing at {path}"
        if not (path / "rpcore").exists():
            return False, "RenderPipeline vendor copy not found (missing rpcore package)"
        if not (path / "config" / "pipeline.yaml").exists():
            return False, "RenderPipeline config folder missing (expected config/pipeline.yaml)"
        if not (path / "data" / "install.flag").exists():
            return False, "RenderPipeline assets not set up (run setup.py in third_party/RenderPipeline)"
        try:
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            from rpcore import render_pipeline as _rp_mod  # noqa: F401
        except Exception as exc:  # pragma: no cover - import guard
            return False, f"Failed to import RenderPipeline: {exc}"
        return True, None

    def _plugin_mask(self, cfg: RenderPipelineConfig) -> Dict[str, bool]:
        mask = {
            "color_correction": True,
            "forward_shading": True,
            "ao": bool(cfg.ao),
            "bloom": bool(cfg.bloom),
            "ssr": bool(cfg.ssr),
            "volumetrics": bool(cfg.volumetrics),
            "motion_blur": bool(cfg.motion_blur),
            "env_probes": bool(cfg.env_probes),
            "scattering": bool(cfg.scattering),
            "pssm": bool(cfg.shadows),
            "skin_shading": True,
            "sky_ao": True,
        }
        mask["smaa"] = cfg.aa == "smaa"
        mask["fxaa"] = cfg.aa == "fxaa"
        return mask

    def _prepare_config_dir(
        self, cfg: RenderPipelineConfig, plugin_mask: Dict[str, bool]
    ) -> Optional[Path]:
        try:
            if str(self._rp_path) not in sys.path:
                sys.path.insert(0, str(self._rp_path))
            from rplibs.yaml import load_yaml_file  # type: ignore
            if sys.version_info < (3, 0):
                from rplibs.yaml.yaml_py2 import safe_dump  # type: ignore
            else:
                from rplibs.yaml.yaml_py3 import safe_dump  # type: ignore
        except Exception:
            return None

        base_cfg_dir = self._rp_path / "config"
        target = self._runtime_config_dir
        target.mkdir(parents=True, exist_ok=True)

        # Copy static config files once (they rarely change)
        for fname in ("daytime.yaml", "debugging.yaml", "panda3d-config.prc", "stages.yaml", "task-scheduler.yaml"):
            src = base_cfg_dir / fname
            dst = target / fname
            if src.exists():
                if not dst.exists():
                    shutil.copy2(src, dst)

        # pipeline.yaml with resolution scale override
        try:
            pipeline_data = load_yaml_file(str(base_cfg_dir / "pipeline.yaml"))
            if "pipeline" in pipeline_data:
                pipeline_data["pipeline"]["resolution_scale"] = float(cfg.resolution_scale)
            with (target / "pipeline.yaml").open("w") as handle:
                safe_dump(pipeline_data, handle, default_flow_style=False, sort_keys=False)
        except Exception:
            return None

        # plugins.yaml with enabled list + keep overrides for enabled plugins
        try:
            plugins_data = load_yaml_file(str(base_cfg_dir / "plugins.yaml"))
        except Exception:
            plugins_data = {}

        enabled = [p for p, enabled_flag in plugin_mask.items() if enabled_flag]
        overrides = plugins_data.get("overrides", {}) if isinstance(plugins_data, dict) else {}
        filtered_overrides = {k: v for k, v in overrides.items() if k in enabled}
        out_plugins = {"enabled": enabled, "overrides": filtered_overrides}
        with (target / "plugins.yaml").open("w") as handle:
            safe_dump(out_plugins, handle, default_flow_style=False, sort_keys=False)

        return target

    def _recreate_pipeline(
        self, cfg: RenderPipelineConfig, plugin_mask: Dict[str, bool]
    ) -> bool:
        self.shutdown()
        config_dir = self._prepare_config_dir(cfg, plugin_mask)
        if config_dir is None:
            self._status_error = "Failed to write RenderPipeline config overlay"
            return False
        try:
            if str(self._rp_path) not in sys.path:
                sys.path.insert(0, str(self._rp_path))
            from rpcore.render_pipeline import RenderPipeline  # type: ignore
        except Exception as exc:
            self._status_error = f"Failed to import RenderPipeline: {exc}"
            return False

        try:
            pipeline = RenderPipeline()
            pipeline.mount_mgr.base_path = str(self._rp_path)
            pipeline.mount_mgr.config_dir = str(config_dir)
            if self._runtime_temp_dir is not None:
                pipeline.mount_mgr.write_path = str(self._runtime_temp_dir)
            pipeline.load_settings(str(config_dir / "pipeline.yaml"))
            pipeline.pre_showbase_init()
            pipeline.create(self._session.base)
        except Exception as exc:  # pragma: no cover - defensive
            self._status_error = f"Failed to start RenderPipeline: {exc}"
            return False

        self._pipeline = pipeline
        self._last_plugin_mask = plugin_mask
        self._last_resolution_scale = float(cfg.resolution_scale)
        self._status_error = None
        return True

    def _apply_runtime_settings(self, cfg: RenderPipelineConfig) -> None:
        if self._pipeline is None:
            return
        self._set_plugin_setting("color_correction", "exposure_scale", float(cfg.exposure))
        self._set_plugin_setting("color_correction", "tonemap_operator", cfg.tonemap)

    def _set_plugin_setting(self, plugin_id: str, setting_id: str, value: Any) -> bool:
        try:
            pmgr = self._pipeline.plugin_mgr
            if plugin_id not in pmgr.settings:
                return False
            if setting_id not in pmgr.settings[plugin_id]:
                return False
            pmgr.on_setting_changed(plugin_id, setting_id, value)
            return True
        except Exception:
            return False


class _LightRigController:
    def __init__(self, session: Any) -> None:
        self._session = session
        self._light_root: Any = None
        self._lights: Dict[str, Any] = {}

    def update(
        self,
        enabled: bool,
        intensity: float,
        enable_shadows: bool,
        shadow_map_size: int,
    ) -> None:
        if not enabled:
            self.clear()
            return
        if AmbientLight is None or DirectionalLight is None:
            return
        if self._light_root is None:
            try:
                self._light_root = self._session.render.attachNewNode("renderer-light-rig")
            except Exception:
                self._light_root = None
        if self._light_root is None:
            return

        if "ambient" not in self._lights:
            amb = AmbientLight("renderer-ambient")
            amb.setColor(Vec4(0.22, 0.22, 0.26, 1.0))
            self._lights["ambient"] = self._light_root.attachNewNode(amb)
            self._session.render.setLight(self._lights["ambient"])

        if "key" not in self._lights:
            key = DirectionalLight("renderer-key")
            key.setColor(Vec4(0.9, 0.9, 0.96, 1.0))
            key_np = self._light_root.attachNewNode(key)
            key_np.setHpr(-35, -45, 0)
            key_np.setPos(6.0, -10.0, 8.0)
            self._lights["key"] = key_np
            self._session.render.setLight(key_np)

        if "fill" not in self._lights:
            fill = DirectionalLight("renderer-fill")
            fill.setColor(Vec4(0.35, 0.38, 0.45, 1.0))
            fill_np = self._light_root.attachNewNode(fill)
            fill_np.setHpr(60, -20, 0)
            self._lights["fill"] = fill_np
            self._session.render.setLight(fill_np)

        base_colors = {
            "ambient": Vec4(0.22, 0.22, 0.26, 1.0),
            "key": Vec4(0.9, 0.9, 0.96, 1.0),
            "fill": Vec4(0.35, 0.38, 0.45, 1.0),
        }
        intensity_scale = max(0.0, float(intensity))
        for name, np in self._lights.items():
            try:
                light_node = np.node()
                base_c = base_colors.get(name) or light_node.getColor()
                light_node.setColor(
                    Vec4(
                        base_c.x * intensity_scale,
                        base_c.y * intensity_scale,
                        base_c.z * intensity_scale,
                        base_c.w,
                    )
                )
            except Exception:
                continue

        self._configure_shadows(enable_shadows, shadow_map_size)

    def _configure_shadows(self, enable_shadows: bool, shadow_map_size: int) -> None:
        key_np = self._lights.get("key")
        if key_np is None:
            return
        try:
            key = key_np.node()
        except Exception:
            return
        try:
            key.setShadowCaster(bool(enable_shadows), int(shadow_map_size), int(shadow_map_size))
        except Exception:
            try:
                key.setShadowCaster(False)
            except Exception:
                pass

    def clear(self) -> None:
        if not self._lights:
            return
        for np in self._lights.values():
            try:
                self._session.render.clearLight(np)
                np.removeNode()
            except Exception:
                pass
        self._lights.clear()
        if self._light_root is not None:
            try:
                self._light_root.removeNode()
            except Exception:
                pass
        self._light_root = None


class Renderer:
    """
    Lightweight render pipeline wrapper.
    Supports a fast CommonFilters backend and an optional RenderPipeline backend.
    """

    def __init__(
        self,
        session: Any,
        config: Optional[RenderConfig] = None,
        presets: Optional[Dict[str, RenderConfig]] = None,
        render_pipeline_config: Optional[RenderPipelineConfig] = None,
        render_pipeline_presets: Optional[Dict[str, RenderPipelineConfig]] = None,
        render_pipeline_path: Optional[Path] = None,
        default_backend: str = "fast",
    ) -> None:
        self._session = session
        self._light_rig = _LightRigController(session)

        self._fast_backend = _FastRendererBackend(session, config=config, presets=presets)
        self._rp_backend = _RenderPipelineBackend(
            session,
            config=render_pipeline_config,
            presets=render_pipeline_presets,
            render_pipeline_path=render_pipeline_path,
        )
        self._backends: Dict[str, _RendererBackend] = {
            self._fast_backend.name: self._fast_backend,
            self._rp_backend.name: self._rp_backend,
        }

        desired_backend = default_backend if default_backend in self._backends else "fast"
        self._active_backend: _RendererBackend = self._select_backend(desired_backend)
        self.apply_config()

    # --- properties ---
    @property
    def config(self) -> Any:
        return self._active_backend.config

    @property
    def fast_config(self) -> RenderConfig:
        return self._fast_backend.config

    @property
    def rp_config(self) -> RenderPipelineConfig:
        return self._rp_backend.config

    @property
    def preset_names(self) -> list[str]:
        return self._active_backend.preset_names

    @property
    def backend_name(self) -> str:
        return self._active_backend.name

    @property
    def filters_available(self) -> bool:
        return self._active_backend.filters_available

    @property
    def render_pipeline_path(self) -> Optional[Path]:
        try:
            return self._rp_backend._rp_path  # type: ignore[attr-defined]
        except Exception:
            return None

    def backend_availability(self, name: str) -> Tuple[bool, Optional[str]]:
        backend = self._backends.get(name)
        if backend is None:
            return False, "Unknown backend"
        return backend.availability()

    # --- panels ---
    def panel_factory(self) -> Any:
        """Convenience: return ImGui RenderSettings panel."""
        try:
            from .ui.panels.render_settings import renderer_panel_factory
        except Exception:
            return lambda *_args, **_kwargs: None
        return renderer_panel_factory(self)

    def render_pipeline_panel_factory(self) -> Any:
        """Separate panel for RenderPipeline controls."""
        try:
            from .ui.panels.render_pipeline import render_pipeline_panel_factory
        except Exception:
            return lambda *_args, **_kwargs: None
        return render_pipeline_panel_factory(self)

    # --- mutations ---
    def set_backend(self, name: str) -> str:
        self._active_backend = self._select_backend(name)
        self.apply_config()
        return self._active_backend.name

    def apply_preset(self, name: str) -> Any:
        cfg = self._active_backend.apply_preset(name)
        self._apply_light_rig(cfg)
        return cfg

    def update_config(self, **kwargs: Any) -> Any:
        cfg = self._active_backend.update_config(**kwargs)
        self._apply_light_rig(cfg)
        return cfg

    def apply_config(self, config: Optional[Any] = None) -> Any:
        cfg = self._active_backend.apply_config(config)
        self._apply_light_rig(cfg)
        return cfg

    def shutdown(self) -> None:
        for backend in self._backends.values():
            backend.shutdown()
        self._light_rig.clear()

    # --- internals ---
    def _select_backend(self, name: str) -> _RendererBackend:
        backend = self._backends.get(name)
        if backend is None:
            return self._fast_backend
        ok, _ = backend.availability()
        if not ok and backend is self._rp_backend:
            return self._fast_backend
        return backend

    def _apply_light_rig(self, cfg: Any) -> None:
        if not hasattr(cfg, "light_rig"):
            return
        self._light_rig.update(
            enabled=bool(getattr(cfg, "light_rig", False)),
            intensity=float(getattr(cfg, "light_intensity", 1.0)),
            enable_shadows=bool(getattr(cfg, "enable_shadows", False)),
            shadow_map_size=int(getattr(cfg, "shadow_map_size", 2048)),
        )
