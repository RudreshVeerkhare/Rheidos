from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from ...rendering import Renderer

if TYPE_CHECKING:  # Avoid runtime circular import
    from ..imgui_manager import PanelFactory


class RenderSettingsPanel:
    """ImGui controls for Renderer presets/effects."""

    id = "render-settings"
    title = "Render Settings"
    order = -50
    separate_window = True

    def __init__(self, renderer: Renderer) -> None:
        self._renderer = renderer

    def draw(self, imgui: Any) -> None:
        backend = self._renderer.backend_name

        imgui.text("Renderer Backend")
        label = "Fast (CommonFilters)" if backend == "fast" else "RenderPipeline"
        if imgui.begin_combo("##renderer-backend", label):
            for name, title in (("fast", "Fast (CommonFilters)"), ("renderpipeline", "RenderPipeline")):
                available, reason = self._renderer.backend_availability(name)
                display = title
                if not available and name != "fast":
                    display = f"{title} (unavailable)"
                selected = name == backend
                if _clicked(imgui.selectable(display, selected)):
                    if not available and name != "fast":
                        backend = self._renderer.backend_name
                        continue
                    chosen = self._renderer.set_backend(name)
                    backend = chosen
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()

        if backend != "fast":
            imgui.separator()
            imgui.text("RenderPipeline active")
            available, reason = self._renderer.backend_availability("renderpipeline")
            if not available and reason:
                imgui.text_disabled(reason)
            imgui.text("Use the Render Pipeline panel for RP-specific controls.")
            return

        cfg = self._renderer.config

        imgui.text("Quality Preset")
        active = cfg.preset or "custom"
        if imgui.begin_combo("##renderer-preset", active):
            for name in self._renderer.preset_names:
                selected = name == active
                if _clicked(imgui.selectable(name, selected)):
                    cfg = self._renderer.apply_preset(name)
                    active = cfg.preset
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()

        imgui.separator()
        imgui.text("Post Effects")
        if not self._renderer.filters_available:
            imgui.text_disabled("CommonFilters unavailable; skipping post stack.")
        else:
            changed, hdr = imgui.checkbox("HDR", cfg.hdr)
            if changed:
                cfg = self._renderer.update_config(hdr=bool(hdr))
            if cfg.hdr:
                changed, exp = imgui.slider_float("Exposure", cfg.exposure, 0.4, 2.5)
                if changed:
                    cfg = self._renderer.update_config(exposure=float(exp))

            changed, bloom = imgui.checkbox("Bloom", cfg.bloom)
            if changed:
                cfg = self._renderer.update_config(bloom=bool(bloom))
            if cfg.bloom:
                changed, strength = imgui.slider_float(
                    "Bloom Strength", cfg.bloom_strength, 0.05, 2.0
                )
                if changed:
                    cfg = self._renderer.update_config(bloom_strength=float(strength))
                current_size = cfg.bloom_size
                if imgui.begin_combo("Bloom Size", current_size):
                    for size in ("small", "medium", "large"):
                        selected = size == current_size
                        if _clicked(imgui.selectable(size, selected)):
                            cfg = self._renderer.update_config(bloom_size=size)
                            current_size = size
                        if selected:
                            imgui.set_item_default_focus()
                    imgui.end_combo()

            changed, ssao = imgui.checkbox("SSAO", cfg.ssao)
            if changed:
                cfg = self._renderer.update_config(ssao=bool(ssao))
            if cfg.ssao:
                changed, radius = imgui.slider_float(
                    "SSAO Radius", cfg.ssao_radius, 0.05, 2.0
                )
                if changed:
                    cfg = self._renderer.update_config(ssao_radius=float(radius))
                changed, strength = imgui.slider_float(
                    "SSAO Strength", cfg.ssao_strength, 0.1, 2.5
                )
                if changed:
                    cfg = self._renderer.update_config(ssao_strength=float(strength))

            changed, sharpen = imgui.slider_float("Sharpen", cfg.sharpen, 0.0, 1.25)
            if changed:
                cfg = self._renderer.update_config(sharpen=float(sharpen))

        imgui.separator()
        imgui.text("Atmosphere")
        changed, fog = imgui.checkbox("Fog", cfg.fog)
        if changed:
            cfg = self._renderer.update_config(fog=bool(fog))
        if cfg.fog:
            changed, density = imgui.slider_float("Fog Density", cfg.fog_density, 0.0, 0.08)
            if changed:
                cfg = self._renderer.update_config(fog_density=float(density))

        imgui.separator()
        imgui.text("Lights")
        changed, rig = imgui.checkbox("Default Light Rig", cfg.light_rig)
        if changed:
            cfg = self._renderer.update_config(light_rig=bool(rig))
        if cfg.light_rig:
            imgui.indent()
            changed, shadows = imgui.checkbox("Shadowed Key", cfg.enable_shadows)
            if changed:
                cfg = self._renderer.update_config(enable_shadows=bool(shadows))
            changed, intensity = imgui.slider_float(
                "Light Intensity", cfg.light_intensity, 0.0, 2.5
            )
            if changed:
                cfg = self._renderer.update_config(light_intensity=float(intensity))
            if cfg.enable_shadows:
                changed, size = imgui.slider_float(
                    "Shadow Map", float(cfg.shadow_map_size), 512.0, 8192.0
                )
                if changed:
                    cfg = self._renderer.update_config(shadow_map_size=int(size))
            imgui.unindent()


def renderer_panel_factory(renderer: Renderer) -> "PanelFactory":
    """Helper: plug into Engine(imgui_panel_factories=...) easily."""

    def factory(session: Any, store: Optional[Any]) -> RenderSettingsPanel:
        return RenderSettingsPanel(renderer)

    return factory


def _clicked(result: Any) -> bool:
    return bool(result[0] if isinstance(result, tuple) else result)
