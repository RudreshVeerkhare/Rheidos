from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from ...rendering import Renderer

if TYPE_CHECKING:  # Avoid runtime circular import
    from ..imgui_manager import PanelFactory


class RenderPipelinePanel:
    """ImGui controls dedicated to the RenderPipeline backend."""

    id = "render-pipeline"
    title = "Render Pipeline"
    order = -49
    separate_window = True

    def __init__(self, renderer: Renderer) -> None:
        self._renderer = renderer

    def draw(self, imgui: Any) -> None:
        available, reason = self._renderer.backend_availability("renderpipeline")
        active = self._renderer.backend_name == "renderpipeline"
        cfg = self._renderer.rp_config

        status = "Active" if active else "Idle"
        imgui.text(f"RenderPipeline status: {status}")
        rp_path = getattr(self._renderer, "render_pipeline_path", None)
        if rp_path:
            imgui.text_disabled(f"Path: {rp_path}")
        if not available and reason:
            imgui.text_colored(reason, 1.0, 0.6, 0.2)

        if not available:
            imgui.separator()
            imgui.text("Install a vendor copy to enable:")
            imgui.bullet_text("Clone latest RenderPipeline into third_party/RenderPipeline")
            imgui.bullet_text("Run `python setup.py install` inside that folder (creates data/install.flag)")
            return

        if not active:
            if imgui.button("Switch to RenderPipeline"):
                self._renderer.set_backend("renderpipeline")
                active = True
            imgui.same_line()
            imgui.text_disabled("Fast renderer stays default; toggle here when ready.")
            if not active:
                return

        imgui.separator()
        imgui.text("Quality Preset")
        active_preset = cfg.preset or "custom"
        if imgui.begin_combo("##rp-preset", active_preset):
            for name in self._renderer.preset_names:
                selected = name == active_preset
                if _clicked(imgui.selectable(name, selected)):
                    self._renderer.apply_preset(name)
                    active_preset = name
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()

        imgui.separator()
        imgui.text("Core")
        changed, exposure = imgui.slider_float("Exposure", cfg.exposure, 0.4, 2.5)
        if changed:
            cfg = self._renderer.update_config(exposure=float(exposure))
        changed, res_scale = imgui.slider_float("Resolution Scale", cfg.resolution_scale, 0.5, 1.5)
        if changed:
            cfg = self._renderer.update_config(resolution_scale=float(res_scale))
        imgui.text_disabled("Resolution scale triggers an RP rebuild")
        if imgui.begin_combo("Tonemap", cfg.tonemap):
            for name in ("optimized", "uncharted2", "reinhard", "exponential", "none"):
                selected = name == cfg.tonemap
                if _clicked(imgui.selectable(name, selected)):
                    cfg = self._renderer.update_config(tonemap=name)
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()

        imgui.separator()
        imgui.text("Effects")
        imgui.text_disabled("Most toggles rebuild the RP stack")
        for label, field in (
            ("Ambient Occlusion", "ao"),
            ("Bloom", "bloom"),
            ("SSR", "ssr"),
            ("Volumetrics", "volumetrics"),
            ("Motion Blur", "motion_blur"),
            ("Environment Probes", "env_probes"),
            ("Atmospheric Scattering", "scattering"),
            ("Shadows (PSSM)", "shadows"),
        ):
            current = bool(getattr(cfg, field))
            changed, val = imgui.checkbox(label, current)
            if changed:
                cfg = self._renderer.update_config(**{field: bool(val)})

        current_aa = cfg.aa
        if imgui.begin_combo("Anti-Aliasing", current_aa):
            for aa in ("smaa", "fxaa", "none"):
                selected = aa == current_aa
                if _clicked(imgui.selectable(aa.upper(), selected)):
                    cfg = self._renderer.update_config(aa=aa)
                    current_aa = aa
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()

        imgui.separator()
        imgui.text("Default Light Rig")
        changed, rig = imgui.checkbox("Enable Light Rig", cfg.light_rig)
        if changed:
            cfg = self._renderer.update_config(light_rig=bool(rig))
        if cfg.light_rig:
            imgui.indent()
            changed, intensity = imgui.slider_float("Light Intensity", cfg.light_intensity, 0.0, 2.5)
            if changed:
                cfg = self._renderer.update_config(light_intensity=float(intensity))
            changed, enable_shadows = imgui.checkbox("Shadowed Key", cfg.enable_shadows)
            if changed:
                cfg = self._renderer.update_config(enable_shadows=bool(enable_shadows))
            if cfg.enable_shadows:
                changed, size = imgui.slider_float("Shadow Map", float(cfg.shadow_map_size), 512.0, 8192.0)
                if changed:
                    cfg = self._renderer.update_config(shadow_map_size=int(size))
            imgui.unindent()


def render_pipeline_panel_factory(renderer: Renderer) -> "PanelFactory":
    def factory(session: Any, store: Optional[Any]) -> RenderPipelinePanel:
        return RenderPipelinePanel(renderer)

    return factory


def _clicked(result: Any) -> bool:
    return bool(result[0] if isinstance(result, tuple) else result)
