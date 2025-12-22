from __future__ import annotations

from typing import Any, Callable, Optional

from ..abc.view import View
from ..store import StoreState
from ..visualization.color_schemes import ColorLegend, ColorScheme

LegendProvider = Callable[[], Optional[ColorLegend]]
SchemeProvider = Callable[[], Optional[ColorScheme]]


class LegendView(View):
    """
    HUD legend that renders ColorScheme metadata via imgui.
    """

    def __init__(
        self,
        *,
        legend_provider: Optional[LegendProvider] = None,
        scheme_provider: Optional[SchemeProvider] = None,
        store: Optional[StoreState] = None,
        visible_store_key: Optional[str] = None,
        name: Optional[str] = None,
        sort: int = 100,
    ) -> None:
        if legend_provider is None and scheme_provider is None:
            raise ValueError("LegendView requires a legend_provider or scheme_provider")
        super().__init__(name=name or "LegendView", sort=sort)
        self._legend_provider = legend_provider
        self._scheme_provider = scheme_provider
        self._store = store
        self._visible_store_key = visible_store_key
        self._event_registered = False

    def setup(self, session: Any) -> None:
        super().setup(session)
        # If p3dimgui is driving the frame, hook into its event to ensure we draw inside an ImGui frame.
        try:
            from direct.showbase.MessengerGlobal import messenger

            messenger.accept("imgui-new-frame", self, self._draw)
            self._event_registered = True
        except Exception:
            self._event_registered = False

    def teardown(self) -> None:
        if self._event_registered:
            try:
                from direct.showbase.MessengerGlobal import messenger

                messenger.ignore("imgui-new-frame", self)
            except Exception:
                pass
        self._event_registered = False

    def update(self, dt: float) -> None:
        # If we are registered with imgui-new-frame, rendering is handled in _draw.
        if self._event_registered:
            return
        self._draw()

    def _draw(self) -> None:
        try:
            from imgui_bundle import imgui
        except Exception:
            return

        if self._store is not None and self._visible_store_key:
            visible = bool(self._store.get(self._visible_store_key, True))
            if not visible:
                return

        legend = self._legend_provider() if self._legend_provider else None
        if legend is None and self._scheme_provider:
            scheme = self._scheme_provider()
            legend = scheme.legend() if scheme is not None else None
        if legend is None:
            return

        imgui.set_next_window_bg_alpha(0.78)
        imgui.set_next_window_size((260, 140), imgui.Cond_.first_use_ever)
        imgui.set_next_window_pos((24, 24), imgui.Cond_.first_use_ever)
        opened, _ = imgui.begin(
            self.name,
            True,
            flags=imgui.WindowFlags_.no_nav | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.always_auto_resize,
        )
        if opened:
            imgui.text(legend.title)
            if legend.units:
                imgui.same_line()
                imgui.text_disabled(f"[{legend.units}]")
            if legend.description:
                imgui.text_disabled(legend.description)
            imgui.separator()

            # Gradient stops preview
            for idx, stop in enumerate(legend.stops):
                imgui.color_button(
                    f"##stop-{idx}",
                    stop.color,
                    flags=imgui.ColorEditFlags_.no_picker
                    | imgui.ColorEditFlags_.no_inputs
                    | imgui.ColorEditFlags_.no_tooltip,
                    size=(18, 18),
                )
                imgui.same_line()
                imgui.text(f"{stop.position:.2f}")
            imgui.separator()

            # Tick labels
            for tick in legend.ticks:
                imgui.bullet_text(f"{tick.label} ({tick.value:g})")
        imgui.end()


__all__ = ["LegendView", "LegendProvider", "SchemeProvider"]
