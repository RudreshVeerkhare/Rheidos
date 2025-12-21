from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

from ..abc.controller import Controller
from ..store import StoreState
from .panels.actions import ControllerActionsPanel


class ImGuiPanel(Protocol):
    """Plugin-provided panel rendered inside the ImGui tools window."""

    id: str
    title: str
    order: int
    separate_window: bool

    def draw(self, imgui: Any) -> None:
        """Render panel content using the provided imgui module."""
        ...


PanelFactory = Callable[[Any, Optional[StoreState]], Optional[ImGuiPanel]]


class ImGuiUIManager:
    """Immediate-mode UI driven by panda3d-imgui."""

    def __init__(
        self,
        session: Any,
        store: Optional[StoreState] = None,
        panel_factories: Optional[Iterable[PanelFactory]] = None,
    ) -> None:
        self._session = session
        self._store = store
        self._controllers: Dict[str, Controller] = {}
        self._panel_factories: List[PanelFactory] = list(panel_factories or [])
        self._panels: List[ImGuiPanel] = []
        self._open = True
        self._panels_enabled = True
        self._panel_visibility: Dict[str, bool] = {}
        self._actions_panel = ControllerActionsPanel(lambda: self._controllers, session)
        self._build_panels()

    def set_controllers(self, controllers: Dict[str, Controller]) -> None:
        self._controllers = dict(controllers)

    def set_panel_factories(self, panel_factories: Sequence[PanelFactory]) -> None:
        """Replace the current panel factory list and rebuild panels."""
        self._panel_factories = list(panel_factories)
        self._build_panels()

    def add_panel_factory(self, panel_factory: PanelFactory) -> None:
        """Append a single panel factory and rebuild panels."""
        self._panel_factories.append(panel_factory)
        self._build_panels()

    def _build_panels(self) -> None:
        """Instantiate panels from factories defensively."""
        self._panels.clear()
        for factory in self._panel_factories:
            try:
                panel = factory(self._session, self._store)
            except Exception:
                continue
            if panel is not None:
                self._panels.append(panel)
        self._panels.sort(key=lambda p: getattr(p, "order", 0))
        # Preserve existing visibilities; default to True for new panels
        for panel in self._panels:
            self._panel_visibility.setdefault(panel.id, True)

    @staticmethod
    def _header_expanded(header_result: Any) -> bool:
        # imgui_bundle may return bool or (bool, opened)
        return header_result[0] if isinstance(header_result, tuple) else bool(header_result)

    def draw_frame(self) -> None:
        if not self._open:
            return
        try:
            from imgui_bundle import imgui
        except Exception:
            return

        imgui.set_next_window_size((360, 520), imgui.Cond_.first_use_ever)
        imgui.begin("Rheidos Tools", True)

        # Core controller actions panel (always on).
        try:
            self._actions_panel.draw(imgui)
        except Exception:
            pass

        embedded_panels = [p for p in self._panels if not getattr(p, "separate_window", False)]
        window_panels = [p for p in self._panels if getattr(p, "separate_window", False)]

        # Plugin panels inside the tools window.
        if embedded_panels:
            changed, enabled = imgui.checkbox("Show Debug Panels", self._panels_enabled)
            if changed:
                self._panels_enabled = bool(enabled)
            imgui.separator()

        if self._panels_enabled:
            for panel in embedded_panels:
                header = imgui.collapsing_header(
                    f"{panel.title}##{panel.id}", flags=imgui.TreeNodeFlags_.default_open
                )
                if not self._header_expanded(header):
                    continue
                try:
                    panel.draw(imgui)
                except Exception:
                    # Panels should not be able to break the host window.
                    pass
                imgui.separator()

        # Toggles for standalone windows.
        if window_panels:
            imgui.separator()
            imgui.text("Debug Windows")
            for panel in window_panels:
                current = self._panel_visibility.get(panel.id, True)
                changed, new_val = imgui.checkbox(f"Show {panel.title}", current)
                if changed:
                    self._panel_visibility[panel.id] = bool(new_val)

        imgui.end()

        # Render separate windows after the main tools window closes.
        for panel in window_panels:
            if not self._panel_visibility.get(panel.id, True):
                continue
            try:
                imgui.begin(f"{panel.title}##{panel.id}")
                panel.draw(imgui)
            except Exception:
                pass
            try:
                imgui.end()
            except Exception:
                pass
