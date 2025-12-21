from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

from ..abc.controller import Controller
from ..store import StoreState


class ImGuiPanel(Protocol):
    """Plugin-provided panel rendered inside the ImGui tools window."""

    id: str
    title: str
    order: int

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

        controllers = sorted(
            self._controllers.values(),
            key=lambda c: (getattr(c, "ui_order", 0), c.name),
        )

        # Controllers retain their existing layout/behavior.
        for ctrl in controllers:
            actions = ctrl.actions()
            if not actions:
                continue

            header = imgui.collapsing_header(
                ctrl.name, flags=imgui.TreeNodeFlags_.default_open
            )
            if not self._header_expanded(header):
                continue

            sorted_actions = sorted(
                actions,
                key=lambda a: (a.group, getattr(a, "order", 0), a.label or a.id),
            )

            for act in sorted_actions:
                label = act.label or act.id
                tooltip = act.tooltip
                if act.kind == "toggle":
                    current = False
                    if act.get_value is not None:
                        try:
                            current = bool(act.get_value(self._session))
                        except Exception:
                            current = False
                    changed, new_val = imgui.checkbox(label, current)
                    if changed:
                        try:
                            if act.set_value is not None:
                                act.set_value(self._session, bool(new_val))
                            act.invoke(self._session, bool(new_val))
                        except Exception:
                            pass
                else:
                    if imgui.button(label):
                        try:
                            act.invoke(self._session, None)
                        except Exception:
                            pass
                if tooltip:
                    try:
                        if imgui.is_item_hovered():
                            imgui.set_tooltip(tooltip + f" [{act.shortcut}]" if act.shortcut else "")
                    except Exception:
                        pass
            imgui.separator()

        # Plugin panels share the same window; each handles its own drawing.
        for panel in self._panels:
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

        imgui.end()
