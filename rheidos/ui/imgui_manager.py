from __future__ import annotations

from typing import Any, Dict

from ..abc.controller import Controller


class ImGuiUIManager:
    """Immediate-mode UI driven by panda3d-imgui."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._controllers: Dict[str, Controller] = {}
        self._open = True

    def set_controllers(self, controllers: Dict[str, Controller]) -> None:
        self._controllers = dict(controllers)

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

        for ctrl in controllers:
            actions = ctrl.actions()
            if not actions:
                continue

            header = imgui.collapsing_header(
                ctrl.name, flags=imgui.TreeNodeFlags_.default_open
            )
            # imgui_bundle may return bool or (bool, opened)
            expanded = header[0] if isinstance(header, tuple) else bool(header)
            if not expanded:
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

        imgui.end()
