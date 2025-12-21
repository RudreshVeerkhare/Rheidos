from __future__ import annotations

from typing import Any, Callable, Dict

from ...abc.controller import Controller


class ControllerActionsPanel:
    """Renders controller actions inside the ImGui tools window."""

    id = "controller-actions"
    title = "Controller Actions"
    order = -100

    def __init__(
        self,
        controllers_fn: Callable[[], Dict[str, Controller]],
        session: Any,
    ) -> None:
        self._controllers_fn = controllers_fn
        self._session = session

    def draw(self, imgui: Any) -> None:
        controllers = sorted(
            self._controllers_fn().values(),
            key=lambda c: (getattr(c, "ui_order", 0), c.name),
        )

        for ctrl in controllers:
            actions = ctrl.actions()
            if not actions:
                continue

            header = imgui.collapsing_header(
                ctrl.name, flags=imgui.TreeNodeFlags_.default_open
            )
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
