from __future__ import annotations

from typing import Dict, Iterable, Optional

from .abc.action import Action


class InputRouter:
    """Bind controller-declared actions to Panda3D input events."""

    def __init__(self, session) -> None:
        self._session = session
        self._bindings: Dict[str, list[str]] = {}

    def bind_actions(self, controller_name: str, actions: Iterable[Action]) -> None:
        base = getattr(self._session, "base", None)
        if base is None:
            return

        # Clear previous bindings for this controller to avoid duplicates.
        self.unbind_controller(controller_name)

        shortcuts: list[str] = []
        for action in actions:
            shortcut = action.shortcut
            if not shortcut:
                continue
            callback = self._make_callback(action)
            try:
                base.accept(shortcut, callback)
                shortcuts.append(shortcut)
            except Exception:
                # Keep the engine running even if a binding fails.
                pass

        if shortcuts:
            self._bindings[controller_name] = shortcuts

    def unbind_controller(self, controller_name: str) -> None:
        base = getattr(self._session, "base", None)
        shortcuts = self._bindings.pop(controller_name, ())
        if base is None:
            return
        for shortcut in shortcuts:
            try:
                base.ignore(shortcut)
            except Exception:
                pass

    def _make_callback(self, action: Action):
        def _blocked() -> bool:
            # If ImGui wants keyboard/mouse, skip hotkeys.
            try:
                from imgui_bundle import imgui
                io = imgui.get_io()
                return bool(io.want_capture_keyboard or io.want_capture_mouse)
            except Exception:
                return False

        if action.kind == "toggle":

            def _cb(value: Optional[object] = None) -> None:
                if _blocked():
                    return
                # If getters/setters are provided, toggle the stored value; otherwise just invoke.
                if action.get_value is not None and action.set_value is not None:
                    try:
                        current = bool(action.get_value(self._session))
                        new_val = not current
                        action.set_value(self._session, new_val)
                        action.invoke(self._session, new_val)
                        return
                    except Exception:
                        pass
                action.invoke(self._session, value)

        else:

            def _cb(value: Optional[object] = None) -> None:
                if _blocked():
                    return
                action.invoke(self._session, value)

        return _cb
