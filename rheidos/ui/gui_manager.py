from __future__ import annotations

import time
from typing import Dict, Any, List, Tuple

from ..abc.controller import Controller
from ..abc.action import Action


class GUIManager:
    """
    Minimal GUI host: rebuilds a simple vertical panel from controller actions.
    Designed to be replaced/extended with a richer layout later.
    """

    def __init__(self, session: Any, surface: Any) -> None:
        self._session = session
        self._surface = surface
        self._root = None
        self._toggle_buttons: List[Tuple[Action, Any]] = []
        self._refresh_task_name = f"gui-refresh-{id(self)}"
        self._last_refresh = 0.0
        self._refresh_interval = 0.2

    def rebuild(self, controllers: Dict[str, Controller]) -> None:
        base = getattr(self._session, "base", None)
        aspect2d = getattr(self._surface, "root", None) if self._surface else None
        if base is None or aspect2d is None:
            return

        # Clear previous UI
        self.clear()
        self._toggle_buttons.clear()

        try:
            from direct.gui.DirectGui import DirectFrame, DirectButton, DirectLabel
            from panda3d.core import TextNode
        except Exception:
            return

        # Root frame pinned to top-left-ish with simple vertical flow
        self._root = DirectFrame(
            parent=aspect2d,
            frameColor=(0, 0, 0, 0.25),
            frameSize=(-0.6, 0.6, -1.0, 1.0),
            pos=(-1.2, 0, 0.8),
        )

        y = -0.08  # start below top edge
        x_pad = 0.05
        y_step = 0.1

        for name, controller in controllers.items():
            # Header label
            DirectLabel(
                parent=self._root,
                text=controller.name,
                text_align=TextNode.A_left,
                frameColor=(0, 0, 0, 0),
                scale=0.05,
                pos=(-0.55, 0, y),
            )
            y -= y_step * 0.8

            for action in controller.actions():
                initial_state = self._get_toggle_state(action)
                label = self._format_label(action, initial_state)
                if action.kind == "toggle":
                    def _make_toggle_callback(act, button_ref):
                        def _cb():
                            if act.get_value is not None and act.set_value is not None:
                                try:
                                    current = bool(act.get_value(self._session))
                                    new_val = not current
                                    act.set_value(self._session, new_val)
                                    act.invoke(self._session, new_val)
                                    self._set_button_state(button_ref, act, new_val)
                                    return
                                except Exception:
                                    pass
                            act.invoke(self._session, None)
                            self._set_button_state(button_ref, act, None)

                        return _cb

                    # placeholder; will set command after creation to capture button reference
                    cb = None  # type: ignore
                else:
                    cb = lambda act=action: act.invoke(self._session, None)

                btn = DirectButton(
                    parent=self._root,
                    text=label,
                    scale=0.045,
                    frameColor=(0.25, 0.25, 0.28, 0.9),
                    relief=1,
                    pos=(-0.5 + x_pad, 0, y),
                    command=cb,
                )
                if action.kind == "toggle":
                    btn["command"] = _make_toggle_callback(action, btn)
                    self._toggle_buttons.append((action, btn))
                y -= y_step

            y -= y_step * 0.6  # extra gap between controllers

        self._ensure_refresh_task()

    def clear(self) -> None:
        self._remove_refresh_task()
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
        self._root = None
        self._toggle_buttons.clear()

    # --- internal helpers ---

    def _format_label(self, action: Action, state: Any) -> str:
        base_label = action.label or action.id
        if action.kind != "toggle":
            return base_label
        marker = "[x]" if state else "[ ]"
        return f"{marker} {base_label}"

    def _get_toggle_state(self, action: Action) -> bool:
        if action.kind != "toggle":
            return False
        if action.get_value is None:
            return False
        try:
            return bool(action.get_value(self._session))
        except Exception:
            return False

    def _set_button_state(self, button: Any, action: Action, state: Any) -> None:
        if action.kind != "toggle":
            return
        label = self._format_label(action, state)
        try:
            button["text"] = label
        except Exception:
            pass

    def _ensure_refresh_task(self) -> None:
        task_mgr = getattr(self._session, "task_mgr", None)
        if task_mgr is None or not self._toggle_buttons:
            return
        try:
            task_mgr.remove(self._refresh_task_name)
        except Exception:
            pass
        task_mgr.add(self._refresh_task, name=self._refresh_task_name, sort=200)

    def _remove_refresh_task(self) -> None:
        task_mgr = getattr(self._session, "task_mgr", None)
        if task_mgr is None:
            return
        try:
            task_mgr.remove(self._refresh_task_name)
        except Exception:
            pass

    def _refresh_task(self, task: Any) -> int:
        now = time.perf_counter()
        if now - self._last_refresh < self._refresh_interval:
            return task.cont
        self._last_refresh = now
        for action, btn in list(self._toggle_buttons):
            state = self._get_toggle_state(action)
            self._set_button_state(btn, action, state)
        return task.cont
