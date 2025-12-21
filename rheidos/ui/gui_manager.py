from __future__ import annotations

import re
import time
from typing import Dict, Any, List, Tuple

from ..abc.controller import Controller
from ..abc.action import Action


def _prettify(name: str) -> str:
    # MeshVisibilityController -> Mesh Visibility
    name = re.sub(r"Controller$", "", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return name.strip()


class GUIManager:
    """
    Cleaner debug/prototype UI:
      - anchored to top-left
      - fixed-size panel + scroll
      - DirectCheckButton for toggles (no "[x]" text hacks)
      - consistent padding/row sizing
    """

    # --- theme-ish constants (tweak freely) ---
    PANEL_W = 0.78
    PANEL_H = 1.55
    PAD_X = 0.04
    PAD_TOP = 0.06
    PAD_BOTTOM = 0.06

    HEADER_SCALE = 0.055
    ROW_SCALE = 0.048
    ROW_H = 0.11
    GAP_AFTER_HEADER = 0.07
    GAP_BETWEEN_CONTROLLERS = 0.10

    def __init__(self, session: Any, surface: Any) -> None:
        self._session = session
        self._surface = surface

        self._root = None
        self._scroll = None

        self._toggle_widgets: List[Tuple[Action, Any]] = []
        self._refresh_task_name = f"gui-refresh-{id(self)}"
        self._last_refresh = 0.0
        self._refresh_interval = 0.2
        self._tooltip = None

    def rebuild(self, controllers: Dict[str, Controller]) -> None:
        base = getattr(self._session, "base", None)
        if base is None:
            return

        # Prefer an anchored corner node for stable placement across aspect ratios
        anchor = getattr(base, "a2dTopLeft", None)
        if anchor is None:
            # fallback: surface root or aspect2d
            anchor = getattr(self._surface, "root", None) or getattr(base, "aspect2d", None)
        if anchor is None:
            return

        self.clear()
        self._toggle_widgets.clear()

        try:
            from direct.gui.DirectGui import (
                DirectFrame,
                DirectButton,
                DirectLabel,
                DirectCheckButton,
                DirectScrolledFrame,
            )
            from panda3d.core import TextNode
            from direct.gui import DirectGuiGlobals as DGG
        except Exception:
            return

        # --- Panel root (top-left anchored) ---
        self._root = DirectFrame(
            parent=anchor,
            frameColor=(0.08, 0.08, 0.10, 0.85),
            frameSize=(0, self.PANEL_W, -self.PANEL_H, 0),
            pos=(0.02, 0, -0.02),
        )

        # --- Two-pass: compute content height to size the scroll canvas ---
        sorted_controllers = sorted(
            controllers.values(), key=lambda c: (getattr(c, "ui_order", 0), c.name)
        )

        total_h = self.PAD_TOP + self.PAD_BOTTOM
        for controller in sorted_controllers:
            actions = list(controller.actions())
            if not actions:
                continue
            total_h += self.GAP_AFTER_HEADER + self.ROW_H * len(actions) + self.GAP_BETWEEN_CONTROLLERS

        # Canvas size: (left, right, bottom, top) in parent coords
        # Top is 0, content extends downward (negative Z).
        canvas_bottom = -max(total_h, self.PANEL_H)

        self._scroll = DirectScrolledFrame(
            parent=self._root,
            frameColor=(0, 0, 0, 0),
            frameSize=(self.PAD_X, self.PANEL_W - self.PAD_X, -self.PANEL_H + self.PAD_BOTTOM, -self.PAD_TOP),
            canvasSize=(0, self.PANEL_W, canvas_bottom, 0),
            scrollBarWidth=0.035,
        )
        canvas = self._scroll.getCanvas()

        # Tooltip label (shared)
        try:
            self._tooltip = DirectLabel(
                parent=self._root,
                text="",
                text_fg=(1, 1, 1, 1),
                text_scale=0.04,
                frameColor=(0, 0, 0, 0.85),
                pad=(0.02, 0.01),
                relief=1,
            )
            self._tooltip.hide()
        except Exception:
            self._tooltip = None

        # --- Layout pass ---
        y = -self.PAD_TOP
        row_width = self.PANEL_W - 2 * self.PAD_X

        for controller in sorted_controllers:
            actions = sorted(
                controller.actions(),
                key=lambda a: (a.group, getattr(a, "order", 0), a.label or a.id),
            )
            if not actions:
                continue

            # Header
            DirectLabel(
                parent=canvas,
                text=_prettify(controller.name),
                text_align=TextNode.A_left,
                text_wordwrap=18,  # helps long names
                frameColor=(0, 0, 0, 0),
                scale=self.HEADER_SCALE,
                pos=(self.PAD_X, 0, y),
            )
            y -= self.GAP_AFTER_HEADER

            for action in actions:
                pretty = action.label or _prettify(action.id)
                if action.shortcut:
                    pretty = f"[{action.shortcut}] {pretty}"
                tooltip = action.tooltip

                if action.kind == "toggle":
                    initial = self._get_toggle_state(action)

                    w = DirectCheckButton(
                        parent=canvas,
                        # Give the row a predictable “hit box”
                        frameSize=(0, row_width, -self.ROW_H, 0),
                        pos=(self.PAD_X, 0, y),
                        frameColor=(0.18, 0.18, 0.22, 0.9),
                        relief=1,

                        # Check indicator is the state; label stays constant
                        indicatorValue=bool(initial),
                        text=pretty,
                        text_align=TextNode.A_left,
                        text_wordwrap=22,
                        text_scale=self.ROW_SCALE,
                        # Nudge text a bit right so it doesn’t overlap the checkbox
                        text_pos=(0.10, -0.035),
                    )

                    def _on_toggle(v, act=action, widget=w):
                        # v is 0/1 from DirectCheckButton
                        new_val = bool(v)
                        try:
                            if act.set_value is not None:
                                act.set_value(self._session, new_val)
                            act.invoke(self._session, new_val)
                        except Exception:
                            pass
                        # ensure UI reflects real state
                        widget["indicatorValue"] = bool(self._get_toggle_state(act))

                    w["command"] = _on_toggle
                    self._toggle_widgets.append((action, w))
                else:
                    w = DirectButton(
                        parent=canvas,
                        frameSize=(0, row_width, -self.ROW_H, 0),
                        pos=(self.PAD_X, 0, y),
                        frameColor=(0.18, 0.18, 0.22, 0.9),
                        relief=1,

                        text=pretty,
                        text_align=TextNode.A_left,
                        text_wordwrap=22,
                        text_scale=self.ROW_SCALE,
                        text_pos=(0.06, -0.035),
                        command=(lambda act=action: act.invoke(self._session, None)),
                    )

                y -= self.ROW_H

                if tooltip and self._tooltip is not None:
                    w.bind(
                        DGG.ENTER,
                        lambda evt, t=tooltip, widget=w: self._show_tooltip(t, widget),
                    )
                    w.bind(DGG.EXIT, lambda evt: self._hide_tooltip())

            y -= self.GAP_BETWEEN_CONTROLLERS

        self._ensure_refresh_task()

    def set_controllers(self, controllers: Dict[str, Controller]) -> None:
        """Compatibility with ImGui manager interface: rebuild in one call."""
        self.rebuild(controllers)

    def clear(self) -> None:
        self._remove_refresh_task()
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
        self._root = None
        self._scroll = None
        self._toggle_widgets.clear()
        self._tooltip = None

    # --- internal helpers ---

    def _get_toggle_state(self, action: Action) -> bool:
        if action.kind != "toggle" or action.get_value is None:
            return False
        try:
            return bool(action.get_value(self._session))
        except Exception:
            return False

    def _ensure_refresh_task(self) -> None:
        task_mgr = getattr(self._session, "task_mgr", None)
        if task_mgr is None or not self._toggle_widgets:
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

        for action, widget in list(self._toggle_widgets):
            state = self._get_toggle_state(action)
            try:
                widget["indicatorValue"] = bool(state)
            except Exception:
                pass

        return task.cont

    def _show_tooltip(self, text: str, widget: Any) -> None:
        if self._tooltip is None or self._root is None:
            return
        try:
            pos = widget.getPos(self._root)
            # Offset slightly above the widget row
            self._tooltip.setText(text)
            self._tooltip.setPos(pos[0], 0, pos[2] + 0.06)
            self._tooltip.show()
        except Exception:
            pass

    def _hide_tooltip(self) -> None:
        if self._tooltip is None:
            return
        try:
            self._tooltip.hide()
        except Exception:
            pass
