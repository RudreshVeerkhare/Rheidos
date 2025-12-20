from __future__ import annotations

from typing import Dict, Any

from ..abc.controller import Controller


class GUIManager:
    """
    Minimal GUI host: rebuilds a simple vertical panel from controller actions.
    Designed to be replaced/extended with a richer layout later.
    """

    def __init__(self, session: Any, surface: Any) -> None:
        self._session = session
        self._surface = surface
        self._root = None

    def rebuild(self, controllers: Dict[str, Controller]) -> None:
        base = getattr(self._session, "base", None)
        aspect2d = getattr(self._surface, "root", None) if self._surface else None
        if base is None or aspect2d is None:
            return

        # Clear previous UI
        self.clear()

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
                label = action.label or action.id
                if action.kind == "toggle":
                    # For now, treat as a button that flips state; richer widgets later.
                    def _make_toggle_callback(act):
                        def _cb():
                            if act.get_value is not None and act.set_value is not None:
                                try:
                                    current = bool(act.get_value(self._session))
                                    new_val = not current
                                    act.set_value(self._session, new_val)
                                    act.invoke(self._session, new_val)
                                    return
                                except Exception:
                                    pass
                            act.invoke(self._session, None)

                        return _cb

                    cb = _make_toggle_callback(action)
                else:
                    cb = lambda act=action: act.invoke(self._session, None)

                DirectButton(
                    parent=self._root,
                    text=label,
                    scale=0.045,
                    frameColor=(0.25, 0.25, 0.28, 0.9),
                    relief=1,
                    pos=(-0.5 + x_pad, 0, y),
                    command=cb,
                )
                y -= y_step

            y -= y_step * 0.6  # extra gap between controllers

    def clear(self) -> None:
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
            self._root = None
