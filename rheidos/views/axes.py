from __future__ import annotations

from typing import Any

try:
    from panda3d.core import LineSegs
except Exception as e:  # pragma: no cover
    LineSegs = None  # type: ignore

from ..abc.view import View


class AxesView(View):
    def __init__(
        self, name: str | None = None, axis_length: float = 1.0, sort: int = 0
    ) -> None:
        super().__init__(name=name, sort=sort)
        self.axis_length = float(axis_length)
        self._node = None

    def setup(self, session: Any) -> None:
        super().setup(session)
        if LineSegs is None:
            return
        ls = LineSegs()
        ls.setThickness(2.0)
        L = self.axis_length
        # X - red
        ls.setColor(1, 0, 0, 1)
        ls.moveTo(0, 0, 0)
        ls.drawTo(L, 0, 0)
        # Y - green
        ls.setColor(0, 1, 0, 1)
        ls.moveTo(0, 0, 0)
        ls.drawTo(0, L, 0)
        # Z - blue
        ls.setColor(0, 0, 1, 1)
        ls.moveTo(0, 0, 0)
        ls.drawTo(0, 0, L)

        node = ls.create(False)
        self._node = self._session.render.attachNewNode(node)
        self._node.setName(self.name)

    def update(self, dt: float) -> None:
        pass

    def teardown(self) -> None:
        if self._node is not None:
            self._node.removeNode()
            self._node = None

    def on_enable(self) -> None:
        if self._node is not None:
            self._node.show()

    def on_disable(self) -> None:
        if self._node is not None:
            self._node.hide()
