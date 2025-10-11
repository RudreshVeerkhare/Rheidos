from __future__ import annotations

from ..abc.controller import Controller


class ScreenshotController(Controller):
    def __init__(self, engine, key: str = "s", filename: str = "screenshot.png", name: str | None = None) -> None:
        super().__init__(name=name)
        self.engine = engine
        self.key = key
        self.filename = filename

    def attach(self, session) -> None:
        super().attach(session)
        session.accept(self.key, self._on_shot)

    def detach(self) -> None:
        self._session.ignore(self.key)

    def _on_shot(self) -> None:
        self.engine.screenshot(self.filename)

