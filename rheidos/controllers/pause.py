from __future__ import annotations

from ..abc.controller import Controller


class PauseController(Controller):
    def __init__(self, engine, key: str = "space", name: str | None = None) -> None:
        super().__init__(name=name)
        self.engine = engine
        self.key = key

    def attach(self, session) -> None:
        super().attach(session)
        session.accept(self.key, self._on_toggle)

    def detach(self) -> None:
        self._session.ignore(self.key)

    def _on_toggle(self) -> None:
        self.engine.set_paused(not self.engine.is_paused())

