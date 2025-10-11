from __future__ import annotations

from ..abc.controller import Controller


class ExitController(Controller):
    def __init__(self, engine, key: str = "escape", name: str | None = None) -> None:
        super().__init__(name=name or "ExitController")
        self.engine = engine
        self.key = key

    def attach(self, session) -> None:
        super().attach(session)
        session.accept(self.key, self._on_exit)

    def detach(self) -> None:
        self._session.ignore(self.key)

    def _on_exit(self) -> None:
        try:
            self.engine.stop()
        except Exception:
            pass
        self._session.base.userExit()

