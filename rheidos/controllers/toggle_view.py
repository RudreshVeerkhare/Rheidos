from __future__ import annotations

from typing import Dict, List

from ..abc.controller import Controller


class ToggleViewController(Controller):
    def __init__(self, engine, groups: List[List[str]], key: str = "v", name: str | None = None) -> None:
        super().__init__(name=name)
        self.engine = engine
        self.groups = groups
        self.key = key
        self._state = 0

    def attach(self, session) -> None:
        super().attach(session)
        session.accept(self.key, self._on_toggle)
        self._apply_state()

    def detach(self) -> None:
        self._session.ignore(self.key)

    def _on_toggle(self) -> None:
        self._state = (self._state + 1) % len(self.groups)
        self._apply_state()

    def _apply_state(self) -> None:
        if not self.groups:
            return
        managed = {v for group in self.groups for v in group}
        current = set(self.groups[self._state])
        for v in managed:
            self.engine.enable_view(v, False)
        for v in current:
            self.engine.enable_view(v, True)
