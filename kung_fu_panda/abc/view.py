from __future__ import annotations

from typing import Any, Optional


class View:
    name: str

    def __init__(self, name: Optional[str] = None, sort: int = 0) -> None:
        self.name = name or self.__class__.__name__
        self.sort = sort
        self._session: Any = None
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def setup(self, session: Any) -> None:
        self._session = session

    def update(self, dt: float) -> None:  # per-frame
        pass

    def teardown(self) -> None:
        pass

    def on_enable(self) -> None:
        pass

    def on_disable(self) -> None:
        pass
