from __future__ import annotations

from typing import Any, Optional


class Observer:
    name: str

    def __init__(self, name: Optional[str] = None, sort: int = -10) -> None:
        self.name = name or self.__class__.__name__
        self.sort = sort
        self._session: Any = None

    def setup(self, session: Any) -> None:
        self._session = session

    def update(self, dt: float) -> None:
        pass

    def teardown(self) -> None:
        pass

