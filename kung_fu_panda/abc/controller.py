from __future__ import annotations

from typing import Any, Optional


class Controller:
    name: str

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__
        self._session: Any = None

    def attach(self, session: Any) -> None:
        self._session = session

    def detach(self) -> None:
        pass

