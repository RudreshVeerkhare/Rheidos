from __future__ import annotations

from typing import Any, Optional

from .action import Action


class Controller:
    name: str

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__
        self._session: Any = None
        self.ui_order: int = 0

    def attach(self, session: Any) -> None:
        self._session = session

    def detach(self) -> None:
        pass

    def actions(self) -> tuple[Action, ...]:
        """Declare available actions for UI layers; override in subclasses."""
        return ()
