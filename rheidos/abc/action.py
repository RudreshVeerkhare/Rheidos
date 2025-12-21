from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Any


ActionKind = Literal["button", "toggle"]


@dataclass(frozen=True)
class Action:
    id: str
    label: str
    kind: ActionKind
    group: str = "General"
    order: int = 0
    shortcut: Optional[str] = None
    tooltip: Optional[str] = None
    invoke: Callable[[Any, Optional[Any]], None] = lambda session, value=None: None
    get_value: Optional[Callable[[Any], Any]] = None
    set_value: Optional[Callable[[Any, Any], None]] = None

    def __post_init__(self) -> None:
        # Defensive: ensure kind stays within the supported set.
        if self.kind not in ("button", "toggle"):
            raise ValueError(f"Unsupported action kind '{self.kind}'")
