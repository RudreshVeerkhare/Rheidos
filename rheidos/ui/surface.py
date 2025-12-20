from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GUISurface:
    """Mount point for UI elements (e.g., Panda3D aspect2d)."""

    name: str
    root: Any
