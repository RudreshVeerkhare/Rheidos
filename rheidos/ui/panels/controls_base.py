from __future__ import annotations

from typing import Any, Optional

from ...store import StoreState


class StoreBoundControls:
    """
    Small helpers to bind imgui widgets to StoreState keys.
    """

    def __init__(self, store: Optional[StoreState]) -> None:
        self._store = store

    def checkbox(self, imgui: Any, label: str, key: str, default: bool = False) -> bool:
        current = bool(self._store.get(key, default)) if self._store else bool(default)
        changed, value = imgui.checkbox(label, current)
        if changed and self._store is not None:
            self._store.set(key, bool(value))
        return bool(value)

    def slider_float(
        self,
        imgui: Any,
        label: str,
        key: str,
        min_val: float,
        max_val: float,
        default: float = 0.0,
        **kwargs: Any,
    ) -> float:
        current = float(self._store.get(key, default)) if self._store else float(default)
        changed, value = imgui.slider_float(label, current, min_val, max_val, **kwargs)
        if changed and self._store is not None:
            self._store.set(key, float(value))
        return float(value)

    def slider_int(
        self,
        imgui: Any,
        label: str,
        key: str,
        min_val: int,
        max_val: int,
        default: int = 0,
        **kwargs: Any,
    ) -> int:
        current = int(self._store.get(key, default)) if self._store else int(default)
        changed, value = imgui.slider_int(label, current, min_val, max_val, **kwargs)
        if changed and self._store is not None:
            self._store.set(key, int(value))
        return int(value)


__all__ = ["StoreBoundControls"]
