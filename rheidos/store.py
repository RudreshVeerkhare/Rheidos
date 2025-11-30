from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional


class StoreState:
    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._listeners: Dict[str, list[Callable[[Any], None]]] = {}

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            listeners = list(self._listeners.get(key, ()))
        for fn in listeners:
            try:
                fn(value)
            except Exception:
                pass

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for k, v in kwargs.items():
                self._data[k] = v
                listeners = list(self._listeners.get(k, ()))
            # Callbacks are executed after releasing lock
        for k, _ in kwargs.items():
            for fn in list(self._listeners.get(k, ())):
                try:
                    fn(self._data[k])
                except Exception:
                    pass

    def subscribe(self, key: str, callback: Callable[[Any], None]) -> Callable[[], None]:
        with self._lock:
            self._listeners.setdefault(key, []).append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if key in self._listeners and callback in self._listeners[key]:
                    self._listeners[key].remove(callback)

        return unsubscribe

    def as_dict(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data)

