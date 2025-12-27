from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict

log = logging.getLogger(__name__)


class StoreState:
    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._listeners: Dict[str, list[Callable[[Any], None]]] = {}

    def get(self, key: Any, default: Any = None) -> Any:
        parts = self._split_key(key)
        with self._lock:
            cur: Any = self._data
            for part in parts:
                if not isinstance(cur, dict) or part not in cur:
                    return default
                cur = cur[part]
            return cur

    def set(self, key: str, value: Any) -> None:
        parts = self._split_key(key)
        listener_key = self._listener_key(key, parts)
        with self._lock:
            self._set_path(parts, value)
            listeners = list(self._listeners.get(listener_key, ()))

        for fn in listeners:
            try:
                fn(value)
            except Exception:
                log.exception(
                    "StoreState listener failed: key=%r callback=%r value=%r",
                    listener_key,
                    getattr(fn, "__name__", repr(fn)),
                    value,
                )

    def update(self, **kwargs: Any) -> None:
        callbacks: list[tuple[str, Any]] = []
        with self._lock:
            for k, v in kwargs.items():
                parts = self._split_key(k)
                self._set_path(parts, v)
                callbacks.append((self._listener_key(k, parts), v))

        for listener_key, value in callbacks:
            for fn in list(self._listeners.get(listener_key, ())):
                try:
                    fn(value)
                except Exception:
                    log.exception(
                        "StoreState listener failed: key=%r callback=%r value=%r",
                        listener_key,
                        getattr(fn, "__name__", repr(fn)),
                        value,
                    )

    def subscribe(self, key: str, callback: Callable[[Any], None]) -> Callable[[], None]:
        listener_key = self._listener_key(key, self._split_key(key))
        with self._lock:
            self._listeners.setdefault(listener_key, []).append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if listener_key in self._listeners and callback in self._listeners[listener_key]:
                    self._listeners[listener_key].remove(callback)

        return unsubscribe

    def as_dict(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def _split_key(self, key: Any) -> tuple[Any, ...]:
        if isinstance(key, (tuple, list)):
            return tuple(key)
        if isinstance(key, str) and "/" in key:
            parts = key.split("/")
            if any(part == "" for part in parts):
                raise ValueError(f"Invalid store key path: {key!r}")
            return tuple(parts)
        return (key,)

    def _listener_key(self, key: Any, parts: tuple[Any, ...]) -> str:
        if isinstance(key, str):
            return key
        if len(parts) > 1:
            return "/".join(str(part) for part in parts)
        return str(parts[0])

    def _set_path(self, parts: tuple[Any, ...], value: Any) -> None:
        if not parts:
            return
        cur: Dict[Any, Any] = self._data
        for part in parts[:-1]:
            nxt = cur.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[part] = nxt
            cur = nxt
        cur[parts[-1]] = value
