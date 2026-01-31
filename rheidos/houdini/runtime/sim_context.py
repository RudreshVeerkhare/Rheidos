"""Global Houdini simulation context with explicit teardown."""

from __future__ import annotations

from typing import Any, Callable, List, Optional
import gc


class SimContext:
    """Single-root owner for simulation state in Houdini."""

    def __init__(self) -> None:
        from .session import ComputeRuntime

        self.runtime = ComputeRuntime()
        self._cleanup_callbacks: List[Callable[[], Any]] = []
        self._cache_clearers: List[Callable[[], Any]] = []

    def register_cleanup(self, fn: Callable[[], Any]) -> None:
        self._cleanup_callbacks.append(fn)

    def register_cache_clear(self, fn: Callable[[], Any]) -> None:
        self._cache_clearers.append(fn)

    def close(self, reason: str = "reset") -> None:
        for cb in reversed(self._cleanup_callbacks):
            try:
                cb()
            except Exception:
                pass
        self._cleanup_callbacks.clear()

        for clear in reversed(self._cache_clearers):
            try:
                clear()
            except Exception:
                pass
        self._cache_clearers.clear()

        runtime = getattr(self, "runtime", None)
        if runtime is not None:
            try:
                runtime.nuke_all(reason, reset_taichi=False)
            except Exception:
                pass
        self.runtime = None
        gc.collect()


__all__ = ["SimContext"]
