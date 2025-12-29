"""Session cache and runtime for Houdini nodes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Deque, Dict, Optional, Tuple, TYPE_CHECKING
import time

import numpy as np

from rheidos.compute.world import World

from .taichi_reset import reset_taichi_hard

if TYPE_CHECKING:
    import hou

__all__ = [
    "ComputeRuntime",
    "SessionKey",
    "WorldSession",
    "get_runtime",
    "make_session_key",
]


@dataclass(frozen=True)
class SessionKey:
    hip_path: str
    node_path: str


def make_session_key(node: "hou.Node") -> SessionKey:
    try:
        import hou  # type: ignore
    except Exception as exc:  # pragma: no cover - only runs in Houdini
        raise RuntimeError("Houdini 'hou' module not available") from exc

    return SessionKey(hip_path=hou.hipFile.path(), node_path=node.path())


@dataclass
class WorldSession:
    world: Optional[World] = None
    user_module: Optional[ModuleType] = None
    user_module_key: Optional[str] = None
    did_setup: bool = False
    last_step_key: Optional[Tuple[Any, ...]] = None
    last_output_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    last_geo_snapshot: Optional[Any] = None
    last_triangles: Optional[np.ndarray] = None
    last_topology_sig: Optional[Tuple[int, int, int]] = None
    last_topology_key: Optional[Tuple[Any, ...]] = None
    last_error: Optional[BaseException] = None
    last_traceback: Optional[str] = None
    log_entries: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=200))
    stats: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_cook_at: Optional[float] = None

    def reset(self, reason: str) -> None:
        self.world = None
        self.user_module = None
        self.user_module_key = None
        self.did_setup = False
        self.last_step_key = None
        self.last_output_cache.clear()
        self.last_geo_snapshot = None
        self.last_triangles = None
        self.last_topology_sig = None
        self.last_topology_key = None
        self.clear_error()
        self.clear_log()
        self.stats.clear()
        self.last_cook_at = None
        self.stats["last_reset_reason"] = reason
        self.stats["last_reset_at"] = time.time()

    def record_error(self, exc: BaseException, tb_str: str) -> None:
        self.last_error = exc
        self.last_traceback = tb_str
        self.stats["last_error_at"] = time.time()

    def clear_error(self) -> None:
        self.last_error = None
        self.last_traceback = None

    def log_event(self, message: str, **payload: Any) -> None:
        entry = {"message": message, "ts": time.time()}
        entry.update(payload)
        self.log_entries.append(entry)

    def clear_log(self) -> None:
        self.log_entries.clear()


class ComputeRuntime:
    def __init__(self) -> None:
        self.sessions: Dict[SessionKey, WorldSession] = {}

    def get_or_create_session(self, node: "hou.Node") -> WorldSession:
        key = make_session_key(node)
        session = self.sessions.get(key)
        if session is None:
            session = WorldSession()
            self.sessions[key] = session
        return session

    def reset_session(self, node: "hou.Node", reason: str) -> None:
        key = make_session_key(node)
        session = self.sessions.get(key)
        if session is not None:
            session.reset(reason)

    def nuke_all(self, reason: str) -> None:
        for session in self.sessions.values():
            session.reset(reason)
        self.sessions.clear()
        reset_taichi_hard()


_RUNTIME = ComputeRuntime()


def get_runtime() -> ComputeRuntime:
    return _RUNTIME
