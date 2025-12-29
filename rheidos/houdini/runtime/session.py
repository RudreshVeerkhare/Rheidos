"""Session cache and runtime for Houdini nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING
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
    did_setup: bool = False
    last_step_key: Optional[Tuple[Any, ...]] = None
    last_output_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    last_error: Optional[BaseException] = None
    last_traceback: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_cook_at: Optional[float] = None

    def reset(self, reason: str) -> None:
        self.world = None
        self.did_setup = False
        self.last_step_key = None
        self.last_output_cache.clear()
        self.clear_error()
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
