"""Session cache and runtime for Houdini nodes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Deque, Dict, Optional, Tuple, TYPE_CHECKING, Literal
import time

import numpy as np

from rheidos.compute.world import World

from .taichi_reset import reset_taichi_hard

if TYPE_CHECKING:
    import hou

__all__ = [
    "AccessMode",
    "ComputeRuntime",
    "SessionAccess",
    "SessionKey",
    "WorldSession",
    "get_runtime",
    "make_session_key",
    "make_session_key_for_path",
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


def make_session_key_for_path(node_path: str) -> SessionKey:
    try:
        import hou  # type: ignore
    except Exception as exc:  # pragma: no cover - only runs in Houdini
        raise RuntimeError("Houdini 'hou' module not available") from exc

    return SessionKey(hip_path=hou.hipFile.path(), node_path=node_path)


AccessMode = Literal["read", "write"]


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
    last_triangles_by_input: Dict[int, np.ndarray] = field(default_factory=dict)
    last_topology_sig_by_input: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)
    last_topology_key_by_input: Dict[int, Tuple[Any, ...]] = field(default_factory=dict)
    last_error: Optional[BaseException] = None
    last_traceback: Optional[str] = None
    log_entries: Deque[Dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=200)
    )
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
        self.last_triangles_by_input.clear()
        self.last_topology_sig_by_input.clear()
        self.last_topology_key_by_input.clear()
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


class RegistryAccess:
    def __init__(self, reg: Any, *, mode: AccessMode) -> None:
        self._reg = reg
        self._mode = mode

    def _require_write(self) -> None:
        if self._mode != "write":
            raise PermissionError("Session access is read-only; use mode='write'")

    def read(self, name: str, *, ensure: bool = True) -> Any:
        return self._reg.read(name, ensure=ensure)

    def ensure(self, name: str) -> None:
        self._reg.ensure(name)

    def declare(self, *args: Any, **kwargs: Any) -> Any:
        self._require_write()
        return self._reg.declare(*args, **kwargs)

    def commit(self, *args: Any, **kwargs: Any) -> Any:
        self._require_write()
        return self._reg.commit(*args, **kwargs)

    def commit_many(self, *args: Any, **kwargs: Any) -> Any:
        self._require_write()
        return self._reg.commit_many(*args, **kwargs)

    def bump(self, *args: Any, **kwargs: Any) -> Any:
        self._require_write()
        return self._reg.bump(*args, **kwargs)

    def set_buffer(self, *args: Any, **kwargs: Any) -> Any:
        self._require_write()
        return self._reg.set_buffer(*args, **kwargs)


@dataclass
class SessionAccess:
    session: WorldSession
    node_path: str
    mode: AccessMode = "read"

    def __post_init__(self) -> None:
        if self.mode not in ("read", "write"):
            raise ValueError(f"Unknown session access mode: {self.mode}")
        self._reg_view = RegistryAccess(self._get_world().reg, mode=self.mode)

    def _get_world(self) -> World:
        if self.session.world is None:
            self.session.world = World()
        return self.session.world

    @property
    def reg(self) -> RegistryAccess:
        return self._reg_view

    def log(self, message: str, **payload: Any) -> None:
        self.session.log_event(message, node_path=self.node_path, **payload)

    def __enter__(self) -> "SessionAccess":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


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

    def get_session_by_path(
        self, node_path: str, *, create: bool = False
    ) -> WorldSession:
        key = make_session_key_for_path(node_path)
        session = self.sessions.get(key)
        if session is None:
            if not create:
                raise KeyError(f"No session for node_path='{node_path}'")
            session = WorldSession()
            self.sessions[key] = session
        return session

    def session_access(
        self,
        node_path: str,
        *,
        mode: AccessMode = "read",
        create: bool = False,
    ) -> SessionAccess:
        session = self.get_session_by_path(node_path, create=create)
        return SessionAccess(session=session, node_path=node_path, mode=mode)

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
