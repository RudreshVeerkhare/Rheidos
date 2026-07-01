from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import contextvars
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Iterator, Mapping, Optional

import numpy as np

from rheidos.compute.profiler.tb import TBConfig, TBLogger

_RUN_DIR_RE = re.compile(r"^run-(?:(?P<name>.+)-)?(?P<id>\d{4})__")
_RUN_NAME_RE = re.compile(r"[^a-z0-9]+")
_CURRENT_SCOPE: contextvars.ContextVar[Optional["_ActiveLoggerContext"]] = (
    contextvars.ContextVar("rheidos_logger_scope", default=None)
)


def _normalize_logdir(value: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(value).strip()))
    return os.path.abspath(os.path.normpath(expanded))


def _normalize_run_name(value: str) -> Optional[str]:
    text = str(value).strip()
    return text or None


def _sanitize_run_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    lowered = value.lower()
    sanitized = _RUN_NAME_RE.sub("-", lowered).strip("-")
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    return sanitized or None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(payload), fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp_path, path)


def _next_run_id(base_dir: Path) -> int:
    max_id = 0
    if not base_dir.is_dir():
        return 1
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        match = _RUN_DIR_RE.match(child.name)
        if match is None:
            continue
        max_id = max(max_id, int(match.group("id")))
    return max_id + 1


def _make_run_dir_name(run_name: Optional[str], run_id: int, timestamp: str) -> str:
    sanitized = _sanitize_run_name(run_name)
    if sanitized is None:
        return f"run-{run_id:04d}__{timestamp}"
    return f"run-{sanitized}-{run_id:04d}__{timestamp}"


def _coerce_scalar(value: Any) -> float:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise TypeError("logger.log() only accepts scalar values")
        return float(value.item())
    if isinstance(value, np.generic):
        return float(value.item())
    if isinstance(value, (bool, int, float)):
        return float(value)
    raise TypeError("logger.log() only accepts scalar values")


def _normalize_tag(name: str, *, category: str) -> str:
    tag = str(name).strip()
    if not tag:
        raise ValueError("logger.log() requires a non-empty metric name")
    if "/" in tag:
        return tag
    prefix = str(category).strip() or "simulation"
    return f"{prefix}/{tag}"


@dataclass
class _LoggerScope:
    configured_logdir: Optional[str] = None
    configured_run_name: Optional[str] = None
    default_logdir: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tb: Optional[TBLogger] = None
    run_dir: Optional[str] = None
    run_id: Optional[int] = None
    timestamp: Optional[str] = None
    started: bool = False
    locked_logdir: Optional[str] = None
    locked_run_name: Optional[str] = None
    step_counter: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def close(self) -> None:
        tb = self.tb
        self.tb = None
        if tb is not None:
            tb.close()


@dataclass(frozen=True)
class _ActiveLoggerContext:
    scope: _LoggerScope
    step_hint: Optional[int] = None


_GLOBAL_SCOPE = _LoggerScope()


def _make_scope(
    *, default_logdir: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None
) -> _LoggerScope:
    scope = _LoggerScope()
    if default_logdir:
        scope.default_logdir = _normalize_logdir(default_logdir)
    if metadata:
        scope.metadata.update(dict(metadata))
    return scope


@contextmanager
def _activate_scope(
    scope: _LoggerScope,
    *,
    default_logdir: Optional[str] = None,
    step_hint: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Iterator[_LoggerScope]:
    if default_logdir:
        scope.default_logdir = _normalize_logdir(default_logdir)
    if metadata:
        scope.metadata.update(dict(metadata))
    resolved_step_hint = None if step_hint is None else int(step_hint)
    token = _CURRENT_SCOPE.set(
        _ActiveLoggerContext(scope=scope, step_hint=resolved_step_hint)
    )
    try:
        yield scope
    finally:
        _CURRENT_SCOPE.reset(token)


def _reset_for_tests() -> None:
    global _GLOBAL_SCOPE
    _GLOBAL_SCOPE.close()
    _GLOBAL_SCOPE = _LoggerScope()
    _CURRENT_SCOPE.set(None)


class SimulationLogger:
    def _active_context(self) -> Optional[_ActiveLoggerContext]:
        return _CURRENT_SCOPE.get()

    def _select_scope(self) -> tuple[_LoggerScope, Optional[_ActiveLoggerContext]]:
        active = self._active_context()
        if active is not None:
            return active.scope, active
        return _GLOBAL_SCOPE, None

    def _effective_logdir(self, scope: _LoggerScope) -> Optional[str]:
        if scope is _GLOBAL_SCOPE:
            return scope.configured_logdir
        if scope.configured_logdir is not None:
            return scope.configured_logdir
        if _GLOBAL_SCOPE.configured_logdir is not None:
            return _GLOBAL_SCOPE.configured_logdir
        return scope.default_logdir

    def _effective_run_name(self, scope: _LoggerScope) -> Optional[str]:
        if scope.configured_run_name is not None:
            return scope.configured_run_name
        if scope is not _GLOBAL_SCOPE and _GLOBAL_SCOPE.configured_run_name is not None:
            return _GLOBAL_SCOPE.configured_run_name
        return None

    def _ensure_mutable_config(
        self,
        scope: _LoggerScope,
        *,
        logdir: Optional[str],
        run_name: Optional[str],
    ) -> None:
        if not scope.started:
            return
        if logdir is not None:
            normalized = _normalize_logdir(logdir)
            if normalized != scope.locked_logdir:
                raise RuntimeError(
                    "logger.configure() cannot change logdir after the run has started"
                )
        if run_name is not None:
            normalized_name = _normalize_run_name(run_name)
            if normalized_name != scope.locked_run_name:
                raise RuntimeError(
                    "logger.configure() cannot change run_name after the run has started"
                )

    def _ensure_run(self, scope: _LoggerScope) -> _LoggerScope:
        if scope.started and scope.tb is not None:
            return scope
        with scope._lock:
            if scope.started and scope.tb is not None:
                return scope

            base_root = self._effective_logdir(scope)
            if base_root is None:
                raise RuntimeError(
                    "logger.log() requires a configured logdir or an active runtime log scope"
                )

            base_path = Path(base_root)
            base_path.mkdir(parents=True, exist_ok=True)
            run_name = self._effective_run_name(scope)

            while True:
                run_id = _next_run_id(base_path)
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                run_dir_name = _make_run_dir_name(run_name, run_id, timestamp)
                run_path = base_path / run_dir_name
                try:
                    run_path.mkdir(parents=False, exist_ok=False)
                    break
                except FileExistsError:
                    time.sleep(0.001)

            scope.tb = TBLogger(TBConfig(logdir=str(run_path)))
            scope.run_id = run_id
            scope.run_dir = str(run_path)
            scope.timestamp = timestamp
            scope.started = True
            scope.locked_logdir = str(base_path)
            scope.locked_run_name = run_name

            metadata = {
                "custom_name": run_name,
                "run_dir": str(run_path),
                "run_id": run_id,
                "timestamp": timestamp,
            }
            metadata.update(_json_safe(scope.metadata))
            _write_json_atomic(run_path / "run.json", metadata)
            _write_json_atomic(base_path / "latest-run.json", metadata)
        return scope

    def _resolve_step(
        self,
        scope: _LoggerScope,
        *,
        explicit_step: Optional[int],
        active: Optional[_ActiveLoggerContext],
    ) -> int:
        if explicit_step is not None:
            step = int(explicit_step)
            scope.step_counter = max(scope.step_counter, step)
            return step
        if active is not None and active.step_hint is not None:
            step = int(active.step_hint)
            scope.step_counter = max(scope.step_counter, step)
            return step
        scope.step_counter += 1
        return scope.step_counter

    def configure(
        self,
        *,
        logdir: str | None = None,
        run_name: str | None = None,
    ) -> None:
        scope, _ = self._select_scope()
        self._ensure_mutable_config(scope, logdir=logdir, run_name=run_name)
        if logdir is not None:
            scope.configured_logdir = _normalize_logdir(logdir)
        if run_name is not None:
            scope.configured_run_name = _normalize_run_name(run_name)

    def log(
        self,
        name: str,
        value: Any,
        *,
        category: str = "simulation",
        step: int | None = None,
        flush: bool = False,
    ) -> None:
        scope, active = self._select_scope()
        tag = _normalize_tag(name, category=category)
        scalar = _coerce_scalar(value)
        scope = self._ensure_run(scope)
        resolved_step = self._resolve_step(scope, explicit_step=step, active=active)
        tb = scope.tb
        if tb is None:
            raise RuntimeError("logger failed to initialize the TensorBoard writer")
        tb.add_scalar(tag, scalar, resolved_step)
        if flush:
            tb.flush()


logger = SimulationLogger()

__all__ = ["SimulationLogger", "logger"]
