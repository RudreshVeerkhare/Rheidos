"""Houdini-scoped dev state for reload/reset coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
import os
from typing import Any, Callable, List, Optional, TypeVar, cast

_STATE_ATTR = "_RHEIDOS_DEV_STATE"
_FALLBACK_STATE: Optional["DevState"] = None
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class DevState:
    pid: int = field(default_factory=os.getpid)
    busy_count: int = 0
    busy_reasons: List[str] = field(default_factory=list)
    reloading: bool = False
    last_reload_error: Optional[str] = None
    last_reload_at: Optional[float] = None


def _get_hou():
    try:
        import hou  # type: ignore
    except Exception:
        return None
    return hou


def get_dev_state() -> DevState:
    hou = _get_hou()
    if hou is not None:
        state = getattr(hou.session, _STATE_ATTR, None)
        if not isinstance(state, DevState) or state.pid != os.getpid():
            state = DevState()
            setattr(hou.session, _STATE_ATTR, state)
        return state

    global _FALLBACK_STATE
    if _FALLBACK_STATE is None or _FALLBACK_STATE.pid != os.getpid():
        _FALLBACK_STATE = DevState()
    return _FALLBACK_STATE


def push_busy(reason: str) -> None:
    state = get_dev_state()
    state.busy_count += 1
    state.busy_reasons.append(reason)


def pop_busy() -> None:
    state = get_dev_state()
    if state.busy_count > 0:
        state.busy_count -= 1
    if state.busy_reasons:
        state.busy_reasons.pop()
    if state.busy_count < 0:
        state.busy_count = 0


def is_busy() -> bool:
    return get_dev_state().busy_count > 0


def current_busy_reason() -> Optional[str]:
    state = get_dev_state()
    if not state.busy_reasons:
        return None
    return state.busy_reasons[-1]


def reset_guard(reason: str) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            push_busy(reason)
            try:
                return fn(*args, **kwargs)
            finally:
                pop_busy()

        return cast(F, wrapper)

    return decorator


__all__ = [
    "DevState",
    "reset_guard",
    "get_dev_state",
    "push_busy",
    "pop_busy",
    "is_busy",
    "current_busy_reason",
]
