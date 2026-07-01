from __future__ import annotations

import contextvars

from rheidos.compute.profiler.core import Profiler, ProfilerConfig


_CURRENT: contextvars.ContextVar[Profiler] = contextvars.ContextVar(
    "rheidos_prof", default=Profiler(ProfilerConfig(enabled=False))
)


def set_current_profiler(p: Profiler):
    return _CURRENT.set(p)


def reset_current_profiler(token) -> None:
    _CURRENT.reset(token)


def get_current_profiler() -> Profiler:
    return _CURRENT.get()
