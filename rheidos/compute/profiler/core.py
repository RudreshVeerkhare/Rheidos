from __future__ import annotations

from dataclasses import dataclass
import functools
from time import perf_counter_ns
from typing import Dict, Optional, Tuple
import threading


@dataclass
class ProfilerConfig:
    enabled: bool = False
    export_hz: float = 5.0
    taichi_enabled: bool = False
    taichi_sample_every_n_cooks: int = 30
    taichi_sync_on_sample: bool = True


StatsKey = Tuple[str, str, Optional[str]]


@dataclass
class Stats:
    last_ns: int = 0
    ema_ns: float = 0.0
    count: int = 0

    def update(self, dur_ns: int, ema_alpha: float = 0.2) -> None:
        self.last_ns = dur_ns
        self.count += 1
        if self.count == 1:
            self.ema_ns = float(dur_ns)
        else:
            self.ema_ns = (1.0 - ema_alpha) * self.ema_ns + ema_alpha * float(dur_ns)


class NullSpan:
    __slots__ = ()

    def __enter__(self) -> "NullSpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class Span:
    __slots__ = ("_prof", "_cat", "_name", "_producer", "_t0")

    def __init__(self, prof: "Profiler", cat: str, name: str, producer: Optional[str]):
        self._prof = prof
        self._cat = cat
        self._name = name
        self._producer = producer
        self._t0 = 0

    def __enter__(self) -> "Span":
        self._t0 = perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        t1 = perf_counter_ns()
        self._prof._record_span(self._cat, self._name, self._producer, t1 - self._t0)
        return False


class Profiler:
    def __init__(self, cfg: ProfilerConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._stats: Dict[StatsKey, Stats] = {}
        self._dropped: int = 0
        self._cook_index: int = 0
        self._taichi_sample: bool = False
        self.taichi_probe = None

    def span(self, name: str, *, cat: str = "python", producer: Optional[str] = None):
        if not self.cfg.enabled:
            return NullSpan()
        return Span(self, cat, name, producer)

    def record_value(
        self,
        cat: str,
        name: str,
        producer: Optional[str],
        value_ms: float,
    ) -> None:
        dur_ns = int(value_ms * 1e6)
        self._record_span(cat, name, producer, dur_ns)

    def set_taichi_sample(self, enabled: bool) -> None:
        self._taichi_sample = enabled

    def is_taichi_sample(self) -> bool:
        return self._taichi_sample

    def _record_span(
        self, cat: str, name: str, producer: Optional[str], dur_ns: int
    ) -> None:
        key = (cat, name, producer)
        with self._lock:
            s = self._stats.get(key)
            if s is None:
                s = Stats()
                self._stats[key] = s
            s.update(dur_ns)

    def snapshot_stats(self) -> Dict[StatsKey, Stats]:
        with self._lock:
            return dict(self._stats)

    def next_cook_index(self) -> int:
        with self._lock:
            self._cook_index += 1
            return self._cook_index


def profiled(name: Optional[str] = None, *, cat: str = "python"):
    def deco(fn):
        nm = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            from rheidos.compute.profiler.runtime import get_current_profiler

            p = get_current_profiler()
            with p.span(nm, cat=cat):
                return fn(*args, **kwargs)

        return wrapped

    return deco
