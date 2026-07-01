from __future__ import annotations

from dataclasses import dataclass
import functools
from time import perf_counter_ns
from typing import Dict, Optional, Tuple, TYPE_CHECKING
import threading

if TYPE_CHECKING:
    from .summary_store import SummaryStore
from .ids import PRODUCER_IDS
from .trace_store import TraceConfig, TraceStore


@dataclass
class ProfilerConfig:
    enabled: bool = False
    mode: str = "coarse"
    export_hz: float = 5.0
    taichi_enabled: bool = False
    taichi_sample_every_n_cooks: int = 30
    taichi_sync_on_sample: bool = True
    trace_cooks: int = 64
    trace_max_edges: int = 20000
    overhead_enabled: bool = False


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
    __slots__ = ("_prof", "_cat", "_name", "_producer", "_t0", "_run", "_is_root")

    def __init__(self, prof: "Profiler", cat: str, name: str, producer: Optional[str]):
        self._prof = prof
        self._cat = cat
        self._name = name
        self._producer = producer
        self._t0 = 0
        self._run = None
        self._is_root = False

    def __enter__(self) -> "Span":
        self._t0, self._run, self._is_root = self._prof._span_enter(
            self._cat, self._name, self._producer
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        t1 = perf_counter_ns()
        self._prof._span_exit(
            self._cat,
            self._name,
            self._producer,
            t1 - self._t0,
            self._run,
            self._is_root,
        )
        return False


class _ProducerRun:
    __slots__ = ("producer", "producer_id", "child_durations_ns", "trace_node_index")

    def __init__(self, producer: str, producer_id: int, trace_node_index: Optional[int]):
        self.producer = producer
        self.producer_id = producer_id
        self.child_durations_ns: Dict[str, int] = {}
        self.trace_node_index = trace_node_index

    def record_child(self, cat: str, name: str, dur_ns: int) -> None:
        key = f"{cat}/{name}"
        self.child_durations_ns[key] = self.child_durations_ns.get(key, 0) + dur_ns


class _ThreadState:
    __slots__ = ("runs",)

    def __init__(self) -> None:
        self.runs: list[_ProducerRun] = []


class Profiler:
    def __init__(self, cfg: ProfilerConfig, summary_store: Optional["SummaryStore"] = None):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._stats: Dict[StatsKey, Stats] = {}
        self._dropped: int = 0
        self._cook_index: int = 0
        self._producer_classes: Dict[int, str] = {}
        self._taichi_sample: bool = False
        self.taichi_probe = None
        self.summary_store = summary_store
        self._local = threading.local()
        self._trace_store = TraceStore(
            TraceConfig(
                max_cooks=cfg.trace_cooks,
                max_edges_per_cook=cfg.trace_max_edges,
            )
        )
        self._overhead_ns: int = 0
        self._edges_recorded: int = 0

    def span(self, name: str, *, cat: str = "python", producer: Optional[str] = None):
        if not self.cfg.enabled or self._mode() == "off":
            return NullSpan()
        if self._mode() == "deps_only" and cat != "producer":
            return NullSpan()
        return Span(self, cat, name, producer)

    def record_value(
        self,
        cat: str,
        name: str,
        producer: Optional[str],
        value_ms: float,
    ) -> None:
        if not self._timing_enabled():
            return
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
        cook_id = 0
        with self._lock:
            s = self._stats.get(key)
            if s is None:
                s = Stats()
                self._stats[key] = s
            s.update(dur_ns)
            cook_id = self._cook_index
        if self.summary_store is not None:
            self.summary_store.record_span(cat, name, producer, dur_ns, cook_id)

    def _span_enter(
        self, cat: str, name: str, producer: Optional[str]
    ) -> tuple[int, Optional["_ProducerRun"], bool]:
        state = self._get_local_state()
        is_root = False
        if cat == "producer" and name == "compute" and producer:
            producer_id = PRODUCER_IDS.intern(producer)
            parent_index = None
            if state.runs:
                parent_index = state.runs[-1].trace_node_index
            trace_index = None
            if self._tree_enabled():
                if any(run.producer_id == producer_id for run in state.runs):
                    self._dropped += 1
                    if self.summary_store is not None:
                        self.summary_store.update_global(dropped_events=self._dropped)
                else:
                    trace_index = self._trace_store.record_producer_begin(
                        producer_id=producer_id,
                        parent_index=parent_index,
                        cook_id=self.current_cook_index(),
                    )
            run = _ProducerRun(producer, producer_id, trace_index)
            state.runs.append(run)
            is_root = True
        else:
            run = state.runs[-1] if state.runs else None
        return perf_counter_ns(), run, is_root

    def _span_exit(
        self,
        cat: str,
        name: str,
        producer: Optional[str],
        dur_ns: int,
        run: Optional["_ProducerRun"],
        is_root: bool,
    ) -> None:
        if self._timing_enabled():
            self._record_span(cat, name, producer, dur_ns)
        if run is not None:
            if is_root:
                state = self._get_local_state()
                if state.runs:
                    state.runs.pop()
                if self._tree_enabled() and run.trace_node_index is not None:
                    self._trace_store.record_producer_end(
                        cook_id=self.current_cook_index(),
                        node_index=run.trace_node_index,
                        inclusive_ns=dur_ns,
                    )
                if self.summary_store is not None and self._timing_enabled():
                    self.summary_store.update_producer_children(
                        run.producer,
                        child_durations_ns=run.child_durations_ns,
                        last_update_id=self.current_cook_index(),
                    )
            else:
                if self._timing_enabled():
                    run.record_child(cat, name, dur_ns)

    def snapshot_stats(self) -> Dict[StatsKey, Stats]:
        with self._lock:
            return dict(self._stats)

    def next_cook_index(self) -> int:
        with self._lock:
            self._cook_index += 1
            cook_id = self._cook_index
            if self._edges_enabled() or self._tree_enabled():
                self._trace_store.begin_cook(cook_id)
            self._flush_overhead_locked()
        if self.summary_store is not None:
            self.summary_store.update_global(cook_id=cook_id)
        return cook_id

    def current_cook_index(self) -> int:
        with self._lock:
            return self._cook_index

    def _get_local_state(self) -> "_ThreadState":
        state = getattr(self._local, "state", None)
        if state is None:
            state = _ThreadState()
            self._local.state = state
        return state

    def attach_summary_store(self, summary_store: Optional["SummaryStore"]) -> None:
        self.summary_store = summary_store

    def register_producer_metadata(self, *, full_name: str, class_name: str) -> None:
        if not full_name:
            return
        if not class_name:
            return
        producer_id = PRODUCER_IDS.intern(full_name)
        with self._lock:
            self._producer_classes[producer_id] = class_name
        if self.summary_store is not None:
            self.summary_store.register_producer_metadata(
                producer=full_name,
                class_name=class_name,
            )

    def configure(self, cfg: ProfilerConfig) -> None:
        self.cfg = cfg
        self._trace_store.configure(
            TraceConfig(
                max_cooks=cfg.trace_cooks,
                max_edges_per_cook=cfg.trace_max_edges,
            )
        )

    def current_producer_id(self) -> Optional[int]:
        state = self._get_local_state()
        if not state.runs:
            return None
        return state.runs[-1].producer_id

    def record_resource_read(
        self, *, resource_id: int, producer_id: Optional[int]
    ) -> None:
        if not self._edges_enabled():
            return
        cur = self.current_producer_id()
        if cur is None:
            return
        t0 = perf_counter_ns() if self.cfg.overhead_enabled else None
        edges_added, dropped = self._trace_store.record_resource_read(
            producer_id=cur,
            resource_id=resource_id,
            resource_producer_id=producer_id,
        )
        if edges_added:
            self._edges_recorded += edges_added
        if dropped:
            self._dropped += dropped
            if self.summary_store is not None:
                self.summary_store.update_global(dropped_events=self._dropped)
        if self.cfg.overhead_enabled and t0 is not None:
            self._overhead_ns += perf_counter_ns() - t0

    def snapshot_dag(self, *, mode: str = "union") -> dict:
        with self._lock:
            class_map = dict(self._producer_classes)
        return self._trace_store.snapshot_dag(mode=mode, producer_classes=class_map)

    def snapshot_exec_tree(self, *, cook_id: Optional[int] = None) -> dict:
        with self._lock:
            class_map = dict(self._producer_classes)
        return self._trace_store.snapshot_exec_tree(
            cook_id=cook_id, producer_classes=class_map
        )

    def snapshot_metrics(self) -> dict:
        if self.summary_store is None:
            return {"cook_id": self.current_cook_index(), "rows": []}
        return self.summary_store.snapshot_metrics()

    def snapshot_node_details(self, producer_id: int) -> Optional[dict]:
        if self.summary_store is None:
            return None
        with self._lock:
            class_map = dict(self._producer_classes)
        return self._trace_store.snapshot_node_details(
            producer_id=producer_id,
            summary_store=self.summary_store,
            producer_classes=class_map,
        )

    def _mode(self) -> str:
        if not self.cfg.enabled:
            return "off"
        mode = (self.cfg.mode or "coarse").strip().lower()
        if mode not in ("off", "coarse", "deps_only", "sampled_taichi"):
            return "coarse"
        return mode

    def _timing_enabled(self) -> bool:
        return self._mode() in ("coarse", "sampled_taichi")

    def _edges_enabled(self) -> bool:
        return self._mode() in ("coarse", "deps_only", "sampled_taichi")

    def _tree_enabled(self) -> bool:
        return self._mode() in ("coarse", "sampled_taichi")

    def _flush_overhead_locked(self) -> None:
        if self.summary_store is None:
            self._overhead_ns = 0
            self._edges_recorded = 0
            return
        overhead_us = self._overhead_ns / 1000.0
        self.summary_store.update_global(
            dropped_events=self._dropped,
            profiler_overhead_us=overhead_us,
            edges_recorded=self._edges_recorded,
        )
        self._overhead_ns = 0
        self._edges_recorded = 0


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
