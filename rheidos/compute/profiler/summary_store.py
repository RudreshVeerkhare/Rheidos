from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Dict, List, Optional

from .core import Stats
from .ids import PRODUCER_IDS


@dataclass
class ChildSpanSummary:
    name: str
    last_ms: float
    ema_ms: float


@dataclass
class ProducerSummary:
    id: str
    name: str
    class_name: str = ""
    stats: Stats = field(default_factory=Stats)
    last_update_id: int = 0
    kernel_ms: float = 0.0
    kernel_frac: float = 0.0
    overhead_est_ms: float = 0.0


@dataclass
class ProducerDetails:
    last_update_id: int = 0
    inputs: List[dict] = field(default_factory=list)
    outputs: List[dict] = field(default_factory=list)
    staleness_reason: str = ""
    top_child_spans: List[ChildSpanSummary] = field(default_factory=list)


@dataclass
class SummaryGlobal:
    cook_id: int = 0
    frame: float = 0.0
    substep: int = 0
    dropped_events: int = 0
    profiler_overhead_us: float = 0.0
    edges_recorded: int = 0
    wall_stats: Stats = field(default_factory=Stats)
    kernel_ms: float = 0.0
    kernel_fraction: float = 0.0


@dataclass
class SummaryDag:
    nodes: List[dict] = field(default_factory=list)
    edges: List[dict] = field(default_factory=list)
    dag_version: int = 0


class SummaryStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._producers: Dict[str, ProducerSummary] = {}
        self._producer_class_names: Dict[str, str] = {}
        self._producer_details: Dict[str, ProducerDetails] = {}
        self._producer_child_stats: Dict[str, Dict[str, Stats]] = {}
        self._category_stats: Dict[str, Dict[tuple[str, Optional[str]], Stats]] = {}
        self._global = SummaryGlobal()
        self._dag = SummaryDag()
        self._tick = 0

    def reset(self) -> None:
        with self._lock:
            self._producers.clear()
            self._producer_class_names.clear()
            self._producer_details.clear()
            self._producer_child_stats.clear()
            self._category_stats.clear()
            self._global = SummaryGlobal()
            self._dag = SummaryDag()
            self._tick = 0

    def update_global(
        self,
        *,
        cook_id: Optional[int] = None,
        frame: Optional[float] = None,
        substep: Optional[int] = None,
        dropped_events: Optional[int] = None,
        profiler_overhead_us: Optional[float] = None,
        edges_recorded: Optional[int] = None,
    ) -> None:
        with self._lock:
            if cook_id is not None:
                self._global.cook_id = cook_id
            if frame is not None:
                self._global.frame = frame
            if substep is not None:
                self._global.substep = substep
            if dropped_events is not None:
                self._global.dropped_events = dropped_events
            if profiler_overhead_us is not None:
                self._global.profiler_overhead_us = profiler_overhead_us
            if edges_recorded is not None:
                self._global.edges_recorded = edges_recorded

    def record_span(
        self,
        cat: str,
        name: str,
        producer: Optional[str],
        dur_ns: int,
        cook_id: int,
    ) -> None:
        if cat == "producer" and name == "compute" and producer:
            self._record_producer_span(producer, dur_ns, cook_id)
            return
        if cat == "cook" and name in ("cook_total", "solver_total"):
            self._record_wall_ns(dur_ns)
        if cat == "taichi":
            self._record_taichi_value(name, producer, dur_ns)
        if cat != "producer":
            self._record_category(cat, name, producer, dur_ns)

    def register_producer_metadata(self, producer: str, *, class_name: str) -> None:
        if not producer:
            return
        if not class_name:
            class_name = self._infer_class_name(producer)
        with self._lock:
            self._producer_class_names[producer] = class_name
            summary = self._producers.get(producer)
            if summary is not None:
                summary.class_name = class_name

    def _record_wall_ns(self, dur_ns: int) -> None:
        with self._lock:
            self._global.wall_stats.update(dur_ns)
            self._update_kernel_fraction_locked()

    def _record_taichi_value(
        self, name: str, producer: Optional[str], dur_ns: int
    ) -> None:
        value_ms = dur_ns / 1e6
        with self._lock:
            if name == "kernel_total" and producer is None:
                self._global.kernel_ms = value_ms
                self._update_kernel_fraction_locked()
                return
            if not producer:
                return
            summary = self._producers.get(producer)
            if summary is None:
                summary = ProducerSummary(
                    id=producer,
                    name=producer,
                    class_name=self._producer_class_names.get(
                        producer, self._infer_class_name(producer)
                    ),
                )
                self._producers[producer] = summary
            elif not summary.class_name:
                summary.class_name = self._producer_class_names.get(
                    producer, self._infer_class_name(producer)
                )
            if name == "producer_kernel_ms":
                summary.kernel_ms = value_ms
                summary.kernel_frac = self._compute_kernel_frac(
                    value_ms, summary.stats.last_ns
                )
            elif name == "producer_overhead_ms":
                summary.overhead_est_ms = value_ms

    def _record_producer_span(
        self, producer: str, dur_ns: int, cook_id: int
    ) -> None:
        with self._lock:
            summary = self._producers.get(producer)
            if summary is None:
                summary = ProducerSummary(
                    id=producer,
                    name=producer,
                    class_name=self._producer_class_names.get(
                        producer, self._infer_class_name(producer)
                    ),
                )
                self._producers[producer] = summary
            elif not summary.class_name:
                summary.class_name = self._producer_class_names.get(
                    producer, self._infer_class_name(producer)
                )
            summary.stats.update(dur_ns)
            summary.last_update_id = cook_id
            summary.kernel_frac = self._compute_kernel_frac(
                summary.kernel_ms, summary.stats.last_ns
            )
            details = self._producer_details.get(producer)
            if details is not None:
                details.last_update_id = cook_id

    def _record_category(
        self, cat: str, name: str, producer: Optional[str], dur_ns: int
    ) -> None:
        with self._lock:
            cat_map = self._category_stats.setdefault(cat, {})
            key = (name, producer)
            stats = cat_map.get(key)
            if stats is None:
                stats = Stats()
                cat_map[key] = stats
            stats.update(dur_ns)

    def update_producer_details(
        self,
        producer: str,
        *,
        last_update_id: int,
        inputs: List[dict],
        outputs: List[dict],
        staleness_reason: str = "",
        class_name: Optional[str] = None,
    ) -> None:
        with self._lock:
            if class_name:
                self._producer_class_names[producer] = class_name
                summary = self._producers.get(producer)
                if summary is not None:
                    summary.class_name = class_name
            details = self._producer_details.get(producer)
            if details is None:
                details = ProducerDetails()
                self._producer_details[producer] = details
            details.last_update_id = last_update_id
            details.inputs = inputs
            details.outputs = outputs
            if staleness_reason:
                details.staleness_reason = staleness_reason

    def update_producer_children(
        self,
        producer: str,
        *,
        child_durations_ns: Dict[str, int],
        last_update_id: int,
        top_n: int = 5,
    ) -> None:
        with self._lock:
            stats_map = self._producer_child_stats.setdefault(producer, {})
            for name, dur_ns in child_durations_ns.items():
                stats = stats_map.get(name)
                if stats is None:
                    stats = Stats()
                    stats_map[name] = stats
                stats.update(dur_ns)
            top_current = sorted(
                child_durations_ns.items(), key=lambda item: item[1], reverse=True
            )[:top_n]
            top_child = []
            for child_name, last_ns in top_current:
                child_stats = stats_map.get(child_name)
                if child_stats is None:
                    continue
                top_child.append(
                    ChildSpanSummary(
                        name=child_name,
                        last_ms=last_ns / 1e6,
                        ema_ms=child_stats.ema_ns / 1e6,
                    )
                )
            details = self._producer_details.get(producer)
            if details is None:
                details = ProducerDetails()
                self._producer_details[producer] = details
            details.last_update_id = last_update_id
            details.top_child_spans = top_child

    def snapshot_compact(self) -> dict:
        with self._lock:
            self._tick += 1
            rows = [
                {
                    "id": summary.id,
                    "name": summary.name,
                    "full_name": summary.name,
                    "class_name": summary.class_name
                    or self._producer_class_names.get(
                        summary.name, self._infer_class_name(summary.name)
                    ),
                    "ema_ms": summary.stats.ema_ns / 1e6,
                    "last_ms": summary.stats.last_ns / 1e6,
                    "last_update": summary.last_update_id,
                    "calls": summary.stats.count,
                    "kernel_ms": summary.kernel_ms,
                    "kernel_frac": summary.kernel_frac,
                    "overhead_est_ms": summary.overhead_est_ms,
                }
                for summary in self._producers.values()
            ]
            categories: Dict[str, list] = {}
            for cat, entries in self._category_stats.items():
                cat_rows = []
                for (name, producer), stats in entries.items():
                    cat_rows.append(
                        {
                            "id": f"{cat}:{name}:{producer or ''}",
                            "name": name,
                            "producer": producer,
                            "ema_ms": stats.ema_ns / 1e6,
                            "last_ms": stats.last_ns / 1e6,
                            "calls": stats.count,
                        }
                    )
                categories[cat] = cat_rows
            return {
                "tick": self._tick,
                "cook_id": self._global.cook_id,
                "frame": self._global.frame,
                "substep": self._global.substep,
                "dropped_events": self._global.dropped_events,
                "profiler_overhead_us": self._global.profiler_overhead_us,
                "edges_recorded": self._global.edges_recorded,
                "wall_ms": self._global.wall_stats.last_ns / 1e6,
                "kernel_ms": self._global.kernel_ms,
                "kernel_fraction": self._global.kernel_fraction,
                "dag_version": self._dag.dag_version,
                "categories": categories,
                "rows": rows,
            }

    def snapshot_metrics(self) -> dict:
        with self._lock:
            cook_id = self._global.cook_id
            rows = []
            for summary in self._producers.values():
                producer_id = PRODUCER_IDS.intern(summary.name)
                class_name = summary.class_name or self._producer_class_names.get(
                    summary.name, self._infer_class_name(summary.name)
                )
                rows.append(
                    {
                        "id": producer_id,
                        "name": summary.name,
                        "full_name": summary.name,
                        "class_name": class_name,
                        "ema_ms": summary.stats.ema_ns / 1e6,
                        "last_ms": summary.stats.last_ns / 1e6,
                        "last_update_id": summary.last_update_id,
                        "calls": summary.stats.count,
                        "kernel_ms": summary.kernel_ms,
                        "kernel_frac": summary.kernel_frac,
                        "overhead_est_ms": summary.overhead_est_ms,
                        "executed_this_cook": summary.last_update_id == cook_id,
                    }
                )
            return {
                "cook_id": cook_id,
                "frame": self._global.frame,
                "substep": self._global.substep,
                "rows": rows,
            }

    def snapshot_producer_metrics(self, producer: str) -> Optional[dict]:
        with self._lock:
            summary = self._producers.get(producer)
            if summary is None:
                return None
            cook_id = self._global.cook_id
            producer_id = PRODUCER_IDS.intern(summary.name)
            class_name = summary.class_name or self._producer_class_names.get(
                summary.name, self._infer_class_name(summary.name)
            )
            return {
                "id": producer_id,
                "name": summary.name,
                "full_name": summary.name,
                "class_name": class_name,
                "ema_ms": summary.stats.ema_ns / 1e6,
                "last_ms": summary.stats.last_ns / 1e6,
                "last_update_id": summary.last_update_id,
                "calls": summary.stats.count,
                "kernel_ms": summary.kernel_ms,
                "kernel_frac": summary.kernel_frac,
                "overhead_est_ms": summary.overhead_est_ms,
                "executed_this_cook": summary.last_update_id == cook_id,
            }

    def update_dag(
        self,
        *,
        nodes: List[dict],
        edges: List[dict],
        dag_version: int,
    ) -> None:
        with self._lock:
            self._dag.nodes = nodes
            self._dag.edges = edges
            self._dag.dag_version = dag_version

    def snapshot_producer_details(self, producer: str) -> Optional[dict]:
        with self._lock:
            details = self._producer_details.get(producer)
            if details is None:
                return None
            class_name = self._producer_class_names.get(
                producer, self._infer_class_name(producer)
            )
            return {
                "id": producer,
                "full_name": producer,
                "class_name": class_name,
                "last_update": details.last_update_id,
                "inputs": list(details.inputs),
                "outputs": list(details.outputs),
                "staleness_reason": details.staleness_reason,
                "top_child_spans": [
                    {
                        "name": span.name,
                        "last_ms": span.last_ms,
                        "ema_ms": span.ema_ms,
                    }
                    for span in details.top_child_spans
                ],
            }

    def _compute_kernel_frac(self, kernel_ms: float, last_ns: int) -> float:
        if last_ns <= 0:
            return 0.0
        wall_ms = last_ns / 1e6
        if wall_ms <= 0:
            return 0.0
        return max(0.0, min(1.0, kernel_ms / wall_ms))

    def _update_kernel_fraction_locked(self) -> None:
        wall_ms = self._global.wall_stats.last_ns / 1e6
        if wall_ms <= 0:
            self._global.kernel_fraction = 0.0
        else:
            self._global.kernel_fraction = max(
                0.0, min(1.0, self._global.kernel_ms / wall_ms)
            )

    @staticmethod
    def _infer_class_name(full_name: str) -> str:
        if not full_name:
            return ""
        return full_name.rsplit(".", 1)[-1]
