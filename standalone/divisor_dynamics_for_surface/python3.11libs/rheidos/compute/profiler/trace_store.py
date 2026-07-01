from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import threading
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from .ids import PRODUCER_IDS, RESOURCE_IDS

if TYPE_CHECKING:
    from .summary_store import SummaryStore


@dataclass(frozen=True)
class TraceConfig:
    max_cooks: int = 64
    max_edges_per_cook: int = 20000


@dataclass
class ExecNode:
    producer_id: int
    parent_index: Optional[int]
    inclusive_ns: int = 0


@dataclass
class CookTrace:
    cook_id: int
    nodes: List[ExecNode] = field(default_factory=list)
    edges_pr: Set[Tuple[int, int]] = field(default_factory=set)
    edges_pp: Set[Tuple[int, int]] = field(default_factory=set)
    edges_pp_meta: Dict[Tuple[int, int], Set[int]] = field(default_factory=dict)
    dropped_edges: int = 0
    edges_enabled: bool = True


class TraceStore:
    def __init__(self, cfg: TraceConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._traces: Dict[int, CookTrace] = {}
        self._order: deque[int] = deque()
        self._active_cook_id = 0
        self._union_pp_counts: Dict[Tuple[int, int], int] = {}
        self._known_producers: Set[int] = set()
        self._dag_version = 0
        self._dropped_edges_total = 0

    def configure(self, cfg: TraceConfig) -> None:
        with self._lock:
            self.cfg = cfg
            self._evict_locked()

    def begin_cook(self, cook_id: int) -> None:
        with self._lock:
            self._active_cook_id = cook_id
            if cook_id in self._traces:
                return
            self._traces[cook_id] = CookTrace(cook_id=cook_id)
            self._order.append(cook_id)
            self._evict_locked()

    def record_producer_begin(
        self, *, producer_id: int, parent_index: Optional[int], cook_id: int
    ) -> int:
        with self._lock:
            if producer_id not in self._known_producers:
                self._known_producers.add(producer_id)
                self._dag_version += 1
            trace = self._get_or_create_trace_locked(cook_id)
            node_index = len(trace.nodes)
            trace.nodes.append(
                ExecNode(
                    producer_id=producer_id,
                    parent_index=parent_index,
                    inclusive_ns=0,
                )
            )
            return node_index

    def record_producer_end(
        self, *, cook_id: int, node_index: int, inclusive_ns: int
    ) -> None:
        with self._lock:
            trace = self._traces.get(cook_id)
            if trace is None:
                return
            if node_index < 0 or node_index >= len(trace.nodes):
                return
            trace.nodes[node_index].inclusive_ns = inclusive_ns

    def record_resource_read(
        self,
        *,
        producer_id: int,
        resource_id: int,
        resource_producer_id: Optional[int],
    ) -> tuple[int, int]:
        edges_added = 0
        dropped = 0
        with self._lock:
            trace = self._get_or_create_trace_locked(self._active_cook_id)
            if not trace.edges_enabled:
                return (0, 0)
            total_edges = len(trace.edges_pr) + len(trace.edges_pp)
            if total_edges >= self.cfg.max_edges_per_cook:
                trace.edges_enabled = False
                trace.dropped_edges += 1
                self._dropped_edges_total += 1
                return (0, 1)
            edge_pr = (producer_id, resource_id)
            if edge_pr not in trace.edges_pr:
                trace.edges_pr.add(edge_pr)
                edges_added += 1
            if resource_producer_id is None:
                return (edges_added, dropped)
            edge_pp = (producer_id, resource_producer_id)
            if edge_pp not in trace.edges_pp:
                trace.edges_pp.add(edge_pp)
                self._union_pp_counts[edge_pp] = self._union_pp_counts.get(edge_pp, 0) + 1
                self._dag_version += 1
                edges_added += 1
            meta = trace.edges_pp_meta.setdefault(edge_pp, set())
            meta.add(resource_id)
        return (edges_added, dropped)

    def snapshot_dag(
        self, *, mode: str = "union", producer_classes: Optional[Dict[int, str]] = None
    ) -> dict:
        return self._snapshot_dag(mode=mode, producer_classes=producer_classes)

    def _snapshot_dag(
        self,
        *,
        mode: str = "union",
        producer_classes: Optional[Dict[int, str]] = None,
    ) -> dict:
        with self._lock:
            active_id = self._active_cook_id
            dag_version = self._dag_version
            known_producers = set(self._known_producers)
            union_counts = dict(self._union_pp_counts)
            trace = self._traces.get(active_id)
            if trace is None and self._order:
                trace = self._traces.get(self._order[-1])
            observed_edges = set(trace.edges_pp) if trace is not None else set()
            observed_meta = (
                {k: set(v) for k, v in trace.edges_pp_meta.items()} if trace else {}
            )
        if mode == "observed":
            edges = observed_edges
            counts = {edge: 1 for edge in edges}
            meta = observed_meta
        else:
            edges = set(union_counts.keys())
            counts = union_counts
            meta = {}
        nodes_set: Set[int] = set(known_producers)
        for src, dst in edges:
            nodes_set.add(src)
            nodes_set.add(dst)
        nodes = []
        for pid in sorted(nodes_set):
            full_name = PRODUCER_IDS.name_for(pid) or str(pid)
            class_name = (producer_classes or {}).get(
                pid, _infer_class_name(full_name)
            )
            nodes.append(
                {
                    "id": pid,
                    "name": full_name,
                    "full_name": full_name,
                    "class_name": class_name,
                }
            )
        edge_rows = []
        for (src, dst), count in counts.items():
            row = {"source": src, "target": dst, "seen_count": count}
            via = meta.get((src, dst))
            if via:
                row["via_resources"] = [
                    RESOURCE_IDS.name_for(rid) or str(rid) for rid in sorted(via)
                ]
            edge_rows.append(row)
        return {
            "cook_id": active_id,
            "dag_version": dag_version,
            "nodes": nodes,
            "edges": edge_rows,
        }

    def snapshot_exec_tree(
        self,
        *,
        cook_id: Optional[int] = None,
        producer_classes: Optional[Dict[int, str]] = None,
    ) -> dict:
        return self._snapshot_exec_tree(
            cook_id=cook_id, producer_classes=producer_classes
        )

    def _snapshot_exec_tree(
        self,
        *,
        cook_id: Optional[int] = None,
        producer_classes: Optional[Dict[int, str]] = None,
    ) -> dict:
        with self._lock:
            trace = None
            if cook_id is not None:
                trace = self._traces.get(cook_id)
            if trace is None and self._order:
                trace = self._traces.get(self._order[-1])
            if trace is None:
                return {"cook_id": None, "nodes": []}
            nodes_raw = [
                (node.producer_id, node.parent_index, node.inclusive_ns)
                for node in trace.nodes
            ]
            cook_id = trace.cook_id
        child_sum = [0] * len(nodes_raw)
        for idx, (_pid, parent, inclusive_ns) in enumerate(nodes_raw):
            if parent is None:
                continue
            if 0 <= parent < len(nodes_raw):
                child_sum[parent] += inclusive_ns
        depths = [0] * len(nodes_raw)
        for idx, (_pid, parent, _inclusive_ns) in enumerate(nodes_raw):
            if parent is None:
                continue
            if 0 <= parent < len(nodes_raw):
                depths[idx] = depths[parent] + 1
        nodes = []
        for idx, (pid, parent, inclusive_ns) in enumerate(nodes_raw):
            exclusive_ns = max(0, inclusive_ns - child_sum[idx])
            full_name = PRODUCER_IDS.name_for(pid) or str(pid)
            class_name = (producer_classes or {}).get(
                pid, _infer_class_name(full_name)
            )
            nodes.append(
                {
                    "id": idx,
                    "producer_id": pid,
                    "name": full_name,
                    "full_name": full_name,
                    "class_name": class_name,
                    "parent": parent,
                    "depth": depths[idx],
                    "inclusive_ms": inclusive_ns / 1e6,
                    "exclusive_ms": exclusive_ns / 1e6,
                }
            )
        return {"cook_id": cook_id, "nodes": nodes}

    def snapshot_node_details(
        self,
        *,
        producer_id: int,
        summary_store: "SummaryStore",
        producer_classes: Optional[Dict[int, str]] = None,
    ) -> Optional[dict]:
        producer_name = PRODUCER_IDS.name_for(producer_id)
        if producer_name is None:
            return None
        class_name = (producer_classes or {}).get(
            producer_id, _infer_class_name(producer_name)
        )
        metrics = summary_store.snapshot_producer_metrics(producer_name)
        details = summary_store.snapshot_producer_details(producer_name)
        with self._lock:
            trace = None
            if self._order:
                trace = self._traces.get(self._order[-1])
            if trace is None:
                if metrics is None and details is None:
                    return None
                inputs = details["inputs"] if details else []
                outputs = details["outputs"] if details else []
                staleness_reason = details.get("staleness_reason") if details else ""
                return {
                    "id": producer_id,
                    "name": producer_name,
                    "full_name": producer_name,
                    "class_name": class_name,
                    "metrics": metrics,
                    "inputs": inputs,
                    "outputs": outputs,
                    "staleness_reason": staleness_reason,
                    "resources_read": [],
                    "last_exec_subtree": None,
                }
            nodes_raw = [
                (node.producer_id, node.parent_index, node.inclusive_ns)
                for node in trace.nodes
            ]
            edges_pr = set(trace.edges_pr)
        resources_read = sorted(
            {
                rid
                for (pid, rid) in edges_pr
                if pid == producer_id
            }
        )
        inputs = details["inputs"] if details else []
        outputs = details["outputs"] if details else []
        staleness_reason = details.get("staleness_reason") if details else ""
        child_sum = [0] * len(nodes_raw)
        for idx, (_pid, parent, inclusive_ns) in enumerate(nodes_raw):
            if parent is None:
                continue
            if 0 <= parent < len(nodes_raw):
                child_sum[parent] += inclusive_ns
        last_index = None
        for idx, (pid, _parent, _inclusive_ns) in enumerate(nodes_raw):
            if pid == producer_id:
                last_index = idx
        subtree = None
        if last_index is not None:
            children_map: Dict[int, List[int]] = {}
            for idx, (_pid, parent, _inclusive_ns) in enumerate(nodes_raw):
                if parent is None:
                    continue
                children_map.setdefault(parent, []).append(idx)

            def build(idx: int) -> dict:
                pid, _parent, inclusive_ns = nodes_raw[idx]
                exclusive_ns = max(0, inclusive_ns - child_sum[idx])
                name = PRODUCER_IDS.name_for(pid) or str(pid)
                kids = children_map.get(idx, [])
                kids.sort(key=lambda i: nodes_raw[i][2], reverse=True)
                return {
                    "node_id": idx,
                    "producer_id": pid,
                    "name": name,
                    "inclusive_ms": inclusive_ns / 1e6,
                    "exclusive_ms": exclusive_ns / 1e6,
                    "children": [build(child) for child in kids],
                }

            subtree = build(last_index)
        return {
            "id": producer_id,
            "name": producer_name,
            "full_name": producer_name,
            "class_name": class_name,
            "metrics": metrics,
            "inputs": inputs,
            "outputs": outputs,
            "staleness_reason": staleness_reason,
            "resources_read": [
                RESOURCE_IDS.name_for(rid) or str(rid) for rid in resources_read
            ],
            "last_exec_subtree": subtree,
        }

    def _get_or_create_trace_locked(self, cook_id: int) -> CookTrace:
        trace = self._traces.get(cook_id)
        if trace is None:
            trace = CookTrace(cook_id=cook_id)
            self._traces[cook_id] = trace
            self._order.append(cook_id)
            self._evict_locked()
        return trace

    def _evict_locked(self) -> None:
        removed_edge = False
        while len(self._order) > max(1, self.cfg.max_cooks):
            old_id = self._order.popleft()
            trace = self._traces.pop(old_id, None)
            if trace is None:
                continue
            for edge in trace.edges_pp:
                count = self._union_pp_counts.get(edge, 0)
                if count <= 1:
                    self._union_pp_counts.pop(edge, None)
                    removed_edge = True
                else:
                    self._union_pp_counts[edge] = count - 1
                    removed_edge = True
        if removed_edge:
            self._dag_version += 1


def _infer_class_name(full_name: str) -> str:
    if not full_name:
        return ""
    return full_name.rsplit(".", 1)[-1]
