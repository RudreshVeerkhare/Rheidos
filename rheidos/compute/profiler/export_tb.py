from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Callable, Optional

from rheidos.compute.profiler.summary_store import SummaryStore
from rheidos.compute.profiler.tb import TBConfig, make_writer


@dataclass
class TBExporterConfig:
    logdir: str
    flush_secs: int = 5
    max_queue: int = 1000
    export_hz: float = 5.0


class TensorboardExporter:
    def __init__(
        self,
        summary_store: SummaryStore,
        cfg: TBExporterConfig,
        *,
        dag_provider: Optional[Callable[[], str]] = None,
    ):
        self.summary_store = summary_store
        self.cfg = cfg
        self._writer = make_writer(TBConfig(cfg.logdir, cfg.flush_secs, cfg.max_queue))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._step = 0
        self._dag_provider = dag_provider
        self._last_dag_hash: Optional[int] = None
        self._pending_texts: queue.SimpleQueue[tuple[str, str]] = queue.SimpleQueue()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="rheidos-tb-exporter", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._thread = None
        self._drain_pending_texts()
        self._writer.flush()
        self._writer.close()

    def enqueue_text(self, tag: str, text: str) -> None:
        if not text:
            return
        self._pending_texts.put((tag, text))

    def _drain_pending_texts(self) -> None:
        while True:
            try:
                tag, text = self._pending_texts.get_nowait()
            except queue.Empty:
                return
            self._writer.add_text(tag, text, self._step)

    def _export_dag(self) -> None:
        if self._dag_provider is None:
            return
        try:
            dot = self._dag_provider()
        except Exception:
            return
        if not dot:
            return
        dag_hash = hash(dot)
        if dag_hash == self._last_dag_hash:
            return
        self._last_dag_hash = dag_hash
        self._writer.add_text("dag/dot", dot, self._step)

    def _run(self) -> None:
        period = 1.0 / max(1e-6, self.cfg.export_hz)
        while not self._stop.is_set():
            time.sleep(period)
            self._step += 1
            snap = self.summary_store.snapshot_compact()
            self._writer.add_scalar("cook/wall_ms", snap.get("wall_ms", 0.0), self._step)
            self._writer.add_scalar(
                "cook/kernel_ms", snap.get("kernel_ms", 0.0), self._step
            )
            self._writer.add_scalar(
                "cook/kernel_fraction",
                snap.get("kernel_fraction", 0.0),
                self._step,
            )
            self._writer.add_scalar(
                "profiler/dropped_events",
                snap.get("dropped_events", 0),
                self._step,
            )
            self._writer.add_scalar(
                "profiler/overhead_us",
                snap.get("profiler_overhead_us", 0.0),
                self._step,
            )
            self._writer.add_scalar(
                "profiler/edges_recorded",
                snap.get("edges_recorded", 0),
                self._step,
            )
            rows = snap.get("rows", [])
            if rows:
                top = sorted(rows, key=lambda r: r.get("ema_ms", 0.0), reverse=True)[
                    :10
                ]
                lines = ["name | ema_ms | last_ms", "--- | --- | ---"]
                for row in top:
                    name = row.get("name", "")
                    ema_ms = row.get("ema_ms", 0.0)
                    last_ms = row.get("last_ms", 0.0)
                    lines.append(f"{name} | {ema_ms:.3f} | {last_ms:.3f}")
                self._writer.add_text("producer/top_ema_ms", "\n".join(lines), self._step)
            self._export_dag()
            self._drain_pending_texts()
