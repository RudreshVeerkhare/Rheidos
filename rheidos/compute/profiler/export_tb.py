from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Callable, Optional

from rheidos.compute.profiler.core import Profiler
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
        prof: Profiler,
        cfg: TBExporterConfig,
        *,
        dag_provider: Optional[Callable[[], str]] = None,
    ):
        self.prof = prof
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
            stats = self.prof.snapshot_stats()
            self._step += 1
            for (cat, name, producer), s in stats.items():
                base = f"{cat}/{name}" if producer is None else f"{cat}/{producer}/{name}"
                self._writer.add_scalar(base + "/last_ms", s.last_ns / 1e6, self._step)
                self._writer.add_scalar(base + "/ema_ms", s.ema_ns / 1e6, self._step)
                self._writer.add_scalar(base + "/count", s.count, self._step)
            self._export_dag()
            self._drain_pending_texts()
