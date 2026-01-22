from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Callable, Dict, Optional

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


@dataclass(frozen=True)
class TBConfig:
    logdir: str
    flush_secs: int = 5
    max_queue: int = 1000


def make_writer(cfg: TBConfig) -> SummaryWriter:
    if SummaryWriter is None:
        raise RuntimeError("tensorboardX not installed. pip install tensorboardX")
    Path(cfg.logdir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(
        log_dir=cfg.logdir, flush_secs=cfg.flush_secs, max_queue=cfg.max_queue
    )


CustomFn = Callable[..., Any]


class TBLogger:
    def __init__(self, cfg: Optional[TBConfig] = None, *, enabled: bool = True) -> None:
        self._cfg = cfg
        self._enabled = enabled
        self._writer: Optional[Any] = None
        self._custom: Dict[str, CustomFn] = {}
        self._lock = threading.Lock()
        self._step = 0

    @property
    def cfg(self) -> Optional[TBConfig]:
        return self._cfg

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, value: int) -> None:
        self._step = int(value)

    def next_step(self) -> int:
        self._step += 1
        return self._step

    def configure(self, cfg: Optional[TBConfig], *, enabled: Optional[bool] = None) -> None:
        prev_enabled = self._enabled
        if enabled is not None:
            self._enabled = enabled
        if cfg == self._cfg:
            if prev_enabled and not self._enabled:
                self._close_writer()
            return
        self._cfg = cfg
        self._close_writer()

    def reset(self) -> None:
        self._close_writer()
        self._cfg = None
        self._enabled = False
        self._custom.clear()
        self._step = 0

    def close(self) -> None:
        self._close_writer()

    def flush(self) -> None:
        writer = self._ensure_writer()
        if writer is None:
            return
        writer.flush()

    def register(self, name: Optional[str] = None, fn: Optional[CustomFn] = None):
        if fn is None and callable(name) and not isinstance(name, str):
            func = name
            self._register_fn(func.__name__, func)
            return func
        if fn is None:
            def decorator(func: CustomFn) -> CustomFn:
                self._register_fn(name or func.__name__, func)
                return func

            return decorator
        self._register_fn(name or fn.__name__, fn)
        return fn

    def unregister(self, name: str) -> None:
        self._custom.pop(name, None)

    @property
    def writer(self) -> Optional[Any]:
        return self._ensure_writer()

    def _register_fn(self, name: str, fn: CustomFn) -> None:
        if not name:
            raise ValueError("Custom tensorboard function name cannot be empty")
        if name in self._custom or getattr(TBLogger, name, None) is not None:
            raise KeyError(f"Custom tensorboard function '{name}' already exists")
        self._custom[name] = fn

    def _close_writer(self) -> None:
        with self._lock:
            if self._writer is None:
                return
            writer = self._writer
            self._writer = None
        try:
            writer.flush()
        except Exception:
            pass
        try:
            writer.close()
        except Exception:
            pass

    def _ensure_writer(self) -> Optional[Any]:
        if not self._enabled or self._cfg is None:
            return None
        if self._writer is not None:
            return self._writer
        with self._lock:
            if self._writer is None:
                self._writer = make_writer(self._cfg)
        return self._writer

    def __getattr__(self, name: str):
        if name in self._custom:
            def _custom_call(*args: Any, **kwargs: Any):
                writer = self._ensure_writer()
                if writer is None:
                    return None
                return self._custom[name](self, writer, *args, **kwargs)

            return _custom_call

        def _proxy(*args: Any, **kwargs: Any):
            writer = self._ensure_writer()
            if writer is None:
                return None
            return getattr(writer, name)(*args, **kwargs)

        return _proxy
