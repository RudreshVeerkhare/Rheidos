from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
