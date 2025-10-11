from __future__ import annotations

from typing import Any

try:
    from panda3d.core import ClockObject
except Exception:
    ClockObject = None  # type: ignore


class PandaSession:
    def __init__(self, base: Any) -> None:
        self.base = base
        self.task_mgr = base.task_mgr if hasattr(base, "task_mgr") else base.taskMgr
        self.render = base.render
        self.win = base.win
        if ClockObject is not None:
            self.clock = ClockObject.getGlobalClock()
        else:
            self.clock = None

    def accept(self, *args: Any, **kwargs: Any) -> Any:
        return self.base.accept(*args, **kwargs)

    def ignore(self, *args: Any, **kwargs: Any) -> Any:
        return self.base.ignore(*args, **kwargs)
