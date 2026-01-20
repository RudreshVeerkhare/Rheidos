from __future__ import annotations

from typing import Optional

from rheidos.compute.profiler.runtime import get_current_profiler


def span(name: str, *, cat: str = "python", producer: Optional[str] = None):
    return get_current_profiler().span(name, cat=cat, producer=producer)
