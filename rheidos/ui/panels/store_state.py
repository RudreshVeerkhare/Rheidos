from __future__ import annotations

import pprint
import time
from typing import Any, Dict, Optional

from ...store import StoreState


class StoreStatePanel:
    """Pretty-print StoreState contents inside the ImGui tools window."""

    id = "store-state"
    title = "Store State"
    order = 100

    def __init__(
        self,
        store: Optional[StoreState],
        refresh_interval: float = 0.5,
        max_items: int = 50,
        max_depth: int = 3,
        max_repr_len: int = 140,
    ) -> None:
        self._store = store
        self._refresh_interval = max(0.05, float(refresh_interval))
        self._max_items = max_items
        self._max_depth = max_depth
        self._max_repr_len = max_repr_len
        self._last_refresh = 0.0
        self._cached: Dict[str, Any] = {}

    def draw(self, imgui: Any) -> None:
        state = self._get_state_snapshot()
        pretty = self._format_state(state)
        imgui.push_text_wrap_pos()
        imgui.text_unformatted(pretty)
        imgui.pop_text_wrap_pos()

    def _get_state_snapshot(self) -> Dict[str, Any]:
        now = time.perf_counter()
        if self._cached and (now - self._last_refresh) < self._refresh_interval:
            return self._cached

        self._last_refresh = now
        if self._store is None:
            self._cached = {}
            return self._cached

        try:
            data = self._store.as_dict()
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}

        self._cached = data
        return self._cached

    def _format_state(self, data: Dict[str, Any]) -> str:
        if not data:
            return "(empty)"
        sanitized = self._sanitize(data, depth=self._max_depth)
        return pprint.pformat(sanitized, width=80, sort_dicts=True, compact=False)

    def _sanitize(self, value: Any, depth: int) -> Any:
        if depth <= 0:
            return "..."

        if isinstance(value, dict):
            items = sorted(value.items(), key=lambda kv: self._short_repr(kv[0]))
            trimmed = items[: self._max_items]
            result = {
                self._short_repr(k): self._sanitize(v, depth - 1) for k, v in trimmed
            }
            if len(items) > len(trimmed):
                result["..."] = f"+{len(items) - len(trimmed)} more"
            return result

        if isinstance(value, (list, tuple)):
            seq = list(value)
            trimmed = seq[: self._max_items]
            sanitized = [self._sanitize(v, depth - 1) for v in trimmed]
            if len(seq) > len(trimmed):
                sanitized.append(f"... +{len(seq) - len(trimmed)} more")
            return sanitized if isinstance(value, list) else tuple(sanitized)

        if isinstance(value, set):
            seq = sorted(list(value), key=self._short_repr)
            trimmed = seq[: self._max_items]
            sanitized = [self._sanitize(v, depth - 1) for v in trimmed]
            if len(seq) > len(trimmed):
                sanitized.append(f"... +{len(seq) - len(trimmed)} more")
            return sanitized

        return self._short_repr(value)

    def _short_repr(self, value: Any) -> str:
        try:
            text = repr(value)
        except Exception:
            text = "<unrepr>"
        if len(text) > self._max_repr_len:
            text = text[: self._max_repr_len] + "..."
        return text
