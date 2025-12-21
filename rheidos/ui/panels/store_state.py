from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from ...store import StoreState


class StoreStatePanel:
    """Pretty-print StoreState contents inside the ImGui tools window."""

    id = "store-state"
    title = "Store State"
    order = 100
    separate_window = True

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
        if not state:
            imgui.text_disabled("(empty)")
            return

        imgui.text("Store contents")
        imgui.same_line()
        imgui.text_disabled(f"(refresh {self._refresh_interval:.2f}s)")

        imgui.separator()
        self._render_node(imgui, label="root", value=state, depth=0, is_root=True)

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

    def _render_node(
        self,
        imgui: Any,
        label: str,
        value: Any,
        depth: int,
        is_root: bool = False,
    ) -> None:
        if depth >= self._max_depth:
            imgui.bullet_text(f"{label}: ...")
            return

        kind, summary = self._describe(value)
        display_label = f"{label} ({kind}{summary})" if not is_root else "store"

        if isinstance(value, dict):
            open_node = imgui.tree_node(display_label)
            if open_node:
                items = sorted(value.items(), key=lambda kv: self._safe_str(kv[0]))
                trimmed = items[: self._max_items]
                for idx, (k, v) in enumerate(trimmed):
                    with imgui_ctx_id(imgui, f"{label}-{idx}"):
                        self._render_node(imgui, label=self._safe_str(k), value=v, depth=depth + 1)
                if len(items) > len(trimmed):
                    imgui.bullet_text(f"... (+{len(items) - len(trimmed)} more)")
                imgui.tree_pop()
        elif isinstance(value, (list, tuple)):
            open_node = imgui.tree_node(display_label)
            if open_node:
                seq = list(value)
                trimmed = seq[: self._max_items]
                for idx, v in enumerate(trimmed):
                    with imgui_ctx_id(imgui, f"{label}-{idx}"):
                        self._render_node(imgui, label=str(idx), value=v, depth=depth + 1)
                if len(seq) > len(trimmed):
                    imgui.bullet_text(f"... (+{len(seq) - len(trimmed)} more)")
                imgui.tree_pop()
        elif isinstance(value, set):
            open_node = imgui.tree_node(display_label)
            if open_node:
                seq = sorted(list(value), key=self._safe_str)
                trimmed = seq[: self._max_items]
                for idx, v in enumerate(trimmed):
                    with imgui_ctx_id(imgui, f"{label}-{idx}"):
                        self._render_node(imgui, label=str(idx), value=v, depth=depth + 1)
                if len(seq) > len(trimmed):
                    imgui.bullet_text(f"... (+{len(seq) - len(trimmed)} more)")
                imgui.tree_pop()
        else:
            imgui.bullet_text(f"{label}: {self._short_repr(value)}")

    def _describe(self, value: Any) -> Tuple[str, str]:
        if isinstance(value, dict):
            return "object", f", {len(value)} keys"
        if isinstance(value, list):
            return "list", f", {len(value)} items"
        if isinstance(value, tuple):
            return "tuple", f", {len(value)} items"
        if isinstance(value, set):
            return "set", f", {len(value)} items"
        return type(value).__name__, ""

    def _short_repr(self, value: Any) -> str:
        try:
            text = repr(value)
        except Exception:
            text = "<unrepr>"
        if len(text) > self._max_repr_len:
            text = text[: self._max_repr_len] + "..."
        return text

    def _safe_str(self, value: Any) -> str:
        try:
            return str(value)
        except Exception:
            return "<unstr>"


class imgui_ctx_id:
    """Context manager to pair push_id/pop_id when available."""

    def __init__(self, imgui: Any, ident: str) -> None:
        self._imgui = imgui
        self._ident = ident

    def __enter__(self) -> None:
        try:
            self._imgui.push_id(self._ident)
        except Exception:
            pass

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._imgui.pop_id()
        except Exception:
            pass
