from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ...scene_config_live import SceneConfigLiveManager


class SceneConfigPanel:
    """Editable scene-config panel with diff/force-apply controls."""

    id = "scene-config"
    title = "Scene Config"
    order = 45
    separate_window = True

    def __init__(
        self,
        engine,
        config_path: Path,
        default_pick_mask: Optional[Any] = None,
        initial_config: Optional[dict] = None,
        initial_result: Optional[Any] = None,
    ) -> None:
        self._engine = engine
        self._path = Path(config_path).expanduser()
        self._manager = SceneConfigLiveManager(
            engine,
            self._path,
            default_pick_mask=default_pick_mask,
            initial_cfg=initial_config,
            initial_result=initial_result,
        )
        self._buffer = self._safe_read()
        self._status: str = ""
        self._error: str = ""
        self._save_on_apply: bool = False

    def draw(self, imgui: Any) -> None:
        try:
            changed, new_text = imgui.input_text_multiline(
                "##scene-config-text",
                self._buffer,
                (520, 420),
            )
            if changed:
                self._buffer = new_text
        except Exception as exc:
            self._error = f"Input error: {exc}"
            self._status = ""
            # Keep going to show error text and buttons.

        if imgui.button("Apply Diff"):
            self._apply(force=False, save=self._save_on_apply)
        imgui.same_line()
        if imgui.button("Force Reload"):
            self._apply(force=True, save=self._save_on_apply)
        imgui.same_line()
        if imgui.button("Reload From Disk"):
            self._reload_buffer_from_disk()
        imgui.same_line()
        if imgui.button("Save Buffer to Disk"):
            self._save_buffer()

        changed_save, save_val = imgui.checkbox("Save to disk on apply", self._save_on_apply)
        if changed_save:
            self._save_on_apply = bool(save_val)

        if self._error:
            imgui.text_colored(self._error, 1.0, 0.4, 0.4, 1.0)
        elif self._status:
            imgui.text_disabled(self._status)

        imgui.separator()
        imgui.text_disabled(str(self._path))

    # --- internals --------------------------------------------------

    def _apply(self, *, force: bool, save: bool) -> None:
        self._error = ""
        self._status = "Applying..."
        try:
            if save:
                self._save_buffer()
            summary = self._manager.apply_text(self._buffer, force=force)
            self._status = f"Applied: {summary.describe()}"
        except Exception as exc:
            self._error = f"Error: {exc}"
            self._status = ""

    def _reload_buffer_from_disk(self) -> None:
        try:
            self._buffer = self._path.read_text()
            self._status = "Reloaded from disk."
            self._error = ""
        except Exception as exc:
            self._error = f"Failed to read: {exc}"
            self._status = ""

    def _safe_read(self) -> str:
        try:
            return self._path.read_text()
        except Exception:
            return ""

    def _save_buffer(self) -> None:
        try:
            self._path.write_text(self._buffer)
            self._status = f"Saved to {self._path}"
            self._error = ""
        except Exception as exc:
            self._error = f"Failed to save: {exc}"
            self._status = ""
