"""Reset + reload orchestration for Houdini dev workflows."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional
import gc
import importlib
import time

from .dev_state import current_busy_reason, get_dev_state, is_busy
from .module_reloader import reload_project
from .session import get_runtime, get_sim_context, set_sim_context
from .taichi_runtime import taichi_init, taichi_reset, taichi_sync


def _get_hou():
    try:
        import hou  # type: ignore
    except Exception:
        return None
    return hou


def _show_ui_message(title: str, message: str) -> None:
    hou = _get_hou()
    if hou is None:
        return
    try:
        if not hou.isUIAvailable():
            return
        hou.ui.displayMessage(message, title=title)
    except Exception:
        return


def _rehydrate_context(pkg_name: str) -> None:
    try:
        session_mod = importlib.import_module(f"{pkg_name}.houdini.runtime.session")
        get_ctx = getattr(session_mod, "get_sim_context", None)
        if callable(get_ctx):
            get_ctx(create=True)
    except Exception:
        return


def reset_and_reload(
    *,
    pkg: str = "rheidos",
    taichi_cfg: Optional[Dict[str, Any]] = None,
    rebuild_fn: Optional[Callable[[], Any]] = None,
) -> Any:
    """Reset runtime + taichi and reload project package."""
    state = get_dev_state()
    if state.reloading:
        raise RuntimeError("Reload already in progress.")
    if is_busy():
        reason = current_busy_reason() or "active cook"
        raise RuntimeError(f"Cannot reload during {reason}.")

    state.reloading = True
    state.last_reload_error = None
    try:
        # Step 2: teardown old simulation state.
        sim = get_sim_context(create=False)
        if sim is not None:
            close = getattr(sim, "close", None)
            if callable(close):
                close("reset_and_reload")
            else:
                runtime = getattr(sim, "runtime", None)
                if runtime is not None:
                    runtime.nuke_all("reset_and_reload", reset_taichi=False)
        else:
            runtime = get_runtime(create=False)
            if runtime is not None:
                runtime.nuke_all("reset_and_reload", reset_taichi=False)
        set_sim_context(None)
        gc.collect()

        # Step 3: sync (best effort).
        taichi_sync()

        # Step 4: reset taichi.
        taichi_reset()

        # Step 5: init taichi with dev config.
        taichi_init(taichi_cfg)

        # Step 6: purge + reload project package.
        module = reload_project(pkg)

        # Step 7: rebuild sim state (optional).
        _rehydrate_context(pkg)
        if rebuild_fn is not None:
            rebuild_fn()

        return module
    except Exception as exc:
        state.last_reload_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        state.last_reload_at = time.time()
        state.reloading = False


def reset_and_reload_with_ui(
    *,
    pkg: str = "rheidos",
    taichi_cfg: Optional[Dict[str, Any]] = None,
    rebuild_fn: Optional[Callable[[], Any]] = None,
    title: str = "Rheidos",
    success_message: str = "Reload + Reset complete.",
    raise_on_error: bool = False,
) -> Optional[Any]:
    try:
        module = reset_and_reload(
            pkg=pkg, taichi_cfg=taichi_cfg, rebuild_fn=rebuild_fn
        )
        _show_ui_message(title, success_message)
        return module
    except Exception as exc:
        _show_ui_message(title, f"Reload failed:\n{exc}")
        if raise_on_error:
            raise
        return None


__all__ = ["reset_and_reload", "reset_and_reload_with_ui"]
