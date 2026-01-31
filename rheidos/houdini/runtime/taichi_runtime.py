"""Taichi lifecycle helpers for Houdini integration."""

from __future__ import annotations

from typing import Any, Dict, Optional
import os


def _get_taichi():
    try:
        import taichi as ti
    except Exception as exc:  # pragma: no cover - only runs with Taichi
        raise RuntimeError("Taichi is not available") from exc
    return ti


def _resolve_arch(value: Any, ti: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return getattr(ti, value, value)
    return value


def taichi_sync() -> bool:
    try:
        ti = _get_taichi()
        ti.sync()
    except Exception:
        return False
    return True


def taichi_reset() -> None:
    ti = _get_taichi()
    checker = getattr(ti, "is_initialized", None)
    if callable(checker):
        try:
            if not bool(checker()):
                return
        except Exception:
            pass
    ti.reset()


def taichi_init(config: Optional[Dict[str, Any]] = None) -> None:
    cfg: Dict[str, Any] = dict(config or {})

    if "offline_cache" not in cfg:
        cfg["offline_cache"] = False
    if cfg.get("offline_cache") is False:
        os.environ["TI_OFFLINE_CACHE"] = "0"

    ti = _get_taichi()
    if "arch" in cfg:
        cfg["arch"] = _resolve_arch(cfg["arch"], ti)

    checker = getattr(ti, "is_initialized", None)
    if callable(checker):
        try:
            if bool(checker()):
                return
        except Exception:
            pass

    ti.init(**cfg)


__all__ = ["taichi_sync", "taichi_reset", "taichi_init"]
