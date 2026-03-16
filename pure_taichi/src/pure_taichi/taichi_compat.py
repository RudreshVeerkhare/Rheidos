from __future__ import annotations

from typing import Any

_INIT_DONE = False
_INIT_ARCH: str | None = None


def import_taichi() -> Any:
    try:
        import taichi as ti
    except Exception as exc:
        raise RuntimeError(
            "Taichi is required for this operation. Install taichi>=1.7 in pure_taichi env."
        ) from exc
    return ti


def taichi_is_available() -> bool:
    try:
        import taichi  # noqa: F401
    except Exception:
        return False
    return True


def ensure_taichi_initialized(arch: str | None = None, *, offline_cache: bool = False) -> Any:
    global _INIT_DONE, _INIT_ARCH

    ti = import_taichi()

    if _INIT_DONE:
        return ti

    checker = getattr(ti, "is_initialized", None)
    if callable(checker):
        try:
            if bool(checker()):
                _INIT_DONE = True
                return ti
        except Exception:
            pass

    core = getattr(ti, "core", None)
    if core is not None and hasattr(core, "is_initialized"):
        try:
            if bool(core.is_initialized()):
                _INIT_DONE = True
                return ti
        except Exception:
            pass

    cfg: dict[str, Any] = {"offline_cache": bool(offline_cache)}
    if arch is not None:
        cfg["arch"] = getattr(ti, arch, arch)

    ti.init(**cfg)
    _INIT_DONE = True
    _INIT_ARCH = str(arch) if arch is not None else None
    return ti
