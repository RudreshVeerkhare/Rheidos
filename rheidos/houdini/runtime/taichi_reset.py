"""Taichi reset helpers for Houdini integration."""

from __future__ import annotations


def reset_taichi_hard() -> None:
    try:
        import taichi as ti
    except Exception as exc:  # pragma: no cover - only runs in Houdini
        raise RuntimeError("Taichi is not available for reset") from exc

    ti.reset()
