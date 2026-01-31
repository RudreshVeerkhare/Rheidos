"""Taichi reset helpers for Houdini integration."""

from __future__ import annotations


def reset_taichi_hard() -> None:
    from .taichi_runtime import taichi_reset

    taichi_reset()


__all__ = ["reset_taichi_hard"]
