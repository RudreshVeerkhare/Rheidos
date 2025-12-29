"""Minimal cook demo for Houdini nodes."""

from __future__ import annotations

import numpy as np

from rheidos.houdini.geo import OWNER_POINT


def cook(ctx) -> None:
    """Color points based on normalized position.

    Args:
        ctx: Cook context providing access to geometry IO.
    """
    P = ctx.P()
    if P.size == 0:
        return
    mins = P.min(axis=0)
    span = np.ptp(P, axis=0)
    colors = (P - mins) / (span + 1e-6)
    ctx.write(OWNER_POINT, "Cd", colors)
