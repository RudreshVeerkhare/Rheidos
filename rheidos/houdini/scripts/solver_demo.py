"""Minimal solver demo for Houdini nodes."""

from __future__ import annotations

import numpy as np

STATE_KEY = "state.P"


def setup(ctx) -> None:
    """Initialize solver state in the session.

    Args:
        ctx: Cook context providing access to geometry and registry.
    """
    P = ctx.P()
    ctx.publish(STATE_KEY, P.copy())


def step(ctx) -> None:
    """Advance the demo solver and write updated positions.

    Args:
        ctx: Cook context providing access to geometry and registry.
    """
    P = ctx.fetch(STATE_KEY)
    if isinstance(P, np.ndarray) and P.size:
        offset = np.sin(ctx.time) * 0.05
        P = P + offset
    ctx.publish(STATE_KEY, P)
    ctx.set_P(P)
