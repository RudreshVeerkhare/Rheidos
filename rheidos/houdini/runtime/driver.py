"""Cook and solver drivers for Houdini node integration."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Optional, TYPE_CHECKING
import time
import traceback

import numpy as np

from ..nodes.config import read_node_config
from ..debug import (
    consume_break_next_button,
    debug_config_from_node,
    ensure_debug_server,
    maybe_break_now,
    request_break_next,
)
from .cook_context import CookContext, build_cook_context
from .publish import publish_geometry_minimal
from .resource_keys import SIM_DT, SIM_FRAME, SIM_SUBSTEP, SIM_TIME
from .session import WorldSession, get_runtime
from .user_script import resolve_user_module

if TYPE_CHECKING:
    import hou

OUT_P = "out.P"


def _get_hou() -> "hou":
    """Return the Houdini Python module or raise if unavailable."""
    try:
        import hou  # type: ignore
    except Exception as exc:  # pragma: no cover - only runs in Houdini
        raise RuntimeError("Houdini 'hou' module not available") from exc
    return hou


def _resource_exists(reg: Any, name: str) -> bool:
    """Check whether a registry-like object contains a named resource.

    Args:
        reg: Registry-like object with a get(name) method.
        name: Resource name to check.
    """
    try:
        reg.get(name)
    except KeyError:
        return False
    return True


def _debug(enabled: bool, message: str) -> None:
    """Print a debug message if enabled.

    Args:
        enabled: True to print the message.
        message: Message to emit.
    """
    if enabled:
        print(message)


def _report_error(node: "hou.Node", message: str, tb_str: str, *, debug_log: bool) -> None:
    """Report a cook/solver error to Houdini and optionally print a traceback.

    Args:
        node: Houdini node receiving the error.
        message: Short error summary for the node UI/console.
        tb_str: Full traceback string.
        debug_log: If True, also print the traceback.
    """
    try:
        node.addError(message)
    except Exception:
        pass
    _set_last_error(node, message)
    print(message)
    if debug_log:
        print(tb_str)


def _set_last_error(node: "hou.Node", message: str) -> None:
    parm = node.parm("last_error")
    if parm is None:
        return
    try:
        parm.set(message)
    except Exception:
        return


def _clear_action_parm(node: "hou.Node", name: str) -> None:
    parm = node.parm(name)
    if parm is None:
        return
    try:
        if parm.eval():
            parm.set(0)
    except Exception:
        return


def _seed_geo_out(geo_out: "hou.Geometry", source: Optional["hou.Geometry"]) -> None:
    """Overwrite output geometry with a source geometry snapshot.

    Args:
        geo_out: Output geometry to clear and fill.
        source: Source geometry to merge into geo_out.
    """
    if source is None:
        return
    geo_out.clear()
    geo_out.merge(source)


def _collect_input_geos(node: "hou.Node") -> list[Optional["hou.Geometry"]]:
    inputs = node.inputs()
    if not inputs:
        return []
    geos: list[Optional["hou.Geometry"]] = []
    for input_node in inputs:
        if input_node is None:
            geos.append(None)
            continue
        try:
            geo = input_node.geometry()
        except Exception:
            geo = None
        geos.append(geo)
    return geos


def _copy_geometry(geo: "hou.Geometry") -> "hou.Geometry":
    """Return a deep copy of a Houdini geometry object.

    Args:
        geo: Geometry to clone.
    """
    hou = _get_hou()
    snapshot = hou.Geometry()
    snapshot.merge(geo)
    return snapshot


def _apply_snapshot(geo_out: "hou.Geometry", snapshot: "hou.Geometry") -> None:
    """Replace output geometry with a cached snapshot.

    Args:
        geo_out: Geometry to overwrite.
        snapshot: Cached geometry copy to apply.
    """
    geo_out.clear()
    geo_out.merge(snapshot)


def _get_callable(module: Any, name: str, *, required: bool) -> Optional[Callable[..., Any]]:
    """Return a callable attribute from a user module or raise if required.

    Args:
        module: User module holding entrypoints.
        name: Attribute name to look up.
        required: If True, raise when missing.
    """
    fn = getattr(module, name, None)
    if fn is None:
        if required:
            raise AttributeError(f"User script is missing required '{name}(ctx)'")
        return None
    if not callable(fn):
        raise TypeError(f"User script '{name}' is not callable")
    return fn


def _apply_out_P(ctx: CookContext, session: WorldSession) -> None:
    """Apply the optional out.P resource to point positions.

    Args:
        ctx: Cook context with world/geometry access.
        session: Session used for caching the last output.
    """
    reg = ctx.world().reg
    if not _resource_exists(reg, OUT_P):
        return
    values = ctx.fetch(OUT_P)
    ctx.set_P(values)
    if isinstance(values, np.ndarray):
        session.last_output_cache[OUT_P] = values.copy()
    else:
        session.last_output_cache[OUT_P] = values


def _publish_sim_keys(ctx: CookContext) -> None:
    """Publish simulation timing keys into the compute registry.

    Args:
        ctx: Cook context with timing data.
    """
    ctx.publish(SIM_FRAME, ctx.frame)
    ctx.publish(SIM_TIME, ctx.time)
    ctx.publish(SIM_DT, ctx.dt)
    ctx.publish(SIM_SUBSTEP, ctx.substep)


def _prepare_session(node: "hou.Node") -> tuple[WorldSession, Any]:
    """Read node config, apply reset/nuke, and return the session plus config.

    Args:
        node: Houdini node to read parameters from.
    """
    config = read_node_config(node)
    runtime = get_runtime()

    if config.nuke_all:
        runtime.nuke_all(reason="nuke_all button")
        _clear_action_parm(node, "nuke_all")

    if config.reset_node:
        runtime.reset_session(node, reason="reset_node button")
        _clear_action_parm(node, "reset_node")

    session = runtime.get_or_create_session(node)
    return session, config


def _warn_mode_mismatch(node: "hou.Node", mode: str, expected: str) -> None:
    if not mode or mode == expected:
        return
    message = f"Mode '{mode}' does not match {expected} node behavior."
    try:
        node.addWarning(message)
    except Exception:
        pass
    print(message)


def _start_timings(session: WorldSession, enabled: bool) -> Optional[list[dict[str, Any]]]:
    if not enabled:
        session.stats.pop("timings", None)
        return None
    timings: list[dict[str, Any]] = []
    session.stats["timings"] = timings
    return timings


@contextmanager
def _time_span(
    session: WorldSession,
    timings: Optional[list[dict[str, Any]]],
    name: str,
) -> None:
    if timings is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings.append({"name": name, "ms": elapsed_ms})
        session.stats["last_timings"] = timings


def _maybe_debug(node: "hou.Node") -> None:
    try:
        cfg = debug_config_from_node(node)
        ensure_debug_server(cfg, node=node)
        if consume_break_next_button(node):
            request_break_next(node=node)
        maybe_break_now(node=node)
    except Exception:
        return


def run_cook(
    node: "hou.Node",
    geo_in: Optional["hou.Geometry"],
    geo_out: "hou.Geometry",
) -> None:
    """Run a stateless cook for a Houdini node.

    Args:
        node: Houdini node executing the cook.
        geo_in: Input geometry (if any).
        geo_out: Output geometry to populate.
    """
    session: Optional[WorldSession] = None
    config = None
    try:
        session, config = _prepare_session(node)
        session.clear_error()
        _warn_mode_mismatch(node, config.mode, "cook")
        timings = _start_timings(session, config.profile)
        _maybe_debug(node)

        module = resolve_user_module(session, config, node)
        cook_fn = _get_callable(module, "cook", required=True)

        input_geo = geo_in if geo_in is not None else geo_out
        if geo_in is not None:
            _seed_geo_out(geo_out, geo_in)
        input_geos = _collect_input_geos(node)
        if input_geos:
            if geo_in is not None:
                input_geos[0] = geo_in
        elif geo_in is not None:
            input_geos = [geo_in]
        ctx = build_cook_context(node, input_geo, geo_out, session, geo_inputs=input_geos)

        _debug(config.debug_log, f"[cook] node={node.path()} module={module.__name__}")
        with _time_span(session, timings, "publish_geometry"):
            publish_geometry_minimal(ctx)
        with _time_span(session, timings, "user_cook"):
            cook_fn(ctx)
        with _time_span(session, timings, "apply_output"):
            _apply_out_P(ctx, session)

        session.last_cook_at = time.time()
        _set_last_error(node, "")
    except Exception as exc:
        tb_str = traceback.format_exc()
        if session is not None:
            session.record_error(exc, tb_str)
        debug_log = bool(getattr(config, "debug_log", False))
        _report_error(node, f"[cook] {exc}", tb_str, debug_log=debug_log)
        if geo_in is not None:
            _seed_geo_out(geo_out, geo_in)


def run_solver(
    node: "hou.Node",
    geo_prev: Optional["hou.Geometry"],
    geo_in: Optional["hou.Geometry"],
    geo_out: "hou.Geometry",
    *,
    substep: int = 0,
) -> None:
    """Run a stateful solver step for a Houdini node.

    Args:
        node: Houdini node executing the solver cook.
        geo_prev: Previous-frame geometry (solver input 0).
        geo_in: Current-frame geometry (solver input 1).
        geo_out: Output geometry to populate.
        substep: Optional substep index for multi-step solvers.
    """
    session: Optional[WorldSession] = None
    config = None
    try:
        session, config = _prepare_session(node)
        session.clear_error()
        _warn_mode_mismatch(node, config.mode, "solver")
        timings = _start_timings(session, config.profile)
        _maybe_debug(node)

        module = resolve_user_module(session, config, node)
        setup_fn = _get_callable(module, "setup", required=False)
        step_fn = _get_callable(module, "step", required=True)

        source_geo = geo_prev if geo_prev is not None else geo_in
        if source_geo is not None:
            _seed_geo_out(geo_out, source_geo)
        input_geo = source_geo if source_geo is not None else geo_out
        input_geos = _collect_input_geos(node)
        if input_geos:
            if geo_prev is not None:
                input_geos[0] = geo_prev
            if len(input_geos) > 1 and geo_in is not None:
                input_geos[1] = geo_in
        else:
            input_geos = [geo_prev, geo_in]
        ctx = build_cook_context(
            node,
            input_geo,
            geo_out,
            session,
            geo_inputs=input_geos,
            substep=substep,
            is_solver=True,
        )

        _debug(config.debug_log, f"[solver] node={node.path()} module={module.__name__}")
        with _time_span(session, timings, "publish_geometry"):
            publish_geometry_minimal(ctx)
        _publish_sim_keys(ctx)

        if not session.did_setup:
            if setup_fn is not None:
                with _time_span(session, timings, "user_setup"):
                    setup_fn(ctx)
            session.did_setup = True

        step_key = (ctx.frame, ctx.substep)
        if step_key == session.last_step_key:
            if session.last_geo_snapshot is not None:
                with _time_span(session, timings, "apply_snapshot"):
                    _apply_snapshot(geo_out, session.last_geo_snapshot)
            elif OUT_P in session.last_output_cache:
                with _time_span(session, timings, "apply_cached_out"):
                    ctx.set_P(session.last_output_cache[OUT_P])
            session.last_cook_at = time.time()
            _set_last_error(node, "")
            return

        with _time_span(session, timings, "user_step"):
            step_fn(ctx)
        session.last_step_key = step_key
        with _time_span(session, timings, "apply_output"):
            _apply_out_P(ctx, session)
        session.last_geo_snapshot = _copy_geometry(geo_out)
        session.last_cook_at = time.time()
        _set_last_error(node, "")
    except Exception as exc:
        tb_str = traceback.format_exc()
        if session is not None:
            session.record_error(exc, tb_str)
        debug_log = bool(getattr(config, "debug_log", False))
        _report_error(node, f"[solver] {exc}", tb_str, debug_log=debug_log)
        source_geo = geo_prev if geo_prev is not None else geo_in
        if source_geo is not None:
            _seed_geo_out(geo_out, source_geo)


__all__ = ["run_cook", "run_solver"]
