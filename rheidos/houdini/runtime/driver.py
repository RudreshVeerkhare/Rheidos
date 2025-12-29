"""Cook and solver drivers for Houdini node integration."""

from __future__ import annotations

from typing import Any, Callable, Optional, TYPE_CHECKING
import time
import traceback

import numpy as np

from ..nodes.config import read_node_config
from .cook_context import CookContext, build_cook_context
from .publish import publish_geometry_minimal
from .resource_keys import SIM_DT, SIM_FRAME, SIM_SUBSTEP, SIM_TIME
from .session import WorldSession, get_runtime
from .user_script import resolve_user_module

if TYPE_CHECKING:
    import hou

OUT_P = "out.P"


def _get_hou() -> "hou":
    try:
        import hou  # type: ignore
    except Exception as exc:  # pragma: no cover - only runs in Houdini
        raise RuntimeError("Houdini 'hou' module not available") from exc
    return hou


def _resource_exists(reg: Any, name: str) -> bool:
    try:
        reg.get(name)
    except KeyError:
        return False
    return True


def _debug(enabled: bool, message: str) -> None:
    if enabled:
        print(message)


def _report_error(node: "hou.Node", message: str, tb_str: str, *, debug_log: bool) -> None:
    try:
        node.addError(message)
    except Exception:
        pass
    print(message)
    if debug_log:
        print(tb_str)


def _seed_geo_out(geo_out: "hou.Geometry", source: Optional["hou.Geometry"]) -> None:
    if source is None:
        return
    geo_out.clear()
    geo_out.merge(source)


def _copy_geometry(geo: "hou.Geometry") -> "hou.Geometry":
    hou = _get_hou()
    snapshot = hou.Geometry()
    snapshot.merge(geo)
    return snapshot


def _apply_snapshot(geo_out: "hou.Geometry", snapshot: "hou.Geometry") -> None:
    geo_out.clear()
    geo_out.merge(snapshot)


def _get_callable(module: Any, name: str, *, required: bool) -> Optional[Callable[..., Any]]:
    fn = getattr(module, name, None)
    if fn is None:
        if required:
            raise AttributeError(f"User script is missing required '{name}(ctx)'")
        return None
    if not callable(fn):
        raise TypeError(f"User script '{name}' is not callable")
    return fn


def _apply_out_P(ctx: CookContext, session: WorldSession) -> None:
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
    ctx.publish(SIM_FRAME, ctx.frame)
    ctx.publish(SIM_TIME, ctx.time)
    ctx.publish(SIM_DT, ctx.dt)
    ctx.publish(SIM_SUBSTEP, ctx.substep)


def _prepare_session(node: "hou.Node") -> tuple[WorldSession, Any]:
    config = read_node_config(node)
    runtime = get_runtime()

    if config.nuke_all:
        runtime.nuke_all(reason="nuke_all button")

    if config.reset_node:
        runtime.reset_session(node, reason="reset_node button")

    session = runtime.get_or_create_session(node)
    return session, config


def run_cook(
    node: "hou.Node",
    geo_in: Optional["hou.Geometry"],
    geo_out: "hou.Geometry",
) -> None:
    session: Optional[WorldSession] = None
    config = None
    try:
        session, config = _prepare_session(node)
        session.clear_error()

        module = resolve_user_module(session, config, node)
        cook_fn = _get_callable(module, "cook", required=True)

        input_geo = geo_in if geo_in is not None else geo_out
        if geo_in is not None:
            _seed_geo_out(geo_out, geo_in)
        ctx = build_cook_context(node, input_geo, geo_out, session)

        _debug(config.debug_log, f"[cook] node={node.path()} module={module.__name__}")
        publish_geometry_minimal(ctx)
        cook_fn(ctx)
        _apply_out_P(ctx, session)

        session.last_cook_at = time.time()
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
    session: Optional[WorldSession] = None
    config = None
    try:
        session, config = _prepare_session(node)
        session.clear_error()

        module = resolve_user_module(session, config, node)
        setup_fn = _get_callable(module, "setup", required=False)
        step_fn = _get_callable(module, "step", required=True)

        source_geo = geo_prev if geo_prev is not None else geo_in
        if source_geo is not None:
            _seed_geo_out(geo_out, source_geo)
        input_geo = source_geo if source_geo is not None else geo_out
        ctx = build_cook_context(node, input_geo, geo_out, session, substep=substep)

        _debug(config.debug_log, f"[solver] node={node.path()} module={module.__name__}")
        publish_geometry_minimal(ctx)
        _publish_sim_keys(ctx)

        if not session.did_setup:
            if setup_fn is not None:
                setup_fn(ctx)
            session.did_setup = True

        step_key = (ctx.frame, ctx.substep)
        if step_key == session.last_step_key:
            if session.last_geo_snapshot is not None:
                _apply_snapshot(geo_out, session.last_geo_snapshot)
            elif OUT_P in session.last_output_cache:
                ctx.set_P(session.last_output_cache[OUT_P])
            session.last_cook_at = time.time()
            return

        step_fn(ctx)
        session.last_step_key = step_key
        _apply_out_P(ctx, session)
        session.last_geo_snapshot = _copy_geometry(geo_out)
        session.last_cook_at = time.time()
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
