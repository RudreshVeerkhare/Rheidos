"""Cook and solver drivers for Houdini node integration."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import io
import os
from typing import Any, Callable, Optional, TYPE_CHECKING
import time
import traceback
from urllib.parse import urlparse

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
from .dev_state import reset_guard
from .user_script import resolve_user_module
from rheidos.compute.profiler.core import ProfilerConfig
from rheidos.compute.profiler.runtime import (
    reset_current_profiler,
    set_current_profiler,
)
from rheidos.compute.profiler.summary_server import (
    SummaryServer,
    SummaryServerConfig,
    SummaryWriter,
    SummaryWriterConfig,
)
from rheidos.compute.profiler.taichi_probe import TaichiProbe
from rheidos.logger import _activate_scope, _make_scope

if TYPE_CHECKING:
    import hou

OUT_P = "out.P"
_LOG_SESSION_ATTR = "_rheidos_log_session_id"
_LOG_SESSION_ID: Optional[str] = None
_LOG_SESSION_PID: Optional[int] = None


def _get_hou() -> "hou":
    """Return the Houdini Python module or raise if unavailable."""
    try:
        import hou  # type: ignore
    except Exception as exc:  # pragma: no cover - only runs in Houdini
        raise RuntimeError("Houdini 'hou' module not available") from exc
    return hou


def _maybe_show_ui_message(title: str, message: str) -> None:
    try:
        hou = _get_hou()
    except Exception:
        return
    try:
        if not hou.isUIAvailable():
            return
        hou.ui.displayMessage(message, title=title)
    except Exception:
        return


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


def _report_error(
    node: "hou.Node", message: str, tb_str: str, *, debug_log: bool
) -> None:
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


def _get_callable(
    module: Any, name: str, *, required: bool
) -> Optional[Callable[..., Any]]:
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
    _configure_profiler(session, config, node)
    return session, config


def _sanitize_tb_component(value: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in value)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "")


def _make_log_session_id(pid: int) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"session-{ts}-{pid}"


def _get_log_session_id() -> str:
    pid = os.getpid()
    try:
        import hou  # type: ignore
    except Exception:
        hou = None

    if hou is not None:
        value = getattr(hou.session, _LOG_SESSION_ATTR, None)
        if isinstance(value, str) and value:
            return value
        value = _make_log_session_id(pid)
        setattr(hou.session, _LOG_SESSION_ATTR, value)
        return value

    global _LOG_SESSION_ID, _LOG_SESSION_PID
    if _LOG_SESSION_ID is None or _LOG_SESSION_PID != pid:
        _LOG_SESSION_ID = _make_log_session_id(pid)
        _LOG_SESSION_PID = pid
    return _LOG_SESSION_ID


def _resolve_profile_root_logdir(config: Any) -> str:
    base = getattr(config, "profile_logdir", None) or ""
    if base:
        base = os.path.expanduser(os.path.expandvars(base))
    else:
        hip_dir = ""
        try:
            import hou  # type: ignore

            hip_dir = os.path.dirname(hou.hipFile.path())
        except Exception:
            hip_dir = ""
        base = os.path.join(hip_dir or os.getcwd(), "_tb_logs")

    hip_name = "untitled"
    try:
        import hou  # type: ignore

        hip_path = hou.hipFile.path()
        if hip_path:
            hip_name = os.path.splitext(os.path.basename(hip_path))[0] or hip_name
    except Exception:
        pass

    safe_hip = _sanitize_tb_component(hip_name)
    return os.path.join(base, safe_hip)


def _resolve_profile_logdir(node: "hou.Node", config: Any) -> str:
    base = _resolve_profile_root_logdir(config)
    node_name = "node"
    try:
        node_name = node.path().strip("/") or node_name
    except Exception:
        pass

    safe_node = _sanitize_tb_component(node_name.replace("/", "_"))
    safe_session = _sanitize_tb_component(_get_log_session_id())
    return os.path.join(base, safe_node, safe_session)


def _resolve_logger_logdir(node: "hou.Node", config: Any) -> str:
    del node
    return _resolve_profile_root_logdir(config)


def _logger_metadata(node: "hou.Node") -> dict[str, Any]:
    hip_path = ""
    try:
        import hou  # type: ignore

        hip_path = hou.hipFile.path()
    except Exception:
        hip_path = ""
    try:
        node_path = node.path()
    except Exception:
        node_path = ""
    return {
        "hip_path": hip_path or None,
        "node_path": node_path or None,
    }


def _logger_step_hint(ctx: CookContext) -> Optional[int]:
    if ctx.substep != 0:
        return None
    return int(ctx.frame)


@contextmanager
def _activate_session_logger_scope(
    session: WorldSession,
    *,
    node: "hou.Node",
    config: Any,
    ctx: CookContext,
):
    scope = session.logger_scope
    if scope is None:
        scope = _make_scope()
        session.logger_scope = scope
    with _activate_scope(
        scope,
        default_logdir=_resolve_logger_logdir(node, config),
        step_hint=_logger_step_hint(ctx),
        metadata=_logger_metadata(node),
    ):
        yield scope


def _configure_profiler(session: WorldSession, config: Any, node: "hou.Node") -> None:
    enabled = bool(config.profile)
    mode = getattr(config, "profile_mode", None) or ("coarse" if enabled else "off")
    taichi_enabled = bool(getattr(config, "profile_taichi", True))
    if mode == "sampled_taichi":
        taichi_enabled = True
    session.profiler.configure(
        ProfilerConfig(
            enabled=enabled,
            mode=mode,
            export_hz=float(getattr(config, "profile_export_hz", 5.0)),
            taichi_enabled=taichi_enabled,
            taichi_sample_every_n_cooks=int(
                getattr(config, "profile_taichi_every", 30)
            ),
            taichi_sync_on_sample=bool(getattr(config, "profile_taichi_sync", True)),
            trace_cooks=int(getattr(config, "profile_trace_cooks", 64)),
            trace_max_edges=int(getattr(config, "profile_trace_edges", 20000)),
            overhead_enabled=bool(getattr(config, "profile_overhead", False)),
        )
    )
    session.profiler.set_taichi_sample(False)
    if session.profiler.cfg.enabled and session.profiler.cfg.taichi_enabled:
        session.taichi_probe = TaichiProbe(
            enabled=True, sync_on_sample=session.profiler.cfg.taichi_sync_on_sample
        )
        session.profiler.taichi_probe = session.taichi_probe
    else:
        session.taichi_probe = None
        session.profiler.taichi_probe = None

    logdir = _resolve_profile_logdir(node, config)
    ui_enabled = _env_flag("RHEIDOS_UI", True)
    if not session.profiler.cfg.enabled:
        if session.summary_writer is not None:
            session.summary_writer.stop()
        session.summary_writer = None
        if session.summary_server is not None:
            session.summary_server.stop()
        session.summary_server = None
        return

    if ui_enabled:
        summary_hz = max(1e-6, session.profiler.cfg.export_hz)
        writer_cfg = SummaryWriterConfig(logdir=logdir, export_hz=summary_hz)
        needs_writer = (
            session.summary_writer is None or session.summary_writer.cfg != writer_cfg
        )
        if needs_writer:
            if session.summary_writer is not None:
                session.summary_writer.stop()
            session.summary_writer = SummaryWriter(session.summary_store, writer_cfg)
            session.summary_writer.start()

        static_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "compute",
                "profiler",
                "ui",
                "dist",
            )
        )
        server_cfg = SummaryServerConfig()
        needs_server = (
            session.summary_server is None or session.summary_server.cfg != server_cfg
        )
        if needs_server:
            if session.summary_server is not None:
                session.summary_server.stop()
            session.summary_server = SummaryServer(
                session.summary_store, server_cfg, trace_provider=session.profiler
            )
            try:
                session.summary_server.start(static_dir=static_dir)
                session.stats["profile_ui_url"] = session.summary_server.url
                url = session.summary_server.url or ""
                parsed = urlparse(url) if url else None
                actual_port = parsed.port if parsed else None
                host = parsed.hostname if parsed else server_cfg.host
                owner = "<unknown>"
                try:
                    owner = node.path()
                except Exception:
                    pass
                requested = server_cfg.port
                port_detail = (
                    f"{actual_port} (requested {requested})"
                    if actual_port is not None and actual_port != requested
                    else f"{actual_port or requested}"
                )
                message = "\n".join(
                    [
                        "Profiler UI server started",
                        f"Owner: {owner}",
                        f"Host: {host}",
                        f"Port: {port_detail}",
                        f"URL: {url}" if url else "URL: <unknown>",
                        "Status: Success",
                    ]
                )
                if session.stats.get("profile_ui_notice") != message:
                    print("[rheidos] Profiler UI server started")
                    print(f"[rheidos] Owner: {owner}")
                    print(f"[rheidos] Host: {host}")
                    print(f"[rheidos] Port: {port_detail}")
                    if url:
                        print(f"[rheidos] URL: {url}")
                    _maybe_show_ui_message("Rheidos Profiler", message)
                    session.stats["profile_ui_notice"] = message
            except Exception as exc:
                try:
                    session.summary_server.stop()
                except Exception:
                    pass
                session.summary_server = None
                session.stats.pop("profile_ui_url", None)
                owner = "<unknown>"
                try:
                    owner = node.path()
                except Exception:
                    pass
                message = "\n".join(
                    [
                        "Profiler UI server failed to start",
                        f"Owner: {owner}",
                        f"Host: {server_cfg.host}",
                        f"Port: {server_cfg.port}",
                        f"Error: {exc}",
                        "Status: Failed",
                    ]
                )
                if session.stats.get("profile_ui_notice_failed") != message:
                    print("[rheidos] Failed to start profiler UI server")
                    print(f"[rheidos] Owner: {owner}")
                    print(f"[rheidos] Host: {server_cfg.host}")
                    print(f"[rheidos] Port: {server_cfg.port}")
                    print(f"[rheidos] Error: {exc}")
                    _maybe_show_ui_message("Rheidos Profiler", message)
                    session.stats["profile_ui_notice_failed"] = message
    else:
        if session.summary_writer is not None:
            session.summary_writer.stop()
        session.summary_writer = None
        if session.summary_server is not None:
            session.summary_server.stop()
        session.summary_server = None


def _warn_mode_mismatch(node: "hou.Node", mode: str, expected: str) -> None:
    if not mode or mode == expected:
        return
    message = f"Mode '{mode}' does not match {expected} node behavior."
    try:
        node.addWarning(message)
    except Exception:
        pass
    print(message)


def _start_timings(
    session: WorldSession, enabled: bool
) -> Optional[list[dict[str, Any]]]:
    if not enabled:
        session.stats.pop("timings", None)
        return None
    timings: list[dict[str, Any]] = []
    session.stats["timings"] = timings
    return timings


def _maybe_log_taichi_scoped(session: WorldSession, config: Any) -> None:
    if not bool(getattr(config, "profile_taichi_scoped_once", False)):
        return
    if session.stats.get("taichi_scoped_logged"):
        return
    try:
        import taichi as ti
    except Exception:
        return

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            ti.profiler.print_scoped_profiler_info()
    except Exception:
        return
    text = buf.getvalue()
    session.stats["taichi_scoped_logged"] = True
    if text:
        print(text)


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


@reset_guard("cook")
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
    prof_token = None
    try:
        session, config = _prepare_session(node)
        session.clear_error()
        _warn_mode_mismatch(node, config.mode, "cook")
        timings = _start_timings(session, config.profile)
        _maybe_debug(node)

        prof_token = set_current_profiler(session.profiler)
        cook_index = session.profiler.next_cook_index()
        sample_every = max(1, session.profiler.cfg.taichi_sample_every_n_cooks)
        is_sample = (
            session.profiler.cfg.enabled
            and session.profiler.cfg.taichi_enabled
            and (cook_index % sample_every == 0)
        )
        session.profiler.set_taichi_sample(is_sample)
        probe = session.taichi_probe if is_sample else None
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
        ctx = build_cook_context(
            node, input_geo, geo_out, session, geo_inputs=input_geos
        )

        _debug(config.debug_log, f"[cook] node={node.path()} module={module.__name__}")
        with _activate_session_logger_scope(session, node=node, config=config, ctx=ctx):
            with session.profiler.span("run_cook", cat="houdini"):
                with session.profiler.span("cook_total", cat="cook"):
                    if probe is not None:
                        probe.clear()
                    with _time_span(session, timings, "publish_geometry"):
                        publish_geometry_minimal(ctx)
                    with _time_span(session, timings, "user_cook"):
                        cook_fn(ctx)
                    with _time_span(session, timings, "apply_output"):
                        _apply_out_P(ctx, session)
                    if probe is not None:
                        probe.sync()
                        k_ms = probe.kernel_total_ms()
                        session.profiler.record_value("taichi", "kernel_total", None, k_ms)

        session.last_cook_at = time.time()
        _maybe_log_taichi_scoped(session, config)
        _set_last_error(node, "")
    except Exception as exc:
        tb_str = traceback.format_exc()
        if session is not None:
            session.record_error(exc, tb_str)
        debug_log = bool(getattr(config, "debug_log", False))
        _report_error(node, f"[cook] {exc}", tb_str, debug_log=debug_log)
        if geo_in is not None:
            _seed_geo_out(geo_out, geo_in)
    finally:
        if prof_token is not None:
            reset_current_profiler(prof_token)
        if session is not None:
            session.profiler.set_taichi_sample(False)


@reset_guard("solver")
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
    prof_token = None
    try:
        session, config = _prepare_session(node)
        session.clear_error()
        _warn_mode_mismatch(node, config.mode, "solver")
        timings = _start_timings(session, config.profile)
        _maybe_debug(node)

        prof_token = set_current_profiler(session.profiler)
        cook_index = session.profiler.next_cook_index()
        sample_every = max(1, session.profiler.cfg.taichi_sample_every_n_cooks)
        is_sample = (
            session.profiler.cfg.enabled
            and session.profiler.cfg.taichi_enabled
            and (cook_index % sample_every == 0)
        )
        session.profiler.set_taichi_sample(is_sample)
        probe = session.taichi_probe if is_sample else None
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

        _debug(
            config.debug_log, f"[solver] node={node.path()} module={module.__name__}"
        )
        with _activate_session_logger_scope(session, node=node, config=config, ctx=ctx):
            with session.profiler.span("run_solver", cat="houdini"):
                with session.profiler.span("solver_total", cat="cook"):
                    if probe is not None:
                        probe.clear()
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
                        _maybe_log_taichi_scoped(session, config)
                        _set_last_error(node, "")
                        return

                    with _time_span(session, timings, "user_step"):
                        step_fn(ctx)
                    session.last_step_key = step_key
                    with _time_span(session, timings, "apply_output"):
                        _apply_out_P(ctx, session)
                    session.last_geo_snapshot = _copy_geometry(geo_out)
                    session.last_cook_at = time.time()
                    if probe is not None:
                        probe.sync()
                        k_ms = probe.kernel_total_ms()
                        session.profiler.record_value("taichi", "kernel_total", None, k_ms)
                    _maybe_log_taichi_scoped(session, config)
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
    finally:
        if prof_token is not None:
            reset_current_profiler(prof_token)
        if session is not None:
            session.profiler.set_taichi_sample(False)


__all__ = ["run_cook", "run_solver"]
