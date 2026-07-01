"""Session cache and runtime for Houdini nodes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import wraps
import inspect
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, Literal, overload
import time
import warnings

import numpy as np

from rheidos.compute.world import World
from rheidos.compute.profiler.core import Profiler, ProfilerConfig
from rheidos.compute.profiler.runtime import reset_current_profiler, set_current_profiler
from rheidos.compute.profiler.summary_store import SummaryStore

from .taichi_reset import reset_taichi_hard

if TYPE_CHECKING:
    import hou

__all__ = [
    "AccessMode",
    "ComputeRuntime",
    "SessionAccess",
    "SessionKey",
    "WorldSession",
    "get_sim_context",
    "get_runtime",
    "make_session_key",
    "make_session_key_for_path",
    "session",
    "set_sim_context",
]

_SIM_ATTR = "RHEIDOS_SIM"
_SHARED_SESSION_PREFIX = "::rheidos_shared_session__:"
_SESSION_OWNER_MODULE_KEY = "decorator_session_owner_module"
_SESSION_OWNER_QUALNAME_KEY = "decorator_session_owner_qualname"
_SESSION_OWNER_FILE_KEY = "decorator_session_owner_file"
_SESSION_OWNER_MISMATCH_KEY = "decorator_session_owner_mismatch"
_SESSION_OWNER_WARNING_KEY = "decorator_session_owner_mismatch_warned"


def _get_hou():
    try:
        import hou  # type: ignore
    except Exception:
        return None
    return hou


@dataclass(frozen=True)
class SessionKey:
    hip_path: str
    node_path: str


def make_session_key(node: "hou.Node") -> SessionKey:
    try:
        import hou  # type: ignore
    except Exception as exc:  # pragma: no cover - only runs in Houdini
        raise RuntimeError("Houdini 'hou' module not available") from exc

    return SessionKey(hip_path=hou.hipFile.path(), node_path=node.path())


def make_session_key_for_path(node_path: str) -> SessionKey:
    try:
        import hou  # type: ignore
    except Exception as exc:  # pragma: no cover - only runs in Houdini
        raise RuntimeError("Houdini 'hou' module not available") from exc

    return SessionKey(hip_path=hou.hipFile.path(), node_path=node_path)


def _normalize_session_name(key: Any) -> str:
    if not isinstance(key, str):
        raise TypeError("session key must be a string")
    value = key.strip()
    if not value:
        raise ValueError("session key must be a non-empty string")
    return value


def _make_shared_session_key(key: str) -> SessionKey:
    try:
        import hou  # type: ignore
    except Exception as exc:  # pragma: no cover - only runs in Houdini
        raise RuntimeError("Houdini 'hou' module not available") from exc

    name = _normalize_session_name(key)
    return SessionKey(
        hip_path=hou.hipFile.path(),
        node_path=f"{_SHARED_SESSION_PREFIX}{name}",
    )


def _make_runtime_session_key(node: "hou.Node", *, key: Optional[str] = None) -> SessionKey:
    if key is None:
        return make_session_key(node)
    return _make_shared_session_key(key)


def _session_scope_label(key: Optional[str]) -> str:
    if key is None:
        return "node-local session"
    return f"named session '{key}'"


def _resolve_entrypoint_node(
    fn_name: str, *, key: Optional[str] = None
) -> "hou.Node":
    hou = _get_hou()
    scope = _session_scope_label(key)
    if hou is None:
        raise RuntimeError(
            f"@session could not resolve the {scope} for '{fn_name}': Houdini 'hou' module not available"
        )
    pwd = getattr(hou, "pwd", None)
    if not callable(pwd):
        raise RuntimeError(
            f"@session could not resolve the {scope} for '{fn_name}': hou.pwd() is unavailable"
        )
    try:
        node = pwd()
    except Exception as exc:
        raise RuntimeError(
            f"@session could not resolve the {scope} for '{fn_name}': hou.pwd() failed"
        ) from exc
    if node is None:
        raise RuntimeError(
            f"@session could not resolve the {scope} for '{fn_name}': hou.pwd() returned None"
        )
    return node


def _get_owner_metadata(fn: Callable[..., Any]) -> tuple[str, str, str]:
    module_name = getattr(fn, "__module__", "<unknown>")
    qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", "<unknown>"))
    try:
        filename = inspect.getsourcefile(fn) or inspect.getfile(fn) or "<unknown>"
    except (OSError, TypeError):
        filename = "<unknown>"
    return module_name, qualname, filename


def _register_named_session_owner(
    session_obj: "WorldSession",
    *,
    key: str,
    fn: Callable[..., Any],
) -> None:
    module_name, qualname, filename = _get_owner_metadata(fn)
    first_module = session_obj.stats.get(_SESSION_OWNER_MODULE_KEY)
    first_qualname = session_obj.stats.get(_SESSION_OWNER_QUALNAME_KEY)
    first_file = session_obj.stats.get(_SESSION_OWNER_FILE_KEY)

    if first_module is None and first_file is None:
        session_obj.stats[_SESSION_OWNER_MODULE_KEY] = module_name
        session_obj.stats[_SESSION_OWNER_QUALNAME_KEY] = qualname
        session_obj.stats[_SESSION_OWNER_FILE_KEY] = filename
        return

    if first_module == module_name and first_file == filename:
        return

    if session_obj.stats.get(_SESSION_OWNER_WARNING_KEY):
        return

    first_owner = f"{first_module}.{first_qualname}" if first_module and first_qualname else "<unknown>"
    current_owner = f"{module_name}.{qualname}"
    message = (
        f"Named session '{key}' was first claimed by '{first_owner}'"
        f" ({first_file or '<unknown>'}) and is now reused by "
        f"'{current_owner}' ({filename}). Sharing will continue."
    )
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    session_obj.stats[_SESSION_OWNER_MISMATCH_KEY] = {
        "session_key": key,
        "first_owner_module": first_module,
        "first_owner_qualname": first_qualname,
        "first_owner_file": first_file,
        "current_owner_module": module_name,
        "current_owner_qualname": qualname,
        "current_owner_file": filename,
    }
    session_obj.stats[_SESSION_OWNER_WARNING_KEY] = True


def _collect_input_geos(node: "hou.Node") -> tuple[Optional["hou.Geometry"], ...]:
    inputs = node.inputs()
    if not inputs:
        return ()

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
    return tuple(geos)


def _eval_optional_str_parm(node: object, name: str) -> str:
    parm_getter = getattr(node, "parm", None)
    if not callable(parm_getter):
        return ""
    parm = parm_getter(name)
    if parm is None:
        return ""
    try:
        value = parm.evalAsString()
    except Exception:
        try:
            value = parm.eval()
        except Exception:
            value = ""
    return "" if value is None else str(value)


def _eval_action_parm(node: "hou.Node", name: str) -> bool:
    parm = getattr(node, "parm", None)
    if not callable(parm):
        return False
    try:
        value = parm(name)
    except Exception:
        return False
    if value is None:
        return False
    try:
        return bool(value.eval())
    except Exception:
        return False


def _clear_action_parm(node: "hou.Node", name: str) -> None:
    parm = getattr(node, "parm", None)
    if not callable(parm):
        return
    try:
        value = parm(name)
    except Exception:
        return
    if value is None:
        return
    try:
        if value.eval():
            value.set(0)
    except Exception:
        return


def _prepare_entrypoint_session(node: "hou.Node", *, key: Optional[str]) -> "WorldSession":
    runtime = get_runtime()

    if _eval_action_parm(node, "nuke_all"):
        runtime.nuke_all(reason="nuke_all button")
        _clear_action_parm(node, "nuke_all")

    if _eval_action_parm(node, "reset_node"):
        runtime.reset_session(node, reason="reset_node button", key=key)
        _clear_action_parm(node, "reset_node")

    return runtime.get_or_create_session(node, key=key)


def _build_entrypoint_context(
    node: "hou.Node",
    *,
    session_obj: "WorldSession",
) -> Any:
    from .cook_context import build_cook_context

    try:
        geo_out = node.geometry()
    except Exception as exc:
        raise RuntimeError(
            f"@session could not resolve output geometry for '{node.path()}'"
        ) from exc

    input_geos = _collect_input_geos(node)
    geo_in = geo_out
    if input_geos and input_geos[0] is not None:
        geo_in = input_geos[0]

    return build_cook_context(
        node,
        geo_in,
        geo_out,
        session_obj,
        geo_inputs=input_geos,
    )


def _run_entrypoint_debug(node: "hou.Node") -> None:
    from rheidos.houdini.debug import (
        consume_break_next_button,
        debug_config_from_node,
        ensure_debug_server,
        maybe_break_now,
        request_break_next,
    )

    cfg = replace(debug_config_from_node(node), enabled=True)
    ensure_debug_server(cfg, node=node)
    if consume_break_next_button(node):
        request_break_next(node=node)
    maybe_break_now(node=node)


def _forced_profiler_config(*, profile_logdir: Optional[str] = None) -> SimpleNamespace:
    return SimpleNamespace(
        profile=True,
        profile_logdir=profile_logdir,
        profile_export_hz=5.0,
        profile_mode="coarse",
        profile_trace_cooks=64,
        profile_trace_edges=20000,
        profile_overhead=False,
        profile_taichi=True,
        profile_taichi_every=30,
        profile_taichi_sync=True,
        profile_taichi_scoped_once=False,
    )


def _call_with_forced_profiler(
    fn: Callable[..., Any],
    ctx: Any,
    entry_kwargs: Dict[str, Any],
    *,
    node: "hou.Node",
    session_obj: "WorldSession",
    profile_logdir: Optional[str],
) -> Any:
    from .driver import _configure_profiler

    config = _forced_profiler_config(profile_logdir=profile_logdir)
    _configure_profiler(session_obj, config, node)

    token = set_current_profiler(session_obj.profiler)
    cook_index = session_obj.profiler.next_cook_index()
    sample_every = max(1, session_obj.profiler.cfg.taichi_sample_every_n_cooks)
    is_sample = (
        session_obj.profiler.cfg.enabled
        and session_obj.profiler.cfg.taichi_enabled
        and (cook_index % sample_every == 0)
    )
    session_obj.profiler.set_taichi_sample(is_sample)
    probe = session_obj.taichi_probe if is_sample else None

    try:
        if probe is not None:
            probe.clear()
        result = fn(ctx, **entry_kwargs)
        if probe is not None:
            probe.sync()
            k_ms = probe.kernel_total_ms()
            session_obj.profiler.record_value("taichi", "kernel_total", None, k_ms)
        return result
    finally:
        reset_current_profiler(token)
        session_obj.profiler.set_taichi_sample(False)


def _session_signature_error(fn: Callable[..., Any]) -> TypeError:
    return TypeError(
        f"@session target '{fn.__qualname__}' must be defined as "
        f"'def {fn.__name__}(ctx, optional=value)'"
    )


def _session_optional_param_names(fn: Callable[..., Any]) -> set[str]:
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if (
        len(params) < 1
        or params[0].name != "ctx"
        or params[0].kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ):
        raise _session_signature_error(fn)

    optional_param_names: set[str] = set()
    for param in params[1:]:
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ):
            raise _session_signature_error(fn)
        if param.default is inspect.Parameter.empty:
            raise _session_signature_error(fn)
        optional_param_names.add(param.name)
    return optional_param_names


def _decorate_session_entrypoint(
    fn: Callable[..., Any],
    *,
    key: Optional[str],
    debugger: bool,
    profiler: bool,
) -> Callable[..., Any]:
    optional_param_names = _session_optional_param_names(fn)

    @wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise TypeError(
                f"@session target '{fn.__qualname__}' injects 'ctx'; "
                "do not pass arguments explicitly; only keyword entrypoint "
                "parameters are supported"
            )
        unknown_kwargs = set(kwargs) - optional_param_names
        if unknown_kwargs:
            names = ", ".join(sorted(unknown_kwargs))
            raise TypeError(
                f"@session target '{fn.__qualname__}' got unexpected keyword "
                f"argument(s): {names}"
            )
        entry_kwargs = dict(kwargs)

        node = _resolve_entrypoint_node(fn.__qualname__, key=key)
        session_obj = _prepare_entrypoint_session(node, key=key)
        if key is not None:
            _register_named_session_owner(session_obj, key=key, fn=fn)
        if debugger:
            _run_entrypoint_debug(node)
        ctx = _build_entrypoint_context(node, session_obj=session_obj)
        profile_logdir = _eval_optional_str_parm(node, "profile_logdir") or None
        logger_config = SimpleNamespace(profile_logdir=profile_logdir)
        from .driver import _activate_session_logger_scope

        with _activate_session_logger_scope(
            session_obj,
            node=node,
            config=logger_config,
            ctx=ctx,
        ):
            if profiler:
                return _call_with_forced_profiler(
                    fn,
                    ctx,
                    entry_kwargs,
                    node=node,
                    session_obj=session_obj,
                    profile_logdir=profile_logdir,
                )
            return fn(ctx, **entry_kwargs)

    return wrapped


@overload
def session(fn: Callable[..., Any]) -> Callable[..., Any]:
    ...


@overload
def session(
    name: str,
    *,
    debugger: bool = False,
    profiler: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...


@overload
def session(
    fn: None = None,
    *,
    key: str,
    debugger: bool = False,
    profiler: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...


def session(
    fn: Optional[Callable[..., Any]] = None,
    *,
    key: Optional[str] = None,
    debugger: bool = False,
    profiler: bool = False,
) -> Any:
    """Inject a cook context into a manual Houdini entrypoint.

    Supported forms:
    - ``@session`` for node-local sessions
    - ``@session("name")`` for shared named sessions
    - ``@session(key="name")`` for shared named sessions
    """

    if callable(fn) and key is None:
        return _decorate_session_entrypoint(
            fn,
            key=None,
            debugger=debugger,
            profiler=profiler,
        )

    if fn is None:
        normalized_key = None if key is None else _normalize_session_name(key)

        def decorator(target: Callable[..., Any]) -> Callable[..., Any]:
            return _decorate_session_entrypoint(
                target,
                key=normalized_key,
                debugger=debugger,
                profiler=profiler,
            )

        return decorator

    if key is None:
        normalized_key = _normalize_session_name(fn)

        def decorator(target: Callable[..., Any]) -> Callable[..., Any]:
            return _decorate_session_entrypoint(
                target,
                key=normalized_key,
                debugger=debugger,
                profiler=profiler,
            )

        return decorator

    raise TypeError(
        "session() supports only @session, @session('name'), or @session(key='name') with optional debugger/profiler flags"
    )


AccessMode = Literal["read", "write"]


@dataclass
class WorldSession:
    world: Optional[World] = None
    user_module: Optional[ModuleType] = None
    user_module_key: Optional[str] = None
    did_setup: bool = False
    last_step_key: Optional[Tuple[Any, ...]] = None
    last_output_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    last_geo_snapshot: Optional[Any] = None
    last_triangles: Optional[np.ndarray] = None
    last_topology_sig: Optional[Tuple[int, int, int]] = None
    last_topology_key: Optional[Tuple[Any, ...]] = None
    last_triangles_by_input: Dict[int, np.ndarray] = field(default_factory=dict)
    last_topology_sig_by_input: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)
    last_topology_key_by_input: Dict[int, Tuple[Any, ...]] = field(default_factory=dict)
    last_error: Optional[BaseException] = None
    last_traceback: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_cook_at: Optional[float] = None
    summary_store: SummaryStore = field(default_factory=SummaryStore)
    profiler: Profiler = field(
        default_factory=lambda: Profiler(ProfilerConfig(enabled=False))
    )
    logger_scope: Optional[Any] = None
    summary_writer: Optional[Any] = None
    summary_server: Optional[Any] = None
    taichi_probe: Optional[Any] = None

    def __post_init__(self) -> None:
        self.profiler.attach_summary_store(self.summary_store)

    def reset(self, reason: str) -> None:
        logger_scope = getattr(self, "logger_scope", None)
        if logger_scope is not None:
            try:
                logger_scope.close()
            except Exception:
                pass
        self.logger_scope = None
        if self.summary_writer is not None:
            try:
                self.summary_writer.stop()
            except Exception:
                pass
        self.summary_writer = None
        if self.summary_server is not None:
            try:
                self.summary_server.stop()
            except Exception:
                pass
        self.summary_server = None
        self.taichi_probe = None
        self.summary_store.reset()
        self.profiler = Profiler(ProfilerConfig(enabled=False))
        self.profiler.attach_summary_store(self.summary_store)
        self.world = None
        self.user_module = None
        self.user_module_key = None
        self.did_setup = False
        self.last_step_key = None
        self.last_output_cache.clear()
        self.last_geo_snapshot = None
        self.last_triangles = None
        self.last_topology_sig = None
        self.last_topology_key = None
        self.last_triangles_by_input.clear()
        self.last_topology_sig_by_input.clear()
        self.last_topology_key_by_input.clear()
        self.clear_error()
        self.stats.clear()
        self.last_cook_at = None
        self.stats["last_reset_reason"] = reason
        self.stats["last_reset_at"] = time.time()

    def record_error(self, exc: BaseException, tb_str: str) -> None:
        self.last_error = exc
        self.last_traceback = tb_str
        self.stats["last_error_at"] = time.time()

    def clear_error(self) -> None:
        self.last_error = None
        self.last_traceback = None


class RegistryAccess:
    def __init__(self, reg: Any, *, mode: AccessMode) -> None:
        self._reg = reg
        self._mode = mode

    def _require_write(self) -> None:
        if self._mode != "write":
            raise PermissionError("Session access is read-only; use mode='write'")

    def read(self, name: str, *, ensure: bool = True) -> Any:
        return self._reg.read(name, ensure=ensure)

    def ensure(self, name: str) -> None:
        self._reg.ensure(name)

    def declare(self, *args: Any, **kwargs: Any) -> Any:
        self._require_write()
        return self._reg.declare(*args, **kwargs)

    def commit(self, *args: Any, **kwargs: Any) -> Any:
        self._require_write()
        return self._reg.commit(*args, **kwargs)

    def commit_many(self, *args: Any, **kwargs: Any) -> Any:
        self._require_write()
        return self._reg.commit_many(*args, **kwargs)

    def bump(self, *args: Any, **kwargs: Any) -> Any:
        self._require_write()
        return self._reg.bump(*args, **kwargs)

    def set_buffer(self, *args: Any, **kwargs: Any) -> Any:
        self._require_write()
        return self._reg.set_buffer(*args, **kwargs)


@dataclass
class SessionAccess:
    session: WorldSession
    node_path: str
    mode: AccessMode = "read"

    def __post_init__(self) -> None:
        if self.mode not in ("read", "write"):
            raise ValueError(f"Unknown session access mode: {self.mode}")
        self._reg_view = RegistryAccess(self._get_world().reg, mode=self.mode)

    def _get_world(self) -> World:
        if self.session.world is None:
            self.session.world = World()
        return self.session.world

    @property
    def reg(self) -> RegistryAccess:
        return self._reg_view

    def __enter__(self) -> "SessionAccess":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class ComputeRuntime:
    def __init__(self) -> None:
        self.sessions: Dict[SessionKey, WorldSession] = {}

    def get_or_create_session(
        self, node: "hou.Node", *, key: Optional[str] = None
    ) -> WorldSession:
        session_key = _make_runtime_session_key(node, key=key)
        session = self.sessions.get(session_key)
        if session is None:
            session = WorldSession()
            self.sessions[session_key] = session
        return session

    def get_session_by_path(
        self, node_path: str, *, create: bool = False
    ) -> WorldSession:
        key = make_session_key_for_path(node_path)
        session = self.sessions.get(key)
        if session is None:
            if not create:
                raise KeyError(f"No session for node_path='{node_path}'")
            session = WorldSession()
            self.sessions[key] = session
        return session

    def session_access(
        self,
        node_path: str,
        *,
        mode: AccessMode = "read",
        create: bool = False,
    ) -> SessionAccess:
        session = self.get_session_by_path(node_path, create=create)
        return SessionAccess(session=session, node_path=node_path, mode=mode)

    def reset_session(
        self, node: "hou.Node", reason: str, *, key: Optional[str] = None
    ) -> None:
        session_key = _make_runtime_session_key(node, key=key)
        session = self.sessions.get(session_key)
        if session is not None:
            session.reset(reason)

    def nuke_all(self, reason: str, *, reset_taichi: bool = True) -> None:
        for session in self.sessions.values():
            session.reset(reason)
        self.sessions.clear()
        if reset_taichi:
            reset_taichi_hard()


_RUNTIME = ComputeRuntime()


def get_sim_context(*, create: bool = False) -> Optional[Any]:
    hou = _get_hou()
    if hou is None:
        return None
    sim = getattr(hou.session, _SIM_ATTR, None)
    if sim is None and create:
        from .sim_context import SimContext

        sim = SimContext()
        setattr(hou.session, _SIM_ATTR, sim)
    return sim


def set_sim_context(sim: Optional[Any]) -> None:
    hou = _get_hou()
    if hou is None:
        return
    setattr(hou.session, _SIM_ATTR, sim)


def get_runtime(*, create: bool = True) -> ComputeRuntime:
    sim = get_sim_context(create=create)
    if sim is not None:
        runtime = getattr(sim, "runtime", None)
        if runtime is None and create:
            runtime = ComputeRuntime()
            try:
                setattr(sim, "runtime", runtime)
            except Exception:
                pass
        if runtime is not None:
            return runtime
    return _RUNTIME
