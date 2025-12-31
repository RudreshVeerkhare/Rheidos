"""Remote debugger support for Houdini via debugpy."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import socket
import sys
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import hou

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 5678
_STATE_ATTR = "_rheidos_debug_state"

_ENV_PREFIXES = ("RHEDIOS", "RHEIDOS")


@dataclass(frozen=True)
class DebugConfig:
    enabled: bool
    host: str = _DEFAULT_HOST
    port: int = _DEFAULT_PORT
    port_strategy: Literal["fixed", "fallback", "auto"] = "fallback"
    allow_remote: bool = False
    take_ownership: bool = False
    owner_hint: Optional[str] = None
    log: bool = True


@dataclass
class DebugState:
    started: bool = False
    host: str = _DEFAULT_HOST
    port: int = _DEFAULT_PORT
    owner_node_path: Optional[str] = None
    pid: int = field(default_factory=os.getpid)
    warned_missing_debugpy: bool = False
    warned_port_bind: bool = False
    warned_remote_host: bool = False
    warned_break_unattached: bool = False
    warned_break_failed: bool = False
    break_next: bool = False
    break_owner: Optional[str] = None
    info_printed: bool = False
    owner_notice_nodes: set[str] = field(default_factory=set)


_FALLBACK_STATE: Optional[DebugState] = None


def debug_config_from_node(node: "hou.Node") -> DebugConfig:
    enabled = _eval_parm_bool(node, "debug_enable", default=False)
    env_enabled = _read_env_bool("DEBUG")
    if env_enabled is not None:
        enabled = env_enabled

    port = _eval_parm_int(node, "debug_port", default=_DEFAULT_PORT)
    env_port = _read_env_int("DEBUG_PORT")
    if env_port is not None:
        port = env_port
    port = _sanitize_port(port)

    host = _eval_parm_str(node, "debug_host", default=_DEFAULT_HOST)
    env_host = _read_env_str("DEBUG_HOST")
    if env_host:
        host = env_host
    host = host.strip() or _DEFAULT_HOST

    strategy_value = _eval_parm_str(node, "debug_port_strategy", default="")
    env_strategy = _read_env_str("DEBUG_PORT_STRATEGY")
    if env_strategy:
        strategy_value = env_strategy
    port_strategy = _normalize_port_strategy(strategy_value)

    allow_remote = _eval_parm_bool(node, "debug_allow_remote", default=False)
    env_allow_remote = _read_env_bool("DEBUG_REMOTE", "DEBUG_ALLOW_REMOTE")
    if env_allow_remote is not None:
        allow_remote = env_allow_remote

    take_ownership = _eval_parm_bool(node, "debug_take_ownership", default=False)
    owner_hint = _node_path(node)

    return DebugConfig(
        enabled=enabled,
        host=host,
        port=port,
        port_strategy=port_strategy,
        allow_remote=allow_remote,
        take_ownership=take_ownership,
        owner_hint=owner_hint,
    )


def ensure_debug_server(
    cfg: DebugConfig, *, node: Optional["hou.Node"] = None
) -> DebugState:
    state = _get_state()
    if not cfg.enabled:
        return state

    _update_owner(state, cfg, node=node)

    host = cfg.host
    if host != _DEFAULT_HOST and not cfg.allow_remote:
        if not state.warned_remote_host:
            print(
                "[rheidos] Remote debugging requires explicit opt-in; "
                "falling back to 127.0.0.1."
            )
            state.warned_remote_host = True
        host = _DEFAULT_HOST

    if state.started:
        _maybe_notice_owner(state, cfg)
        return state

    try:
        import debugpy  # type: ignore
    except Exception:
        _warn_missing_debugpy(state)
        return state

    attempted_ports: list[int] = []
    for port in _candidate_ports(cfg, host):
        attempted_ports.append(port)
        try:
            debugpy.listen((host, port), in_process_debug_adapter=True)
        except Exception:
            continue
        state.started = True
        state.host = host
        state.port = port
        state.pid = os.getpid()
        _print_startup_message(state)
        return state

    if not state.warned_port_bind:
        attempted = ", ".join(str(p) for p in attempted_ports if p)
        if attempted:
            detail = f" Attempted ports: {attempted}."
        else:
            detail = ""
        print(
            "[rheidos] Failed to start debug server."
            f"{detail} Try a different port or strategy."
        )
        state.warned_port_bind = True
    return state


def request_break_next(*, node: Optional["hou.Node"] = None) -> None:
    state = _get_state()
    state.break_next = True
    state.break_owner = _node_path(node)
    state.warned_break_unattached = False


def maybe_break_now(*, node: Optional["hou.Node"] = None) -> None:
    state = _get_state()
    if not state.break_next:
        return

    try:
        import debugpy  # type: ignore
    except Exception:
        _warn_missing_debugpy(state)
        return

    try:
        connected = bool(debugpy.is_client_connected())
    except Exception:
        connected = False

    if not connected:
        if not state.warned_break_unattached:
            owner = state.break_owner or _node_path(node) or "<unknown>"
            print(
                f"[rheidos] Break requested by {owner}. "
                "Attach a debugger and recook to break."
            )
            state.warned_break_unattached = True
        return

    try:
        debugpy.breakpoint()
    except Exception:
        if not state.warned_break_failed:
            print("[rheidos] Failed to trigger debugger breakpoint.")
            state.warned_break_failed = True
        return

    state.break_next = False
    state.break_owner = None
    state.warned_break_unattached = False
    state.warned_break_failed = False


def consume_break_next_button(node: Optional["hou.Node"]) -> bool:
    if node is None:
        return False
    parm = _safe_parm(node, "debug_break_next")
    if parm is None:
        return False
    try:
        pressed = bool(parm.eval())
    except Exception:
        return False
    if not pressed:
        return False
    try:
        parm.set(0)
    except Exception:
        pass
    return True


def _get_state() -> DebugState:
    hou = _get_hou()
    if hou is not None:
        state = getattr(hou.session, _STATE_ATTR, None)
        if not isinstance(state, DebugState) or state.pid != os.getpid():
            state = DebugState()
            setattr(hou.session, _STATE_ATTR, state)
        return state

    global _FALLBACK_STATE
    if _FALLBACK_STATE is None or _FALLBACK_STATE.pid != os.getpid():
        _FALLBACK_STATE = DebugState()
    return _FALLBACK_STATE


def _get_hou() -> Optional["hou"]:
    try:
        import hou  # type: ignore
    except Exception:
        return None
    return hou


def _node_path(node: Optional["hou.Node"]) -> Optional[str]:
    if node is None:
        return None
    try:
        return node.path()
    except Exception:
        return None


def _owner_is_stale(owner_path: Optional[str]) -> bool:
    if not owner_path:
        return False
    hou = _get_hou()
    if hou is None:
        return False
    try:
        return hou.node(owner_path) is None
    except Exception:
        return False


def _update_owner(
    state: DebugState, cfg: DebugConfig, *, node: Optional["hou.Node"]
) -> None:
    owner_hint = cfg.owner_hint or _node_path(node)
    if state.owner_node_path is None and owner_hint:
        state.owner_node_path = owner_hint
        return
    if cfg.take_ownership or _owner_is_stale(state.owner_node_path):
        if owner_hint:
            state.owner_node_path = owner_hint


def _maybe_notice_owner(state: DebugState, cfg: DebugConfig) -> None:
    owner = state.owner_node_path
    node_path = cfg.owner_hint
    if not owner or not node_path or owner == node_path:
        return
    if node_path in state.owner_notice_nodes:
        return
    print(f"[rheidos] Debug server owned by {owner}.")
    state.owner_notice_nodes.add(node_path)


def _print_startup_message(state: DebugState) -> None:
    if state.info_printed:
        return
    owner = state.owner_node_path or "<unknown>"
    print("[rheidos] Debug server started")
    print(f"[rheidos] Owner: {owner}")
    print(f"[rheidos] Attach: {state.host}:{state.port}")
    print(
        "[rheidos] VS Code: Python: Attach -> " f"host {state.host} port {state.port}"
    )
    _maybe_show_ui_message(
        "\n".join(
            [
                "Debug server started",
                f"Owner: {owner}",
                f"Attach: {state.host}:{state.port}",
            ]
        )
    )
    state.info_printed = True


def _maybe_show_ui_message(message: str) -> None:
    hou = _get_hou()
    if hou is None:
        return
    try:
        if not hou.isUIAvailable():
            return
        hou.ui.displayMessage(message, title="Rheidos Debugger")
    except Exception:
        return


def _warn_missing_debugpy(state: DebugState) -> None:
    if state.warned_missing_debugpy:
        return
    exe = sys.executable or "hython"
    print("[rheidos] debugpy is not available.")
    print(f"[rheidos] Python executable: {exe}")
    print(f"[rheidos] Install: {exe} -m pip install --user debugpy")
    state.warned_missing_debugpy = True


def _candidate_ports(cfg: DebugConfig, host: str) -> list[int]:
    ports: list[int] = []

    def _add(port: Optional[int]) -> None:
        if port is None:
            return
        if port <= 0 or port > 65535:
            return
        if port in ports:
            return
        ports.append(port)

    base_port = _sanitize_port(cfg.port)
    if cfg.port_strategy == "fixed":
        _add(base_port)
        return ports

    if cfg.port_strategy == "fallback":
        _add(base_port)
        for offset in range(1, 21):
            _add(base_port + offset)
        _add(_find_free_port(host))
        return ports

    if cfg.port_strategy == "auto":
        _add(_find_free_port(host))
        return ports

    _add(base_port)
    return ports


def _find_free_port(host: str) -> Optional[int]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])
    except OSError:
        return None


def _sanitize_port(port: Optional[int]) -> int:
    try:
        port_int = int(port) if port is not None else _DEFAULT_PORT
    except (TypeError, ValueError):
        port_int = _DEFAULT_PORT
    if port_int <= 0 or port_int > 65535:
        return _DEFAULT_PORT
    return port_int


def _normalize_port_strategy(value: object) -> Literal["fixed", "fallback", "auto"]:
    if isinstance(value, bytes):
        try:
            value = value.decode()
        except Exception:
            value = ""
    if isinstance(value, str):
        text = value.strip().lower()
        if text.isdigit():
            value = int(text)
        elif text in ("fixed", "fix"):
            return "fixed"
        elif text in ("fallback", "fall_back", "fallbacks", "fallbacks"):
            return "fallback"
        elif text in ("auto", "automatic"):
            return "auto"
        else:
            return "fallback"
    if isinstance(value, (int, float)):
        idx = int(value)
        if idx == 0:
            return "fixed"
        if idx == 1:
            return "fallback"
        if idx == 2:
            return "auto"
    return "fallback"


def _safe_parm(node: Optional["hou.Node"], name: str) -> Optional["hou.Parm"]:
    if node is None:
        return None
    try:
        return node.parm(name)
    except Exception:
        return None


def _eval_parm_bool(node: Optional["hou.Node"], name: str, *, default: bool) -> bool:
    parm = _safe_parm(node, name)
    if parm is None:
        return default
    try:
        return bool(parm.eval())
    except Exception:
        return default


def _eval_parm_int(node: Optional["hou.Node"], name: str, *, default: int) -> int:
    parm = _safe_parm(node, name)
    if parm is None:
        return default
    try:
        return int(parm.eval())
    except Exception:
        return default


def _eval_parm_str(node: Optional["hou.Node"], name: str, *, default: str) -> str:
    parm = _safe_parm(node, name)
    if parm is None:
        return default
    try:
        value = parm.evalAsString()
    except Exception:
        try:
            value = parm.eval()
        except Exception:
            return default
    return "" if value is None else str(value)


def _read_env_str(name: str) -> Optional[str]:
    for prefix in _ENV_PREFIXES:
        value = os.getenv(f"{prefix}_{name}")
        if value:
            return value.strip()
    return None


def _read_env_int(name: str) -> Optional[int]:
    for prefix in _ENV_PREFIXES:
        value = os.getenv(f"{prefix}_{name}")
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return None


def _read_env_bool(*names: str) -> Optional[bool]:
    for name in names:
        for prefix in _ENV_PREFIXES:
            value = os.getenv(f"{prefix}_{name}")
            parsed = _parse_bool(value)
            if parsed is not None:
                return parsed
    return None


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    text = value.strip().lower()
    if text in ("1", "true", "yes", "on", "y"):
        return True
    if text in ("0", "false", "no", "off", "n"):
        return False
    return None
