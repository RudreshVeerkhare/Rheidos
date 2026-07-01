"""Houdini node parameter parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import hou


@dataclass(frozen=True)
class NodeConfig:
    script_path: Optional[str]
    module_path: Optional[str]
    mode: str
    reset_node: bool
    nuke_all: bool
    profile: bool
    profile_logdir: Optional[str]
    profile_export_hz: float
    profile_taichi: bool
    profile_taichi_every: int
    profile_taichi_sync: bool
    profile_taichi_scoped_once: bool
    debug_log: bool


def _require_parm(node: "hou.Node", name: str) -> "hou.Parm":
    parm = node.parm(name)
    if parm is None:
        raise KeyError(f"Missing parm '{name}' on node '{node.path()}'")
    return parm


def _eval_parm_str(node: "hou.Node", name: str) -> str:
    parm = _require_parm(node, name)
    try:
        value = parm.evalAsString()
    except Exception:
        value = parm.eval()
    return "" if value is None else str(value)


def _eval_parm_bool(node: "hou.Node", name: str) -> bool:
    return bool(_require_parm(node, name).eval())


def _eval_parm_optional(node: "hou.Node", name: str):
    parm = node.parm(name)
    if parm is None:
        return None
    try:
        return parm.eval()
    except Exception:
        return None


def _eval_parm_optional_str(node: "hou.Node", name: str, default: str = "") -> str:
    value = _eval_parm_optional(node, name)
    if value is None:
        return default
    return "" if value is None else str(value)


def _eval_parm_optional_bool(
    node: "hou.Node", name: str, default: bool = False
) -> bool:
    value = _eval_parm_optional(node, name)
    if value is None:
        return default
    return bool(value)


def _eval_parm_optional_int(node: "hou.Node", name: str, default: int) -> int:
    value = _eval_parm_optional(node, name)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _eval_parm_optional_float(node: "hou.Node", name: str, default: float) -> float:
    value = _eval_parm_optional(node, name)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def read_node_config(node: "hou.Node") -> NodeConfig:
    script_path = _eval_parm_str(node, "script_path") or None
    module_path = _eval_parm_str(node, "module_path") or None

    return NodeConfig(
        script_path=script_path,
        module_path=module_path,
        mode=_eval_parm_str(node, "mode"),
        reset_node=_eval_parm_bool(node, "reset_node"),
        nuke_all=_eval_parm_bool(node, "nuke_all"),
        profile=_eval_parm_bool(node, "profile"),
        profile_logdir=_eval_parm_optional_str(node, "profile_logdir") or None,
        profile_export_hz=_eval_parm_optional_float(
            node, "profile_export_hz", 5.0
        ),
        profile_taichi=_eval_parm_optional_bool(node, "profile_taichi", True),
        profile_taichi_every=_eval_parm_optional_int(
            node, "profile_taichi_every", 30
        ),
        profile_taichi_sync=_eval_parm_optional_bool(
            node, "profile_taichi_sync", True
        ),
        profile_taichi_scoped_once=_eval_parm_optional_bool(
            node, "profile_taichi_scoped_once", False
        ),
        debug_log=_eval_parm_bool(node, "debug_log"),
    )
