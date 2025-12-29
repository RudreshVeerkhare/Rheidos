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
        debug_log=_eval_parm_bool(node, "debug_log"),
    )
