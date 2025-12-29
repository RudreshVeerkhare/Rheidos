"""User script loading for Houdini node drivers."""

from __future__ import annotations

from types import ModuleType
from typing import Optional, TYPE_CHECKING
import hashlib
import importlib
import importlib.util
import os
import sys

from ..nodes.config import NodeConfig
from .session import WorldSession

if TYPE_CHECKING:
    import hou


def _try_get_hou() -> Optional["hou"]:
    try:
        import hou  # type: ignore
    except Exception:
        return None
    return hou


def _expand_path(path: str) -> str:
    value = path
    hou = _try_get_hou()
    if hou is not None:
        try:
            value = hou.expandString(value)
        except Exception:
            pass
    value = os.path.expandvars(os.path.expanduser(value))
    if not os.path.isabs(value):
        base = ""
        if hou is not None:
            try:
                base = os.path.dirname(hou.hipFile.path())
            except Exception:
                base = ""
        if not base:
            base = os.getcwd()
        value = os.path.join(base, value)
    return os.path.normpath(value)


def _module_name_from_path(path: str) -> str:
    digest = hashlib.md5(path.encode("utf-8")).hexdigest()[:12]
    base = os.path.splitext(os.path.basename(path))[0]
    safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in base)
    return f"rheidos_houdini_user_{safe}_{digest}"


def _load_module_from_path(path: str) -> ModuleType:
    module_name = _module_name_from_path(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from '{path}'")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_module_by_name(module_name: str, *, force_reload: bool) -> ModuleType:
    existing = sys.modules.get(module_name)
    module = importlib.import_module(module_name)
    if force_reload and existing is not None:
        module = importlib.reload(module)
    return module


def resolve_user_module(
    session: WorldSession,
    config: NodeConfig,
    node: "hou.Node",
) -> ModuleType:
    del node  # reserved for future use
    if config.script_path and config.module_path:
        raise ValueError("Provide exactly one of script_path or module_path")
    if not config.script_path and not config.module_path:
        raise ValueError("Missing script_path or module_path")

    if config.script_path:
        path = _expand_path(config.script_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Script not found: {path}")
        key = f"script:{path}"
        loader = lambda: _load_module_from_path(path)
    else:
        module_name = (config.module_path or "").strip()
        if not module_name:
            raise ValueError("module_path is empty")
        key = f"module:{module_name}"
        force_reload = bool(config.reset_node or config.nuke_all)
        loader = lambda: _load_module_by_name(module_name, force_reload=force_reload)

    if session.user_module_key is not None and session.user_module_key != key:
        raise RuntimeError(
            "User script changed; press Reset Node or Nuke All to reload"
        )

    if session.user_module is not None and session.user_module_key == key:
        return session.user_module

    module = loader()
    session.user_module = module
    session.user_module_key = key
    return module


__all__ = ["resolve_user_module"]
