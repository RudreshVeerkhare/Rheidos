"""Helpers for purging and reloading project packages."""

from __future__ import annotations

from types import ModuleType
from typing import Iterable, Optional
import importlib
import sys


def _purge_package(pkg_name: str, *, keep: Optional[Iterable[str]] = None) -> list[str]:
    keep_set = set(keep or ())
    removed = []
    prefix = pkg_name + "."
    for name in list(sys.modules.keys()):
        if name == pkg_name or name.startswith(prefix):
            if name in keep_set:
                continue
            sys.modules.pop(name, None)
            removed.append(name)
    return removed


def reload_project(pkg_name: str, *, keep: Optional[Iterable[str]] = None) -> ModuleType:
    importlib.invalidate_caches()
    _purge_package(pkg_name, keep=keep)
    return importlib.import_module(pkg_name)


__all__ = ["_purge_package", "reload_project"]
