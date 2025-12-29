"""Smoke test for Houdini + rheidos imports."""

from __future__ import annotations

import sys


def _print_houdini_version() -> None:
    try:
        import hou  # type: ignore
    except Exception as exc:
        print(f"houdini import failed: {exc}")
        return
    print(f"houdini: {hou.applicationVersionString()}")


def _print_python_version() -> None:
    print(f"python: {sys.version}")


def _print_taichi_version() -> None:
    try:
        import taichi as ti
    except Exception as exc:
        print(f"taichi import failed: {exc}")
        return
    print(f"taichi: {ti.__version__}")


def _print_rheidos_imports() -> None:
    try:
        import rheidos.compute  # noqa: F401
        import rheidos.houdini  # noqa: F401
    except Exception as exc:
        print(f"rheidos import failed: {exc}")
        return
    print("rheidos: import ok")


def _print_houdini_helpers() -> None:
    try:
        from rheidos.houdini.geo import GeometryIO, GeometrySchema  # noqa: F401
        from rheidos.houdini.runtime import (  # noqa: F401
            CookContext,
            GEO_P,
            GEO_TRIANGLES,
            publish_geometry_minimal,
        )
    except Exception as exc:
        print(f"houdini helpers import failed: {exc}")
        return
    print("houdini helpers: import ok")


def main() -> None:
    _print_houdini_version()
    _print_python_version()
    _print_taichi_version()
    _print_rheidos_imports()
    _print_houdini_helpers()


if __name__ == "__main__":
    main()
