"""Houdini Python SOP entrypoint for stateless cook."""

from __future__ import annotations


def main() -> None:
    import hou  # type: ignore

    from rheidos.houdini.runtime import run_cook

    node = hou.pwd()
    geo_out = node.geometry()
    inputs = node.inputs()
    geo_in = inputs[0].geometry() if inputs and inputs[0] is not None else None
    run_cook(node, geo_in, geo_out)


if __name__ == "__main__":
    main()
