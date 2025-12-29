"""Houdini Python SOP entrypoint for solver cook."""

from __future__ import annotations


def _eval_substep(node) -> int:
    parm = node.parm("substep")
    if parm is None:
        return 0
    try:
        return int(parm.eval())
    except Exception:
        return 0


def main() -> None:
    import hou  # type: ignore

    from rheidos.houdini.runtime import run_solver

    node = hou.pwd()
    geo_out = node.geometry()
    inputs = node.inputs()
    geo_prev = inputs[0].geometry() if len(inputs) > 0 and inputs[0] is not None else None
    geo_in = inputs[1].geometry() if len(inputs) > 1 and inputs[1] is not None else None
    run_solver(node, geo_prev, geo_in, geo_out, substep=_eval_substep(node))


if __name__ == "__main__":
    main()
