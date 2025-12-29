"""Build Houdini HDAs for Rheidos cook/solver SOP nodes."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import os
from typing import Optional


COOK_TYPE_NAME = "rheidos::cook_sop"
SOLVER_TYPE_NAME = "rheidos::solver_sop"

COOK_LABEL = "Rheidos Cook SOP"
SOLVER_LABEL = "Rheidos Solver SOP"

COOK_SCRIPT = "from rheidos.houdini.scripts.cook_sop import main\nmain()\n"
SOLVER_SCRIPT = "from rheidos.houdini.scripts.solver_sop import main\nmain()\n"


def _get_hou():
    try:
        import hou  # type: ignore
    except Exception as exc:  # pragma: no cover - only runs in Houdini
        raise RuntimeError("Houdini 'hou' module not available") from exc
    return hou


def _default_output_path() -> str:
    base = os.path.dirname(__file__)
    path = os.path.join(base, "..", "otls", "rheidos_houdini.otl")
    return os.path.normpath(path)


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _disable_when_nonempty(template, other_name: str) -> None:
    hou = _get_hou()
    template.setConditional(
        hou.parmCondType.DisableWhen,
        f"{{ {other_name} != '' }}",
    )


def _replace_or_append(ptg: "hou.ParmTemplateGroup", template: "hou.ParmTemplate") -> None:
    existing = ptg.find(template.name())
    if existing is None:
        ptg.append(template)
    else:
        ptg.replace(template.name(), template)


def _build_parm_group(
    base: "hou.ParmTemplateGroup",
    *,
    default_mode: str,
    include_substep: bool,
) -> "hou.ParmTemplateGroup":
    hou = _get_hou()
    ptg = base

    script_path = hou.StringParmTemplate(
        "script_path",
        "Script Path",
        1,
        default_value=("",),
        string_type=hou.stringParmType.FileReference,
    )
    script_path.setHelp("Path to a Python script defining cook(ctx) or setup/step(ctx).")
    _disable_when_nonempty(script_path, "module_path")

    module_path = hou.StringParmTemplate(
        "module_path",
        "Module Path",
        1,
        default_value=("",),
        string_type=hou.stringParmType.Regular,
    )
    module_path.setHelp("Python module path defining cook(ctx) or setup/step(ctx).")
    _disable_when_nonempty(module_path, "script_path")

    mode_items = ("cook", "solver")
    mode_labels = ("Cook", "Solver")
    mode_default = 0 if default_mode == "cook" else 1
    mode = hou.MenuParmTemplate(
        "mode",
        "Mode",
        menu_items=mode_items,
        menu_labels=mode_labels,
        default_value=mode_default,
    )
    mode.setHelp("Node execution mode (primarily informational for now).")

    script_folder = hou.FolderParmTemplate(
        "rheidos_script",
        "Script",
        parm_templates=[script_path, module_path, mode],
    )

    reset_node = hou.ToggleParmTemplate("reset_node", "Reset Node", default_value=False)
    nuke_all = hou.ToggleParmTemplate("nuke_all", "Nuke All", default_value=False)
    profile = hou.ToggleParmTemplate("profile", "Profile", default_value=False)
    debug_log = hou.ToggleParmTemplate("debug_log", "Debug Log", default_value=False)

    runtime_parms = [reset_node, nuke_all]
    if include_substep:
        runtime_parms.insert(0, hou.IntParmTemplate("substep", "Substep", 1, default_value=(0,)))

    runtime_folder = hou.FolderParmTemplate(
        "rheidos_runtime",
        "Runtime",
        parm_templates=runtime_parms,
    )

    diag_folder = hou.FolderParmTemplate(
        "rheidos_diagnostics",
        "Diagnostics",
        parm_templates=[profile, debug_log],
    )

    last_error = hou.StringParmTemplate(
        "last_error",
        "Last Error",
        1,
        default_value=("",),
        string_type=hou.stringParmType.Regular,
    )
    last_error.setReadOnly(True)
    last_error.setHelp("Last error from cook/solver (cleared on success).")
    diag_folder.addParmTemplate(last_error)

    _replace_or_append(ptg, script_folder)
    _replace_or_append(ptg, runtime_folder)
    _replace_or_append(ptg, diag_folder)
    return ptg


def _hide_python_param(definition: "hou.HDADefinition") -> None:
    try:
        ptg = definition.parmTemplateGroup()
        python_parm = ptg.find("python")
        if python_parm is None:
            return
        python_parm.setHidden(True)
        ptg.replace("python", python_parm)
        definition.setParmTemplateGroup(ptg)
    except Exception:
        return


def _apply_parm_group(definition: "hou.HDADefinition", ptg: "hou.ParmTemplateGroup") -> None:
    definition.setParmTemplateGroup(ptg)
    _hide_python_param(definition)


@dataclass(frozen=True)
class _AssetSpec:
    type_name: str
    label: str
    script: str
    default_mode: str
    include_substep: bool
    min_inputs: int
    max_inputs: int


def _create_asset(
    parent: "hou.Node",
    spec: _AssetSpec,
    output_path: str,
) -> None:
    hou = _get_hou()
    node = parent.createNode("python", node_name=spec.type_name.replace("::", "_"))
    node.parm("python").set(spec.script)

    try:
        hda_node = node.createDigitalAsset(
            name=spec.type_name,
            hda_file_name=output_path,
            description=spec.label,
            min_num_inputs=spec.min_inputs,
            max_num_inputs=spec.max_inputs,
        )
    except hou.OperationFailed as exc:
        raise RuntimeError(
            f"Failed to create {spec.type_name} asset. "
            "If an existing definition conflicts, remove it or choose a new output path."
        ) from exc

    definition = hda_node.type().definition()
    if definition is None:
        raise RuntimeError(f"Missing HDA definition for {spec.type_name}")
    base_group = definition.parmTemplateGroup()
    ptg = _build_parm_group(
        base_group,
        default_mode=spec.default_mode,
        include_substep=spec.include_substep,
    )
    _apply_parm_group(definition, ptg)


def build_assets(output_path: Optional[str] = None) -> str:
    """Build Rheidos cook/solver SOP HDAs.

    Returns the output HDA path.
    """
    hou = _get_hou()
    path = output_path or _default_output_path()
    _ensure_parent_dir(path)

    obj = hou.node("/obj")
    if obj is None:
        raise RuntimeError("Could not find /obj in current Houdini session")

    tmp_geo = obj.createNode("geo", node_name="rheidos_hda_build")
    try:
        for child in tmp_geo.children():
            child.destroy()

        assets = [
            _AssetSpec(
                type_name=COOK_TYPE_NAME,
                label=COOK_LABEL,
                script=COOK_SCRIPT,
                default_mode="cook",
                include_substep=False,
                min_inputs=0,
                max_inputs=1,
            ),
            _AssetSpec(
                type_name=SOLVER_TYPE_NAME,
                label=SOLVER_LABEL,
                script=SOLVER_SCRIPT,
                default_mode="solver",
                include_substep=True,
                min_inputs=0,
                max_inputs=2,
            ),
        ]

        for spec in assets:
            _create_asset(tmp_geo, spec, path)

    finally:
        try:
            tmp_geo.destroy()
        except Exception:
            pass

    return path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build Rheidos Houdini HDAs")
    parser.add_argument(
        "--output",
        dest="output",
        default=None,
        help="Output .otl path (defaults to repo rheidos/houdini/otls)",
    )
    args = parser.parse_args(argv)
    path = build_assets(args.output)
    print(f"Built HDAs at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
