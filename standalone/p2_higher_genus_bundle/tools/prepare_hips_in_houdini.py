"""Prepare copied HIP files using Houdini's Python API.

Run from the bundle root with Houdini's Python:

    hython tools/prepare_hips_in_houdini.py

This script intentionally uses HOM instead of editing HIP bytes directly.
"""

from __future__ import annotations

from pathlib import Path

try:
    import hou  # type: ignore
except Exception as exc:  # pragma: no cover - must run inside hython
    raise SystemExit(
        "Run this with Houdini's Python, for example:\n"
        "  hython tools/prepare_hips_in_houdini.py"
    ) from exc


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
PYTHON_LIBS_EXPR = "$HIP/python3.11libs"
ASSET_EXPR = "$HIP/assets/double_torus.obj"

HIP_FILES = [
    "tree_cotree_torus.hipnc",
    "harmonic_basis_torus.hipnc",
    "hero_torus.hipnc",
    "opposite_dipole.hipnc",
    "vortex_dynamics_on_higher_genus_surface.hipnc",
    "point_vortex_obstacles.hipnc",
    "cached_island.hipnc",
]

APP_IMPORT_PREFIXES = (
    "import rheidos.apps.p2.higher_genus",
    "import rheidos.apps.p2.double_obstacle",
    "import rheidos.apps.p2.p1_sphere_app",
)

SESSION_MARKER = "P2 higher-genus standalone bundle bootstrap"
SESSION_VERSION_MARKER = f"{SESSION_MARKER} v2"
NODE_MARKER = "P2 higher-genus standalone node bootstrap v2"
SESSION_BOOTSTRAP = f"""
# {SESSION_VERSION_MARKER}
import os
import sys
try:
    import hou
    _p2_bundle_root = os.environ.get("RHEIDOS_STANDALONE_BUNDLE", "")
    if not _p2_bundle_root:
        _p2_hip_path = hou.hipFile.path()
        _p2_bundle_root = os.path.dirname(_p2_hip_path) if _p2_hip_path else ""
    _p2_bundle_python = (
        os.path.join(_p2_bundle_root, "python3.11libs")
        if _p2_bundle_root
        else hou.expandString("{PYTHON_LIBS_EXPR}")
    )
except Exception:
    _p2_bundle_python = ""
if _p2_bundle_python:
    sys.path[:] = [p for p in sys.path if p != _p2_bundle_python]
    sys.path.insert(0, _p2_bundle_python)
    _p2_loaded = sys.modules.get("rheidos")
    _p2_loaded_file = str(getattr(_p2_loaded, "__file__", ""))
    if _p2_loaded is not None and not _p2_loaded_file.startswith(_p2_bundle_python):
        for _p2_name in list(sys.modules):
            if _p2_name == "rheidos" or _p2_name.startswith("rheidos."):
                del sys.modules[_p2_name]
""".strip()

NODE_BOOTSTRAP = f"""
# {NODE_MARKER}
import os
import sys
import hou
_p2_bundle_root = os.environ.get("RHEIDOS_STANDALONE_BUNDLE", "")
if not _p2_bundle_root:
    _p2_hip_path = hou.hipFile.path()
    _p2_bundle_root = os.path.dirname(_p2_hip_path) if _p2_hip_path else ""
_p2_bundle_python = (
    os.path.join(_p2_bundle_root, "python3.11libs")
    if _p2_bundle_root
    else hou.expandString("{PYTHON_LIBS_EXPR}")
)
if _p2_bundle_python:
    sys.path[:] = [p for p in sys.path if p != _p2_bundle_python]
    sys.path.insert(0, _p2_bundle_python)
    _p2_loaded = sys.modules.get("rheidos")
    _p2_loaded_file = str(getattr(_p2_loaded, "__file__", ""))
    if _p2_loaded is not None and not _p2_loaded_file.startswith(_p2_bundle_python):
        for _p2_name in list(sys.modules):
            if _p2_name == "rheidos" or _p2_name.startswith("rheidos."):
                del sys.modules[_p2_name]
""".lstrip()

OLD_PATH_FRAGMENTS = (
    "/" + "/".join(("Users", "codebox", "dev", "mesh_viz", "double_torus.obj")),
    "/"
    + "/".join(
        (
            "Users",
            "codebox",
            "dev",
            "kung_fu_panda",
            "rheidos",
            "apps",
            "p2",
            "higher_genus",
            "vortex_dynamics",
            "torus.obj",
        )
    ),
)


def _session_source() -> str:
    getter = getattr(hou, "sessionModuleSource", None)
    if callable(getter):
        return getter()
    return ""


def _set_session_source(source: str) -> None:
    setter = getattr(hou, "setSessionModuleSource", None)
    if callable(setter):
        setter(source)
        return
    appender = getattr(hou, "appendSessionModuleSource", None)
    if callable(appender):
        appender(source)
        return
    raise RuntimeError("This Houdini build does not expose session module source APIs")


def _patch_session_module() -> bool:
    source = _session_source()
    if SESSION_VERSION_MARKER in source:
        return False
    next_source = SESSION_BOOTSTRAP if not source else f"{SESSION_BOOTSTRAP}\n\n{source}"
    _set_session_source(next_source)
    return True


def _patch_python_sops() -> int:
    changed = 0
    root = hou.node("/")
    if root is None:
        return changed

    for node in root.allSubChildren():
        parm = node.parm("python")
        if parm is None:
            continue
        try:
            script = parm.unexpandedString()
        except Exception:
            script = parm.evalAsString()
        if not any(prefix in script for prefix in APP_IMPORT_PREFIXES):
            continue
        if NODE_MARKER in script:
            continue
        parm.set(NODE_BOOTSTRAP + script)
        changed += 1
    return changed


def _patch_file_paths() -> int:
    changed = 0
    root = hou.node("/")
    if root is None:
        return changed

    for node in root.allSubChildren():
        for parm in node.parms():
            try:
                raw = parm.unexpandedString()
            except Exception:
                continue
            next_value = raw
            for fragment in OLD_PATH_FRAGMENTS:
                next_value = next_value.replace(fragment, ASSET_EXPR)
            if next_value == raw:
                continue
            parm.set(next_value)
            changed += 1
    return changed


def _install_bundle_hdas() -> int:
    hda_count = 0
    otls_dir = BUNDLE_ROOT / "otls"
    if not otls_dir.is_dir():
        return hda_count

    for path in sorted(otls_dir.glob("*.hda*")):
        hou.hda.installFile(
            str(path),
            change_oplibraries_file=False,
            force_use_assets=True,
        )
        hda_count += 1
    return hda_count


def prepare_hip(path: Path) -> None:
    hou.hipFile.load(str(path), suppress_save_prompt=True, ignore_load_warnings=True)
    session_changed = _patch_session_module()
    sop_count = _patch_python_sops()
    path_count = _patch_file_paths()
    hou.hipFile.save(str(path))
    print(
        f"{path.name}: session={int(session_changed)} "
        f"python_sops={sop_count} file_paths={path_count}"
    )


def main() -> None:
    hda_count = _install_bundle_hdas()
    if hda_count:
        print(f"Installed {hda_count} bundle HDA file(s) for this hython session")

    for name in HIP_FILES:
        path = BUNDLE_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        prepare_hip(path)


if __name__ == "__main__":
    main()
