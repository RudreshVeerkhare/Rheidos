# P2 Houdini Standalone Bundle

This folder is a standalone copy of the P2 higher-genus and double-obstacle
Houdini scenes, plus the local `rheidos` source they need. You do not need to
run `pip install rheidos`.

## Requirements

- Houdini 21.x, using Python 3.11.
- Houdini's command-line Python, `hython`.
- Third-party Python packages installed into this bundle with `hython`.

## Setup

1. Put this whole `p2_higher_genus_bundle` folder anywhere on the target machine.
2. Open a terminal in this folder.
3. Install third-party dependencies into the local `python3.11libs` folder:

```bash
hython -m pip install -r requirements-houdini.txt -t python3.11libs
```

If Houdini's Python does not have pip enabled, run this first:

```bash
hython -m ensurepip --upgrade
```

4. Prepare the copied HIP files once using Houdini's Python API:

```bash
hython tools/prepare_hips_in_houdini.py
```

This updates Python SOPs and file paths through Houdini's own APIs. Do not edit
the `.hipnc` files with a text or binary patcher.

5. Open one of the HIP files from this folder:

```text
tree_cotree_torus.hipnc
harmonic_basis_torus.hipnc
hero_torus.hipnc
opposite_dipole.hipnc
vortex_dynamics_on_higher_genus_surface.hipnc
point_vortex_obstacles.hipnc
cached_island.hipnc
```

The preparation script patches the copied HIP files so their Python SOPs add
this bundle's `python3.11libs` to `sys.path` before importing `rheidos`, clears
any already-loaded `rheidos` modules from a different location, updates mesh
nodes to use `$HIP/assets/double_torus.obj`, and loads local HDAs from `otls`
for this preparation session.

If you want to open Houdini with the bundle path set without editing your global
environment, run:

```bash
./open_houdini_with_bundle.sh hero_torus.hipnc
```

## Quick Import Check

Inside Houdini's Python Shell, run:

```python
import sys, hou
bundle_python = hou.expandString("$HIP/python3.11libs")
if bundle_python not in sys.path:
    sys.path.insert(0, bundle_python)

import rheidos
import scipy
print(rheidos.__file__)
print(scipy.__version__)
```

`rheidos.__file__` should point inside this bundle.

## Running the Scenes

- For `tree_cotree_torus.hipnc`, cook the Python SOPs named for setup, dual
  tree, primal tree, and generators.
- For `harmonic_basis_torus.hipnc`, cook setup, tree-cotree, then harmonic basis.
- For the vortex dynamics scenes, cook setup first, then the solver/interpolation
  nodes used by the scene.
- For `point_vortex_obstacles.hipnc`, cook `setup_mesh_and_point_vortices`,
  then the solver/interpolation nodes used by the scene.
- `cached_island.hipnc` is the lightweight cached double-obstacle scene. Its
  surface cache lives at `geo/surface.bgeo`.

If a node reports `No module named scipy`, rerun the `hython -m pip install ...`
command above from the bundle root.

If Houdini reports `invalid .hip file header`, replace that `.hipnc` with a
fresh copy from the original scene before running the preparation script again.
That error means the binary HIP file was edited outside Houdini/HOM.

If a traceback points at a path outside this bundle, for example
`/Users/codebox/dev/kung_fu_panda/rheidos/...`, restart Houdini and reopen the
scene through `./open_houdini_with_bundle.sh`, or run the preparation script
again. Houdini caches imported Python modules for the life of the process.

## Bundle Contents

- `python3.11libs/rheidos`: vendored local source required by the bundled P2
  Houdini apps.
- `assets/double_torus.obj`: mesh asset used by the copied HIP files.
- `geo/surface.bgeo`: local surface cache used by `cached_island.hipnc`.
- `otls/fc2d.hdanc`: local HDA used by the double-obstacle scene.
- `requirements-houdini.txt`: third-party dependencies to install into
  `python3.11libs`.
- `tools/prepare_hips_in_houdini.py`: Houdini-side utility used to patch copied
  HIP paths and Python import bootstraps safely through HOM.
- `open_houdini_with_bundle.sh`: optional launcher that sets `PYTHONPATH`,
  `RHEIDOS_STANDALONE_BUNDLE`, and `HOUDINI_PATH` for this bundle.

## Notes

This bundle intentionally excludes development artifacts such as tests, docs,
`__pycache__`, `.DS_Store`, backup HIP files, MP4 renders, old ZIP exports,
TensorBoard logs, and frontend profiler UI files.

`duck_torus` is intentionally not included in this bundle.
