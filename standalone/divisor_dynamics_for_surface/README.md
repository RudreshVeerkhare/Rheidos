# Bundled Code for Divisor Dynamics for Surfaces

## Requirements

- Houdini 21.x, using Python 3.11. (Might work with older versions as well, but can't guarantee)
- Houdini's command-line Python, `hython`.
- Third-party Python packages installed into this bundle with `hython`.

## Hython Path

`hython` is just a fancy alias used for python binary which comes pre-bundled with Houdini. To be able to run this code, you need to have certain python packages installed in Houdini's internal Python.

The `hython` binary is located inside `$HFS/bin` folder. To get the value for `$HFS` use the Houdini's Python shell and run `os.path.expandvars('$HFS')`.

## Setup

1. Put this whole folder anywhere on the target machine.
2. Open a terminal in this folder.
3. Install third-party dependencies into the local `python3.11libs` folder:

```bash
hython -m pip install -r requirements.txt -t python3.11libs
```

If Houdini's Python does not have pip enabled, run this first:

```bash
hython -m ensurepip --upgrade
```

5. Open one of the HIP files from this folder:

```text
tree_cotree.hipnc
harmonic_basis.hipnc
vortex_sheet.hipnc
vortex_dynamics_on_higher_genus_surface.hipnc
point_vortex_island.hipnc
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

- For `tree_cotree.hipnc`, cook the Python SOPs named for setup, dual
  tree, primal tree, and generators.
- For `harmonic_basis.hipnc`, cook setup, tree-cotree, then harmonic basis.
- For the vortex dynamics scenes, cook setup first, then the solver/interpolation
  nodes used by the scene.
- For `point_vortex_island.hipnc`, has the setup for point vortex passing between obstacles.

If a node reports `No module named ...`, rerun the `hython -m pip install ...`
command above from the bundle root.

## Bundle Contents

- `python3.11libs/rheidos`: vendored local source required by the bundled P2
  Houdini apps.
- `assets/double_torus.obj`: mesh asset used by the copied HIP files.
- `requirements.txt`: third-party dependencies to install into
  `python3.11libs`.
