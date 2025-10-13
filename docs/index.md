Kung Fu Panda — A DDG + Taichi + Panda3D Playground

Welcome! This is a tiny, pragmatic framework for quickly prototyping physics and geometry experiments that render in real time. It stitches together three worlds:

- Panda3D for rendering and input (window, camera, lights, scene graph)
- Taichi for high‑performance compute kernels (optional, but great for simulation)
- Numpy as the glue for data exchange

The goal: keep you in “flow”. Spin up a window fast, drop in a mesh, write a few kernels, and iterate from a notebook or a small script. The abstractions are intentionally light so you can keep creating your own layers of abstraction as you experiment.

If you’re new to Panda3D or Taichi, don’t worry — the docs include short intros and walk‑throughs. Think developer‑to‑developer chat with runnable snippets.

What you get
- Engine that runs Panda3D in blocking mode (scripts) or async stepping mode (Jupyter/REPL)
- Views (renderers), Controllers (input), Observers (per‑frame workers/loggers)
- Mesh/Texture helpers to bridge Numpy/Taichi data into Panda3D buffers efficiently
- A tiny global StoreState to share flags/data across components

Quick Start
1) Install dependencies (Python 3.9+):

   - Core: `pip install panda3d numpy`
   - Optional (recommended): `pip install taichi trimesh`
   - Project (editable): `pip install -e .[all]`

2) Run a demo script:

   - `python kung_fu_panda/examples/demo_script.py`
   - `python kung_fu_panda/examples/mesh_preview.py <optional_mesh_path>`

3) Or boot interactively in a notebook:

   ```python
   from kung_fu_panda.engine import Engine
   eng = Engine(window_title="Kung Fu Panda — Interactive", interactive=True)
   # now add/remove views/controllers live from cells
   ```

Controls (mesh_preview)
- WASD move, QE vertical strafe, Shift sprint, hold left mouse to look
- Space toggles surface/wireframe, L toggles labels, P screenshot, ESC exit

Docs Map
- Getting Started: installation, first runs, controls
- Concepts: Engine, Session, Views, Observers, Controllers, StoreState
- Rendering: meshes, materials, lighting, labels
- Input: FPV camera and other controllers
- Taichi: kernels, data flow, per‑frame updates
- Notebooks: interactive loop and workflow
- Recipes: step‑by‑step examples
- Philosophy: how to build abstractions to move fast
- FAQ: common hiccups and fixes

Links into this repo
- README: README.md
- Examples: kung_fu_panda/examples
- Core Engine: kung_fu_panda/engine.py
- Abstractions (base classes): kung_fu_panda/abc
- Views: kung_fu_panda/views
- Controllers: kung_fu_panda/controllers
- Resources (meshes/textures): kung_fu_panda/resources
- Taichi bridge: kung_fu_panda/utils/taichi_bridge.py

Next up: read Getting Started (docs/getting-started.md), then Concepts (docs/concepts.md).

