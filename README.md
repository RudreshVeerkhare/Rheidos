Kung Fu Panda — DDG + Taichi + Panda3D Playground

Overview
- A small framework to prototype physics simulations using discrete differential geometry (DDG), Taichi for compute kernels, and Panda3D for rendering.
- Two usage modes:
  - Interactive: drive from a Jupyter notebook while a Panda3D window renders in a background loop.
  - Script/Demo: run a single script that sets up views/controllers and blocks in Panda3D’s main loop.

Core Abstractions
- Views: Frame-updated tasks with full access to the Panda3D session. Use to render meshes, scalar fields, vector fields, overlays, legends, etc.
- Controllers: Input/event handlers. Can attach/detach views, pause, capture screenshots, etc.
- Observers: Like views but intended for logging/metrics or pre/post steps. Scheduled with task sort priorities.
- Resources (Mesh/Texture): Efficiently bridge Numpy/Taichi data into Panda3D buffers. Mesh uses separate vertex arrays for positions, normals, colors, texcoords.
- StoreState: Global, thread-safe store that views/controllers/observers can query/update.

Install
- Requires Python 3.9+.
- Suggested deps: panda3d, taichi, numpy.

Using pip:

  pip install -e .[all]

Examples
- Script mode:

  python kung_fu_panda/examples/demo_script.py

- Interactive loop (suitable for notebooks or REPL):

  python kung_fu_panda/examples/run_interactive.py

- Jupyter notebook walkthrough:

  jupyter notebook notebooks/interactive_demo.ipynb

- Mesh preview with FPV controls (space toggles surface/wireframe, ESC exits):

  python kung_fu_panda/examples/mesh_preview.py

- Load a custom OBJ/PLY/STL file and preview it with glossy shading:

  python kung_fu_panda/examples/mesh_preview.py path/to/model.obj

- Additional flags:

  python kung_fu_panda/examples/mesh_preview.py path/to/model.obj --no-center

Controls: WASD to fly, QE vertical strafe, Shift to sprint, mouse to look, Space toggle wireframe, P screenshot, ESC exit.

Jupyter Basics
- In a notebook, you can import and boot the Engine in interactive mode (the loop auto-starts):

  from kung_fu_panda.engine import Engine
  eng = Engine(window_title="Kung Fu Panda — Interactive", interactive=True)
  # Now add/remove views/controllers dynamically from cells

- Stop the interactive loop when done:

  eng.stop()

Notes
- Video capture is stubbed; screenshots are supported.
- Mesh utilities expect float32 arrays for positions/normals/texcoords and float32/uint8 for colors.
- Trimesh-powered mesh loading is optional; install via `pip install -e .[mesh]` or `.[all]`.
