Getting Started

Install
- Python 3.9+
- Install packages:
  - Minimum: `pip install panda3d numpy`
  - Optional: `pip install taichi trimesh`
  - Project (editable with extras): `pip install -e .[all]`

Verify your environment
- Panda3D window opens and shows axes:

  ```bash
  python kung_fu_panda/examples/demo_script.py
  ```

- Preview a cube (use mouse + keyboard controls below):

  ```bash
  python kung_fu_panda/examples/mesh_preview.py
  ```

- Preview your own mesh (OBJ/PLY/STL via trimesh):

  ```bash
  python kung_fu_panda/examples/mesh_preview.py ~/models/bunny.obj
  # Skip recentre: --no-center
  ```

Keyboard & mouse
- Hold left mouse to enable mouselook
- `WASD` move, `Q/E` vertical strafe, `Shift` sprint
- `Space` toggle surface/wireframe
- `L` toggle vertex labels
- `P` screenshot
- `ESC` exit

Two usage modes
1) Script/demo mode (blocking)

   ```python
   from kung_fu_panda.engine import Engine
   from kung_fu_panda.views.axes import AxesView
   from kung_fu_panda.controllers import PauseController, ScreenshotController

   eng = Engine(window_title="Kung Fu Panda — Demo", interactive=False)
   eng.add_view(AxesView(axis_length=1.5))
   eng.add_controller(PauseController(eng, key="space"))
   eng.add_controller(ScreenshotController(eng, key="s", filename="shot.png"))
   eng.start()  # blocks until window closes
   ```

2) Interactive mode (Jupyter/REPL friendly)

   ```python
   from kung_fu_panda.engine import Engine
   eng = Engine(window_title="Kung Fu Panda — Interactive", interactive=True)
   # The engine starts an async loop and the notebook stays responsive.
   # You can add/remove views/controllers live across cells.
   # Stop when you’re done:
   # await eng.stop_async()  # in async cells
   # or eng.stop() in a script
   ```

Notebook tips
- If the window doesn’t show, make sure the Jupyter kernel allows GUI event loops (standard JupyterLab/Notebook does)
- Prefer `eng.start_async()` and `await eng.stop_async()` in async‑enabled cells; otherwise `eng.start()`/`eng.stop()` work too
- You can dynamically attach controllers (e.g., FPV camera) and toggle views while the loop runs

Project layout
- Engine: `kung_fu_panda/engine.py`
- Session wrapper: `kung_fu_panda/session.py`
- Base classes: `kung_fu_panda/abc/*.py`
- Views: `kung_fu_panda/views/*.py`
- Controllers: `kung_fu_panda/controllers/*.py`
- Mesh/Texture/Loaders: `kung_fu_panda/resources/*.py`
- Taichi <-> Numpy bridge: `kung_fu_panda/utils/taichi_bridge.py`

