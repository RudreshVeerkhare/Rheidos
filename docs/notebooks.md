Notebooks & Interactive Loop

Why interactive?
- Faster iteration: tweak a kernel, rerun a cell, see results immediately
- Keep the render loop running cooperatively alongside the notebook’s event loop

Engine setup

```python
from rheidos.engine import Engine
eng = Engine(window_title="Rheidos — Interactive", interactive=True)
```

What happens under the hood
- The engine creates Panda3D’s `ShowBase` and a window
- It schedules two internal tasks: a service/dispatch task and an FPS title updater
- In interactive mode, it starts an asyncio task that calls `taskMgr.step()` roughly at your target FPS
- You can still call `eng.start()` from a regular cell; it’ll run an async helper that pumps until you stop

Working style
- Define views/controllers/observers in cells and attach them:

  ```python
  from rheidos.views import AxesView
  from rheidos.controllers import FpvCameraController
  eng.add_view(AxesView(axis_length=1.5))
  eng.add_controller(FpvCameraController())
  ```

- Toggle views on/off without tearing them down: `eng.enable_view("labels", False)`
- Clean up at the end:
  - Async cells: `await eng.stop_async()`
  - Non‑async: `eng.stop()` (best‑effort)

Common gotchas
- If the window doesn’t respond: ensure your notebook server allows GUI windows and that you aren’t starting competing event loops
- Avoid long blocking work on the main thread; put heavy work in observers that do quick increments per frame
- If you see GL errors after very early exceptions, restart the kernel (Panda3D doesn’t like half‑constructed contexts)

