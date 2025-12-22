Input & Controllers

Philosophy
- Keep input logic out of rendering. Controllers attach to the session, bind keys/mouse, and update the scene or engine state.
- Swap controllers freely (e.g., disable FPV in favor of a trackball down the line).

FPV camera
- Class: `rheidos/controllers/fpv_camera.py:FpvCameraController`
- Features:
  - Hold left mouse to capture and look around (cursor stays visible/moves by default; set `lock_mouse=True` if you want it hidden/centered)
  - WASD for motion, Q/E for vertical thrust, Z/C to roll left/right, Shift for faster speed
  - World‑up yaw with clamped pitch (no pole flips) and optional roll for a true flycam feel
  - Clean attach/detach: camera re‑parents under a rig while active, restored on detach

Usage:

```python
from rheidos.controllers import FpvCameraController
eng.add_controller(FpvCameraController(speed=6.0, speed_fast=12.0, invert_y=False))
```

ToggleViewController
- Cycle between groups of view names. Great for switching render modes or overlays.

```python
from rheidos.controllers import ToggleViewController
eng.add_controller(ToggleViewController(eng, groups=[["surface"],["wireframe"]], key="space"))
eng.add_controller(ToggleViewController(eng, groups=[["labels"],[]], key="l"))
```

Pause, Screenshot, Exit
- `PauseController(eng, key="space")`: flips `Engine.is_paused()` — exposed to Views/Observers via `dt == 0` in wrappers
- `ScreenshotController(eng, key="p", filename="frame.png")`: saves a screenshot
- `ExitController(eng, key="escape")`: quits cleanly

Custom controller template

```python
from rheidos.abc.controller import Controller

class MyController(Controller):
    def __init__(self, key="f"):
        super().__init__(name="MyController")
        self.key = key
    def attach(self, session):
        super().attach(session)
        session.accept(self.key, self._on)
    def detach(self):
        self._session.ignore(self.key)
    def _on(self):
        print("Toggled!")
```
