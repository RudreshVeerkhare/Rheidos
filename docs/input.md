Input & Controllers

Philosophy
- Keep input logic out of rendering. Controllers attach to the session, bind keys/mouse, and update the scene or engine state.
- Swap controllers freely (e.g., disable FPV in favor of a trackball down the line).

FPV camera
- Class: `kung_fu_panda/controllers/fpv_camera.py:FpvCameraController`
- Features:
  - Hold left mouse to capture and look around
  - WASD for planar motion, Q/E for vertical, Shift for faster speed
  - Robust yaw/pitch with global‑up clamping (prevents gimbal weirdness)
  - Clean attach/detach: camera re‑parents under a rig while active, restored on detach

Usage:

```python
from kung_fu_panda.controllers import FpvCameraController
eng.add_controller(FpvCameraController(speed=6.0, speed_fast=12.0, invert_y=False))
```

ToggleViewController
- Cycle between groups of view names. Great for switching render modes or overlays.

```python
from kung_fu_panda.controllers import ToggleViewController
eng.add_controller(ToggleViewController(eng, groups=[["surface"],["wireframe"]], key="space"))
eng.add_controller(ToggleViewController(eng, groups=[["labels"],[]], key="l"))
```

Pause, Screenshot, Exit
- `PauseController(eng, key="space")`: flips `Engine.is_paused()` — exposed to Views/Observers via `dt == 0` in wrappers
- `ScreenshotController(eng, key="p", filename="frame.png")`: saves a screenshot
- `ExitController(eng, key="escape")`: quits cleanly

Custom controller template

```python
from kung_fu_panda.abc.controller import Controller

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

