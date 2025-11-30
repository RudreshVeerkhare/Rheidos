Recipes

1) Minimal surface + FPV + toggle

```python
from rheidos.engine import Engine
from rheidos.resources import cube
from rheidos.views import MeshSurfaceView, MeshWireframeView, AxesView
from rheidos.controllers import FpvCameraController, ToggleViewController

eng = Engine("Minimal", interactive=False)
prim = cube(size=2.0)
surface = MeshSurfaceView(prim.mesh, name="surface")
wire = MeshWireframeView(prim.mesh, name="wire")

eng.add_view(AxesView(axis_length=1.5, sort=-10))
eng.add_view(surface)
eng.add_view(wire)
eng.add_controller(FpvCameraController())
eng.add_controller(ToggleViewController(eng, groups=[["surface"],["wire"]], key="space"))
eng.start()
```

2) Live color pulsing (view update)

```python
import numpy as np
from rheidos.abc.view import View

class PulseColors(View):
    def __init__(self, mesh):
        super().__init__(name="PulseColors")
        self.mesh = mesh
        self.t = 0.0
        self._base = None
    def setup(self, session):
        # Cache a baseline copy
        handle = self.mesh.vdata.getArray(2).getHandle()  # colors
        raw = memoryview(handle.getData())
        self._base = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4).copy()
    def update(self, dt: float):
        self.t += dt
        cols = self._base.copy()
        cols[:,0] *= 0.7 + 0.3*np.sin(self.t*2.0)
        cols[:,1] *= 0.8
        self.mesh.set_colors(cols)

# Usage
from rheidos.resources import cube
from rheidos.views import MeshSurfaceView
from rheidos.engine import Engine
prim = cube(2.0)
eng = Engine("Pulse", interactive=False)
eng.add_view(MeshSurfaceView(prim.mesh))
eng.add_view(PulseColors(prim.mesh))
eng.start()
```

3) Pause‑aware simulation (observer)

```python
from rheidos.abc.observer import Observer

class StepSim(Observer):
    def __init__(self):
        super().__init__(name="StepSim")
        self.time = 0.0
    def update(self, dt: float):
        if dt <= 0:  # engine paused
            return
        self.time += dt
        # step your sim with fixed or variable dt

# Add it
eng.add_observer(StepSim())
```

4) Screenshot on condition

```python
def on_peak(val):
    if val > 0.95:
        eng.screenshot("peak.png")
unsub = eng.store.subscribe("signal", on_peak)
eng.store.set("signal", 1.0)
```

