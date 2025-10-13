import numpy as np
from kung_fu_panda.abc.view import View


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
        cols[:, 0] *= 0.7 + 0.3 * np.sin(self.t * 2.0)
        cols[:, 1] *= 0.8
        self.mesh.set_colors(cols)


# Usage
from kung_fu_panda.resources import cube
from kung_fu_panda.views import MeshSurfaceView
from kung_fu_panda.engine import Engine

prim = cube(2.0)
eng = Engine("Pulse", interactive=False)
eng.add_view(MeshSurfaceView(prim.mesh))
eng.add_view(PulseColors(prim.mesh))
eng.start()
