Taichi: Compute Meets Rendering

If you haven’t used Taichi before, think of it as a Pythonic way to write high‑performance kernels (SIMD‑y, GPU‑y or CPU‑vectorized) with simple parallel loops. You define fields (arrays) and kernels (functions) that run fast. Perfect for per‑vertex updates or grid‑based simulations.

Install
- `pip install taichi`
- Minimal intro: https://docs.taichi.graphics/

Data flow in this framework
- Your simulation state often lives in Taichi fields
- To render, you’ll push updated data to Panda3D via Numpy
- Helpers in `rheidos/utils/taichi_bridge.py`:
  - `field_to_numpy(field) -> np.ndarray`
  - `numpy_to_field(np_array, field) -> None`

Basic pattern (per‑frame updates)
1) Initialize a mesh (positions/normals/colors)
2) Mirror positions in a Taichi field
3) Each frame, run a kernel to update positions
4) Copy results back to Numpy and call `mesh.set_vertices(...)`

End‑to‑end example: wavy cube

```python
import numpy as np
import taichi as ti
from rheidos.engine import Engine
from rheidos.resources import cube
from rheidos.views import MeshSurfaceView
from rheidos.abc.observer import Observer
from rheidos.utils.taichi_bridge import numpy_to_field, field_to_numpy

ti.init()

primitive = cube(size=2.0)
V_np = primitive.mesh.vdata.getArray(0).getHandle().getData()
V = np.frombuffer(memoryview(V_np), dtype=np.float32).reshape(-1, 3).copy()

N = V.shape[0]
pos = ti.field(dtype=ti.f32, shape=(N, 3))
numpy_to_field(V, pos)

@ti.kernel
def wave(t: ti.f32):
    for i in range(N):
        x = pos[i, 0]
        y = pos[i, 1]
        z0 = pos[i, 2]
        dz = 0.25 * ti.sin(2.0 * x + 2.5 * y + 1.5 * t)
        pos[i, 2] = z0 + dz

class WaveObserver(Observer):
    def __init__(self, mesh):
        super().__init__(name="WaveObserver", sort=-5)
        self.mesh = mesh
        self.t = 0.0
    def update(self, dt: float) -> None:
        if dt <= 0:  # paused
            return
        self.t += dt
        wave(self.t)
        V_updated = field_to_numpy(pos)
        self.mesh.set_vertices(V_updated.astype(np.float32, copy=False))

eng = Engine("Taichi Wave", interactive=False)
eng.add_view(MeshSurfaceView(primitive.mesh))
eng.add_observer(WaveObserver(primitive.mesh))
eng.start()
```

Performance tips
- Copying fields <-> Numpy each frame has a cost. For small meshes it’s fine; for larger meshes consider batching updates (e.g., only every few frames) or using Taichi’s interop features to write directly into a contiguous Numpy buffer that you reuse.
- Keep arrays contiguous (`np.ascontiguousarray`) and dtypes exact (`float32` for positions/normals/uvs, `uint8` or `float32` for colors).
- If you also update normals, recompute them in your kernel or cache them on the Taichi side; then call `mesh.set_normals(...)` along with vertices.

Stepping logic vs. rendering
- Put your kernels in an `Observer` so they run irrespective of which views are enabled
- Use the engine’s pause flag to gate updates (dt==0 when paused in wrappers)

Debugging
- Print small samples from fields using `field.to_numpy()[:5]`
- For stability sensitive sims, separate “simulation dt” from “render dt” and integrate using a fixed step size accumulator in your observer
