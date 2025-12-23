Taichi: Compute Meets Rendering

If you haven’t used Taichi before, think of it as a Pythonic way to write high‑performance kernels (SIMD‑y, GPU‑y or CPU‑vectorized) with simple parallel loops. You define fields (arrays) and kernels (functions) that run fast. Perfect for per‑vertex updates or grid‑based simulations.

Install
- `pip install taichi`
- Minimal intro: https://docs.taichi.graphics/

Data flow in this framework
- Your simulation state often lives in Taichi fields (or external arrays)
- Rendering consumes contiguous NumPy buffers; use external arrays to avoid extra hops
- Helpers in `rheidos/utils/taichi_bridge.py`:
  - `field_to_numpy(field) -> np.ndarray`
  - `numpy_to_field(np_array, field) -> None`
  - `external_array(shape, dtype=np.float32, zero=False) -> np.ndarray`
  - `ensure_external_array(arr, dtype=np.float32) -> np.ndarray` (contiguous/writeable)

Basic pattern (per‑frame updates)
1) Initialize a mesh (positions/normals/colors) — meshes are `Geom.UHDynamic` by default
2) Allocate a reusable external NumPy buffer for the vertex data
3) Run a Taichi kernel that writes **directly** into that external array (`ti.types.ndarray`)
4) Copy the external array into Panda3D once via memoryview-backed `mesh.set_vertices(...)`

End‑to‑end example: wavy cube (external array path, 1 copy CPU → Panda)

```python
import numpy as np
import taichi as ti
from rheidos.engine import Engine
from rheidos.resources import cube
from rheidos.views import MeshSurfaceView
from rheidos.abc.observer import Observer
from rheidos.utils.taichi_bridge import external_array, ensure_external_array

ti.init()

primitive = cube(size=2.0)
v_handle = primitive.mesh.vdata.getArray(0).getHandle().getData()
V = np.frombuffer(memoryview(v_handle), dtype=np.float32).reshape(-1, 3)

N = V.shape[0]
positions_ext = external_array((N, 3))
# Seed the external buffer once from the mesh
np.copyto(positions_ext, V)

@ti.kernel
def wave(out: ti.types.ndarray(ndim=2, dtype=ti.f32), t: ti.f32):
    for i in range(out.shape[0]):
        x = out[i, 0]
        y = out[i, 1]
        z0 = out[i, 2]
        dz = 0.25 * ti.sin(2.0 * x + 2.5 * y + 1.5 * t)
        out[i, 2] = z0 + dz

class WaveObserver(Observer):
    def __init__(self, mesh):
        super().__init__(name="WaveObserver", sort=-5)
        self.mesh = mesh
        self.t = 0.0
    def update(self, dt: float) -> None:
        if dt <= 0:  # paused
            return
        self.t += dt
        wave(positions_ext, self.t)
        # Single memcpy into Panda3D’s dynamic vertex buffer (via memoryview)
        self.mesh.set_vertices(positions_ext)

eng = Engine("Taichi Wave", interactive=False)
eng.add_view(MeshSurfaceView(primitive.mesh))
eng.add_observer(WaveObserver(primitive.mesh))
eng.start()
```

Performance tips
- External arrays avoid the Taichi field -> NumPy -> Panda3D chain. Kernels that take `ti.types.ndarray` arguments can write straight into a reusable NumPy buffer; `mesh.set_vertices`/`set_normals` copy that buffer into Panda3D with one `np.copyto` and no intermediate allocations.
- If you’re updating your own `GeomVertexData` instead of a `Mesh`, call `rheidos.utils.panda_arrays.copy_numpy_to_vertex_array(...)` to fill its arrays via `memoryview`.
- If you still need fields (e.g., gradients, sparse), keep them internal and only expose external arrays for the data you render.
- Keep arrays contiguous (`np.ascontiguousarray`) and dtypes exact (`float32` for positions/normals/uvs, `uint8` or `float32` for colors).
- If you build your own `GeomVertexData`, mark usage as `Geom.UHDynamic` so Panda3D doesn’t over‑optimize for static data.
- If you also update normals, recompute them in your kernel or cache them on the Taichi side; then call `mesh.set_normals(...)` along with vertices.

Stepping logic vs. rendering
- Put your kernels in an `Observer` so they run irrespective of which views are enabled
- Use the engine’s pause flag to gate updates (dt==0 when paused in wrappers)

Debugging
- Print small samples from fields using `field.to_numpy()[:5]`
- For stability sensitive sims, separate “simulation dt” from “render dt” and integrate using a fixed step size accumulator in your observer

Further reading: see `docs/direct-copy.md` for a focused guide and examples of the single-copy pipeline.
