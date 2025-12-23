# Direct Copy Pipeline (Taichi → NumPy external array → Panda3D)

Goal: avoid the Taichi field → new NumPy array → Panda3D buffer chain. With external arrays and memoryviews, the CPU path becomes:

```
Taichi kernel writes external array
→ single memcpy into Panda3D’s dynamic vertex buffer
```

This keeps sims decoupled (still trading in NumPy arrays) while removing extra allocations/conversions.

## Prerequisites
- Panda3D vertex data marked dynamic (`Geom.UHDynamic`)
- Contiguous `float32` buffers for positions/normals/uvs, `float32` or `uint8` for colors
- Taichi kernels accepting `ti.types.ndarray` for external arrays

Helpers:
- `rheidos.utils.taichi_bridge.external_array(shape, dtype=np.float32, zero=False)`
- `rheidos.utils.taichi_bridge.ensure_external_array(arr, dtype=np.float32)`
- `rheidos.utils.panda_arrays.copy_numpy_to_vertex_array(vdata, array_index, src, cols)`
- `Mesh.set_vertices/set_normals/set_texcoords/set_colors` already use the fast copy path internally

## Example A: Mesh convenience path (positions only)

```python
import numpy as np
import taichi as ti
from rheidos.engine import Engine
from rheidos.resources import cube
from rheidos.views import MeshSurfaceView
from rheidos.abc.observer import Observer
from rheidos.utils.taichi_bridge import external_array

ti.init()

primitive = cube(size=2.0)  # Mesh uses Geom.UHDynamic
v_handle = primitive.mesh.vdata.getArray(0).getHandle().getData()
V = np.frombuffer(memoryview(v_handle), dtype=np.float32).reshape(-1, 3)

positions_ext = external_array(V.shape)  # contiguous, writable NumPy
np.copyto(positions_ext, V)              # seed from existing mesh data

@ti.kernel
def wave(out: ti.types.ndarray(ndim=2, dtype=ti.f32), t: ti.f32):
    for i in range(out.shape[0]):
        x, y, z0 = out[i, 0], out[i, 1], out[i, 2]
        out[i, 2] = z0 + 0.25 * ti.sin(2.0 * x + 2.5 * y + 1.5 * t)

class WaveObserver(Observer):
    def __init__(self, mesh):
        super().__init__(name="WaveObserver", sort=-5)
        self.mesh = mesh
        self.t = 0.0
    def update(self, dt: float) -> None:
        if dt <= 0:
            return
        self.t += dt
        wave(positions_ext, self.t)      # Taichi writes external array
        self.mesh.set_vertices(positions_ext)  # one copy into Panda buffer

eng = Engine("Wave", interactive=False)
eng.add_view(MeshSurfaceView(primitive.mesh))
eng.add_observer(WaveObserver(primitive.mesh))
eng.start()
```

Notes:
- `Mesh` setters already use memoryview-backed copies; you only provide a contiguous array.
- The only per-frame copy is `np.copyto` inside `set_vertices` into the Panda3D buffer.

## Example B: Custom GeomVertexData (positions + colors)

```python
import numpy as np
import taichi as ti
from panda3d.core import Geom, GeomPoints, GeomVertexData, GeomVertexFormat
from rheidos.utils.taichi_bridge import external_array, ensure_external_array
from rheidos.utils.panda_arrays import copy_numpy_to_vertex_array

ti.init()

# Dynamic vertex data
fmt = GeomVertexFormat.getV3c4()
vdata = GeomVertexData("points", fmt, Geom.UHDynamic)
vdata.setNumRows(0)

# External arrays
N = 1024
positions_ext = external_array((N, 3))
colors_ext = external_array((N, 4))

@ti.kernel
def fill(out_pos: ti.types.ndarray(ndim=2, dtype=ti.f32),
         out_col: ti.types.ndarray(ndim=2, dtype=ti.f32),
         t: ti.f32):
    for i in range(out_pos.shape[0]):
        out_pos[i, 0] = ti.sin(0.01 * i + t)
        out_pos[i, 1] = ti.cos(0.01 * i + t)
        out_pos[i, 2] = 0.0
        out_col[i, :] = ti.Vector([0.2, 0.8, 1.0, 1.0])

def update(t: float) -> None:
    fill(positions_ext, colors_ext, t)
    # Single copies into Panda arrays via memoryview
    copy_numpy_to_vertex_array(vdata, 0, positions_ext, cols=3)
    copy_numpy_to_vertex_array(vdata, 1, colors_ext, cols=4)

# Build a primitive for rendering (points here)
prim = GeomPoints(Geom.UHDynamic)
prim.add_next_vertices(N)
prim.close_primitive()
geom = Geom(vdata)
geom.add_primitive(prim)
```

Notes:
- `copy_numpy_to_vertex_array` reshapes the Panda array view and `np.copyto`s your external buffer—no `tobytes` conversions.
- Use `ensure_external_array` if your producer hands you a non-contiguous buffer.
- Keep `Geom.UHDynamic` for both `GeomVertexData` and primitives when you stream updates.

## Guidelines
- Keep buffers contiguous (`np.ascontiguousarray`) and exact dtype (`float32` for positions/normals/uvs; `float32`/`uint8` for colors).
- Reuse the same external arrays every frame; don’t allocate inside the update loop.
- Seed external arrays from existing Panda buffers via `memoryview` once, then mutate in-place.
- If you need Taichi fields for other reasons, keep them internal; expose only the external arrays needed for rendering.
