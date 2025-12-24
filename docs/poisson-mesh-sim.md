# Poisson on a Triangle Mesh (DEC + Point Picking)

Incremental, runnable steps to build a Poisson solver demo on a surface mesh. You will:
- load a triangle mesh from a scene config
- pick + and - constraint vertices
- build DEC operators (d0, d1, star0, star1, star2)
- solve a Poisson/Laplace system
- visualize u and deform the surface along normals

## Store keys to use throughout
`poisson/pos_points`, `poisson/neg_points`, `poisson/deform_scale`.

## Step 0 - Env prep (once)
- `pip install -e .`

## Step 1 - Scene config (mesh + picking)
Create `apps/poisson_mesh/scene_configs/poisson.yaml`:
```yaml
meshes:
  - path: ../../../models/flat_mesh.obj
    name: domain
    pickable: true
    surface: true
    wireframe: false
    two_sided: true
    material:
      diffuse: [0.8, 0.82, 0.90, 1.0]

camera:
  auto_frame: true

studio:
  enabled: true

controllers:
  - factory: rheidos.examples.scene_factories:make_fpv_camera_controller
    config:
      speed: 6.0
      speed_fast: 12.0
      mouse_sensitivity: 0.15
      roll_speed: 120.0
      pitch_limit_deg: 89.0
      lock_mouse: false
      invert_y: false
  - factory: rheidos.examples.scene_factories:make_exit_controller
    config: { key: escape }

ui:
  scene_config_panel: true
```

Smoke test:
`python -m rheidos.examples.point_selection --scene-config apps/poisson_mesh/scene_configs/poisson.yaml`

## Step 2 - Minimal loader
Create `apps/poisson_mesh/app.py`:
```python
from pathlib import Path

from rheidos.engine import Engine
from rheidos.scene_config import load_scene_from_config


def main() -> None:
    cfg_path = Path(__file__).resolve().parent / "scene_configs" / "poisson.yaml"
    eng = Engine(window_title="Poisson Mesh Demo", interactive=False)
    load_scene_from_config(eng, cfg_path)
    eng.start()


if __name__ == "__main__":
    main()
```

Run:
`python apps/poisson_mesh/app.py`

## Step 3 - Add + / - point picking
Append to `apps/poisson_mesh/app.py`:
```python
from panda3d.core import BitMask32
from rheidos.controllers import SceneVertexPointSelector
from rheidos.views import PointSelectionView

# After load_scene_from_config(...)
pick_mask = BitMask32.bit(4)

pos_markers = PointSelectionView(
    name="pos_markers",
    selected_color=(1.0, 0.25, 0.25, 1.0),
    hover_color=(1.0, 0.9, 0.5, 1.0),
)
neg_markers = PointSelectionView(
    name="neg_markers",
    selected_color=(0.2, 0.4, 1.0, 1.0),
    hover_color=(0.6, 0.8, 1.0, 1.0),
)

eng.add_view(pos_markers)
eng.add_view(neg_markers)

pos_selector = SceneVertexPointSelector(
    engine=eng,
    pick_mask=pick_mask,
    markers_view=pos_markers,
    store_key="poisson/pos_points",
    select_button="mouse1",
    clear_shortcut="c",
)
neg_selector = SceneVertexPointSelector(
    engine=eng,
    pick_mask=pick_mask,
    markers_view=neg_markers,
    store_key="poisson/neg_points",
    select_button="mouse3",
    clear_shortcut="v",
)

eng.add_controller(pos_selector)
eng.add_controller(neg_selector)
```

Run again. Left click adds a positive vertex (red), right click adds a negative vertex (blue). Press `c` or `v` to clear each set.

## Step 4 - DEC operators
Create `apps/poisson_mesh/dec.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class DECOperators:
    vertices: np.ndarray
    faces: np.ndarray
    edges: np.ndarray
    d0: np.ndarray
    d1: np.ndarray
    star0: np.ndarray  # diagonal entries (len V)
    star1: np.ndarray  # diagonal entries (len E)
    star2: np.ndarray  # diagonal entries (len F)

    def laplacian(self) -> np.ndarray:
        weighted_d0 = self.star1[:, None] * self.d0
        return self.d0.T @ weighted_d0


def extract_mesh_arrays(mesh) -> tuple[np.ndarray, np.ndarray]:
    handle = mesh.vdata.getArray(0).getHandle()
    raw = memoryview(handle.getData())
    vertices = np.frombuffer(raw, dtype=np.float32).reshape(-1, 3).copy()

    prim = mesh.prim
    count = prim.getNumVertices()
    indices = np.array([prim.getVertex(i) for i in range(count)], dtype=np.int32)
    faces = indices.reshape(-1, 3)
    return vertices, faces


def _cotangent(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u = a - c
    v = b - c
    cross = np.cross(u, v)
    denom = np.linalg.norm(cross)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(u, v) / denom)


def build_dec(vertices: np.ndarray, faces: np.ndarray) -> DECOperators:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    vcount = vertices.shape[0]
    fcount = faces.shape[0]

    edge_map: dict[tuple[int, int], int] = {}
    edges: list[tuple[int, int]] = []
    face_edges: list[list[tuple[int, float]]] = []

    for tri in faces:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        tri_edges = []
        for a, b in ((i, j), (j, k), (k, i)):
            key = (a, b) if a < b else (b, a)
            if key not in edge_map:
                edge_map[key] = len(edges)
                edges.append(key)
            e_idx = edge_map[key]
            sign = 1.0 if (a, b) == key else -1.0
            tri_edges.append((e_idx, sign))
        face_edges.append(tri_edges)

    edges_arr = np.asarray(edges, dtype=np.int32)
    ecount = edges_arr.shape[0]

    d0 = np.zeros((ecount, vcount), dtype=np.float32)
    for e_idx, (a, b) in enumerate(edges_arr):
        d0[e_idx, a] = -1.0
        d0[e_idx, b] = 1.0

    d1 = np.zeros((fcount, ecount), dtype=np.float32)
    for f_idx, tri_edges in enumerate(face_edges):
        for e_idx, sign in tri_edges:
            d1[f_idx, e_idx] = sign

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_areas = 0.5 * np.linalg.norm(face_normals, axis=1)

    vertex_areas = np.zeros(vcount, dtype=np.float32)
    for f_idx, tri in enumerate(faces):
        vertex_areas[tri] += face_areas[f_idx] / 3.0

    edge_cot = np.zeros(ecount, dtype=np.float32)
    for tri in faces:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        cot_i = _cotangent(vj, vk, vi)  # opposite edge (j, k)
        cot_j = _cotangent(vk, vi, vj)  # opposite edge (k, i)
        cot_k = _cotangent(vi, vj, vk)  # opposite edge (i, j)

        edge_cot[edge_map[(j, k) if j < k else (k, j)]] += cot_i
        edge_cot[edge_map[(k, i) if k < i else (i, k)]] += cot_j
        edge_cot[edge_map[(i, j) if i < j else (j, i)]] += cot_k

    edge_cot *= 0.5

    star0 = vertex_areas.astype(np.float32)
    star1 = edge_cot.astype(np.float32)
    star2 = (1.0 / np.maximum(face_areas, 1e-12)).astype(np.float32)

    return DECOperators(
        vertices=vertices,
        faces=faces,
        edges=edges_arr,
        d0=d0,
        d1=d1,
        star0=star0,
        star1=star1,
        star2=star2,
    )
```

Update `apps/poisson_mesh/app.py` to build DEC and print shapes:
```python
import numpy as np
from dec import extract_mesh_arrays, build_dec

# After load_scene_from_config(...)
scene = load_scene_from_config(eng, cfg_path)
mesh = scene.objects[0].primitive.mesh

base_vertices, faces = extract_mesh_arrays(mesh)
dec = build_dec(base_vertices, faces)

print("V,E,F:", base_vertices.shape[0], dec.edges.shape[0], faces.shape[0])
print("d0:", dec.d0.shape, "d1:", dec.d1.shape)
```

Run and confirm the shapes print in the console.

## Step 5 - Poisson solve (Dirichlet) + vertex colors
Append to `apps/poisson_mesh/app.py`:
```python
from rheidos.abc.observer import Observer


def _indices_from_store(store, key: str) -> list[int]:
    points = store.get(key, []) or []
    indices = []
    for p in points:
        idx = p.get("index")
        if idx is not None:
            indices.append(int(idx))
    return sorted(set(indices))


def _apply_dirichlet(L: np.ndarray, b: np.ndarray, fixed: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    A = L.copy()
    rhs = b.copy()
    for idx, val in fixed.items():
        rhs -= A[:, idx] * val
        A[:, idx] = 0.0
        A[idx, :] = 0.0
        A[idx, idx] = 1.0
        rhs[idx] = val
    return A, rhs


def _cg_solve(A: np.ndarray, b: np.ndarray, tol: float = 1e-6, max_iter: int = 200) -> np.ndarray:
    x = np.zeros_like(b)
    r = b - A @ x
    p = r.copy()
    rs_old = float(np.dot(r, r))
    if rs_old < tol * tol:
        return x
    for _ in range(max_iter):
        Ap = A @ p
        denom = float(np.dot(p, Ap))
        if abs(denom) < 1e-12:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(np.dot(r, r))
        if rs_new < tol * tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


def _values_to_colors(values: np.ndarray) -> np.ndarray:
    vmax = float(max(abs(values.min()), abs(values.max()), 1e-8))
    t = np.clip(values / vmax, -1.0, 1.0)
    r = np.where(t >= 0.0, 1.0, 1.0 + t)
    b = np.where(t <= 0.0, 1.0, 1.0 - t)
    g = 1.0 - np.abs(t)
    colors = np.stack([r, g, b, np.ones_like(r)], axis=1)
    return colors.astype(np.float32)


class PoissonUpdater(Observer):
    def __init__(self, mesh, dec, base_vertices, faces, store):
        super().__init__("PoissonUpdater", sort=-5)
        self.mesh = mesh
        self.dec = dec
        self.base = base_vertices
        self.faces = faces
        self.store = store
        self.L = dec.laplacian().astype(np.float32)
        self.last_key = None

    def update(self, dt: float) -> None:
        pos_idx = _indices_from_store(self.store, "poisson/pos_points")
        neg_idx = _indices_from_store(self.store, "poisson/neg_points")
        key = (tuple(pos_idx), tuple(neg_idx))
        if key == self.last_key:
            return
        self.last_key = key

        if not pos_idx and not neg_idx:
            colors = _values_to_colors(np.zeros(self.base.shape[0], dtype=np.float32))
            self.mesh.set_colors(colors)
            self.mesh.set_vertices(self.base)
            return

        fixed = {i: 1.0 for i in pos_idx}
        fixed.update({i: -1.0 for i in neg_idx})

        b = np.zeros(self.base.shape[0], dtype=np.float32)
        A, rhs = _apply_dirichlet(self.L, b, fixed)
        u = _cg_solve(A, rhs, tol=1e-6, max_iter=200)

        colors = _values_to_colors(u)
        self.mesh.set_colors(colors)
```

Add the observer in `main` after DEC creation:
```python
eng.add_observer(PoissonUpdater(mesh, dec, base_vertices, faces, eng.store))
```

Run again. Pick a few red and blue vertices; the surface colors should update.

## Step 6 - Deform along normals
Extend `PoissonUpdater` to compute normals and displace vertices:
```python
def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for idx in range(faces.shape[0]):
        a, b, c = faces[idx]
        normals[a] += face_normals[idx]
        normals[b] += face_normals[idx]
        normals[c] += face_normals[idx]
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lens, 1e-8)
    return normals.astype(np.float32)
```

Inside `PoissonUpdater.update`, after computing `u`:
```python
scale = float(self.store.get("poisson/deform_scale", 0.15))
normals = _vertex_normals(self.base, self.faces)
deformed = self.base + normals * (u[:, None] * scale)
self.mesh.set_vertices(deformed)
self.mesh.set_normals(_vertex_normals(deformed, self.faces))
```

Seed the store key in `main`:
```python
eng.store.set("poisson/deform_scale", 0.15)
```

Run again and adjust `poisson/deform_scale` in the ImGui Store panel to see the surface move.

## Step 7 - Optional: keep snapping accurate after deformation
`SceneVertexPointSelector` caches vertex positions the first time you pick. If you keep picking after large deformations, clear the cache after updates:
```python
try:
    pos_selector._geom_cache.clear()
    neg_selector._geom_cache.clear()
except Exception:
    pass
```

Drop this at the end of `PoissonUpdater.update` if you see stale snapping.
