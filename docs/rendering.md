Rendering

Panda3D basics (crash course)
- Scene graph of `NodePath` objects under `render`
- Right‑handed, Y forward, Z up
- Per‑frame tasks update and draw
- Materials, lights, and shaders affect appearance

Meshes
- Class: `kung_fu_panda/resources/mesh.py:Mesh`
- Separate vertex arrays for positions, normals, colors, texcoords
- Dynamic updates supported (buffers created with dynamic usage by default)
- Construct and attach quickly:

  ```python
  import numpy as np
  from kung_fu_panda.resources import Mesh

  # A single triangle
  V = np.array([[-1,0,0],[1,0,0],[0,0,1]], dtype=np.float32)
  N = np.tile([0,1,0], (3,1)).astype(np.float32)
  C = np.tile([0.8,0.9,1.0,1.0], (3,1)).astype(np.float32)
  I = np.array([0,1,2], dtype=np.int32)

  mesh = Mesh(vertices=V, indices=I, normals=N, colors=C, name="tri")
  mesh.reparent_to(eng.session.render)
  ```

Updating mesh data
- All setters accept contiguous arrays:
  - `set_vertices((N,3) float32)`
  - `set_normals((N,3) float32)`
  - `set_colors((N,4) float32 in 0..1)` or `set_colors_uint8((N,4) uint8)`
  - `set_texcoords((N,2) float32)`
  - `set_indices((M*3,) or (M,3) int32)`

Primitives and loaders
- Procedural cube: `from kung_fu_panda.resources import cube`
- Load external mesh with trimesh: `load_mesh(path, center=True)`

  ```python
  from kung_fu_panda.resources import load_mesh, cube
  primitive = load_mesh("~/models/bunny.obj")  # or cube(size=2.0)
  eng.session.base.camera.lookAt(0,0,0)
  ```

Ready‑made Views
- `MeshSurfaceView`: glossy shaded surface with `setShaderAuto()` and optional `Material`
- `MeshWireframeView`: wireframe with a nice teal color
- `MeshPositionLabelsView`: shows a label with vertex coordinates near the mouse cursor and highlights the closest vertex

  ```python
  from kung_fu_panda.views import MeshSurfaceView, MeshWireframeView, MeshPositionLabelsView
  from panda3d.core import Material

  mat = Material("Glossy"); mat.setShininess(64)
  surface = MeshSurfaceView(primitive.mesh, material=mat, two_sided=False)
  wire = MeshWireframeView(primitive.mesh)
  labels = MeshPositionLabelsView(primitive.mesh, scale_factor=0.02)

  eng.add_view(surface)
  eng.add_view(wire)
  eng.add_view(labels)
  eng.enable_view("labels", False)  # start hidden
  ```

Lighting
- Panda3D’s lights are standard nodes; attach to scene and enable per‑render root:

  ```python
  from panda3d.core import AmbientLight, DirectionalLight, Vec4
  r = eng.session.render
  r.clearLight()
  amb = AmbientLight("ambient"); amb.setColor(Vec4(0.18,0.18,0.22,1)); r.setLight(r.attachNewNode(amb))
  key = DirectionalLight("key"); key.setColor(Vec4(0.85,0.85,0.9,1)); n = r.attachNewNode(key); n.setHpr(-35,-45,0); r.setLight(n)
  ```

Axes helper
- `AxesView` draws RGB axes (X red, Y green, Z blue). Simple but handy for orientation.

Two‑sided rendering
- For thin surfaces, call `node.setTwoSided(True)` via `MeshSurfaceView(two_sided=True)` or `mesh.set_two_sided(True)`

