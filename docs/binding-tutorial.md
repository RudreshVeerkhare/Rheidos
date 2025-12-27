# Binding Tutorial: Mesh Deformation From a Signal

Audience: you can run Rheidos scripts and are comfortable editing a Python file.
Prereqs: `pip install -e .[all]` (Panda3D, Taichi, NumPy, trimesh).

Goal: wire a compute signal to a mesh deformer using the Adapter/Binder API, and see the mesh move when you select points.

## Step 1: Load a scene and mesh
Action: create a tiny app that loads the Poisson scene and grabs the mesh.

```python
from pathlib import Path
from panda3d.core import BitMask32

from rheidos.engine import Engine
from rheidos.scene_config import load_scene_from_config

from apps.poisson_dec.system import PoissonSystem

def _scene_mesh_arrays(mesh_name: str, scene_result):
    target = None
    for obj in scene_result.objects:
        if obj.name == mesh_name:
            target = obj
            break
    if target is None and scene_result.objects:
        target = scene_result.objects[0]
    if target is None:
        raise RuntimeError("Scene has no mesh objects to bind.")
    mesh = target.primitive.mesh
    vertices = mesh.get_vertices()
    indices = mesh.get_indices()
    if vertices is None or indices is None:
        raise RuntimeError("Scene mesh is missing vertex or index buffers.")
    faces = indices.reshape(-1, 3)
    return mesh, vertices, faces

cfg_path = Path(__file__).resolve().parent / "scene_configs" / "poisson.yaml"
eng = Engine(window_title="Binding Tutorial", interactive=False)
scene_result = load_scene_from_config(eng, cfg_path)
mesh, vertices, faces = _scene_mesh_arrays("domain", scene_result)
```

Result: you have a loaded scene mesh and its vertex/index arrays.

## Step 2: Initialize compute from the scene mesh
Action: create the Poisson system and pass the scene mesh into compute.

```python
import taichi as ti

ti.init()
compute = PoissonSystem()
compute.set_mesh_from_numpy(vertices, faces)
```

Result: compute and render now share the same mesh topology.

## Step 3: Create the adapter and scheduler
Action: build the adapter, then schedule it as an observer so compute runs only when needed.

```python
from rheidos.sim.interaction import ComputeScheduler
from apps.poisson_dec.interaction import PoissonAdapter

adapter = PoissonAdapter(compute, instance="domain")
scheduler = ComputeScheduler(adapter)
eng.add_observer(scheduler)
```

Result: compute is decoupled from rendering; it only runs when queued.

## Step 4: Bind the vertex-scalar signal to a mesh deformer
Action: register a binding rule and bind the adapter to the scene mesh.

```python
from rheidos.sim.interaction import AdapterBinder, BindingRule
from apps.poisson_dec.bindings import PoissonMeshDeformer
from rheidos.resources.mesh import Mesh

def _bind_vertex_scalar(signal, topology, _adapter):
    if not isinstance(topology, Mesh):
        raise RuntimeError("Vertex scalar binding requires a mesh topology.")
    return PoissonMeshDeformer(topology, signal, scale=0.2)

binder = AdapterBinder(eng)
binder.add_rule(BindingRule(domain="vertex", meaning="scalar", factory=_bind_vertex_scalar))

mesh_map = {obj.name: obj.primitive.mesh for obj in scene_result.objects}
binder.bind(adapter, topologies=mesh_map)
```

Result: any `vertex/scalar` signal with topology "domain" is now wired to a mesh deformer.

## Step 5: Queue constraints from selections
Action: enqueue constraints when the user selects points.

```python
from rheidos.views import PointSelectionView
from rheidos.controllers.point_selector import SceneVertexPointSelector
from apps.poisson_dec.interaction import ChargeBatch

POS_STORE_KEY = "poisson/pos_points"
NEG_STORE_KEY = "poisson/neg_points"

pos_markers = PointSelectionView(name="pos_markers")
neg_markers = PointSelectionView(name="neg_markers")
eng.add_view(pos_markers)
eng.add_view(neg_markers)

eng.add_controller(
    SceneVertexPointSelector(
        engine=eng,
        pick_mask=BitMask32.bit(4),
        markers_view=pos_markers,
        store_key=POS_STORE_KEY,
        select_button="mouse1",
        clear_shortcut="c",
    )
)
eng.add_controller(
    SceneVertexPointSelector(
        engine=eng,
        pick_mask=BitMask32.bit(4),
        markers_view=neg_markers,
        store_key=NEG_STORE_KEY,
        select_button="mouse3",
        clear_shortcut="v",
    )
)

def _collect_charges(store):
    charges = []
    for point in store.get(POS_STORE_KEY, ()) or ():
        charges.append((int(point["index"]), 1.0))
    for point in store.get(NEG_STORE_KEY, ()) or ():
        charges.append((int(point["index"]), -1.0))
    return charges

def _on_points_changed(_value):
    scheduler.enqueue("constraints", ChargeBatch(_collect_charges(eng.store)))

eng.store.subscribe(POS_STORE_KEY, _on_points_changed)
eng.store.subscribe(NEG_STORE_KEY, _on_points_changed)
```

Result: clicking on the mesh applies constraints, runs compute, and the mesh deforms.

## Step 6: Run the app
Action: start the engine.

```python
eng.start()
```

Result: you should see the mesh deform around selected points.
