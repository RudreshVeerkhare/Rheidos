from pathlib import Path

import numpy as np
import taichi as ti
from panda3d.core import BitMask32

from rheidos.controllers.point_selector import SceneVertexPointSelector
from rheidos.engine import Engine
from rheidos.resources.mesh import Mesh
from rheidos.scene_config import load_scene_from_config
from rheidos.sim.interaction import AdapterBinder, BindingRule, ComputeScheduler
from rheidos.utils.geom_guard import GeomNanGuard
from rheidos.views import PointSelectionView

from apps.poisson_dec.bindings import PoissonMeshDeformer
from apps.poisson_dec.interaction import ChargeBatch, PoissonAdapter
from apps.poisson_dec.system import Charge, PoissonSystem

POS_STORE_KEY = "poisson/pos_points"
NEG_STORE_KEY = "poisson/neg_points"
DEFORM_SCALE = 0.2


def _scene_mesh_arrays(mesh_name: str, scene_result) -> tuple[Mesh, np.ndarray, np.ndarray]:
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


class PoissonScene:
    def __init__(self, engine: Engine, scheduler: ComputeScheduler, *, pick_mask: BitMask32) -> None:
        self.engine = engine
        self.scheduler = scheduler
        self.pick_mask = pick_mask

        self._setup_views()
        self._setup_selectors()
        self._subscribe_store()

    def _setup_views(self) -> None:
        self.pos_markers = PointSelectionView(
            name="pos_markers",
            selected_color=(1.0, 0.25, 0.25, 1.0),
            hover_color=(1.0, 0.9, 0.5, 1.0),
        )
        self.neg_markers = PointSelectionView(
            name="neg_markers",
            selected_color=(0.2, 0.4, 1.0, 1.0),
            hover_color=(0.6, 0.8, 1.0, 1.0),
        )

        self.engine.add_view(self.pos_markers)
        self.engine.add_view(self.neg_markers)

    def _setup_selectors(self) -> None:
        self.engine.add_controller(
            SceneVertexPointSelector(
                engine=self.engine,
                pick_mask=self.pick_mask,
                markers_view=self.pos_markers,
                store_key=POS_STORE_KEY,
                select_button="mouse1",
                clear_shortcut="c",
            )
        )
        self.engine.add_controller(
            SceneVertexPointSelector(
                engine=self.engine,
                pick_mask=self.pick_mask,
                markers_view=self.neg_markers,
                store_key=NEG_STORE_KEY,
                select_button="mouse3",
                clear_shortcut="v",
            )
        )

    def _subscribe_store(self) -> None:
        self.engine.store.subscribe(POS_STORE_KEY, self._on_points_changed)
        self.engine.store.subscribe(NEG_STORE_KEY, self._on_points_changed)

    def _collect_charges(self) -> list[Charge]:
        charges: list[Charge] = []
        pos_points = self.engine.store.get(POS_STORE_KEY, ())
        neg_points = self.engine.store.get(NEG_STORE_KEY, ())
        for point in pos_points or ():
            charges.append((int(point["index"]), 1.0))
        for point in neg_points or ():
            charges.append((int(point["index"]), -1.0))
        return charges

    def _on_points_changed(self, _value: object) -> None:
        charges = self._collect_charges()
        self.scheduler.enqueue("constraints", ChargeBatch(charges))

    def sync(self) -> None:
        self._on_points_changed(None)


def main() -> None:
    cfg_path = Path(__file__).resolve().parent / "scene_configs" / "poisson.yaml"
    eng = Engine(
        window_title="Poisson Mesh Demo",
        interactive=False,
        enable_imgui=True,
        imgui_use_glfw=False,
    )
    scene_result = load_scene_from_config(eng, cfg_path)
    ti.init()

    compute = PoissonSystem()
    mesh, vertices, faces = _scene_mesh_arrays("domain", scene_result)
    compute.set_mesh_from_numpy(vertices, faces)

    adapter = PoissonAdapter(compute, instance="domain")
    scheduler = ComputeScheduler(adapter)
    eng.add_observer(scheduler)
    eng.add_observer(GeomNanGuard())

    scene = PoissonScene(eng, scheduler, pick_mask=BitMask32.bit(4))
    scene.sync()

    def _bind_vertex_scalar(signal, topology, _adapter):
        if not isinstance(topology, Mesh):
            raise RuntimeError("Vertex scalar binding requires a mesh topology.")
        return PoissonMeshDeformer(topology, signal, scale=DEFORM_SCALE)

    binder = AdapterBinder(eng)
    binder.add_rule(
        BindingRule(
            domain="vertex",
            meaning="scalar",
            factory=_bind_vertex_scalar,
        )
    )
    mesh_map = {obj.name: obj.primitive.mesh for obj in scene_result.objects}
    binder.bind(adapter, topologies=mesh_map)

    eng.start()


if __name__ == "__main__":
    main()
