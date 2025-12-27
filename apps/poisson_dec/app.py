from pathlib import Path

from rheidos.engine import Engine
from rheidos.scene_config import load_scene_from_config

from rheidos.controllers.point_selector import SceneVertexPointSelector
from rheidos.views import PointSelectionView
from panda3d.core import BitMask32


from rheidos.compute.world import World
from compute.mesh import MeshModule
from compute.dec import DECModule
from compute.poisson import PoissonSolverModule

import numpy as np
import taichi as ti
from typing import Optional, Sequence, Tuple

def load_mesh(path: str | Path, name: Optional[str] = None, center: bool = True) -> tuple:
    try:
        import trimesh  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("trimesh is not available. Install 'trimesh'.") from exc

    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type from path '{path}'")

    mesh = mesh.copy()
    if mesh.faces is None or mesh.faces.size == 0:
        raise ValueError("Loaded mesh has no faces")
    if mesh.faces.shape[1] != 3:
        mesh = mesh.triangulate()

    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()

    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)
    normals = mesh.vertex_normals.astype(np.float32)

    colors = None
    if mesh.visual.kind == "vertex" and mesh.visual.vertex_colors is not None:
        vc = np.asarray(mesh.visual.vertex_colors)
        if vc.ndim == 2 and vc.shape[0] == vertices.shape[0]:
            if vc.shape[1] >= 4:
                colors = vc[:, :4]
            elif vc.shape[1] == 3:
                alpha = np.full((vc.shape[0], 1), 255, dtype=vc.dtype)
                colors = np.concatenate([vc, alpha], axis=1)

    if colors is None:
        colors = np.full((vertices.shape[0], 4), [204, 204, 224, 255], dtype=np.uint8)
    else:
        if colors.dtype in (np.float32, np.float64):
            colors = colors.astype(np.float32, copy=False)
            if colors.max() > 1.0:
                colors = (colors / 255.0).astype(np.float32)
        elif colors.dtype != np.uint8:
            colors = colors.astype(np.uint8, copy=False)

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)

    if center:
        offset = (mins + maxs) * 0.5
        vertices = vertices - offset.astype(np.float32)
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
    
    return vertices, faces, normals


from rheidos.compute.world import ModuleBase
from rheidos.compute.resource import ResourceSpec, ResourceRef, ShapeFn, Shape
from rheidos.compute.registry import Registry
from typing import Any


# TODO: Impl DEC Poisson Solve

class PoissonApp(ModuleBase):
    NAME = "PoissonApp"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(MeshModule)
        self.dec = self.require(DECModule)
        self.poisson = self.require(PoissonSolverModule)
        
    def set_charges(self, charges: Sequence[Tuple[int, int]]) -> None:
        np_charges = np.zeros((self.mesh.V_pos.get().shape[0]))

        for idx, charge in charges:
            np_charges[idx] = charge

        self.poisson.constraint_value.get().from_numpy(np_charges.astype(np.float32))
        self.poisson.constraint_mask.get().from_numpy(np.abs(np_charges).astype(np.int32))

        self.poisson.constraint_value.bump()
        self.poisson.constraint_mask.bump()
        

def main() -> None:
    cfg_path = Path(__file__).resolve().parent / "scene_configs" / "poisson.yaml"
    eng = Engine(window_title="Poisson Mesh Demo", interactive=False)
    scene = load_scene_from_config(eng, cfg_path)
    ti.init()
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


    # Compute
    world = World()

    # Renderer side mesh is already defined
    app = world.require(PoissonApp)

    V_np, F_np, normals = load_mesh("/Users/codebox/dev/kung_fu_panda/models/flat_mesh.obj")
    nV, nF = V_np.shape[0], F_np.shape[0]

    V = ti.Vector.field(3, dtype=ti.f32, shape=(nV,))
    F = ti.Vector.field(3, dtype=ti.i32, shape=(nF,))
    V.from_numpy(V_np)
    F.from_numpy(F_np)

    app.mesh.V_pos.set(V)
    app.mesh.F_verts.set(F)

    
    mask = ti.field(dtype=ti.i32, shape=(nV,))
    val = ti.field(dtype=ti.f32, shape=(nV,))
    mask.fill(0.0)
    val.fill(0)

    app.poisson.constraint_mask.set(mask)
    app.poisson.constraint_value.set(val)
    
    def update_charges(_):
        charges = []
        pos_charges = eng.store.get("poisson/pos_points")
        neg_charges = eng.store.get("poisson/neg_points")
        for point in pos_charges:
            charges.append((point['index'], 1.0))
        for point in neg_charges:
            charges.append((point['index'], -1.0))
        print(charges)
        app.set_charges(charges)
        print(app.poisson.u.get()) # Triggers 

    eng.store.subscribe("poisson/pos_points", update_charges)
    eng.store.subscribe("poisson/neg_points", update_charges)


    eng.start()

# TODO: Add a automated sync-mechanism to bing panda3D resources to compute ones
if __name__ == "__main__":
    main()
