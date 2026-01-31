from rheidos.compute import (
    ModuleBase,
    World,
    ResourceSpec,
    shape_of,
    WiredProducer,
    ResourceRef,
    out_field,
)
from rheidos.compute.registry import Registry

from ..point_vortex import PointVortexModule
from ..surface_mesh import SurfaceMeshModule

import taichi as ti
import numpy as np

from collections import deque

from typing import Dict, List
from dataclasses import dataclass


def init_is_comp(reg: Registry, io: "SelfVelBasisProducerIO"):
    f_verts = io.F_verts.peek()
    if f_verts is None:
        return None

    is_comp = ti.field(dtype=ti.int32, shape=f_verts.shape)
    is_comp.fill(-1)
    return is_comp


@dataclass
class SelfVelBasisProducerIO:
    qurey_face_id: ResourceRef[int]
    V_pos: ResourceRef[ti.Field]  # (nV, vec3f)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    V_incident: ResourceRef[Dict[int, List[int]]]

    basis: ResourceRef[ti.Field] = out_field()  # (nV, vec3f)
    is_comp: ResourceRef[ti.Field] = out_field(alloc=init_is_comp)  # (nF, int)


class SelfVelBasisProducer(WiredProducer[SelfVelBasisProducerIO]):

    def bfs_faces(self, root_face_id: int, max_depth=2):

        faces = self.io.F_verts.get().to_numpy()  # (nF, vec3i)
        v_inc = self.io.V_incident.get()

        queue = deque([(root_face_id, 0)])
        visited_faces = set()
        F_patch = []

        while len(queue) > 0:
            face_id, depth = queue.popleft()
            if depth > max_depth or face_id in visited_faces:
                continue

            F_patch.append(faces[face_id])

            for pos_id in faces[face_id]:
                for next_fid in v_inc[pos_id]:
                    if next_fid in visited_faces:
                        continue

                    queue.append((next_fid, depth + 1))

            visited_faces.add(face_id)

        return np.array(F_patch)

    def compute(self, reg: Registry) -> None:

        inputs = self.require_inputs()
        outputs = self.ensure_outputs(reg)

        # step 1: BFS and get faces 2 ring
        is_comp = outputs["is_comp"].get()
        query_face_id = inputs["qurey_face_id"].get()

        if is_comp[query_face_id] != -1:
            # basis already exists
            return

        F_patch = self.bfs_faces(query_face_id)

        # step 2: Reduced Laplacian
        # - boundary/ interior

        # step 3: Rotate grad operator

        # calculate basis
        pass


class SelfVelBasisModule(ModuleBase):
    NAME = "SelfVelBasisModule"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.surface_mesh = world.require(SurfaceMeshModule)
        self.pt_vortex = world.require(PointVortexModule)

        self.query_face_id = self.resource(
            "query_face_id",
            spec=ResourceSpec(kind="python", dtype=int, allow_none=True),
            doc="The query face for which the producer will run computation.",
        )

        self.vel_basis = self.resource(
            "vel_basis",
            spec=ResourceSpec(
                kind="taichi_kind",
                dtype=ti.f32,
                shape_fn=shape_of(self.surface_mesh.V_pos),
                lanes=3,
                allow_none=True,
            ),
            doc="Basis for local self induced velocity by a point vortex inside a triangle. Shape: (nV, vec3f)",
        )

        self.is_computed = self.resource(
            "is_computed",
            spec=ResourceSpec(
                kind="taichi_kind",
                dtype=ti.i32,
                shape_fn=shape_of(self.surface_mesh.F_verts),
                allow_none=True,
            ),
            doc="For lazy computation, keep track of triangles for which the basis are already computed. Shape: (nF, int32)",
        )
