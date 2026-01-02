from rheidos.compute import (
    ModuleBase,
    ResourceSpec,
    World,
    WiredProducer,
    ResourceRef,
    out_field,
)
from rheidos.compute.registry import Registry

from .surface_mesh import SurfaceMeshModule

import taichi as ti

from dataclasses import dataclass
from typing import Tuple


@dataclass
class VortexWorldPositionProducerIO:
    V_pos: ResourceRef[ti.Field]  # (nV, vec3f)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    n_vortices: ResourceRef[ti.Field]  # ()
    face_ids: ResourceRef[ti.Field]  # (100, )
    bary: ResourceRef[ti.Field]  # (100, vec3f)
    pos_world: ResourceRef[ti.Field] = out_field()  # (100, vec3f)


@ti.data_oriented
class VortexWorldPositionProducer(WiredProducer[VortexWorldPositionProducerIO]):

    def __init__(
        self,
        V_pos: ResourceRef[ti.Field],
        F_verts: ResourceRef[ti.Field],
        n_vortices: ResourceRef[ti.Field],
        face_ids: ResourceRef[ti.Field],
        bary: ResourceRef[ti.Field],
        pos_world: ResourceRef[ti.Field],
    ) -> None:
        super().__init__(
            VortexWorldPositionProducerIO(
                V_pos, F_verts, n_vortices, face_ids, bary, pos_world
            )
        )

    @ti.kernel
    def _calculate_world_position(
        self,
        V: ti.template(),
        F: ti.template(),
        N: ti.i32,
        face_ids: ti.template(),
        bary: ti.template(),
        pos_world: ti.template(),
    ):
        for vortex_id in range(N[None]):
            fid = face_ids[vortex_id]
            x1 = V[F[fid][0]]
            x2 = V[F[fid][1]]
            x3 = V[F[fid][2]]

            pos_world[vortex_id] = (
                x1 * bary[vortex_id][0]
                + x2 * bary[vortex_id][1]
                + x3 * bary[vortex_id][2]
            )

    def compute(self, reg: Registry) -> None:
        V = self.io.V_pos.peek()
        F = self.io.F_verts.peek()
        n = self.io.n_vortices.peek()
        face_ids = self.io.face_ids.peek()
        bary = self.io.bary.peek()

        if (
            (V is None)
            or (F is None)
            or (n is None)
            or (face_ids is None)
            or (bary is None)
        ):
            raise RuntimeError(
                "VortexWorldPositionProducer is missing one or more of V_pos/F_verts/n_vortices/face_ids/bary"
            )

        pos_world = self.io.pos_world.peek()

        self._calculate_world_position(V, F, n, face_ids, bary, pos_world)

        self.io.pos_world.commit()


class PointVortexModule(ModuleBase):

    NAME = "PointVortexModule"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)
        print("Here here here...")

        self.n_vortices = self.resource(
            "n_vortices",
            spec=ResourceSpec(
                kind="taichi_field", dtype=ti.i32, shape=(), allow_none=True
            ),
            doc="Count of total point vortices on surface",
            declare=True,
        )

        self.gammas = self.resource(
            "gammas",
            spec=ResourceSpec(
                kind="taichi_field", dtype=ti.f32, shape=(100,), allow_none=False
            ),
            doc="Strengths of point vortices. Max capacity 100 elements.",
            buffer=ti.field(dtype=ti.f32, shape=(100,)),
            declare=True,
        )

        # on-surface coordinates
        self.face_ids = self.resource(
            "face_ids",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.i32,
                shape=(100,),
                allow_none=False,
            ),
            buffer=ti.field(ti.i32, shape=(100,)),
            declare=True,
        )

        self.bary = self.resource(
            "bary",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape=(100,),
                lanes=3,
                allow_none=False,
            ),
            doc="Barycentric coordinates of point vortex inside a face. Shape: (100, vec3f)",
            buffer=ti.Vector.field(3, dtype=ti.f32, shape=(100,)),
            declare=True,
        )

        # TODO: Add derived resources `pos_world` and `vel_world` whose producers depends
        # on simulation end state
        # For this maybe I'll need to implement more fine grained resource level dependency structure

        self.pos_world = self.resource(
            "pos_world",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape=(100,),
                lanes=3,
                allow_none=False,
            ),
            doc="3D world coordinates of points",
            buffer=ti.Vector.field(3, dtype=ti.f32, shape=(100,)),
            declare=False,
        )

        vortex_world_pos_producer = VortexWorldPositionProducer(
            self.mesh.V_pos,
            self.mesh.F_verts,
            self.n_vortices,
            self.face_ids,
            self.bary,
            self.pos_world,
        )

        self.declare_resource(
            self.pos_world,
            deps=(
                self.mesh.V_pos,
                self.mesh.F_verts,
                self.n_vortices,
                self.face_ids,
                self.bary,
            ),
            producer=vortex_world_pos_producer,
        )

    def add_vortex(self, face_id: int, bary: Tuple[float, float, float]):
        N = self.n_vortices.get()
        face_ids = self.face_ids.get()
        barys = self.bary.get()

        vid = N[None]
        face_ids[vid] = face_id
        barys[vid] = bary
        N[None] = vid + 1

        self.n_vortices.bump()
        self.face_ids.bump()
        self.bary.bump()
