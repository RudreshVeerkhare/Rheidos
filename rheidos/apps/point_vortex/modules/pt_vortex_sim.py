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

from .point_vortex import PointVortexModule
from .surface_mesh import SurfaceMeshModule
from .dec_operator import SurfaceDECModule
from .poisson_solver import PoissonSolverModule

import taichi as ti
import numpy as np
from dataclasses import dataclass


@dataclass
class SplatPtVortexProducerIO:
    n_vortices: ResourceRef[ti.Field]  # i32
    gammas: ResourceRef[ti.Field]  # (100, f32)
    face_ids: ResourceRef[ti.Field]  # (100, i32)
    V_pos: ResourceRef[ti.Field]  # (nV, vec3f)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    bary: ResourceRef[ti.Field]  # (100, vec3f)
    omega: ResourceRef[ti.Field] = out_field()  # (nV, f32)


@ti.data_oriented
class SplatPtVortexProducer(WiredProducer[SplatPtVortexProducerIO]):

    def __init__(
        self,
        n_vortices: ResourceRef[ti.Field],
        gammas: ResourceRef[ti.Field],
        face_ids: ResourceRef[ti.Field],
        V_pos: ResourceRef[ti.Field],
        F_verts: ResourceRef[ti.Field],
        bary: ResourceRef[ti.Field],
        omega: ResourceRef[ti.Field],
    ) -> None:
        super().__init__(
            SplatPtVortexProducerIO(
                n_vortices, gammas, face_ids, V_pos, F_verts, bary, omega
            )
        )

    @ti.kernel
    def _splat_gamma_over_vertices_using_bary(
        self,
        n_vortices: ti.template(),  # scalar field, use [None]
        gammas: ti.template(),
        face_ids: ti.template(),
        F_verts: ti.template(),
        bary: ti.template(),
        omega: ti.template(),
    ):
        # Clear omega
        for v in omega:
            omega[v] = 0.0

        for k in range(n_vortices[None]):
            fid = face_ids[k]
            gamma = gammas[k]
            bc = bary[k]  # vec3f
            fv = F_verts[fid]  # vec3i

            ti.atomic_add(omega[fv[0]], gamma * bc[0])
            ti.atomic_add(omega[fv[1]], gamma * bc[1])
            ti.atomic_add(omega[fv[2]], gamma * bc[2])

    # TODO: Hide this repeated broiler plate for check and ensuring buffers existence and shape
    # We already have `out_field` so anyway we only create and update the out_field which is easy to
    # encapsulate
    def compute(self, reg: Registry) -> None:
        io = self.io
        V_pos = io.V_pos.get()
        F_verts = io.F_verts.get()
        N = io.n_vortices.get()
        gammas = io.gammas.get()
        face_ids = io.face_ids.get()
        bary = io.bary.get()

        if (
            (V_pos is None)
            or (F_verts is None)
            or (N is None)
            or (gammas is None)
            or (face_ids is None)
            or (bary is None)
        ):
            raise RuntimeError(
                "SplatPtVortexProducer is missing one or more of V_pos/F_verts/n_vortices/gammas/face_ids/bary"
            )
        nV = V_pos.shape[0]
        omega = io.omega.peek()
        if omega is None or omega.shape != (nV,):
            omega = ti.field(dtype=ti.f32, shape=(nV,))
            io.omega.set_buffer(omega, bump=False)

        self._splat_gamma_over_vertices_using_bary(
            N, gammas, face_ids, F_verts, bary, omega
        )

        io.omega.commit()


@dataclass
class StreamFuncProducerIO:
    omega: ResourceRef[ti.Field]  # (nV, f32)
    n_vortices: ResourceRef[ti.Field]  # (i32)
    vortices_face_ids: ResourceRef[ti.Field]  # (100, i32)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    int_mask: ResourceRef[ti.Field]  # (nV, i32)
    values: ResourceRef[ti.Field]  # (nV, f32)
    u: ResourceRef[ti.Field]  # (nV, f32)
    psi: ResourceRef[ti.Field] = out_field()  # (nV, f32)


@ti.data_oriented
class StreamFuncProducer(WiredProducer[StreamFuncProducerIO]):
    def __init__(
        self,
        omega: ResourceRef[ti.Field],
        n_vortices: ResourceRef[ti.Field],
        vortices_face_ids: ResourceRef[ti.Field],
        F_verts: ResourceRef[ti.Field],
        int_mask: ResourceRef[ti.Field],
        values: ResourceRef[ti.Field],
        u: ResourceRef[ti.Field],
        psi: ResourceRef[ti.Field],
        pin_vertex_id: int = 0,
    ) -> None:
        super().__init__(
            StreamFuncProducerIO(
                omega, n_vortices, vortices_face_ids, F_verts, int_mask, values, u, psi
            )
        )
        self.pin_vertex_id = pin_vertex_id

    @ti.kernel
    def _set_mask(
        self,
        int_mask: ti.template(),
        n_vortices: ti.template(),
        vortices_face_ids: ti.template(),
        F: ti.template(),
    ):
        for vid in range(n_vortices[None]):
            face_id = vortices_face_ids[vid]
            int_mask[F[face_id][0]] = 1
            int_mask[F[face_id][1]] = 1
            int_mask[F[face_id][2]] = 1

    def compute(
        self, reg: Registry
    ) -> None:  # TODO: Remove `reg` and out io as parameter
        io = self.io
        omega = io.omega.get()
        n_vortices = io.n_vortices.get()
        vortices_face_ids = io.vortices_face_ids.get()
        F = io.F_verts.get()

        if (
            omega is None
            or n_vortices is None
            or vortices_face_ids is None
            or F is None
        ):
            raise RuntimeError(
                "StreamFuncProducer is missing one or more of omega/n_vortices/vortices_face_ids/F_verts"
            )

        psi = io.psi.peek()
        nV = omega.shape[0]
        if psi is None or psi.shape != (nV,):
            psi = ti.field(ti.f32, shape=(nV,))
            io.psi.set_buffer(psi, bump=False)

        int_mask = io.int_mask.peek()
        if int_mask is None or int_mask.shape != (nV,):
            int_mask = ti.field(ti.i32, shape=(nV,))
            io.int_mask.set_buffer(int_mask, bump=False)

        values = io.values.peek()
        if values is None or values.shape != (nV,):
            values = ti.field(ti.f32, shape=(nV,))
            io.values.set_buffer(values, bump=False)

        int_mask.fill(0)
        self._set_mask(int_mask, n_vortices, vortices_face_ids, F)
        int_mask[self.pin_vertex_id] = (
            1  # Dirichlet pin one vertex to remove null space
        )

        # trigger poisson solve
        values.copy_from(omega)
        u = io.u.get()  # triggers poisson solve

        psi.copy_from(u)
        io.psi.commit()


class PtVortexSimModule(ModuleBase):
    NAME = "PtVortexSimModule"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = world.require(SurfaceMeshModule)
        self.dec = world.require(SurfaceDECModule)
        self.pt_vortex = world.require(PointVortexModule)
        self.poisson = world.require(PoissonSolverModule)

        ## Dual vorticity 2-form
        self.omega = self.resource(
            "omega",
            spec=ResourceSpec(
                kind="taichi_field", dtype=ti.f32, shape_fn=shape_of(self.mesh.V_pos)
            ),
            doc="Per vertex dual vorticity 2-form. Shape: (nV, )",
            declare=False,
        )

        pt_vortex_splat_producer = SplatPtVortexProducer(
            self.pt_vortex.n_vortices,
            self.pt_vortex.gammas,
            self.pt_vortex.face_ids,
            self.mesh.V_pos,
            self.mesh.F_verts,
            self.pt_vortex.bary,
            self.omega,
        )

        self.declare_resource(
            self.omega,
            deps=(
                self.mesh.V_pos,
                self.mesh.F_verts,
                self.pt_vortex.n_vortices,  # TODO: Create helper to get all fields as dependency for this. Use regex, and negative mask selection
                self.pt_vortex.gammas,
                self.pt_vortex.face_ids,
                self.pt_vortex.bary,
            ),
            producer=pt_vortex_splat_producer,
        )

        # Stream function
        self.psi = self.resource(
            "psi",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape_fn=shape_of(self.mesh.V_pos),
                allow_none=True,
            ),
            doc="0-form stream function for fluids. Shape: (nV, f32)",
            declare=False,
        )

        stream_func_producer = StreamFuncProducer(
            self.omega,
            self.pt_vortex.n_vortices,
            self.pt_vortex.face_ids,
            self.mesh.F_verts,
            self.poisson.constraint_mask,
            self.poisson.constraint_value,
            self.poisson.u,
            self.psi,
            pin_vertex_id=0,
        )

        self.declare_resource(
            self.psi, deps=(self.omega,), producer=stream_func_producer
        )
