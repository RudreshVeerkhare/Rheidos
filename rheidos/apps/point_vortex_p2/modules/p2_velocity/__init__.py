from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rheidos.compute import ModuleBase, ProducerBase, ResourceSpec, World

from ..fe_utils import CENTROID_BARY, CORNER_BARY, bary_to_ref, p2_shape_and_grad_ref
from ..p2_geometry import FaceGeometryModule
from ..p2_poisson import P2PoissonModule, grad_ref_to_surface
from ..p2_space import P2ScalarSpaceModule
from ..surface_mesh import SurfaceMeshModule


def sample_velocity_from_corners(
    vel_corner: np.ndarray,
    face_id: int,
    bary: np.ndarray,
) -> np.ndarray:
    b = np.asarray(bary, dtype=np.float64)
    return b[0] * vel_corner[face_id, 0] + b[1] * vel_corner[face_id, 1] + b[2] * vel_corner[face_id, 2]


def build_p2_velocity_fields(
    psi: np.ndarray,
    face_to_dofs: np.ndarray,
    J: np.ndarray,
    Ginv: np.ndarray,
    normals: np.ndarray,
    n_vertices: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-face corner and centroid velocities from P2 stream coefficients."""
    psi = np.ascontiguousarray(psi, dtype=np.float64)
    face_to_dofs = np.ascontiguousarray(face_to_dofs, dtype=np.int32)
    J = np.ascontiguousarray(J, dtype=np.float64)
    Ginv = np.ascontiguousarray(Ginv, dtype=np.float64)
    normals = np.ascontiguousarray(normals, dtype=np.float64)

    nF = int(face_to_dofs.shape[0])
    vel_corner = np.zeros((nF, 3, 3), dtype=np.float64)
    vel_face = np.zeros((nF, 3), dtype=np.float64)

    for fid in range(nF):
        dofs = face_to_dofs[fid]
        local_psi = psi[dofs]

        n_hat = normals[fid]
        nn = float(np.linalg.norm(n_hat))
        if nn <= 1e-20:
            n_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            n_hat = n_hat / nn

        Jf = J[fid]
        Ginvf = Ginv[fid]

        for corner, bc in enumerate(CORNER_BARY):
            xi, eta = bary_to_ref(bc)
            _, dphi_ref = p2_shape_and_grad_ref(xi, eta)
            dphi = grad_ref_to_surface(Jf, Ginvf, dphi_ref)
            grad_psi = np.sum(local_psi[:, None] * dphi, axis=0)
            vel_corner[fid, corner] = np.cross(n_hat, grad_psi)

        xi, eta = bary_to_ref(CENTROID_BARY)
        _, dphi_ref = p2_shape_and_grad_ref(xi, eta)
        dphi = grad_ref_to_surface(Jf, Ginvf, dphi_ref)
        grad_psi = np.sum(local_psi[:, None] * dphi, axis=0)
        vel_face[fid] = np.cross(n_hat, grad_psi)

    stream_vertex = psi[:n_vertices].copy()
    return vel_corner, vel_face, stream_vertex


@dataclass
class BuildP2VelocityProducer(ProducerBase):
    psi: str
    face_to_dofs: str
    J: str
    Ginv: str
    F_normal: str
    V_pos: str

    vel_corner: str
    vel_face: str
    stream_vertex: str

    @property
    def outputs(self):
        return (self.vel_corner, self.vel_face, self.stream_vertex)

    def compute(self, reg) -> None:
        psi = np.asarray(reg.read(self.psi), dtype=np.float64)
        face_to_dofs = np.asarray(reg.read(self.face_to_dofs), dtype=np.int32)
        J = np.asarray(reg.read(self.J), dtype=np.float64)
        Ginv = np.asarray(reg.read(self.Ginv), dtype=np.float64)
        F_normal = np.asarray(reg.read(self.F_normal), dtype=np.float64)
        V_pos = np.asarray(reg.read(self.V_pos), dtype=np.float64)

        vel_corner, vel_face, stream_vertex = build_p2_velocity_fields(
            psi,
            face_to_dofs,
            J,
            Ginv,
            F_normal,
            n_vertices=int(V_pos.shape[0]),
        )

        reg.commit(self.vel_corner, buffer=vel_corner)
        reg.commit(self.vel_face, buffer=vel_face)
        reg.commit(self.stream_vertex, buffer=stream_vertex)


class P2VelocityModule(ModuleBase):
    NAME = "P2Velocity"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = world.require(SurfaceMeshModule, scope=scope)
        self.space = world.require(P2ScalarSpaceModule, scope=scope)
        self.geom = world.require(FaceGeometryModule, scope=scope)
        self.poisson = world.require(P2PoissonModule, scope=scope)

        self.vel_corner = self.resource(
            "vel_corner",
            spec=ResourceSpec(kind="numpy", dtype=np.float64, allow_none=True),
        )
        self.vel_face = self.resource(
            "vel_face",
            spec=ResourceSpec(kind="numpy", dtype=np.float64, allow_none=True),
        )
        self.stream_vertex = self.resource(
            "stream_vertex",
            spec=ResourceSpec(kind="numpy", dtype=np.float64, allow_none=True),
        )

        prod = BuildP2VelocityProducer(
            psi=self.poisson.psi.name,
            face_to_dofs=self.space.face_to_dofs.name,
            J=self.geom.J.name,
            Ginv=self.geom.Ginv.name,
            F_normal=self.mesh.F_normal.name,
            V_pos=self.mesh.V_pos.name,
            vel_corner=self.vel_corner.name,
            vel_face=self.vel_face.name,
            stream_vertex=self.stream_vertex.name,
        )

        deps = (
            self.poisson.psi,
            self.space.face_to_dofs,
            self.geom.J,
            self.geom.Ginv,
            self.mesh.F_normal,
            self.mesh.V_pos,
        )
        self.declare_resource(self.vel_corner, deps=deps, producer=prod)
        self.declare_resource(self.vel_face, deps=deps, producer=prod)
        self.declare_resource(self.stream_vertex, deps=deps, producer=prod)
