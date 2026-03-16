from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rheidos.compute import ModuleBase, ProducerBase, ResourceSpec, World

from ..surface_mesh import SurfaceMeshModule


def build_face_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-face affine geometry map for embedded triangles."""
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)

    nF = int(f.shape[0])
    J = np.zeros((nF, 3, 2), dtype=np.float64)
    Ginv = np.zeros((nF, 2, 2), dtype=np.float64)
    sqrt_detG = np.zeros((nF,), dtype=np.float64)

    for fid, (i0, i1, i2) in enumerate(f):
        x0 = v[int(i0)]
        x1 = v[int(i1)]
        x2 = v[int(i2)]

        e1 = x1 - x0
        e2 = x2 - x0
        Jf = np.column_stack((e1, e2))
        G = Jf.T @ Jf
        detG = float(np.linalg.det(G))
        if detG <= 1e-24:
            raise ValueError(f"Degenerate face geometry at face {fid}")

        J[fid] = Jf
        Ginv[fid] = np.linalg.inv(G)
        sqrt_detG[fid] = np.sqrt(detG)

    return J, Ginv, sqrt_detG


@dataclass
class BuildFaceGeometryProducer(ProducerBase):
    V_pos: str
    F_verts: str
    J: str
    Ginv: str
    sqrt_detG: str

    @property
    def outputs(self):
        return (self.J, self.Ginv, self.sqrt_detG)

    def compute(self, reg) -> None:
        J, Ginv, sqrt_detG = build_face_geometry(reg.read(self.V_pos), reg.read(self.F_verts))
        reg.commit(self.J, buffer=J)
        reg.commit(self.Ginv, buffer=Ginv)
        reg.commit(self.sqrt_detG, buffer=sqrt_detG)


class FaceGeometryModule(ModuleBase):
    NAME = "P2FaceGeometry"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = world.require(SurfaceMeshModule, scope=scope)

        self.J = self.resource(
            "J",
            spec=ResourceSpec(kind="numpy", dtype=np.float64, allow_none=True),
        )
        self.Ginv = self.resource(
            "Ginv",
            spec=ResourceSpec(kind="numpy", dtype=np.float64, allow_none=True),
        )
        self.sqrt_detG = self.resource(
            "sqrt_detG",
            spec=ResourceSpec(kind="numpy", dtype=np.float64, allow_none=True),
        )

        prod = BuildFaceGeometryProducer(
            V_pos=self.mesh.V_pos.name,
            F_verts=self.mesh.F_verts.name,
            J=self.J.name,
            Ginv=self.Ginv.name,
            sqrt_detG=self.sqrt_detG.name,
        )

        deps = (self.mesh.V_pos, self.mesh.F_verts)
        self.declare_resource(self.J, deps=deps, producer=prod)
        self.declare_resource(self.Ginv, deps=deps, producer=prod)
        self.declare_resource(self.sqrt_detG, deps=deps, producer=prod)
