from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rheidos.compute import ModuleBase, ProducerBase, ResourceSpec, World

from ..surface_mesh import SurfaceMeshModule


def vortex_positions_from_face_bary(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: np.ndarray,
    bary: np.ndarray,
) -> np.ndarray:
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    face_ids = np.ascontiguousarray(face_ids, dtype=np.int32)
    bary = np.ascontiguousarray(bary, dtype=np.float64)

    n = int(face_ids.shape[0])
    out = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        fid = int(face_ids[i])
        if fid < 0 or fid >= f.shape[0]:
            raise ValueError(f"Invalid face id {fid} for vortex index {i}")
        tri = f[fid]
        b = bary[i]
        out[i] = b[0] * v[tri[0]] + b[1] * v[tri[1]] + b[2] * v[tri[2]]
    return out


@dataclass
class VortexWorldPosProducer(ProducerBase):
    V_pos: str
    F_verts: str
    face_ids: str
    bary: str
    pos_world: str

    @property
    def outputs(self):
        return (self.pos_world,)

    def compute(self, reg) -> None:
        pos = vortex_positions_from_face_bary(
            reg.read(self.V_pos),
            reg.read(self.F_verts),
            reg.read(self.face_ids),
            reg.read(self.bary),
        )
        reg.commit(self.pos_world, buffer=pos)


class PointVortexModule(ModuleBase):
    NAME = "P2PointVortex"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = world.require(SurfaceMeshModule, scope=scope)

        self.face_ids = self.resource(
            "face_ids",
            declare=True,
            spec=ResourceSpec(kind="numpy", dtype=np.int32, allow_none=True),
        )
        self.bary = self.resource(
            "bary",
            declare=True,
            spec=ResourceSpec(kind="numpy", dtype=np.float64, allow_none=True),
        )
        self.gamma = self.resource(
            "gamma",
            declare=True,
            spec=ResourceSpec(kind="numpy", dtype=np.float64, allow_none=True),
        )

        self.pos_world = self.resource(
            "pos_world",
            spec=ResourceSpec(kind="numpy", dtype=np.float64, allow_none=True),
        )

        prod = VortexWorldPosProducer(
            V_pos=self.mesh.V_pos.name,
            F_verts=self.mesh.F_verts.name,
            face_ids=self.face_ids.name,
            bary=self.bary.name,
            pos_world=self.pos_world.name,
        )
        self.declare_resource(
            self.pos_world,
            deps=(self.mesh.V_pos, self.mesh.F_verts, self.face_ids, self.bary),
            producer=prod,
        )

    def set_state(
        self,
        face_ids: np.ndarray,
        bary: np.ndarray,
        gamma: np.ndarray,
    ) -> None:
        face_ids_np = np.ascontiguousarray(face_ids, dtype=np.int32)
        bary_np = np.ascontiguousarray(bary, dtype=np.float64)
        gamma_np = np.ascontiguousarray(gamma, dtype=np.float64)

        if face_ids_np.ndim != 1:
            raise ValueError(f"face_ids must be (N,), got {face_ids_np.shape}")
        if bary_np.ndim != 2 or bary_np.shape[1] != 3:
            raise ValueError(f"bary must be (N,3), got {bary_np.shape}")
        if gamma_np.ndim != 1:
            raise ValueError(f"gamma must be (N,), got {gamma_np.shape}")

        n = int(face_ids_np.shape[0])
        if bary_np.shape[0] != n or gamma_np.shape[0] != n:
            raise ValueError(
                "face_ids, bary, gamma must have equal first dimension: "
                f"got {face_ids_np.shape[0]}, {bary_np.shape[0]}, {gamma_np.shape[0]}"
            )

        self.face_ids.set(face_ids_np)
        self.bary.set(bary_np)
        self.gamma.set(gamma_np)
