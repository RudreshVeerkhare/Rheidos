import numpy as np

from rheidos.compute import ModuleBase, ResourceSpec, World


class PointVortexModule(ModuleBase):
    NAME = "PointVortexModule"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.face_ids = self.resource(
            "face_ids",
            spec=ResourceSpec(kind="numpy", dtype=np.int32, allow_none=True),
            doc="Face if of face onto which the point vortex lies",
            declare=True,
        )
        self.bary = self.resource(
            "bary",
            spec=ResourceSpec(kind="numpy", dtype=np.float32, allow_none=True),
            doc="Barycentric co-ordinates of the point inside the triangle",
            declare=True,
        )
        self.gamma = self.resource(
            "gamma",
            spec=ResourceSpec(kind="numpy", dtype=np.float32, allow_none=True),
            doc="Circulation strength of a point vortex",
            declare=True,
        )

        self.pos_world = self.resource(
            "pos_world",
            spec=ResourceSpec(kind="numpy", dtype=np.float32, allow_none=True),
            doc="Position of a point vortex in 3D free space.",
            declare=True,
        )

    def set_vortex(self, faceids, bary, gamma, pos) -> None:
        self.bary.set(np.ascontiguousarray(bary))
        self.face_ids.set(np.ascontiguousarray(faceids))
        self.gamma.set(np.ascontiguousarray(gamma))
        self.pos_world.set(np.ascontiguousarray(pos))
