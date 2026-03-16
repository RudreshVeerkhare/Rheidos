import numpy as np

from rheidos.apps.p2.modules.point_vortex import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh import SurfaceMeshModule
from rheidos.compute import ModuleBase, ResourceSpec, World, shape_of


class P1StreamFunction(ModuleBase):
    NAME = "P1StreamFunction"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)
        self.point_vortex = self.require(PointVortexModule)

        self.psi = self.resource(
            "psi",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float32,
                allow_none=True,
                shape_fn=shape_of(self.mesh.V_pos),
            ),
            declare=True,
            doc="Stream function coefficient to be paired with respective hat basis function. Shape (nV, )",
        )

        self.omega = self.resource(
            "omega",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float32,
                allow_none=True,
                shape_fn=shape_of(self.mesh.V_pos),
            ),
            declare=True,
            doc="Vorticity field coefficient to be paired with basis function. Shape: (nV, )",
        )
