import numpy as np

from rheidos.apps.p2.modules.p2_space.p2_stream_function import P2StreamFunction
from rheidos.apps.p2.modules.p2_space.p2_elements import P2Elements
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule

from rheidos.compute import ModuleBase, World

from .probe_utils import probe_arrays


class P2VelocityField(ModuleBase):
    NAME = "P2VelocityField"

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        p2_space: P2Elements,
        stream: P2StreamFunction,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)

        self.mesh = mesh
        self.p2_space = p2_space
        self.stream = stream

    def interpolate(self, probes):
        """Interpolates the P2 velocity field at the probe locations.

        Args:
           probes (np.ndarray): [[faceid, [b1, b2, b3]], ...]
        """
        faceids, bary = probe_arrays(probes)
        if faceids.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        coeffs = self.stream.psi.get()[self.p2_space.face_dof.get()[faceids]]
        # grad_bary stores [∇λ1, ∇λ2, ∇λ3] as rows.
        j_grad = np.cross(
            self.mesh.F_normal.get()[faceids, None, :],
            self.mesh.grad_bary.get()[faceids],
        )

        b1, b2, b3 = bary.T
        c1, c2, c3, c4, c5, c6 = coeffs.T

        a1 = (4.0 * b1 - 1.0) * c1 + 4.0 * b2 * c4 + 4.0 * b3 * c6
        a2 = 4.0 * b1 * c4 + (4.0 * b2 - 1.0) * c2 + 4.0 * b3 * c5
        a3 = 4.0 * b1 * c6 + 4.0 * b2 * c5 + (4.0 * b3 - 1.0) * c3

        return (
            a1[:, None] * j_grad[:, 0, :]
            + a2[:, None] * j_grad[:, 1, :]
            + a3[:, None] * j_grad[:, 2, :]
        )
