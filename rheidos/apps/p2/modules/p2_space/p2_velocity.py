import numpy as np

from rheidos.apps.p2.modules.p2_space.p2_stream_function import P2StreamFunction
from rheidos.apps.p2.modules.p2_space.p2_elements import P2Elements
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule

from rheidos.compute import ModuleBase, ProducerContext, producer, World

from .probe_utils import probe_arrays


class P2VelocityField(ModuleBase):
    NAME = "P2VelocityField"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)
        self.p2_space = self.require(P2Elements)
        self.stream = self.require(P2StreamFunction)

    def interpolate(self, probes):
        """Interpolates the value of `psi` using P2 lagrange basis

        Args:
           probes (np.ndarray): [[faceid, [b1, b2, b3]], ...]
        """

        grad_bary = self.mesh.grad_bary.get()
        face_normals = self.mesh.F_normal.get()
        psi = self.stream.psi.get()
        face_dofs = self.p2_space.face_dof.get()

        velocities = []

        for faceid, (b1, b2, b3) in probes:
            normal = face_normals[faceid]
            j_grad = np.cross(normal, grad_bary[faceid])
            face_dof = face_dofs[faceid]

            c1, c2, c3, c4, c5, c6 = psi[face_dof]

            coef_mat = 4 * np.array([[c1, c4, c6], [c4, c2, c5], [c6, c5, c3]])
            l = np.array([b1, b2, b3]).T
            c = np.array([c1, c2, c3]).T

            # grad_bary stores [∇λ1, ∇λ2, ∇λ3] as rows
            velocities.append(j_grad.T @ (coef_mat @ l - c))

        return np.array(velocities)
