from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_annulus_harmoic_stream_function import (
    P1AnnulusHarmonicStreamFunction,
)
from rheidos.apps.p2.modules.p1_space.p1_stream_function import P1StreamFunction
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ResourceSpec, World, ModuleBase, shape_map
import numpy as np

from rheidos.compute.wiring import ProducerContext, producer

from .probe_utils import probe_arrays


def area_weighted_face_vectors_to_vertices(
    vel_per_face: np.ndarray,
    face_area: np.ndarray,
    face_verts: np.ndarray,
    n_vertices: int,
) -> np.ndarray:
    """Average facewise tangent vectors to vertices using incident face areas."""
    vel_per_face = np.asarray(vel_per_face, dtype=np.float64)
    face_area = np.asarray(face_area, dtype=np.float64)
    face_verts = np.asarray(face_verts)
    n_vertices = int(n_vertices)

    if vel_per_face.ndim != 2 or vel_per_face.shape[1] != 3:
        raise ValueError(f"vel_per_face must have shape (nF,3), got {vel_per_face.shape}")
    if face_area.shape != (vel_per_face.shape[0],):
        raise ValueError(
            "face_area must have shape "
            f"({vel_per_face.shape[0]},), got {face_area.shape}"
        )
    if face_verts.shape != (vel_per_face.shape[0], 3):
        raise ValueError(
            "face_verts must have shape "
            f"({vel_per_face.shape[0]},3), got {face_verts.shape}"
        )
    if n_vertices < 0:
        raise ValueError("n_vertices must be non-negative")

    vertex_velocity = np.zeros((n_vertices, 3), dtype=np.float64)
    if vel_per_face.shape[0] == 0 or n_vertices == 0:
        return vertex_velocity

    flat_verts = face_verts.reshape(-1)
    flat_area = np.repeat(face_area, 3)
    weighted_face_velocity = vel_per_face * face_area[:, None]
    vertex_area = np.bincount(
        flat_verts,
        weights=flat_area,
        minlength=n_vertices,
    )

    for axis in range(3):
        vertex_velocity[:, axis] = np.bincount(
            flat_verts,
            weights=np.repeat(weighted_face_velocity[:, axis], 3),
            minlength=n_vertices,
        )

    np.divide(
        vertex_velocity,
        vertex_area[:, None],
        out=vertex_velocity,
        where=vertex_area[:, None] > 0,
    )
    return vertex_velocity


class P1VelocityFieldModule(ModuleBase):
    NAME = "P1VelocityFieldModule"

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        dec: DEC,
        stream: P1StreamFunction | P1AnnulusHarmonicStreamFunction,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)

        self.mesh = mesh
        self.dec = dec
        self.stream = stream

        self.vel_per_face = self.resource(
            "vel_per_face",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.mesh.F_verts, lambda s: (s[0], 3)),
            ),
            doc="Facewise constant velocity as gradient of P1 basis stream function. Shape: (nF, 3)",
        )

        # Area-weighted mean of incident face contributions at each vertex.
        self.vel_per_vertex = self.resource(
            "vel_per_vertex",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.mesh.V_pos, lambda s: (s[0], 3)),
            ),
            doc="Per-vertex area-weighted mean of incident-face velocity contributions in R^3. Shape: (nV, 3)",
        )

        self.bind_producers()

    @producer(
        inputs=("vel_per_face", "mesh.F_area", "mesh.F_verts", "mesh.V_pos"),
        outputs=("vel_per_vertex",),
    )
    def per_vertex_vel_calculate(self, ctx: ProducerContext):

        ctx.require_inputs()

        f_vel = self.vel_per_face.get()
        f_area = self.mesh.F_area.get()
        f_verts = self.mesh.F_verts.get()

        ctx.commit(
            vel_per_vertex=area_weighted_face_vectors_to_vertices(
                f_vel,
                f_area,
                f_verts,
                self.mesh.V_pos.get().shape[0],
            )
        )

    @producer(
        inputs=("mesh.F_verts", "mesh.F_normal", "mesh.grad_bary", "stream.psi"),
        outputs=("vel_per_face",),
    )
    def per_face_vel_calculate(self, ctx: ProducerContext):
        ctx.require_inputs()

        coeffs = self.stream.psi.get()[self.mesh.F_verts.get()]
        # grad_bary stores [∇λ1, ∇λ2, ∇λ3] as rows.
        j_grad = np.cross(
            self.mesh.F_normal.get()[:, None, :],
            self.mesh.grad_bary.get(),
        )

        p1, p2, p3 = coeffs.T

        ctx.commit(
            vel_per_face=(
                p1[:, None] * j_grad[:, 0, :]
                + p2[:, None] * j_grad[:, 1, :]
                + p3[:, None] * j_grad[:, 2, :]
            )
        )

    def interpolate(self, probes, smooth=True):
        """Calculates and return facewise constant velocity from P1 basis

        Args:
           probes (np.ndarray): [[faceid, [b1, b2, b3]], ...]
        """
        faceids, bary = probe_arrays(probes)
        if faceids.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        if not smooth:
            return self.vel_per_face.get()[faceids]

        verts = self.mesh.F_verts.get()[faceids]
        vel_verts = self.vel_per_vertex.get()[verts]

        b1, b2, b3 = map(lambda x: x.reshape(-1, 1), bary.T)
        v1, v2, v3 = vel_verts[:, 0, :], vel_verts[:, 1, :], vel_verts[:, 2, :]

        return b1 * v1 + b2 * v2 + b3 * v3
