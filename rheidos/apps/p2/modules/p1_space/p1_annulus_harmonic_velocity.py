from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_annulus_harmoic_stream_function import (
    P1AnnulusHarmonicStreamFunction,
)
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ResourceSpec, World, ModuleBase, shape_map
import numpy as np

from rheidos.compute.wiring import ProducerContext, producer

from .probe_utils import probe_arrays


class P1AnnulusHarmonicVelocityFieldModule(ModuleBase):
    NAME = "P1AnnulusHarmonicVelocityFieldModule"

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        dec: DEC,
        stream: P1AnnulusHarmonicStreamFunction,
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
        inputs=("vel_per_face", "mesh.F_area", "mesh.F_verts"),
        outputs=("vel_per_vertex",),
    )
    def per_vertex_vel_calculate(self, ctx: ProducerContext):

        ctx.require_inputs()

        f_vel = self.vel_per_face.get()
        f_area = self.mesh.F_area.get()
        f_verts = self.mesh.F_verts.get()

        ctx.ensure_outputs()
        v_vel = self.vel_per_vertex.peek()

        face_contrib = f_vel / 3.0
        flat_verts = f_verts.reshape(-1)
        flat_area = np.repeat(f_area, 3)
        weighted_face_contrib = face_contrib * f_area[:, None]
        vertex_area = np.bincount(
            flat_verts,
            weights=flat_area,
            minlength=v_vel.shape[0],
        )

        # Area-weighted mean of per-face contributions at each vertex.
        v_vel[:, 0] = np.bincount(
            flat_verts,
            weights=np.repeat(weighted_face_contrib[:, 0], 3),
            minlength=v_vel.shape[0],
        )
        v_vel[:, 1] = np.bincount(
            flat_verts,
            weights=np.repeat(weighted_face_contrib[:, 1], 3),
            minlength=v_vel.shape[0],
        )
        v_vel[:, 2] = np.bincount(
            flat_verts,
            weights=np.repeat(weighted_face_contrib[:, 2], 3),
            minlength=v_vel.shape[0],
        )
        np.divide(
            v_vel, vertex_area[:, None], out=v_vel, where=vertex_area[:, None] > 0
        )

        ctx.commit(vel_per_vertex=v_vel)

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

    def interpolate(self, probes, smooth=False):
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
