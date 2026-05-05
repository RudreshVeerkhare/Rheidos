import numpy as np

from rheidos.apps.p2.modules.higher_genus.dual_harmonic_field import (
    DualHarmonicFieldModule,
)
from rheidos.apps.p2.modules.p1_space.p1_velocity import (
    area_weighted_face_vectors_to_vertices,
)
from rheidos.apps.p2.modules.p1_space.probe_utils import probe_arrays
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ModuleBase, ProducerContext, ResourceSpec, World, producer
from rheidos.compute import shape_map


def _harmonic_coeff_shape(zeta_ref):
    def shape_fn(reg):
        zeta_face = reg.read(zeta_ref.name, ensure=False)
        if zeta_face is None or not hasattr(zeta_face, "shape"):
            return None
        return (int(zeta_face.shape[0]),)

    return shape_fn


class HarmonicVelocityFieldModule(ModuleBase):
    NAME = "HarmonicVelocityFieldModule"

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        dual_harmonic_field: DualHarmonicFieldModule,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)

        self.mesh = mesh
        self.dual_harmonic_field = dual_harmonic_field

        self.harmonic_c = self.resource(
            "harmonic_c",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=_harmonic_coeff_shape(self.dual_harmonic_field.zeta_face),
            ),
            doc="Current harmonic velocity coefficients. Shape: (K,)",
            declare=True,
        )

        self.vel_per_face = self.resource(
            "vel_per_face",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.mesh.F_verts, lambda s: (s[0], 3)),
            ),
            doc="Facewise harmonic velocity sum_k c[k] zeta_k. Shape: (nF,3)",
        )

        self.vel_per_vertex = self.resource(
            "vel_per_vertex",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.mesh.V_pos, lambda s: (s[0], 3)),
            ),
            doc="Area-weighted smoothed harmonic velocity at vertices. Shape: (nV,3)",
        )

        self.bind_producers()

    def _zero_coefficients(self) -> np.ndarray:
        zeta_face = self.dual_harmonic_field.zeta_face.get()
        return np.zeros((zeta_face.shape[0],), dtype=np.float64)

    def set_coefficients(self, coefficients: np.ndarray) -> None:
        zeta_face = self.dual_harmonic_field.zeta_face.get()
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.shape != (zeta_face.shape[0],):
            raise ValueError(
                "harmonic coefficients must have shape "
                f"({zeta_face.shape[0]},), got {coefficients.shape}"
            )
        self.harmonic_c.set(np.ascontiguousarray(coefficients))

    @producer(
        inputs=("dual_harmonic_field.zeta_face", "harmonic_c"),
        outputs=("vel_per_face",),
        allow_none=("harmonic_c",),
    )
    def per_face_vel_calculate(self, ctx: ProducerContext) -> None:
        ctx.require_inputs(allow_none=("harmonic_c",))
        zeta_face = self.dual_harmonic_field.zeta_face.get()
        coefficients = self.harmonic_c.peek()

        if coefficients is None:
            coefficients = self._zero_coefficients()
            self.harmonic_c.set(coefficients)
        if coefficients.shape != (zeta_face.shape[0],):
            raise ValueError(
                "harmonic coefficients must have shape "
                f"({zeta_face.shape[0]},), got {coefficients.shape}"
            )

        if zeta_face.shape[0] == 0:
            ctx.commit(
                vel_per_face=np.zeros((zeta_face.shape[1], 3), dtype=np.float64)
            )
            return

        ctx.commit(vel_per_face=np.einsum("k,kfi->fi", coefficients, zeta_face))

    @producer(
        inputs=("vel_per_face", "mesh.F_area", "mesh.F_verts", "mesh.V_pos"),
        outputs=("vel_per_vertex",),
    )
    def per_vertex_vel_calculate(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        ctx.commit(
            vel_per_vertex=area_weighted_face_vectors_to_vertices(
                self.vel_per_face.get(),
                self.mesh.F_area.get(),
                self.mesh.F_verts.get(),
                self.mesh.V_pos.get().shape[0],
            )
        )

    def interpolate(self, probes, smooth=True) -> np.ndarray:
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
