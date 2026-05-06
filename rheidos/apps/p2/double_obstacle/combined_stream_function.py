import numpy as np

from rheidos.apps.p2.modules.p1_space.probe_utils import probe_arrays
from rheidos.compute.wiring import ProducerContext, producer

from .harmonic_basis import HarmonicBasisFieldModule, HarmonicBasisModule
from rheidos.apps.p2.modules.p1_space.p1_stream_function import P1StreamFunction
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.compute import shape_map
from rheidos.compute.resource import ResourceSpec
from rheidos.compute.world import ModuleBase, World


class CombinedStreamFunction(ModuleBase):

    def __init__(
        self,
        world: World,
        *,
        point_vortex: PointVortexModule,
        stream: P1StreamFunction,
        harmonic_basis_potential: HarmonicBasisModule,
        harmonic_basis_field: HarmonicBasisFieldModule,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)

        self.point_vortex = point_vortex
        self.stream = stream
        self.harmonic_basis_potential = harmonic_basis_potential
        self.harmonic_basis_field = harmonic_basis_field

        self.harmonic_coefficient = self.resource(
            "harmonic_coefficient",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(
                    self.harmonic_basis_potential.basis, lambda s: (s[0],)
                ),
            ),
            doc="Evolving harmonic coefficient of the harmonic component. Shape: (N, ) where N is dimension of the harmonic subspace",
        )

        self.psi = self.resource(
            "psi",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.stream.psi, lambda s: (s[0],)),
            ),
            doc="Combined stream function. Shape: (nV, )",
        )

        self.initial_coeff = self.resource(
            "initial_coeff",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(
                    self.harmonic_basis_potential.basis, lambda s: (s[0],)
                ),
            ),
            doc="Initial value of coefficient based on vortex initial position",
        )

        self.gram_inv = self.resource(
            "gram_inv",
            spec=ResourceSpec(kind="numpy", dtype=np.float64, shape=(2, 2)),
            doc="Inverse of gram matrix",
        )

        self.bind_producers()

    @producer(
        inputs=("harmonic_basis_field.vel_per_face", "stream.mesh.F_area"),
        outputs=("gram_inv",),
    )
    def build_gram_inverse(self, ctx: ProducerContext):
        print("1")
        F_area = self.stream.mesh.F_area.get()
        F_field = self.harmonic_basis_field.vel_per_face.get()

        G = np.zeros((2, 2))
        for fid, A in enumerate(F_area):
            for k in range(2):
                for l in range(2):
                    G[k, l] += A * np.dot(F_field[k][fid], F_field[l][fid])

        Ginv = np.linalg.inv(G)
        print("2")
        ctx.commit(gram_inv=Ginv)
        pass

    def initialize_harmonic_coeffs(self):
        gammas = self.point_vortex.gamma.get()
        faceids = self.point_vortex.face_ids.get()
        barys = self.point_vortex.bary.get()
        hpsi = np.array(
            [
                self.harmonic_basis_potential.interpolate(
                    list(zip(faceids, barys)), basis_id=basis_id
                )
                for basis_id in range(self.harmonic_basis_potential.dim)
            ]
        )  # (Hdim, nVortices)

        c = np.zeros((hpsi.shape[0],))

        # c = ∑ Gamma_vid * psi(x_vid) - c0
        self.initial_coeff = (hpsi * gammas).sum(axis=1) - c

    @producer(
        inputs=(
            "harmonic_basis_potential.basis",
            "gram_inv",
            "point_vortex.gamma",
            "point_vortex.face_ids",
            "point_vortex.bary",
        ),
        outputs=("harmonic_coefficient",),
    )
    def compute_harmonic_coefficient(self, ctx: ProducerContext):
        if self.initial_coeff is None:
            raise RuntimeError(f"Initial Coefficient is calculated")

        ctx.require_inputs()
        gammas = self.point_vortex.gamma.get()
        faceids = self.point_vortex.face_ids.get()
        barys = self.point_vortex.bary.get()
        hpsi = np.array(
            [
                self.harmonic_basis_potential.interpolate(
                    list(zip(faceids, barys)), basis_id=basis_id
                )
                for basis_id in range(self.harmonic_basis_potential.dim)
            ]
        )  # (Hdim, nVortices)
        Ginv = self.gram_inv.get()

        # c = ∑ Gamma_vid * psi(x_vid) - C0
        harmonic_coeff = (hpsi * gammas).sum(axis=1) - self.initial_coeff

        ctx.commit(harmonic_coefficient=Ginv @ harmonic_coeff)

    @producer(
        inputs=("stream.psi", "harmonic_basis_potential.basis", "harmonic_coefficient"),
        outputs=("psi",),
    )
    def calculate_combined_psi(self, ctx: ProducerContext):
        ctx.require_inputs()
        psi_s = self.stream.psi.get()
        psi_h = self.harmonic_basis_potential.basis.get()  # (Hdim, nV)
        c = self.harmonic_coefficient.get()  # (Hdim, )

        combined_harmonic = (psi_h.T * c).T.sum(
            axis=0
        )  # = c0 * psi_0 + c1 * psi_1 + ...
        ctx.commit(psi=np.array(psi_s + combined_harmonic, dtype=np.float64))

    def interpolate(self, probes) -> np.ndarray:
        psi = self.psi.get()
        faceids, bary = probe_arrays(probes)
        verts = self.stream.mesh.F_verts.get()[faceids]
        return np.einsum("ij,ij->i", psi[verts], bary)
