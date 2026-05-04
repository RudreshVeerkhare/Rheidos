import numpy as np

from rheidos.apps.p2.modules.higher_genus.harmonic_basis import HarmonicBasis
from rheidos.apps.p2.modules.p1_space.probe_utils import probe_arrays
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ProducerContext, ResourceSpec, producer
from rheidos.compute.world import ModuleBase, World


def _basis_face_vector_shape(forms_ref, faces_ref):
    def shape_fn(reg):
        forms = reg.read(forms_ref.name, ensure=False)
        faces = reg.read(faces_ref.name, ensure=False)
        if forms is None or faces is None:
            return None
        if not hasattr(forms, "shape") or not hasattr(faces, "shape"):
            return None
        return (int(forms.shape[0]), int(faces.shape[0]), 3)

    return shape_fn


def _basis_pairing_shape(forms_ref):
    def shape_fn(reg):
        forms = reg.read(forms_ref.name, ensure=False)
        if forms is None or not hasattr(forms, "shape"):
            return None
        return (int(forms.shape[0]), int(forms.shape[0]))

    return shape_fn


class DualHarmonicFieldModule(ModuleBase):
    NAME = "DualHarmonicFieldModule"
    _FIELD_ALIASES = {
        "xi": "xi_face",
        "xi_face": "xi_face",
        "raw_zeta": "raw_zeta_face",
        "zeta_tilde": "raw_zeta_face",
        "zeta_tilde_face": "raw_zeta_face",
        "raw_zeta_face": "raw_zeta_face",
        "zeta": "zeta_face",
        "zeta_face": "zeta_face",
    }

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        harmonic_basis: HarmonicBasis,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)

        self.mesh = mesh
        self.harmonic_basis = harmonic_basis

        self.xi_face = self.resource(
            "xi_face",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=_basis_face_vector_shape(
                    self.harmonic_basis.gamma,
                    self.mesh.F_verts,
                ),
            ),
            doc="Whitney interpolation of the harmonic 1-form basis xi. Shape: (K,nF,3)",
        )

        self.raw_zeta_face = self.resource(
            "zeta_tilde_face",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=_basis_face_vector_shape(
                    self.harmonic_basis.gamma,
                    self.mesh.F_verts,
                ),
            ),
            doc="Raw rotated harmonic velocity basis -star(xi). Shape: (K,nF,3)",
        )

        self.zeta_face = self.resource(
            "zeta_face",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=_basis_face_vector_shape(
                    self.harmonic_basis.gamma,
                    self.mesh.F_verts,
                ),
            ),
            doc="Dual harmonic velocity basis. Shape: (K,nF,3)",
        )

        self.lambda_raw = self.resource(
            "lambda_raw",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=_basis_pairing_shape(self.harmonic_basis.gamma),
            ),
            doc="Raw wedge pairing matrix between zeta_tilde and xi. Shape: (K,K)",
        )

        self.final_pairing = self.resource(
            "final_pairing",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=_basis_pairing_shape(self.harmonic_basis.gamma),
            ),
            doc="Final wedge pairing matrix between zeta and xi. Shape: (K,K)",
        )

        self.bind_producers()

    @staticmethod
    def _star_face_vectors(vectors: np.ndarray, face_normals: np.ndarray) -> np.ndarray:
        return np.cross(face_normals[None, :, :], vectors)

    @classmethod
    def _wedge_pairing_matrix(
        cls,
        alpha_face: np.ndarray,
        beta_face: np.ndarray,
        face_normals: np.ndarray,
        face_areas: np.ndarray,
    ) -> np.ndarray:
        # The exterior pairing alpha wedge beta is represented by
        # <star(alpha), beta> dA, not by the Euclidean dot product directly.
        star_alpha = cls._star_face_vectors(alpha_face, face_normals)
        return np.einsum("kfi,lfi,f->kl", star_alpha, beta_face, face_areas)

    @staticmethod
    def _facewise_whitney_vectors(
        gamma: np.ndarray,
        face_edges: np.ndarray,
        face_edge_sign: np.ndarray,
        face_edge01: np.ndarray,
        face_edge02: np.ndarray,
    ) -> np.ndarray:
        local_values = gamma[:, face_edges] * face_edge_sign[None, :, :]
        alpha_01 = local_values[:, :, 0]

        # F_edges stores the third local edge as k -> i. The Whitney solve uses
        # the edge i -> k, so its cochain value has the opposite sign.
        alpha_02 = -local_values[:, :, 2]

        g00 = np.einsum("fi,fi->f", face_edge01, face_edge01)
        g01 = np.einsum("fi,fi->f", face_edge01, face_edge02)
        g11 = np.einsum("fi,fi->f", face_edge02, face_edge02)
        det = g00 * g11 - g01 * g01
        if np.any(det <= 1e-30):
            raise ValueError("Dual harmonic field cannot use degenerate triangles.")

        coeff_0 = (alpha_01 * g11[None, :] - alpha_02 * g01[None, :]) / det[None, :]
        coeff_1 = (-alpha_01 * g01[None, :] + alpha_02 * g00[None, :]) / det[None, :]
        return (
            coeff_0[:, :, None] * face_edge01[None, :, :]
            + coeff_1[:, :, None] * face_edge02[None, :, :]
        )

    @producer(
        inputs=(
            "harmonic_basis.gamma",
            "mesh.F_edges",
            "mesh.F_edge_sign",
            "mesh.F_edge01",
            "mesh.F_edge02",
            "mesh.F_normal",
            "mesh.F_area",
        ),
        outputs=(
            "xi_face",
            "raw_zeta_face",
            "zeta_face",
            "lambda_raw",
            "final_pairing",
        ),
    )
    def build_dual_harmonic_fields(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        gamma = np.asarray(self.harmonic_basis.gamma.get(), dtype=np.float64)
        if gamma.ndim != 2:
            raise ValueError(
                "DualHarmonicFieldModule expects harmonic_basis.gamma with "
                f"shape (K,nE), got {gamma.shape}"
            )

        face_edges = self.mesh.F_edges.get()
        n_faces = int(face_edges.shape[0])
        k_basis = int(gamma.shape[0])
        if k_basis == 0:
            empty_face_basis = np.empty((0, n_faces, 3), dtype=np.float64)
            empty_pairing = np.empty((0, 0), dtype=np.float64)
            ctx.commit(
                xi_face=empty_face_basis,
                raw_zeta_face=empty_face_basis.copy(),
                zeta_face=empty_face_basis.copy(),
                lambda_raw=empty_pairing,
                final_pairing=empty_pairing.copy(),
            )
            return

        xi_face = self._facewise_whitney_vectors(
            gamma,
            face_edges,
            self.mesh.F_edge_sign.get(),
            self.mesh.F_edge01.get(),
            self.mesh.F_edge02.get(),
        )
        face_normals = self.mesh.F_normal.get()
        face_areas = self.mesh.F_area.get()

        # star(v) = n_f x v for tangent vector proxies of primal 1-forms.
        raw_zeta_face = -self._star_face_vectors(xi_face, face_normals)
        lambda_raw = self._wedge_pairing_matrix(
            raw_zeta_face,
            xi_face,
            face_normals,
            face_areas,
        )

        dualizer = np.linalg.solve(lambda_raw, np.eye(k_basis, dtype=np.float64))
        zeta_face = np.einsum("km,mfi->kfi", dualizer, raw_zeta_face)
        final_pairing = self._wedge_pairing_matrix(
            zeta_face,
            xi_face,
            face_normals,
            face_areas,
        )

        ctx.commit(
            xi_face=xi_face,
            raw_zeta_face=raw_zeta_face,
            zeta_face=zeta_face,
            lambda_raw=lambda_raw,
            final_pairing=final_pairing,
        )

    def integrate_xi_same_face(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        faceids: np.ndarray,
    ) -> np.ndarray:
        start = np.asarray(p0, dtype=np.float64)
        end = np.asarray(p1, dtype=np.float64)
        faceids = np.asarray(faceids, dtype=np.int32)
        if start.shape != end.shape or start.ndim != 2 or start.shape[1] != 3:
            raise ValueError(
                "integrate_xi_same_face expects p0 and p1 with matching "
                f"shape (N,3), got {start.shape} and {end.shape}"
            )
        if faceids.shape != (start.shape[0],):
            raise ValueError(
                "integrate_xi_same_face expects faceids with shape "
                f"({start.shape[0]},), got {faceids.shape}"
            )

        xi_face = self.xi_face.get()
        return np.einsum("kni,ni->nk", xi_face[:, faceids, :], end - start)

    def harmonic_velocity_at_faces(
        self,
        coefficients: np.ndarray,
        faceids: np.ndarray,
    ) -> np.ndarray:
        coefficients = np.asarray(coefficients, dtype=np.float64)
        faceids = np.asarray(faceids, dtype=np.int32)
        zeta_face = self.zeta_face.get()
        if coefficients.shape != (zeta_face.shape[0],):
            raise ValueError(
                "harmonic_velocity_at_faces expects coefficients with shape "
                f"({zeta_face.shape[0]},), got {coefficients.shape}"
            )
        if faceids.ndim != 1:
            raise ValueError(
                "harmonic_velocity_at_faces expects faceids with shape (N,), "
                f"got {faceids.shape}"
            )
        return np.einsum("k,kni->ni", coefficients, zeta_face[:, faceids, :])

    def interpolate(self, probes, field: str = "zeta") -> np.ndarray:
        """Sample one facewise harmonic basis field at probe faces.

        ``probes`` follows ``Whitney1FormInterpolator.interpolate``: either
        ``(faceids, bary)`` arrays or ``[(faceid, bary), ...]``. Barycentric
        coordinates are accepted for API consistency, but these basis fields
        are facewise constant after construction.
        """

        faceids, _bary = probe_arrays(probes)
        field_name = self._FIELD_ALIASES.get(str(field))
        if field_name is None:
            valid = ", ".join(sorted(self._FIELD_ALIASES))
            raise ValueError(
                f"Unknown harmonic field {field!r}; expected one of: {valid}"
            )

        field_values = getattr(self, field_name).get()
        return field_values[:, faceids, :]
