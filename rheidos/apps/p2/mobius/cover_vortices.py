"""State ownership and deck pairing for Mobius point vortices."""

from __future__ import annotations

import numpy as np

from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ModuleBase, ProducerContext, ResourceSpec, World, producer

from .intrinsic_advection import IntrinsicVortexState, positions_from_state
from .orientation_double_cover import OrientationDoubleCover


def remap_barycentrics(
    source_vertex_ids: np.ndarray,
    target_vertex_ids: np.ndarray,
    bary: np.ndarray,
) -> np.ndarray:
    """Move weights between two local orderings of the same projected face.

    Cover faces reverse their local winding under the deck transformation.
    Matching projected vertex ids avoids embedding that ordering assumption in
    the advection code.
    """
    matches = source_vertex_ids[:, :, None] == target_vertex_ids[:, None, :]
    if np.any(matches.sum(axis=2) != 1) or np.any(matches.sum(axis=1) != 1):
        raise RuntimeError("Face corner maps are not one-to-one by projected vertex id")
    target_corner = np.argmax(matches, axis=2)
    remapped = np.zeros_like(bary)
    rows = np.arange(bary.shape[0])
    for source_corner in range(3):
        remapped[rows, target_corner[:, source_corner]] = bary[:, source_corner]
    return remapped


class DoubleCoverPointVortex(ModuleBase):
    """A derived 2N odd-parity view of N live lifted vortices."""

    NAME = "DoubleCoverPointVortex"

    def __init__(
        self,
        world: World,
        *,
        source_point_vortex: PointVortexModule,
        double_cover: OrientationDoubleCover,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)
        self.source_point_vortex = source_point_vortex
        self.cover = double_cover

        # These three resources intentionally match PointVortexModule's public
        # interface, allowing existing stream solvers to consume the paired
        # vortices without knowing how the cover state is represented.
        self.face_ids = self.resource(
            "face_ids",
            spec=ResourceSpec(kind="numpy", dtype=np.int32, allow_none=True),
            doc="Interleaved original and deck-paired cover face ids. Shape: (2N,)",
        )
        self.bary = self.resource(
            "bary",
            spec=ResourceSpec(kind="numpy", dtype=np.float64, allow_none=True),
            doc="Interleaved original and deck-paired barycentrics. Shape: (2N,3)",
        )
        self.gamma = self.resource(
            "gamma",
            spec=ResourceSpec(kind="numpy", dtype=np.float64, allow_none=True),
            doc="Odd deck-paired circulation strengths. Shape: (2N,)",
        )
        self.bind_producers()

    @producer(
        inputs=(
            "source_point_vortex.face_ids",
            "source_point_vortex.bary",
            "source_point_vortex.gamma",
            "cover.cover_faces",
            "cover.pi_vertex",
            "cover.tau_face",
        ),
        outputs=("face_ids", "bary", "gamma"),
    )
    def duplicate_vortices(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        source_face_ids = ctx.inputs.source_point_vortex.face_ids.get()
        source_bary = ctx.inputs.source_point_vortex.bary.get()
        source_gamma = ctx.inputs.source_point_vortex.gamma.get()
        cover_faces = ctx.inputs.cover.cover_faces.get()
        pi_vertex = ctx.inputs.cover.pi_vertex.get()
        tau_face = ctx.inputs.cover.tau_face.get()

        vortex_count = source_face_ids.shape[0] if source_face_ids.ndim == 1 else -1
        if source_face_ids.ndim != 1:
            raise ValueError(f"face_ids must have shape (N,), got {source_face_ids.shape}")
        if source_bary.shape != (vortex_count, 3):
            raise ValueError(
                f"bary must have shape ({vortex_count},3), got {source_bary.shape}"
            )
        if source_gamma.shape != (vortex_count,):
            raise ValueError(
                f"gamma must have shape ({vortex_count},), got {source_gamma.shape}"
            )
        if np.any(source_face_ids < 0) or np.any(source_face_ids >= tau_face.shape[0]):
            raise ValueError(
                f"face_ids must be in [0, {tau_face.shape[0]}), got {source_face_ids}"
            )

        paired_face_ids = tau_face[source_face_ids]
        source_vertices = pi_vertex[cover_faces[source_face_ids]]
        paired_vertices = pi_vertex[cover_faces[paired_face_ids]]
        paired_bary = remap_barycentrics(
            source_vertices,
            paired_vertices,
            source_bary,
        )

        # Vorticity is odd under the orientation-reversing deck map.  Pairing
        # circulation here makes the downstream P1 right-hand side odd before
        # the Poisson solve rather than repairing it after the fact.
        face_ids = np.stack((source_face_ids, paired_face_ids), axis=1).reshape(-1)
        bary = np.stack((source_bary, paired_bary), axis=1).reshape(-1, 3)
        gamma = np.stack((source_gamma, -source_gamma), axis=1).reshape(-1)

        # These checks protect the parity contract at the point where it is
        # created.  They are cheap O(N) array comparisons, not mesh searches.
        if not np.array_equal(face_ids[1::2], tau_face[face_ids[0::2]]):
            raise RuntimeError("Deck-paired vortex faces do not match tau_face")
        if not np.array_equal(gamma[1::2], -gamma[0::2]):
            raise RuntimeError("Deck-paired vortex circulation is not odd")

        ctx.commit(face_ids=face_ids, bary=bary, gamma=gamma)


class OrientationCoverVortexState(ModuleBase):
    """Own N lifted vortices and synchronize their derived representations."""

    NAME = "OrientationCoverVortexState"

    def __init__(
        self,
        world: World,
        *,
        base_mesh: SurfaceMeshModule,
        base_point_vortex: PointVortexModule,
        double_cover: OrientationDoubleCover,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)
        self.base_mesh = base_mesh
        self.base_point_vortex = base_point_vortex
        self.cover = double_cover
        self.cover_mesh = double_cover.cover_mesh

        self.lifted_point_vortex = self.require(
            PointVortexModule,
            child=True,
            child_name="lifted",
        )
        self.paired_point_vortex = self.require(
            DoubleCoverPointVortex,
            child=True,
            child_name="paired",
            source_point_vortex=self.lifted_point_vortex,
            double_cover=self.cover,
        )

    def current_state(self) -> IntrinsicVortexState:
        """Return a snapshot so rejected RK4 stages can restore it safely."""
        return IntrinsicVortexState(
            np.array(self.lifted_point_vortex.face_ids.get(), dtype=np.int32, copy=True),
            np.array(self.lifted_point_vortex.bary.get(), dtype=np.float64, copy=True),
        )

    def deck_state(self) -> IntrinsicVortexState:
        """Return the deck mate of every live lift in the same vortex order."""
        paired_face_ids = self.paired_point_vortex.face_ids.get()
        paired_bary = self.paired_point_vortex.bary.get()
        return IntrinsicVortexState(
            np.ascontiguousarray(paired_face_ids[1::2], dtype=np.int32),
            np.ascontiguousarray(paired_bary[1::2], dtype=np.float64),
        )

    def set_state(
        self,
        state: IntrinsicVortexState,
        gamma: np.ndarray | None = None,
    ) -> None:
        if gamma is None:
            gamma = self.lifted_point_vortex.gamma.get()
        self.lifted_point_vortex.set_vortex(
            state.face_ids,
            state.bary,
            gamma,
            positions_from_state(self.cover_mesh, state),
        )

    def initialize_from_base(self) -> IntrinsicVortexState:
        """Choose the canonical lift of each input base vortex."""
        base_face_ids = self.base_point_vortex.face_ids.get()
        base_bary = self.base_point_vortex.bary.get()
        base_face_count = self.base_mesh.F_verts.get().shape[0]
        if np.any(base_face_ids < 0) or np.any(base_face_ids >= base_face_count):
            raise ValueError(
                f"face_ids must be in [0, {base_face_count}), got {base_face_ids}"
            )

        # The cover builder stores the two lifts of base face f at 2f and
        # 2f+1.  We select the first only for initialization; subsequent
        # advection is free to cross to either sheet.
        cover_face_ids = 2 * base_face_ids
        base_vertices = self.cover.base_vertex_representative.get()[
            self.base_mesh.F_verts.get()[base_face_ids]
        ]
        cover_vertices = self.cover.pi_vertex.get()[
            self.cover_mesh.F_verts.get()[cover_face_ids]
        ]
        cover_bary = remap_barycentrics(base_vertices, cover_vertices, base_bary)

        state = IntrinsicVortexState(
            np.ascontiguousarray(cover_face_ids, dtype=np.int32),
            np.ascontiguousarray(cover_bary, dtype=np.float64),
        )
        self.set_state(state, self.base_point_vortex.gamma.get())
        self.sync_base()
        return state

    def sync_base(self) -> IntrinsicVortexState:
        """Update the N-vortex base view through pi, without a face search."""
        cover_state = self.current_state()
        base_face_ids = self.cover.pi_face.get()[cover_state.face_ids]
        cover_vertices = self.cover.pi_vertex.get()[
            self.cover_mesh.F_verts.get()[cover_state.face_ids]
        ]
        base_vertices = self.cover.base_vertex_representative.get()[
            self.base_mesh.F_verts.get()[base_face_ids]
        ]
        base_bary = remap_barycentrics(
            cover_vertices,
            base_vertices,
            cover_state.bary,
        )

        base_state = IntrinsicVortexState(
            np.ascontiguousarray(base_face_ids, dtype=np.int32),
            np.ascontiguousarray(base_bary, dtype=np.float64),
        )
        self.base_point_vortex.set_vortex(
            base_state.face_ids,
            base_state.bary,
            self.lifted_point_vortex.gamma.get(),
            positions_from_state(self.base_mesh, base_state),
        )
        return base_state


__all__ = [
    "DoubleCoverPointVortex",
    "OrientationCoverVortexState",
    "remap_barycentrics",
]
