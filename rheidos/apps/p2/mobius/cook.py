import numpy as np

from rheidos.apps.p2._io import read_probe_input
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_stream_function import P1StreamFunction
from rheidos.apps.p2.modules.p1_space.p1_velocity import P1VelocityFieldModule
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute.resource import ResourceSpec
from rheidos.compute.wiring import ProducerContext, producer
from rheidos.compute.world import ModuleBase, World
from rheidos.houdini import CookContext, session

from .io import load_mesh_input, load_point_vortex_input
from .orientation_double_cover import OrientationDoubleCover

SESSION_NAME = "mobius_strip_vortex_dynamics"


@session(SESSION_NAME, debugger=True)
def write_cover_point_vortices(ctx: CookContext):
    mods = ctx.world().require(App)
    pt_vortex = mods.point_vortex

    face_ids = pt_vortex.face_ids.get()
    bary = pt_vortex.bary.get()
    gamma = pt_vortex.gamma.get()

    # Houdini points require P even though the lifted vortices are represented
    # intrinsically by (faceid, bary). A later SOP can compute their positions.
    placeholder_positions = np.zeros((face_ids.shape[0], 3), dtype=np.float64)

    ctx.clear_output()
    ctx.create_points(placeholder_positions)  # Shape: (2N, 3)

    ctx.write_point("faceid", face_ids, create=True)  # (2N, )
    ctx.write_point("bary", bary, create=True)  # (2N, 3)
    ctx.write_point("gamma", gamma, create=True)  # (2N, )


@session(SESSION_NAME, debugger=True)
def setup_mesh_node(ctx: CookContext):
    mods = ctx.world().require(App)
    load_mesh_input(
        ctx, mods.base_mesh, missing_message="Input 0 has to be a mesh input geometry"
    )
    load_point_vortex_input(ctx, mods.base_point_vortex, index=1)
    mods.stream_function.set_homo_dirichlet_boundary()

    # Add cover attributes to the mesh geometry
    cover_vertices = mods.cover.cover_vertices.get()
    cover_faces = mods.cover.cover_faces.get()
    output = ctx.output_io()
    output.clear_output()
    output.create_points(cover_vertices)
    output.create_polygons(cover_faces)
    output.write_point("base_vertex", mods.cover.pi_vertex.get(), create=True)
    output.write_point("deck_vertex", mods.cover.tau_vertex.get(), create=True)
    output.write_prim("base_face", mods.cover.pi_face.get(), create=True)
    output.write_prim("deck_face", mods.cover.tau_face.get(), create=True)
    output.write_point("psi", mods.stream_function.psi.get(), create=True)


@session(SESSION_NAME, debugger=True)
def interpolate_stream_velocity_node(ctx: CookContext):
    mods = ctx.world().require(App)
    faceids, bary = read_probe_input(ctx, index=0)
    stream_velocity_field = mods.stream_velocity.interpolate((faceids, bary))
    ctx.write_point("stream_velocity_field", stream_velocity_field)


### MODULES BELOW


class App(ModuleBase):
    NAME = "MobiusOrientationCoverApp"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.base_mesh = self.require(
            SurfaceMeshModule,
            child=True,
            child_name="base_mesh",
        )

        # Double cover
        self.cover = self.require(
            OrientationDoubleCover,
            child=True,
            child_name="orientation_cover",
            parent_mesh=self.base_mesh,
        )
        self.cover_mesh: SurfaceMeshModule = self.cover.cover_mesh
        self.dec: DEC = self.cover.dec

        # point vortex
        self.base_point_vortex = self.require(PointVortexModule)
        self.point_vortex = self.require(
            DoubleCoverPointVortex,
            base_point_vortex=self.base_point_vortex,
            base_mesh=self.base_mesh,
            double_cover=self.cover,
        )

        # Stream function
        self.stream_function = self.require(
            P1StreamFunction,
            mesh=self.cover_mesh,
            point_vortex=self.point_vortex,
            dec=self.dec,
        )
        self.stream_velocity = self.require(
            P1VelocityFieldModule,
            mesh=self.cover_mesh,
            dec=self.dec,
            stream=self.stream_function,
        )


class DoubleCoverPointVortex(ModuleBase):
    NAME = "DoubleCoverPointVortex"

    def __init__(
        self,
        world: World,
        *,
        base_point_vortex: PointVortexModule,
        base_mesh: SurfaceMeshModule,
        double_cover: OrientationDoubleCover,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)

        self.base_mesh: SurfaceMeshModule = base_mesh
        self.base_point_vortex: PointVortexModule = base_point_vortex
        self.cover: OrientationDoubleCover = double_cover

        # Fields copy from PointVortexModule
        # TODO: See if there's a better way to wrap module classes for such cases.
        self.face_ids = self.resource(
            "face_ids",
            spec=ResourceSpec(kind="numpy", dtype=np.int32, allow_none=True),
            doc="Face id of the face onto which the point vortex lies",
        )
        self.bary = self.resource(
            "bary",
            spec=ResourceSpec(kind="numpy", dtype=np.float64, allow_none=True),
            doc="Barycentric co-ordinates of the point inside the triangle",
        )
        self.gamma = self.resource(
            "gamma",
            spec=ResourceSpec(kind="numpy", dtype=np.float64, allow_none=True),
            doc="Circulation strength of a point vortex",
        )

        self.bind_producers()

    @producer(
        inputs=(
            "base_point_vortex.face_ids",
            "base_point_vortex.bary",
            "base_point_vortex.gamma",
            "cover.tau_face",
        ),
        outputs=("face_ids", "bary", "gamma"),
    )
    def duplicate_vortices(self, ctx: ProducerContext):
        # Lift each vortex to both deck-related faces, reorder barycentrics for
        # the reversed face, and negate circulation on the minus sheet.

        ctx.require_inputs()

        base_faceids = ctx.inputs.base_point_vortex.face_ids.get()
        base_barys = ctx.inputs.base_point_vortex.bary.get()
        base_gammas = ctx.inputs.base_point_vortex.gamma.get()
        tau_face = ctx.inputs.cover.tau_face.get()

        vortex_count = base_faceids.shape[0] if base_faceids.ndim == 1 else -1
        if base_faceids.ndim != 1:
            raise ValueError(f"face_ids must have shape (N,), got {base_faceids.shape}")
        if base_barys.shape != (vortex_count, 3):
            raise ValueError(
                f"bary must have shape ({vortex_count}, 3), got {base_barys.shape}"
            )
        if base_gammas.shape != (vortex_count,):
            raise ValueError(
                f"gamma must have shape ({vortex_count},), got {base_gammas.shape}"
            )
        if not np.all(np.isfinite(base_barys)):
            raise ValueError("bary contains non-finite values")
        if not np.all(np.isfinite(base_gammas)):
            raise ValueError("gamma contains non-finite values")

        base_face_count = tau_face.shape[0] // 2
        if np.any(base_faceids < 0) or np.any(base_faceids >= base_face_count):
            raise ValueError(
                f"face_ids must be in [0, {base_face_count}), got {base_faceids}"
            )

        # mapping faceids is f -> 2f (plus) and f -> 2f+1 (minus)
        plus_faceids = 2 * base_faceids

        # Get its lift using deck transform
        minus_faceids = tau_face[plus_faceids]

        # Minus faces reverse the last two corners: (v0, v1, v2) -> (v0, v2, v1).
        plus_bary = base_barys
        minus_bary = base_barys[:, [0, 2, 1]]

        # Circulation reverse as vorticity is odd parity
        plus_gamma = base_gammas
        minus_gamma = -base_gammas

        # Combine both +- copies in [pt0+, pt0-, pt1+, pt1-, ....] order
        cover_faceids = np.stack((plus_faceids, minus_faceids), axis=1).reshape(-1)
        cover_bary = np.stack((plus_bary, minus_bary), axis=1).reshape(-1, 3)
        cover_gamma = np.stack((plus_gamma, minus_gamma), axis=1).reshape(-1)

        ctx.commit(face_ids=cover_faceids, bary=cover_bary, gamma=cover_gamma)
