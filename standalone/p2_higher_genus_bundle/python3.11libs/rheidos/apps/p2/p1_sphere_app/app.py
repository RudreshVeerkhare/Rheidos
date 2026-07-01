import numpy as np
from dataclasses import dataclass
from typing import Callable

from rheidos.apps.p2._io import (
    load_mesh_input,
    load_point_vortex_input,
    read_probe_input,
)
from rheidos.apps.p2.modules.intergrator.rk4 import RK4IntegratorModule
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_stream_function import P1StreamFunction
from rheidos.apps.p2.modules.p1_space.p1_velocity import P1VelocityFieldModule
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute.world import ModuleBase, World
from rheidos.houdini.runtime.cook_context import CookContext
from rheidos.houdini.sop import (
    CallGeo,
    CtxInputGeo,
    SopFunctionModule,
    point_attrib_to_numpy,
)

DEFAULT_REFERENCE_SURFACE_PROJECT_SOP = "/obj/geo1/solver1/d/s/ray1"
REFERENCE_SURFACE_PROJECT_SOP_PARM = "reference_surface_project_sop"
RAY_HIT_PRIM_ATTR = "hitprim"
RAY_HIT_UVW_ATTR = "hitprimuv"


def triangle_bary_from_hituvw(
    hituvw: np.ndarray,
) -> np.ndarray:
    hituvw = np.asarray(hituvw, dtype=np.float64)
    if hituvw.ndim != 2 or hituvw.shape[1] != 3:
        raise ValueError(f"hituvw must have shape (N, 3), got {hituvw.shape}")
    bary = np.empty((hituvw.shape[0], 3), dtype=np.float64)
    bary[:, 0] = 1.0 - hituvw[:, 0] - hituvw[:, 1]
    bary[:, 1] = hituvw[:, 0]
    bary[:, 2] = hituvw[:, 1]
    return bary


@dataclass(frozen=True)
class ReferenceSurfaceProjection:
    pos: np.ndarray
    faceids: np.ndarray
    bary: np.ndarray


class ProjectPointsToReferenceSurface(SopFunctionModule):
    NAME = "ProjectPointsToReferenceSurface"
    SOP_INPUTS = {
        0: CallGeo("query"),
        1: CtxInputGeo(1, cache="cook"),
    }

    def project_points(self, points: np.ndarray) -> ReferenceSurfaceProjection:
        query_geo = self.points_to_geo(np.asarray(points, dtype=np.float64))
        return self.run(query=query_geo)

    def postprocess(self, out_geo, meta) -> ReferenceSurfaceProjection:
        del meta
        pos = point_attrib_to_numpy(out_geo, "P", dtype=np.float64, components=3)
        faceids = point_attrib_to_numpy(out_geo, RAY_HIT_PRIM_ATTR, dtype=np.int32)
        hituvw = point_attrib_to_numpy(
            out_geo,
            RAY_HIT_UVW_ATTR,
            dtype=np.float64,
            components=3,
        )
        return ReferenceSurfaceProjection(
            pos,
            faceids,
            triangle_bary_from_hituvw(hituvw),
        )


def _eval_parm_str(node, name: str, default: str) -> str:
    parm = node.parm(name)
    if parm is None:
        return default
    try:
        value = parm.evalAsString()
    except Exception:
        try:
            value = parm.eval()
        except Exception:
            value = default
    value = "" if value is None else str(value).strip()
    return value or default


def _projection_sop_path(ctx: CookContext) -> str:
    return _eval_parm_str(
        ctx.node,
        REFERENCE_SURFACE_PROJECT_SOP_PARM,
        DEFAULT_REFERENCE_SURFACE_PROJECT_SOP,
    )


def _setup_reference_surface_projector(ctx: CookContext) -> "App":
    mods = ctx.world().require(App)
    mods.reference_surface_projector.configure(node_path=_projection_sop_path(ctx))
    mods.reference_surface_projector.setup(ctx)
    return mods


class App(ModuleBase):
    NAME = "P1SphereApp"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)
        self.dec = self.require(DEC, mesh=self.mesh)

        # Point Vortices
        self.point_vortex = self.require(PointVortexModule)
        self.coexact_stream_function = self.require(
            P1StreamFunction,
            mesh=self.mesh,
            point_vortex=self.point_vortex,
            dec=self.dec,
        )
        self.coexact_vel = self.require(
            P1VelocityFieldModule,
            mesh=self.mesh,
            stream=self.coexact_stream_function,
            dec=self.dec,
        )

        # Advection
        self.rk4 = self.require(RK4IntegratorModule)
        self.reference_surface_projector = self.require(
            ProjectPointsToReferenceSurface,
            child=True,
            child_name="reference_surface_projector",
            node_path=DEFAULT_REFERENCE_SURFACE_PROJECT_SOP,
        )

    # y_dot
    @staticmethod
    def rk4_step(
        ctx: CookContext, project_to_faces=True
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        mods = (
            _setup_reference_surface_projector(ctx)
            if project_to_faces
            else ctx.world().require(App)
        )

        def _fn(y: np.ndarray, t: float) -> np.ndarray:
            if project_to_faces:
                projected = mods.reference_surface_projector.project_points(y)
                faceids, barys, pos = (
                    projected.faceids,
                    projected.bary,
                    projected.pos,
                )
            else:
                faceids, barys, pos = mods.mesh.project_on_nearest_face(y)
            gammas = mods.point_vortex.gamma.get()
            mods.point_vortex.set_vortex(
                faceids,
                barys,
                gammas,
                pos,
            )
            return mods.coexact_vel.interpolate((faceids, barys))

        return _fn


def setup_coexact_stream_function(ctx: CookContext):
    mods = ctx.world().require(App)
    load_mesh_input(
        ctx, mods.mesh, missing_message="Input 0 has to be mesh input geometry"
    )
    load_point_vortex_input(ctx, mods.point_vortex, index=1)
    mods.coexact_stream_function.set_homo_dirichlet_boundary()

    is_closed_surface = mods.mesh.boundary_edge_count.get() == 0
    if is_closed_surface:
        mods.coexact_stream_function.distribute_excess_vorticity = True


def interpolate_coexact_stream_function(ctx: CookContext) -> None:
    mods = ctx.world().require(App)
    faceids, bary = read_probe_input(ctx, index=0)
    stream_func = mods.coexact_stream_function.interpolate((faceids, bary))
    ctx.write_point("coexact_stream_func", stream_func)


def interpolate_coexact_velocity(ctx: CookContext) -> None:
    mods = ctx.world().require(App)
    faceids, bary = read_probe_input(ctx, index=0)
    vel = mods.coexact_vel.interpolate((faceids, bary))
    ctx.write_point("coexact_vel", vel)


def rk4_advect(ctx: CookContext, dt=0.001, project_to_faces=True) -> None:
    mods = (
        _setup_reference_surface_projector(ctx)
        if project_to_faces
        else ctx.world().require(App)
    )
    y_dot = mods.rk4_step(ctx, project_to_faces=project_to_faces)
    mods.rk4.configure(y_dot=y_dot, timestep=dt)
    load_point_vortex_input(ctx, mods.point_vortex, index=0)
    y0 = mods.point_vortex.pos_world.get()
    y = mods.rk4.step(y0)
    if project_to_faces:
        projected = mods.reference_surface_projector.project_points(y)
        faceids, barys, pos = projected.faceids, projected.bary, projected.pos
    else:
        faceids, barys, pos = mods.mesh.project_on_nearest_face(y)
    gammas = mods.point_vortex.gamma.get()
    mods.point_vortex.set_vortex(
        faceids,
        barys,
        gammas,
        pos,
    )
    ctx.write_point("P", pos)
    ctx.write_point("bary", barys)
    ctx.write_point("faceid", faceids)


def project_points_to_reference_surface(ctx: CookContext) -> None:
    mods = _setup_reference_surface_projector(ctx)
    projected = mods.reference_surface_projector.project_points(ctx.P())
    ctx.write_point("P", projected.pos)
    ctx.write_point("bary", projected.bary)
    ctx.write_point("faceid", projected.faceids)


def read_coexact_stream_function_per_vertex(ctx: CookContext):
    mods = ctx.world().require(App)
    # read stream function on mesh vertex
    # NOTE: We are assuming that the order is consistent between python mesh and houdini mesh
    stream_func = mods.coexact_stream_function.psi.get()
    ctx.write_point("coexact_stream_func", stream_func)


def read_facewise_velocity_field(ctx: CookContext):
    mods = ctx.world().require(App)
    # NOTE: We are assuming that the order/indexing is consistent between python mesh and houdini mesh
    facewise_vel = mods.coexact_vel.vel_per_face.get()
    ctx.write_prim("velocity", facewise_vel)


def read_per_vertex_velocity_field(ctx: CookContext):
    mods = ctx.world().require(App)
    # NOTE: We are assuming that the order/indexing is consistent between python mesh and houdini mesh
    per_vertex_vel = mods.coexact_vel.vel_per_vertex.get()
    ctx.write_point("velocity", per_vertex_vel)
