from rheidos.houdini import CookContext, session
from .._io import copy_input_to_output

from .app import setup_mesh_and_point_vortices, rk4_advect

SESSION_NAME = "point_vortex_obstacle"


@session(SESSION_NAME, debugger=True)
def setup_mesh_and_point_vortices_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    setup_mesh_and_point_vortices(ctx)


@session(SESSION_NAME, debugger=True)
def rk4_advection_node(ctx: CookContext, dt=0.01):
    copy_input_to_output(ctx, 0)
    rk4_advect(ctx, dt=dt)
