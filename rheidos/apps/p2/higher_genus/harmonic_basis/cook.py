from rheidos.houdini import CookContext, session
from ..io import copy_input_to_output

from .app import interpolate_harmonic_basis_velocity, setup_mesh, tree_cotree

SESSION_NAME = "torus_harmonic_basis"


@session(SESSION_NAME, debugger=True)
def setup_mesh_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    setup_mesh(ctx)


@session(SESSION_NAME, debugger=True)
def tree_cotree_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    tree_cotree(ctx)


@session(SESSION_NAME, debugger=True)
def interpolate_harmonic_basis_velocity_node(ctx: CookContext, basis_id=0):
    copy_input_to_output(ctx, 1)
    interpolate_harmonic_basis_velocity(ctx, basis_id=basis_id)
