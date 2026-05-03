from rheidos.houdini import CookContext, session
from ..io import copy_input_to_output

from .app import setup_mesh, tree_cotree

SESSION_NAME = "torus_harmonic_basis"


@session(SESSION_NAME, debugger=True)
def setup_mesh_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    setup_mesh(ctx)


@session(SESSION_NAME, debugger=True)
def tree_cotree_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    tree_cotree(ctx)
