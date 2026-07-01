from rheidos.houdini import CookContext, session
from ..io import copy_input_to_output
from .app import (
    export_dual_tree,
    export_generator_dual_loops,
    export_primal_tree,
    mark_generator_faces,
    setup_mesh as setup_mesh_app,
)

SESSION_NAME = "torus_tree_cotree"


@session(SESSION_NAME, debugger=True)
def setup_mesh(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    setup_mesh_app(ctx)


@session(SESSION_NAME, debugger=True)
def export_generator_dual_loops_node(ctx: CookContext):
    export_generator_dual_loops(ctx)


@session(SESSION_NAME, debugger=True)
def export_dual_tree_node(ctx: CookContext):
    export_dual_tree(ctx)


@session(SESSION_NAME, debugger=True)
def export_primal_tree_node(ctx: CookContext):
    export_primal_tree(ctx)


@session(SESSION_NAME, debugger=True)
def mark_generator_faces_node(ctx: CookContext):
    mark_generator_faces(ctx)
