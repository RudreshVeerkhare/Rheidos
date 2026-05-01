from rheidos.houdini import CookContext, session
from .._io import copy_input_to_output

from .app import (
    setup_annulus_stream_functions,
    interpolate_combined_stream_function,
    set_evaluate_vortex_and_core_velocity,
)

SESSION_NAME = "p1_annulus_plot1"


@session(SESSION_NAME, debugger=True)
def p1_setup_annulus_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    setup_annulus_stream_functions(ctx)


@session(SESSION_NAME, debugger=True)
def interpolate_combined_stream_function_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    interpolate_combined_stream_function(ctx)


@session(SESSION_NAME, debugger=True)
def set_evaluate_vortex_and_core_velocity_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    set_evaluate_vortex_and_core_velocity(ctx)


@session(SESSION_NAME, debugger=True)
def export_analytical_vel_at_vortex_as_csv_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)

    df = ctx.input_io(0).to_dataframes()["points"]
    theta = df["theta"].unique()[0]
    r_in = df["Rin"].unique()[0]
    r_out = df["Rout"].unique()[0]

    df.to_csv(
        f"/Users/codebox/dev/kung_fu_panda/rheidos/apps/p2/annulus_plot1/data/analytical/{theta:.2f}_{r_in:.2f}_{r_out:.2f}.csv",
        index=False,
    )


@session(SESSION_NAME, debugger=True)
def export_discrete_vel_at_vortex_as_csv_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)

    df = ctx.input_io(0).to_dataframes()["points"]
    theta = df["theta"].unique()[0]
    r_in = df["Rin"].unique()[0]
    r_out = df["Rout"].unique()[0]

    df.to_csv(
        f"/Users/codebox/dev/kung_fu_panda/rheidos/apps/p2/annulus_plot1/data/discrete/{theta:.2f}_{r_in:.2f}_{r_out:.2f}.csv",
        index=False,
    )
