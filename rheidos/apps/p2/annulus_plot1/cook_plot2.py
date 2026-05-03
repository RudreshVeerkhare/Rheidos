from rheidos.houdini import CookContext, session
from .._io import copy_input_to_output

from .app_plot2 import (
    setup_annulus_stream_functions,
    interpolate_combined_stream_function,
    set_evaluate_vortex_and_core_velocity,
    interpolate_p1_velocity,
    rk4_advect,
)

SESSION_NAME = "p1_annulus_plot2"


@session(SESSION_NAME, debugger=True)
def p1_setup_annulus_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    setup_annulus_stream_functions(ctx)


@session(SESSION_NAME, debugger=True)
def interpolate_combined_stream_function_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    interpolate_combined_stream_function(ctx)


@session(SESSION_NAME, debugger=True)
def interpolate_p1_velocity_node(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    interpolate_p1_velocity(ctx)


@session(SESSION_NAME, debugger=True)
def set_evaluate_vortex_and_core_velocity_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    set_evaluate_vortex_and_core_velocity(ctx)


@session(SESSION_NAME, debugger=True)
def rk4_vortex_advection(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    rk4_advect(ctx)


@session(SESSION_NAME, debugger=True)
def export_analytical_time_evolution_as_csv(ctx: CookContext):
    copy_input_to_output(ctx, 0)

    df = ctx.input_io(0).to_dataframes()["points"]
    df = df.drop(columns=["Cd_x", "Cd_y", "Cd_z"])
    df.to_csv(
        f"/Users/codebox/dev/kung_fu_panda/rheidos/apps/p2/annulus_plot1/data/radius_time_evolution/analytical_Rin_1_Rout_2.csv",
        index=False,
    )

    print("Hello")


@session(SESSION_NAME, debugger=True)
def export_discrete_time_evolution_as_csv(ctx: CookContext):
    copy_input_to_output(ctx, 0)

    df = ctx.input_io(0).to_dataframes()["points"]
    df = df.drop(columns=["Cd_x", "Cd_y", "Cd_z"])
    df.to_csv(
        f"/Users/codebox/dev/kung_fu_panda/rheidos/apps/p2/annulus_plot1/data/radius_time_evolution/discrete_Rin_1_Rout_2.csv",
        index=False,
    )

    print("Hello")
