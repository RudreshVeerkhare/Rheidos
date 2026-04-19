from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_poisson_solver import P1PoissonSolver
from rheidos.apps.p2.modules.p1_space.p1_stream_function import P1StreamFunction
from rheidos.apps.p2.modules.p2_space.p2_elements import P2Elements
from rheidos.apps.p2.modules.p2_space.p2_poisson_solver import P2PoissonSolver
from rheidos.apps.p2.modules.p2_space.p2_stream_function import P2StreamFunction
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import World


def test_p1_stream_function_uses_explicit_poisson_solver() -> None:
    world = World()

    mesh = world.require(SurfaceMeshModule)
    vortex = world.require(PointVortexModule)
    dec = world.require(DEC, mesh=mesh)
    poisson = world.require(
        P1PoissonSolver,
        mesh=mesh,
        dec=dec,
        declare_rhs=False,
    )
    stream = world.require(
        P1StreamFunction,
        mesh=mesh,
        point_vortex=vortex,
        poisson=poisson,
    )

    deps = world.module_dependencies()
    assert stream.poisson is poisson
    assert stream.dec is dec
    assert stream.poisson.mesh is mesh
    assert stream.omega.name == poisson.rhs.name
    assert poisson._module_key in deps[stream._module_key]
    assert dec._module_key in deps[poisson._module_key]


def test_p2_stream_function_uses_explicit_poisson_solver() -> None:
    world = World()

    mesh = world.require(SurfaceMeshModule)
    vortex = world.require(PointVortexModule)
    p2_space = world.require(P2Elements, mesh=mesh)
    poisson = world.require(
        P2PoissonSolver,
        p2_space=p2_space,
        declare_rhs=False,
    )
    stream = world.require(
        P2StreamFunction,
        mesh=mesh,
        point_vortex=vortex,
        p2_elements=p2_space,
        poisson=poisson,
    )

    deps = world.module_dependencies()
    assert stream.poisson is poisson
    assert stream.p2_elements is p2_space
    assert stream.omega.name == poisson.rhs.name
    assert poisson._module_key in deps[stream._module_key]
    assert p2_space._module_key in deps[poisson._module_key]
