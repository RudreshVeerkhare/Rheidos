from rheidos.apps.p2.modules.p1_space.p1_poisson_solver import P1PoissonSolver
from rheidos.apps.p2.modules.p1_space.p1_stream_function import P1StreamFunction
from rheidos.apps.p2.modules.p2_space.p2_poisson_solver import P2PoissonSolver
from rheidos.apps.p2.modules.p2_space.p2_stream_function import P2StreamFunction
from rheidos.compute import World


def test_p1_stream_function_uses_child_poisson_solver_scope() -> None:
    world = World()

    stream = world.require(P1StreamFunction)
    standalone = world.require(P1PoissonSolver)

    assert stream.poisson.prefix == "P1StreamFunction.poisson.P1PoissonSolver"
    assert stream.poisson.lookup_scope == stream.lookup_scope
    assert stream.poisson.mesh is stream.mesh
    assert stream.poisson.dec is stream.dec
    assert stream.omega.name == stream.poisson.rhs.name
    assert standalone.mesh is stream.mesh
    assert standalone.dec is stream.dec


def test_p2_stream_function_uses_child_poisson_solver_scope() -> None:
    world = World()

    stream = world.require(P2StreamFunction)
    standalone = world.require(P2PoissonSolver)

    assert stream.poisson.prefix == "P2StreamFunction.poisson.P2PoissonSolver"
    assert stream.poisson.lookup_scope == stream.lookup_scope
    assert stream.poisson.p2_space is stream.p2_elements
    assert stream.omega.name == stream.poisson.rhs.name
    assert standalone.p2_space is stream.p2_elements
