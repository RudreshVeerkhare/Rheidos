import numpy as np
import pytest

from rheidos.apps.p2.modules.intergrator.rk4 import RK4IntegratorModule
from rheidos.compute import World


def test_rk4_matches_exponential_growth_step() -> None:
    module = RK4IntegratorModule(World(), y_dot=lambda y, t: y, timestep=0.1)

    result = module.step(np.array([1.0]))

    assert np.allclose(result, np.array([np.exp(0.1)]), atol=1e-7)
    assert module.time.get() == pytest.approx(0.1)


def test_rk4_substeps_accumulate_time() -> None:
    module = RK4IntegratorModule(
        World(),
        y_dot=lambda y, t: np.ones_like(y),
        timestep=0.2,
        substeps=5,
    )

    result = module.step(np.array([3.0, 4.0]))

    assert np.allclose(result, np.array([4.0, 5.0]))
    assert module.time.get() == pytest.approx(1.0)


def test_rk4_configure_updates_timestep_and_internal_time() -> None:
    module = RK4IntegratorModule(World())
    module.configure(
        y_dot=lambda y, t: np.full_like(y, t),
        timestep=0.1,
    )

    first = module.step(np.array([0.0]))
    second = module.step(first)

    assert np.allclose(first, np.array([0.005]))
    assert np.allclose(second, np.array([0.02]))
    assert module.time.get() == pytest.approx(0.2)


def test_rk4_step_requires_configured_derivative() -> None:
    module = RK4IntegratorModule(World(), timestep=0.1)

    with pytest.raises(
        RuntimeError,
        match="requires y_dot to be configured before stepping",
    ):
        module.step(np.array([1.0]))


@pytest.mark.parametrize(
    ("kwargs", "expected_error"),
    [
        ({"y_dot": 1}, TypeError),
        ({"y_dot": lambda y, t: y, "timestep": 0.0}, ValueError),
        ({"y_dot": lambda y, t: y, "timestep": -0.1}, ValueError),
        ({"y_dot": lambda y, t: y, "substeps": 0}, ValueError),
        ({"y_dot": lambda y, t: y, "substeps": -1}, ValueError),
        ({"y_dot": lambda y, t: y, "substeps": 1.5}, TypeError),
    ],
)
def test_rk4_rejects_invalid_constructor_arguments(kwargs, expected_error) -> None:
    with pytest.raises(expected_error):
        RK4IntegratorModule(World(), **kwargs)


def test_rk4_rejects_derivative_shape_broadcasting() -> None:
    module = RK4IntegratorModule(
        World(),
        y_dot=lambda y, t: np.array([1.0]),
        timestep=0.1,
    )

    with pytest.raises(ValueError, match="same shape as y"):
        module.step(np.zeros((2, 3)))


def test_rk4_reuses_same_world_module_with_reconfigured_derivatives() -> None:
    world = World()

    module_first = world.require(RK4IntegratorModule)
    module_first.configure(y_dot=lambda y, t: y, timestep=0.1)
    result_first = module_first.step(np.array([1.0]))

    module_second = world.require(RK4IntegratorModule)
    module_second.configure(y_dot=lambda y, t: 2.0 * y, timestep=0.1)
    result_second = module_second.step(np.array([1.0]))

    assert module_second is module_first
    assert np.allclose(result_first, np.array([np.exp(0.1)]), atol=1e-7)
    assert np.allclose(result_second, np.array([np.exp(0.2)]), atol=1e-7)
    assert module_second.time.get() == pytest.approx(0.2)
