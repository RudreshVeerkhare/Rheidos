import numpy as np
import pytest

from rheidos.apps.p2.modules.intergrator.rk4 import RK4IntegratorModule
from rheidos.compute import World


def test_rk4_matches_exponential_growth_step() -> None:
    module = RK4IntegratorModule(World(), y_dot=lambda y, t: y, timestep=0.1)

    result = module.step(np.array([1.0]), 0.0)

    assert np.allclose(result, np.array([np.exp(0.1)]), atol=1e-7)


def test_rk4_substeps_accumulate_time() -> None:
    module = RK4IntegratorModule(
        World(),
        y_dot=lambda y, t: np.ones_like(y),
        timestep=0.2,
        substeps=5,
    )

    result = module.step(np.array([3.0, 4.0]), 0.0)

    assert np.allclose(result, np.array([4.0, 5.0]))


@pytest.mark.parametrize(
    ("kwargs", "expected_error"),
    [
        ({"y_dot": None}, TypeError),
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
        module.step(np.zeros((2, 3)), 0.0)
