import math
import operator
from typing import Callable

import numpy as np

from rheidos.compute import ModuleBase, World


class RK4IntegratorModule(ModuleBase):
    NAME = "RK4IntegratorModule"

    def __init__(
        self,
        world: World,
        *,
        scope: str = "",
        y_dot: Callable[[np.ndarray, float], np.ndarray] | None = None,
        timestep: float = 1e-3,
        substeps: int = 1,
    ) -> None:
        super().__init__(world, scope=scope)

        if not callable(y_dot):
            raise TypeError("y_dot must be a callable with signature y_dot(y, t)")

        timestep = float(timestep)
        if not math.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("timestep must be a finite positive float")

        try:
            substeps = operator.index(substeps)
        except TypeError as exc:
            raise TypeError("substeps must be an integer") from exc
        if substeps < 1:
            raise ValueError("substeps must be >= 1")

        self.y_dot = y_dot
        self.timestep = timestep
        self.substeps = substeps

    def _eval_derivative(self, y: np.ndarray, t: float) -> np.ndarray:
        dydt = np.asarray(self.y_dot(y, t))
        if dydt.shape != y.shape:
            raise ValueError(
                "y_dot must return an array with the same shape as y. "
                f"Expected {y.shape}, got {dydt.shape}."
            )
        return dydt

    def _rk4_step(self, y0: np.ndarray, t0: float, dt: float) -> np.ndarray:
        k1 = self._eval_derivative(y0, t0)
        k2 = self._eval_derivative(y0 + 0.5 * dt * k1, t0 + 0.5 * dt)
        k3 = self._eval_derivative(y0 + 0.5 * dt * k2, t0 + 0.5 * dt)
        k4 = self._eval_derivative(y0 + dt * k3, t0 + dt)

        return y0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(self, y0: np.ndarray, t0: float) -> np.ndarray:
        y = np.array(y0, copy=True)
        t = t0

        for _ in range(self.substeps):
            y = self._rk4_step(y, t, self.timestep)
            t += self.timestep

        return y
