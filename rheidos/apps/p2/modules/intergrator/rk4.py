import math
import operator
from typing import Callable

import numpy as np

from rheidos.compute import ModuleBase, World
from rheidos.compute.resource import ResourceSpec


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
        self.y_dot: Callable[[np.ndarray, float], np.ndarray] | None = None
        self.time = self.resource(
            "time",
            spec=ResourceSpec(kind="python", dtype=float),
            doc="Time elapsed counter",
            buffer=0,
            declare=True,
        )
        self.timestep = self._validate_timestep(timestep)
        self.substeps = self._validate_substeps(substeps)

        if y_dot is not None:
            self.configure(y_dot=y_dot)

    def _validate_y_dot(
        self,
        y_dot: Callable[[np.ndarray, float], np.ndarray] | None,
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        if not callable(y_dot):
            raise TypeError("y_dot must be a callable with signature y_dot(y, t)")
        return y_dot

    def _validate_timestep(self, timestep: float) -> float:
        timestep = float(timestep)
        if not math.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("timestep must be a finite positive float")
        return timestep

    def _validate_substeps(self, substeps: int) -> int:
        try:
            substeps = operator.index(substeps)
        except TypeError as exc:
            raise TypeError("substeps must be an integer") from exc
        if substeps < 1:
            raise ValueError("substeps must be >= 1")
        return substeps

    def configure(
        self,
        *,
        y_dot: Callable[[np.ndarray, float], np.ndarray] | None = None,
        timestep: float | None = None,
    ) -> None:
        if y_dot is not None:
            self.y_dot = self._validate_y_dot(y_dot)
        if timestep is not None:
            self.timestep = self._validate_timestep(timestep)

    def _require_y_dot(self) -> Callable[[np.ndarray, float], np.ndarray]:
        if self.y_dot is None:
            raise RuntimeError(
                "RK4IntegratorModule.step() requires y_dot to be configured "
                "before stepping."
            )
        return self.y_dot

    def _eval_derivative(self, y: np.ndarray, t: float) -> np.ndarray:
        y_dot = self._require_y_dot()
        dydt = np.asarray(y_dot(y, t))
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

    def step(self, y0: np.ndarray) -> np.ndarray:
        y = np.array(y0, copy=True)
        t = self.time.get()

        for _ in range(self.substeps):
            y = self._rk4_step(y, t, self.timestep)
            t += self.timestep

        self.time.set(t)
        return y
