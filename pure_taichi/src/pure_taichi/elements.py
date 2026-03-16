from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


REF_Q_PTS = np.array(
    [
        [1.0 / 6.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0],
    ],
    dtype=np.float64,
)
REF_Q_WTS = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=np.float64)

CORNER_BARY = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
CENTROID_BARY = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)


@runtime_checkable
class ScalarElement(Protocol):
    @property
    def local_dof_count(self) -> int: ...

    def eval_shape(self, xi: float, eta: float) -> np.ndarray: ...

    def eval_grad_ref(self, xi: float, eta: float) -> np.ndarray: ...


@dataclass(frozen=True)
class P2Element(ScalarElement):
    local_dof_count: int = 6

    def eval_shape(self, xi: float, eta: float) -> np.ndarray:
        l0 = 1.0 - xi - eta
        l1 = xi
        l2 = eta
        return np.array(
            [
                l0 * (2.0 * l0 - 1.0),
                l1 * (2.0 * l1 - 1.0),
                l2 * (2.0 * l2 - 1.0),
                4.0 * l0 * l1,
                4.0 * l1 * l2,
                4.0 * l2 * l0,
            ],
            dtype=np.float64,
        )

    def eval_grad_ref(self, xi: float, eta: float) -> np.ndarray:
        l0 = 1.0 - xi - eta
        l1 = xi
        l2 = eta

        dl0 = np.array([-1.0, -1.0], dtype=np.float64)
        dl1 = np.array([1.0, 0.0], dtype=np.float64)
        dl2 = np.array([0.0, 1.0], dtype=np.float64)

        return np.vstack(
            [
                (4.0 * l0 - 1.0) * dl0,
                (4.0 * l1 - 1.0) * dl1,
                (4.0 * l2 - 1.0) * dl2,
                4.0 * (l0 * dl1 + l1 * dl0),
                4.0 * (l1 * dl2 + l2 * dl1),
                4.0 * (l2 * dl0 + l0 * dl2),
            ]
        )


def bary_to_ref(bary: np.ndarray) -> tuple[float, float]:
    return float(bary[1]), float(bary[2])
