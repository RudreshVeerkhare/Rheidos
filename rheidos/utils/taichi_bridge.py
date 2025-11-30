from __future__ import annotations

from typing import Any

import numpy as np


def field_to_numpy(field: Any) -> np.ndarray:
    try:
        import taichi as ti  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Taichi is not available. Install 'taichi'.") from e
    return field.to_numpy()  # type: ignore[attr-defined]


def numpy_to_field(arr: np.ndarray, field: Any) -> None:
    try:
        import taichi as ti  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Taichi is not available. Install 'taichi'.") from e
    field.from_numpy(np.ascontiguousarray(arr))  # type: ignore[attr-defined]

