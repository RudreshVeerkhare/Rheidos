from __future__ import annotations

from typing import Any, Sequence

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


def external_array(shape: Sequence[int], dtype: np.dtype[Any] = np.float32, *, zero: bool = False) -> np.ndarray:
    """
    Allocate a contiguous NumPy array suitable for Taichi external array kernels.

    External arrays are passed into kernels using `ti.types.ndarray` arguments.
    Use `zero=True` when you need predictable initial values; otherwise the array
    is left uninitialised for speed.
    """
    arr = np.zeros(shape, dtype=dtype) if zero else np.empty(shape, dtype=dtype)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def ensure_external_array(arr: np.ndarray, dtype: np.dtype[Any] = np.float32) -> np.ndarray:
    """
    Coerce an input array into a contiguous, writeable external buffer.

    Use this to adapt NumPy/Torch/Paddle outputs before passing them to Taichi
    kernels that accept `ti.types.ndarray` arguments, avoiding surprise copies
    inside the kernel launch.
    """
    out = np.asarray(arr, dtype=dtype)
    if not out.flags["C_CONTIGUOUS"] or not out.flags["WRITEABLE"]:
        out = np.array(out, dtype=dtype, copy=True, order="C")
    return out


__all__ = ["field_to_numpy", "numpy_to_field", "external_array", "ensure_external_array"]
