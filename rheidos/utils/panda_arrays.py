from __future__ import annotations

from typing import Any

import numpy as np

try:
    from panda3d.core import GeomVertexData
except Exception as e:  # pragma: no cover
    GeomVertexData = None  # type: ignore


def copy_numpy_to_vertex_array(
    vdata: Any, array_index: int, src: np.ndarray, cols: int
) -> None:
    """
    Copy a contiguous NumPy array directly into a Panda3D vertex array.

    Uses memoryview to avoid intermediate Python objects/allocations. Intended
    for dynamic buffers that are rewritten every frame.
    """
    if GeomVertexData is None:
        raise RuntimeError("Panda3D is not available. Install 'panda3d'.")
    if not isinstance(vdata, GeomVertexData):
        raise TypeError("vdata must be a GeomVertexData")

    src = np.ascontiguousarray(src)
    n = int(src.shape[0])
    if vdata.getNumRows() != n:
        vdata.setNumRows(n)

    if n == 0:
        return

    arr = vdata.modifyArray(array_index)
    handle = arr.modifyHandle()
    raw = memoryview(handle.getData())
    dest = np.frombuffer(raw, dtype=src.dtype)

    expected = n * cols
    if dest.size < expected:
        raise ValueError(
            f"Vertex array is smaller than source data: {dest.size} < {expected}"
        )

    dest = dest[:expected].reshape((n, cols))
    try:
        np.copyto(dest, src)
    except ValueError as exc:
        if "read-only" not in str(exc).lower():
            raise
        # Static vertex buffers may expose a read-only view; fall back to setData.
        handle.setData(src.reshape(-1).tobytes())


__all__ = ["copy_numpy_to_vertex_array"]
