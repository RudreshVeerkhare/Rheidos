from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from panda3d.core import (
        Geom,
        GeomNode,
        GeomTriangles,
        GeomVertexArrayData,
        GeomVertexArrayFormat,
        GeomVertexData,
        GeomVertexFormat,
        InternalName,
        NodePath,
    )
except Exception as e:  # pragma: no cover
    Geom = None  # type: ignore
    GeomNode = None  # type: ignore
    GeomTriangles = None  # type: ignore
    GeomVertexArrayData = None  # type: ignore
    GeomVertexArrayFormat = None  # type: ignore
    GeomVertexData = None  # type: ignore
    GeomVertexFormat = None  # type: ignore
    InternalName = None  # type: ignore
    NodePath = None  # type: ignore


def _ensure_float32(a: np.ndarray, cols: int) -> np.ndarray:
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    a = np.ascontiguousarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, cols)
    assert a.shape[1] == cols, f"Expected {cols} columns, got {a.shape}"
    return a


def _ensure_uint8(a: np.ndarray, cols: int) -> np.ndarray:
    if a.dtype != np.uint8:
        a = a.astype(np.uint8, copy=False)
    a = np.ascontiguousarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, cols)
    assert a.shape[1] == cols, f"Expected {cols} columns, got {a.shape}"
    return a


class Mesh:
    def __init__(
        self,
        vertices: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        texcoords: Optional[np.ndarray] = None,
        dynamic: bool = True,
        name: str = "mesh",
    ) -> None:
        if Geom is None:
            raise RuntimeError("Panda3D is not available. Install 'panda3d'.")

        # Build separated arrays format: P, N, C, T
        v_arr = GeomVertexArrayFormat()
        v_arr.addColumn(InternalName.getVertex(), 3, Geom.NTFloat32, Geom.CPoint)
        n_arr = GeomVertexArrayFormat()
        n_arr.addColumn(InternalName.getNormal(), 3, Geom.NTFloat32, Geom.CNormal)
        c_arr = GeomVertexArrayFormat()
        c_arr.addColumn(InternalName.getColor(), 4, Geom.NTFloat32, Geom.CColor)
        t_arr = GeomVertexArrayFormat()
        t_arr.addColumn(InternalName.getTexcoord(), 2, Geom.NTFloat32, Geom.CTexcoord)

        fmt = GeomVertexFormat()
        fmt.addArray(v_arr)
        fmt.addArray(n_arr)
        fmt.addArray(c_arr)
        fmt.addArray(t_arr)
        fmt = GeomVertexFormat.registerFormat(fmt)

        usage = Geom.UHDynamic if dynamic else Geom.UHStatic
        self.vdata = GeomVertexData(name, fmt, usage)
        self.vdata.setNumRows(0)
        self.prim = GeomTriangles(usage)
        self.geom = Geom(self.vdata)
        self.geom.addPrimitive(self.prim)
        self.node = GeomNode(name)
        self.node.addGeom(self.geom)
        self.node_path = NodePath(self.node)

        if vertices is not None:
            self.set_vertices(vertices)
        if normals is not None:
            self.set_normals(normals)
        if colors is not None:
            # colors can be float32 (0..1) or uint8
            if colors.dtype == np.uint8:
                self.set_colors_uint8(colors)
            else:
                self.set_colors(colors.astype(np.float32, copy=False))
        if texcoords is not None:
            self.set_texcoords(texcoords)
        if vertices is not None and indices is not None:
            self.set_indices(indices)

    # --- setters
    def set_vertices(self, vertices: np.ndarray) -> None:
        verts = _ensure_float32(vertices, 3)
        n = verts.shape[0]
        if self.vdata.getNumRows() != n:
            self.vdata.setNumRows(n)
        arr = self.vdata.modifyArray(0)
        handle = arr.modifyHandle()
        handle.setData(verts.tobytes())

    def set_normals(self, normals: np.ndarray) -> None:
        norms = _ensure_float32(normals, 3)
        arr = self.vdata.modifyArray(1)
        handle = arr.modifyHandle()
        handle.setData(norms.tobytes())

    def set_colors(self, colors: np.ndarray) -> None:
        cols = _ensure_float32(colors, 4)
        arr = self.vdata.modifyArray(2)
        handle = arr.modifyHandle()
        handle.setData(cols.tobytes())

    def set_colors_uint8(self, colors: np.ndarray) -> None:
        cols = _ensure_uint8(colors, 4)
        # convert to float32 0..1 for the buffer
        cols_f = (cols.astype(np.float32) / 255.0).astype(np.float32)
        self.set_colors(cols_f)

    def set_texcoords(self, texcoords: np.ndarray) -> None:
        uvs = _ensure_float32(texcoords, 2)
        arr = self.vdata.modifyArray(3)
        handle = arr.modifyHandle()
        handle.setData(uvs.tobytes())

    def set_indices(self, indices: np.ndarray) -> None:
        inds = np.ascontiguousarray(indices, dtype=np.int32)
        if inds.ndim == 2:
            inds = inds.reshape(-1)
        if inds.ndim != 1:
            raise ValueError(f"Indices must be a 1D or 2D array, got shape {inds.shape}")
        if inds.size % 3 != 0:
            raise ValueError(
                f"Triangle index buffer must be a multiple of 3 elements, got {inds.size}"
            )
        if inds.size == 0:
            self.prim.clearVertices()
            return

        min_idx = int(inds.min())
        if min_idx < 0:
            raise ValueError("Triangle indices must be non-negative")

        vert_count = self.vdata.getNumRows()
        max_idx = int(inds.max())
        if vert_count > 0 and max_idx >= vert_count:
            raise ValueError(
                f"Triangle index {max_idx} exceeds vertex count {vert_count - 1}"
            )

        prim = self.prim
        prim.clearVertices()
        prim.reserveNumVertices(inds.size)

        # Choose 16-bit or 32-bit index type depending on the range we need.
        try:
            if max_idx < 65536:
                prim.setIndexType(Geom.NTUint16)
                inds_bytes = inds.astype(np.uint16, copy=False).tobytes()
            else:
                prim.setIndexType(Geom.NTUint32)
                inds_bytes = inds.tobytes()
        except Exception:
            inds_bytes = inds.tobytes()

        try:
            handle = prim.modifyVertices().modifyHandle()
            handle.setData(inds_bytes)
            prim.closePrimitive()
        except Exception:
            # Fallback to the safer (but slower) per-triangle insertion path.
            prim.clearVertices()
            for i in range(0, inds.size, 3):
                a, b, c = int(inds[i]), int(inds[i + 1]), int(inds[i + 2])
                prim.addVertices(a, b, c)
            prim.closePrimitive()

    # --- scene helpers
    def reparent_to(self, parent: NodePath) -> None:
        self.node_path.reparentTo(parent)

    def set_two_sided(self, two_sided: bool = True) -> None:
        self.node.setTwoSided(two_sided)
