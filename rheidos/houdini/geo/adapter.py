"""Generic Houdini geometry IO adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from .schema import (
    AttribDesc,
    GeometrySchema,
    OWNER_DETAIL,
    OWNER_POINT,
    OWNER_PRIM,
    OWNER_VERTEX,
    OWNERS,
)

if TYPE_CHECKING:
    import hou
    import pandas as pd

_CacheKey = Tuple[str, str, Optional[int], Optional[str]]


def _get_hou() -> "hou":
    try:
        import hou  # type: ignore
    except Exception as exc:  # pragma: no cover - only runs in Houdini
        raise RuntimeError("Houdini 'hou' module not available") from exc
    return hou


def _enum_name(value: Any) -> str:
    if value is None:
        return "unknown"
    name = getattr(value, "name", None)
    if callable(name):
        try:
            return str(name())
        except Exception:
            pass
    if name is not None:
        return str(name)
    return str(value)


def _attrib_storage_type(attrib: "hou.Attrib") -> str:
    for attr_name in ("storageType", "dataType"):
        getter = getattr(attrib, attr_name, None)
        if getter is None:
            continue
        try:
            value = getter()
        except TypeError:
            value = getter
        if value is not None:
            return _enum_name(value)
    return "unknown"


def _validate_owner(owner: str) -> str:
    if owner not in OWNERS:
        raise ValueError(f"Unknown owner '{owner}'")
    return owner


def _attribs_for_owner(geo: "hou.Geometry", owner: str) -> Sequence["hou.Attrib"]:
    if owner == OWNER_POINT:
        return geo.pointAttribs()
    if owner == OWNER_PRIM:
        return geo.primAttribs()
    if owner == OWNER_VERTEX:
        return geo.vertexAttribs()
    if owner == OWNER_DETAIL:
        return geo.globalAttribs()
    raise ValueError(f"Unknown owner '{owner}'")


def _find_attrib(geo: "hou.Geometry", owner: str, name: str) -> Optional["hou.Attrib"]:
    if owner == OWNER_POINT:
        return geo.findPointAttrib(name)
    if owner == OWNER_PRIM:
        return geo.findPrimAttrib(name)
    if owner == OWNER_VERTEX:
        return geo.findVertexAttrib(name)
    if owner == OWNER_DETAIL:
        return geo.findGlobalAttrib(name)
    raise ValueError(f"Unknown owner '{owner}'")


def _attrib_type_for_owner(owner: str) -> "hou.attribType":
    hou = _get_hou()
    if owner == OWNER_POINT:
        return hou.attribType.Point
    if owner == OWNER_PRIM:
        return hou.attribType.Prim
    if owner == OWNER_VERTEX:
        return hou.attribType.Vertex
    if owner == OWNER_DETAIL:
        return hou.attribType.Global
    raise ValueError(f"Unknown owner '{owner}'")


def _owner_count(geo: "hou.Geometry", owner: str) -> int:
    if owner == OWNER_POINT:
        return len(geo.points())
    if owner == OWNER_PRIM:
        return len(geo.prims())
    if owner == OWNER_VERTEX:
        return len(geo.vertices())
    if owner == OWNER_DETAIL:
        return 1
    raise ValueError(f"Unknown owner '{owner}'")


def _attrib_kind(attrib: "hou.Attrib") -> str:
    hou = _get_hou()
    data_type = attrib.dataType()
    if data_type == hou.attribData.Float:
        return "float"
    if data_type == hou.attribData.Int:
        return "int"
    if data_type == hou.attribData.String:
        return "string"
    return "unsupported"


def _values_kind(values: np.ndarray) -> str:
    if values.dtype.kind in ("U", "S", "O"):
        return "string"
    if values.dtype.kind in ("i", "u", "b"):
        return "int"
    return "float"


def _normalize_values(owner: str, values: Any) -> np.ndarray:
    values_np = np.asarray(values)
    if owner == OWNER_DETAIL:
        if values_np.ndim == 0:
            return values_np.reshape((1, 1))
        if values_np.ndim == 1:
            return values_np.reshape((1, values_np.shape[0]))
        if values_np.ndim == 2 and values_np.shape[0] == 1:
            return values_np
        raise ValueError(
            "Detail attributes must be scalar, (tuple_size,), or (1, tuple_size)"
        )

    if values_np.ndim == 1:
        return values_np.reshape((values_np.shape[0], 1))
    if values_np.ndim == 2:
        return values_np
    raise ValueError("Attribute values must be 1D or 2D")


@dataclass
class GeometryIO:
    """Geometry adapter that reads from `geo_in` and writes to `geo_out`.

    Prefer the owner-specific helpers such as `read_point()` and
    `write_prim()` for most callers. The generic `read()` and `write()`
    methods remain available for compatibility and dynamic owner cases.
    """

    geo_in: "hou.Geometry"
    geo_out: Optional["hou.Geometry"] = None

    def __post_init__(self) -> None:
        if self.geo_out is None:
            self.geo_out = self.geo_in
        self._cache: Dict[_CacheKey, np.ndarray] = {}
        self._schema_cache: Optional[GeometrySchema] = None

    def clear_cache(self) -> None:
        self._cache.clear()
        self._schema_cache = None

    def _require_output_geo(self) -> "hou.Geometry":
        if self.geo_out is None:
            raise RuntimeError("GeometryIO has no output geometry to write")
        return self.geo_out

    def describe(self, owner: Optional[str] = None) -> GeometrySchema:
        if owner is None and self._schema_cache is not None:
            return self._schema_cache

        owners = OWNERS if owner is None else (_validate_owner(owner),)
        descs: Dict[str, Tuple[AttribDesc, ...]] = {}
        for own in owners:
            attribs = []
            for attrib in _attribs_for_owner(self.geo_in, own):
                attribs.append(
                    AttribDesc(
                        name=attrib.name(),
                        owner=own,
                        storage_type=_attrib_storage_type(attrib),
                        tuple_size=attrib.size(),
                    )
                )
            descs[own] = tuple(attribs)

        schema = GeometrySchema(
            point=descs.get(OWNER_POINT, ()),
            prim=descs.get(OWNER_PRIM, ()),
            vertex=descs.get(OWNER_VERTEX, ()),
            detail=descs.get(OWNER_DETAIL, ()),
        )

        if owner is None:
            self._schema_cache = schema
        return schema

    def to_dataframes(
        self,
        *,
        include_prim_intrinsics: bool = False,
    ) -> "dict[str, pd.DataFrame]":
        """Return spreadsheet-style pandas DataFrames for `geo_in`."""
        from .dataframes import geometry_to_dataframes

        return geometry_to_dataframes(
            self.geo_in,
            include_prim_intrinsics=include_prim_intrinsics,
        )

    def clear_output(self) -> None:
        """Clear all topology and attributes from the output geometry."""

        geo = self._require_output_geo()
        geo.clear()
        self.clear_cache()

    def create_point(self, position: Any = None) -> "hou.Point":
        """Create one output point, optionally setting its 3D position."""

        geo = self._require_output_geo()
        point = geo.createPoint()
        if position is not None:
            point.setPosition(position)
        self.clear_cache()
        return point

    def create_points(self, positions: Any) -> tuple["hou.Point", ...]:
        """Create output points from an ``(N, 3)`` array-like of positions."""

        geo = self._require_output_geo()
        positions_np = np.asarray(positions)
        if positions_np.ndim != 2 or positions_np.shape[1] != 3:
            raise ValueError(f"positions must have shape (N,3), got {positions_np.shape}")
        points = tuple(geo.createPoints(tuple(map(tuple, positions_np.tolist()))))
        self.clear_cache()
        return points

    def _point_from_ref(self, point_ref: Any) -> "hou.Point":
        if isinstance(point_ref, (int, np.integer)):
            point_number = int(point_ref)
            if point_number < 0:
                raise ValueError(f"Point number {point_number} does not exist")
            try:
                return self._require_output_geo().points()[point_number]
            except IndexError as exc:
                raise ValueError(f"Point number {point_number} does not exist") from exc
        return point_ref

    def add_vertices(
        self,
        prim: "hou.Polygon",
        points: Iterable[Any],
    ) -> tuple["hou.Vertex", ...]:
        """Append vertices to ``prim`` from point objects or point numbers."""

        vertices = tuple(prim.addVertex(self._point_from_ref(point)) for point in points)
        self.clear_cache()
        return vertices

    def create_polygon(
        self,
        points: Iterable[Any],
        *,
        closed: bool = True,
    ) -> "hou.Polygon":
        """Create a polygon face or polygon curve from point objects or numbers."""

        geo = self._require_output_geo()
        prim = geo.createPolygon(is_closed=bool(closed))
        self.add_vertices(prim, points)
        self.clear_cache()
        return prim

    def create_polygons(
        self,
        polygons: Iterable[Iterable[Any]],
        *,
        closed: bool = True,
    ) -> tuple["hou.Polygon", ...]:
        """Create multiple polygon faces or curves from point objects or numbers."""

        return tuple(
            self.create_polygon(points, closed=closed)
            for points in polygons
        )

    def _read_attrib(self, owner: str, name: str) -> np.ndarray:
        owner = _validate_owner(owner)
        attrib = _find_attrib(self.geo_in, owner, name)
        if attrib is None:
            raise KeyError(f"Missing {owner} attribute '{name}'")

        tuple_size = attrib.size()
        kind = _attrib_kind(attrib)
        if kind == "unsupported":
            raise TypeError(f"Unsupported attribute data type for '{name}'")

        if owner == OWNER_DETAIL:
            value = self.geo_in.attribValue(name)
            if tuple_size == 1:
                return np.asarray([value])
            if isinstance(value, (tuple, list, np.ndarray)):
                data = np.asarray(value)
                if data.shape[0] != tuple_size:
                    raise ValueError(
                        f"Detail attrib '{name}' expected tuple size {tuple_size}"
                    )
                return np.asarray([data])
            raise ValueError(f"Detail attrib '{name}' expected tuple size {tuple_size}")

        if kind == "float":
            values = _read_float_attrib(self.geo_in, owner, name)
        elif kind == "int":
            values = _read_int_attrib(self.geo_in, owner, name)
        else:
            values = _read_string_attrib(self.geo_in, owner, name)

        arr = np.asarray(values)
        if tuple_size > 1:
            arr = arr.reshape((-1, tuple_size))
        return arr

    def read(
        self,
        owner: str,
        name: str,
        *,
        dtype: Optional[Any] = None,
        components: Optional[int] = None,
    ) -> np.ndarray:
        cache_key = (owner, name, components, str(dtype) if dtype is not None else None)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        arr = self._read_attrib(owner, name)
        tuple_size = arr.shape[1] if arr.ndim == 2 else 1
        if components is not None and int(components) != int(tuple_size):
            raise ValueError(
                f"Attrib '{name}' expected tuple size {components}, got {tuple_size}"
            )
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        self._cache[cache_key] = arr
        return arr

    def read_point(
        self,
        name: str,
        *,
        dtype: Optional[Any] = None,
        components: Optional[int] = None,
    ) -> np.ndarray:
        """Read a point attribute from `geo_in`."""
        return self.read(OWNER_POINT, name, dtype=dtype, components=components)

    def read_prim(
        self,
        name: str,
        *,
        dtype: Optional[Any] = None,
        components: Optional[int] = None,
    ) -> np.ndarray:
        """Read a primitive attribute from `geo_in`."""
        return self.read(OWNER_PRIM, name, dtype=dtype, components=components)

    def read_vertex(
        self,
        name: str,
        *,
        dtype: Optional[Any] = None,
        components: Optional[int] = None,
    ) -> np.ndarray:
        """Read a vertex attribute from `geo_in`."""
        return self.read(OWNER_VERTEX, name, dtype=dtype, components=components)

    def read_detail(
        self,
        name: str,
        *,
        dtype: Optional[Any] = None,
        components: Optional[int] = None,
    ) -> np.ndarray:
        """Read a detail attribute from `geo_in`."""
        return self.read(OWNER_DETAIL, name, dtype=dtype, components=components)

    def write(self, owner: str, name: str, values: Any, *, create: bool = True) -> None:
        owner = _validate_owner(owner)
        geo_out = self._require_output_geo()

        values_np = _normalize_values(owner, values)
        tuple_size = values_np.shape[1]
        count_expected = _owner_count(geo_out, owner)
        if owner != OWNER_DETAIL and values_np.shape[0] != count_expected:
            raise ValueError(
                f"{owner} attrib '{name}' expects {count_expected} elements, got {values_np.shape[0]}"
            )

        attrib = _find_attrib(geo_out, owner, name)
        if attrib is None:
            if not create:
                raise KeyError(f"Missing {owner} attribute '{name}'")
            _create_attrib(
                geo_out, owner, name, tuple_size, _values_kind(values_np)
            )
            attrib = _find_attrib(geo_out, owner, name)
            if attrib is None:
                raise RuntimeError(f"Failed to create {owner} attribute '{name}'")

        if attrib.size() != tuple_size:
            raise ValueError(
                f"{owner} attrib '{name}' expected tuple size {attrib.size()}, got {tuple_size}"
            )

        kind = _attrib_kind(attrib)
        if kind == "string":
            _write_string_attrib(geo_out, owner, name, values_np)
        elif kind == "int":
            _write_int_attrib(geo_out, owner, name, values_np)
        elif kind == "float":
            _write_float_attrib(geo_out, owner, name, values_np)
        else:
            raise TypeError(f"Unsupported attribute data type for '{name}'")

    def write_point(
        self,
        name: str,
        values: Any,
        *,
        create: bool = True,
    ) -> None:
        """Write a point attribute to `geo_out`."""
        self.write(OWNER_POINT, name, values, create=create)

    def write_prim(
        self,
        name: str,
        values: Any,
        *,
        create: bool = True,
    ) -> None:
        """Write a primitive attribute to `geo_out`."""
        self.write(OWNER_PRIM, name, values, create=create)

    def write_vertex(
        self,
        name: str,
        values: Any,
        *,
        create: bool = True,
    ) -> None:
        """Write a vertex attribute to `geo_out`."""
        self.write(OWNER_VERTEX, name, values, create=create)

    def write_detail(
        self,
        name: str,
        values: Any,
        *,
        create: bool = True,
    ) -> None:
        """Write a detail attribute to `geo_out`."""
        self.write(OWNER_DETAIL, name, values, create=create)

    def read_prims(self, arity: int = 3) -> np.ndarray:
        hou = _get_hou()
        prims = self.geo_in.prims()
        if not prims:
            return np.zeros((0, arity), dtype=np.int64)

        indices: list[int] = []
        for prim in prims:
            if prim.type() != hou.primType.Polygon or prim.numVertices() != arity:
                raise ValueError(
                    f"Expected polygon prims with arity {arity}; found {prim.type()} with {prim.numVertices()}"
                )
            verts = prim.vertices()
            if len(verts) != arity:
                raise ValueError(f"Expected {arity} vertices, got {len(verts)}")
            indices.extend(v.point().number() for v in verts)

        return np.asarray(indices, dtype=np.int64).reshape((-1, arity))

    def read_group(
        self, owner: str, group_name: str, *, as_mask: bool = False
    ) -> np.ndarray:
        owner = _validate_owner(owner)
        group = _find_group(self.geo_in, owner, group_name)
        if group is None:
            raise KeyError(f"Missing {owner} group '{group_name}'")

        if owner == OWNER_POINT:
            items = group.points()
            indices = [pt.number() for pt in items]
        elif owner == OWNER_PRIM:
            items = group.prims()
            indices = [prim.number() for prim in items]
        elif owner == OWNER_VERTEX:
            items = group.vertices()
            indices = [_vertex_index(v) for v in items]
        else:
            raise ValueError(f"Groups are not supported for owner '{owner}'")

        count = _owner_count(self.geo_in, owner)
        if as_mask:
            mask = np.zeros((count,), dtype=bool)
            mask[indices] = True
            return mask
        return np.asarray(indices, dtype=np.int64)


def _read_float_attrib(geo: "hou.Geometry", owner: str, name: str) -> Sequence[float]:
    if owner == OWNER_POINT:
        return geo.pointFloatAttribValues(name)
    if owner == OWNER_PRIM:
        return geo.primFloatAttribValues(name)
    if owner == OWNER_VERTEX:
        return geo.vertexFloatAttribValues(name)
    raise ValueError(f"Float attribs are not supported for owner '{owner}'")


def _read_int_attrib(geo: "hou.Geometry", owner: str, name: str) -> Sequence[int]:
    if owner == OWNER_POINT:
        return geo.pointIntAttribValues(name)
    if owner == OWNER_PRIM:
        return geo.primIntAttribValues(name)
    if owner == OWNER_VERTEX:
        return geo.vertexIntAttribValues(name)
    raise ValueError(f"Int attribs are not supported for owner '{owner}'")


def _read_string_attrib(geo: "hou.Geometry", owner: str, name: str) -> Sequence[str]:
    if owner == OWNER_POINT:
        return geo.pointStringAttribValues(name)
    if owner == OWNER_PRIM:
        return geo.primStringAttribValues(name)
    if owner == OWNER_VERTEX:
        return geo.vertexStringAttribValues(name)
    raise ValueError(f"String attribs are not supported for owner '{owner}'")


def _write_float_attrib(
    geo: "hou.Geometry", owner: str, name: str, values: np.ndarray
) -> None:
    flat = values.reshape(-1).astype(float).tolist()
    if owner == OWNER_POINT:
        geo.setPointFloatAttribValues(name, flat)
        return
    if owner == OWNER_PRIM:
        geo.setPrimFloatAttribValues(name, flat)
        return
    if owner == OWNER_VERTEX:
        geo.setVertexFloatAttribValues(name, flat)
        return
    if owner == OWNER_DETAIL:
        geo.setGlobalAttribValue(name, _detail_value(values))
        return
    raise ValueError(f"Float attribs are not supported for owner '{owner}'")


def _write_int_attrib(
    geo: "hou.Geometry", owner: str, name: str, values: np.ndarray
) -> None:
    flat = values.reshape(-1).astype(int).tolist()
    if owner == OWNER_POINT:
        geo.setPointIntAttribValues(name, flat)
        return
    if owner == OWNER_PRIM:
        geo.setPrimIntAttribValues(name, flat)
        return
    if owner == OWNER_VERTEX:
        geo.setVertexIntAttribValues(name, flat)
        return
    if owner == OWNER_DETAIL:
        geo.setGlobalAttribValue(name, _detail_value(values))
        return
    raise ValueError(f"Int attribs are not supported for owner '{owner}'")


def _write_string_attrib(
    geo: "hou.Geometry", owner: str, name: str, values: np.ndarray
) -> None:
    if owner == OWNER_DETAIL:
        geo.setGlobalAttribValue(name, _detail_value(values))
        return
    flat = [str(v) for v in values.reshape(-1).tolist()]
    if owner == OWNER_POINT:
        geo.setPointStringAttribValues(name, flat)
        return
    if owner == OWNER_PRIM:
        geo.setPrimStringAttribValues(name, flat)
        return
    if owner == OWNER_VERTEX:
        geo.setVertexStringAttribValues(name, flat)
        return
    raise ValueError(f"String attribs are not supported for owner '{owner}'")


def _detail_value(values: np.ndarray) -> Any:
    values = values.reshape((-1,))
    if values.size == 1:
        value = values[0]
        if isinstance(value, np.generic):
            return value.item()
        return value
    return [
        value.item() if isinstance(value, np.generic) else value
        for value in values.tolist()
    ]


def _create_attrib(
    geo: "hou.Geometry",
    owner: str,
    name: str,
    tuple_size: int,
    kind: str,
) -> None:
    attrib_type = _attrib_type_for_owner(owner)
    if kind == "string":
        default = ""
    elif kind == "int":
        default = 0
    else:
        default = 0.0
    if tuple_size > 1:
        default = tuple([default] * tuple_size)
    geo.addAttrib(attrib_type, name, default)


def _find_group(geo: "hou.Geometry", owner: str, name: str) -> Optional["hou.Group"]:
    if owner == OWNER_POINT:
        return geo.findPointGroup(name)
    if owner == OWNER_PRIM:
        return geo.findPrimGroup(name)
    if owner == OWNER_VERTEX:
        return geo.findVertexGroup(name)
    return None


def _vertex_index(vertex: Any) -> int:
    if hasattr(vertex, "number"):
        return int(vertex.number())
    if hasattr(vertex, "index"):
        return int(vertex.index())
    return int(vertex.point().number())
