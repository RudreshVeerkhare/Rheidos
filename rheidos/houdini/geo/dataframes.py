"""Pandas DataFrame export helpers for Houdini geometry."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import hou
    import pandas as pd


_COMPONENT_SUFFIXES = ("x", "y", "z", "w")
_PANDAS_ERROR = (
    "GeometryIO.to_dataframes() requires pandas. Install pandas into the active "
    "Houdini Python environment to use this helper."
)


def geometry_to_dataframes(
    geo: "hou.Geometry",
    *,
    include_prim_intrinsics: bool = False,
) -> "dict[str, pd.DataFrame]":
    """Return Houdini spreadsheet tabs as pandas DataFrames."""
    pd = _import_pandas()
    return {
        "points": pd.DataFrame(_point_rows(geo)),
        "primitives": pd.DataFrame(
            _primitive_rows(geo, include_prim_intrinsics=include_prim_intrinsics)
        ),
        "vertices": pd.DataFrame(_vertex_rows(geo)),
        "detail": pd.DataFrame([_detail_row(geo)]),
    }


def _import_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError(_PANDAS_ERROR) from exc
    return pd


def _point_rows(geo: "hou.Geometry") -> list[dict[str, Any]]:
    attribs = geo.pointAttribs()
    rows = []
    for point in geo.points():
        row = {"ptnum": point.number()}
        row.update(_attribs_to_row(point, attribs))
        rows.append(row)
    return rows


def _primitive_rows(
    geo: "hou.Geometry",
    *,
    include_prim_intrinsics: bool,
) -> list[dict[str, Any]]:
    attribs = geo.primAttribs()
    rows = []
    for prim in geo.prims():
        vertices = prim.vertices()
        row = {
            "primnum": prim.number(),
            "type": _type_name(prim.type()),
            "num_vertices": len(vertices),
        }
        row.update(_attribs_to_row(prim, attribs))
        if include_prim_intrinsics:
            row.update(_primitive_intrinsics(prim))
        rows.append(row)
    return rows


def _vertex_rows(geo: "hou.Geometry") -> list[dict[str, Any]]:
    attribs = geo.vertexAttribs()
    rows = []
    linear_vtxnum = 0
    for prim in geo.prims():
        for local_vtxnum, vertex in enumerate(prim.vertices()):
            row = {
                "vtxnum": linear_vtxnum,
                "primnum": prim.number(),
                "prim_vtxnum": local_vtxnum,
                "ptnum": vertex.point().number(),
            }
            row.update(_attribs_to_row(vertex, attribs))
            rows.append(row)
            linear_vtxnum += 1
    return rows


def _detail_row(geo: "hou.Geometry") -> dict[str, Any]:
    row = {}
    for attrib in geo.globalAttribs():
        name = attrib.name()
        row.update(_flatten_value(name, _attrib_value(geo, attrib)))
    return row


def _attribs_to_row(element: Any, attribs: Any) -> dict[str, Any]:
    row = {}
    for attrib in attribs:
        name = attrib.name()
        row.update(_flatten_value(name, _attrib_value(element, attrib)))
    return row


def _attrib_value(element: Any, attrib: Any) -> Any:
    name = attrib.name()
    try:
        return element.attribValue(name)
    except Exception as name_exc:
        try:
            return element.attribValue(attrib)
        except Exception:
            raise name_exc


def _primitive_intrinsics(prim: "hou.Prim") -> dict[str, Any]:
    row = {}
    for intrinsic_name in prim.intrinsicNames():
        try:
            value = prim.intrinsicValue(intrinsic_name)
        except Exception:
            continue
        row.update(_flatten_value(f"intrinsic_{intrinsic_name}", value))
    return row


def _flatten_value(name: str, value: Any) -> dict[str, Any]:
    if not _is_sequence(value):
        return {name: value}

    row = {}
    for index, item in enumerate(list(value)):
        suffix = _COMPONENT_SUFFIXES[index] if index < len(_COMPONENT_SUFFIXES) else str(index)
        row[f"{name}_{suffix}"] = item
    return row


def _is_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _type_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if callable(name):
        try:
            return str(name())
        except Exception:
            pass
    if name is not None:
        return str(name)
    return str(value)
