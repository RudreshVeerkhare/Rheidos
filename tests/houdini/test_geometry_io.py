from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import rheidos.houdini.geo.adapter as adapter_mod
from rheidos.houdini.geo import OWNER_DETAIL, OWNER_POINT, OWNER_PRIM, OWNER_VERTEX
from rheidos.houdini.geo.adapter import GeometryIO
from rheidos.houdini.runtime.cook_context import CookContext


class _FakeAttrib:
    def __init__(self, name: str, tuple_size: int, data_type: str) -> None:
        self._name = name
        self._tuple_size = int(tuple_size)
        self._data_type = data_type

    def name(self) -> str:
        return self._name

    def size(self) -> int:
        return self._tuple_size

    def dataType(self) -> str:
        return self._data_type


def _attrib_name(name_or_attrib):
    name = getattr(name_or_attrib, "name", None)
    if callable(name):
        return name()
    return name_or_attrib


class _FakePoint:
    def __init__(self, geo, number: int) -> None:
        self._geo = geo
        self._number = int(number)

    def number(self) -> int:
        return self._number

    def position(self):
        return tuple(self._geo._point_positions[self._number])

    def setPosition(self, position) -> None:
        self._geo._point_positions[self._number] = tuple(position)

    def attribValue(self, name_or_attrib):
        return self._geo._element_attrib_value(OWNER_POINT, self._number, name_or_attrib)


class _FakePrimType:
    def __init__(self, name: str = "Polygon") -> None:
        self._name = name

    def name(self) -> str:
        return self._name


class _FakeVertex:
    def __init__(
        self,
        geo,
        number: int,
        prim_number,
        local_number: int,
        point_number: int,
    ) -> None:
        self._geo = geo
        self._number = int(number)
        self._prim_number = prim_number
        self._local_number = int(local_number)
        self._point_number = int(point_number)

    def number(self) -> int:
        return self._number

    def point(self) -> _FakePoint:
        return _FakePoint(self._geo, self._point_number)

    def attribValue(self, name_or_attrib):
        return self._geo._element_attrib_value(OWNER_VERTEX, self._number, name_or_attrib)


class _FakePrim:
    def __init__(self, geo, number: int) -> None:
        self._geo = geo
        self._number = int(number)

    def number(self) -> int:
        return self._number

    def type(self) -> _FakePrimType:
        return _FakePrimType()

    def vertices(self):
        if self._geo._prim_points is None:
            return []
        start = self._geo._prim_vertex_start(self._number)
        return [
            _FakeVertex(self._geo, start + local, self._number, local, point_number)
            for local, point_number in enumerate(self._geo._prim_points[self._number])
        ]

    def numVertices(self) -> int:
        return len(self.vertices())

    def isClosed(self) -> bool:
        return bool(self._geo._prim_closed[self._number])

    def addVertex(self, point):
        if self._geo._prim_points is None:
            self._geo._prim_points = [[] for _ in range(self._geo._counts[OWNER_PRIM])]
        point_number = int(point.number())
        self._geo._prim_points[self._number].append(point_number)
        vertex_number = self._geo._counts[OWNER_VERTEX]
        self._geo._counts[OWNER_VERTEX] += 1
        self._geo._append_default_values(OWNER_VERTEX, 1)
        return _FakeVertex(
            self._geo,
            vertex_number,
            self._number,
            len(self._geo._prim_points[self._number]) - 1,
            point_number,
        )

    def attribValue(self, name_or_attrib):
        return self._geo._element_attrib_value(OWNER_PRIM, self._number, name_or_attrib)

    def intrinsicNames(self):
        return list(self._geo._prim_intrinsics[self._number].keys())

    def intrinsicValue(self, name: str):
        value = self._geo._prim_intrinsics[self._number][name]
        if isinstance(value, Exception):
            raise value
        return value


class _FakeGeometry:
    def __init__(
        self,
        *,
        point_count: int = 0,
        prim_count: int = 0,
        vertex_count: int = 0,
        prim_points=None,
    ) -> None:
        if prim_points is not None:
            prim_points = [list(points) for points in prim_points]
            prim_count = len(prim_points)
            vertex_count = sum(len(points) for points in prim_points)

        self._counts = {
            OWNER_POINT: int(point_count),
            OWNER_PRIM: int(prim_count),
            OWNER_VERTEX: int(vertex_count),
            OWNER_DETAIL: 1,
        }
        self._attribs = {
            OWNER_POINT: {},
            OWNER_PRIM: {},
            OWNER_VERTEX: {},
            OWNER_DETAIL: {},
        }
        self._prim_points = prim_points
        self._prim_closed = [True] * int(prim_count)
        self._prim_intrinsics = [{} for _ in range(int(prim_count))]
        self._point_positions = [(0.0, 0.0, 0.0)] * int(point_count)
        self._attrib_defaults = {
            OWNER_POINT: {},
            OWNER_PRIM: {},
            OWNER_VERTEX: {},
            OWNER_DETAIL: {},
        }
        self._values = {
            OWNER_POINT: {},
            OWNER_PRIM: {},
            OWNER_VERTEX: {},
            OWNER_DETAIL: {},
        }

    def clear(self) -> None:
        self._counts = {
            OWNER_POINT: 0,
            OWNER_PRIM: 0,
            OWNER_VERTEX: 0,
            OWNER_DETAIL: 1,
        }
        self._attribs = {
            OWNER_POINT: {},
            OWNER_PRIM: {},
            OWNER_VERTEX: {},
            OWNER_DETAIL: {},
        }
        self._attrib_defaults = {
            OWNER_POINT: {},
            OWNER_PRIM: {},
            OWNER_VERTEX: {},
            OWNER_DETAIL: {},
        }
        self._values = {
            OWNER_POINT: {},
            OWNER_PRIM: {},
            OWNER_VERTEX: {},
            OWNER_DETAIL: {},
        }
        self._prim_points = []
        self._prim_closed = []
        self._prim_intrinsics = []
        self._point_positions = []

    def pointAttribs(self):
        return list(self._attribs[OWNER_POINT].values())

    def primAttribs(self):
        return list(self._attribs[OWNER_PRIM].values())

    def vertexAttribs(self):
        return list(self._attribs[OWNER_VERTEX].values())

    def globalAttribs(self):
        return list(self._attribs[OWNER_DETAIL].values())

    def findPointAttrib(self, name: str):
        return self._attribs[OWNER_POINT].get(name)

    def findPrimAttrib(self, name: str):
        return self._attribs[OWNER_PRIM].get(name)

    def findVertexAttrib(self, name: str):
        return self._attribs[OWNER_VERTEX].get(name)

    def findGlobalAttrib(self, name: str):
        return self._attribs[OWNER_DETAIL].get(name)

    def points(self):
        return [_FakePoint(self, number) for number in range(self._counts[OWNER_POINT])]

    def prims(self):
        return [_FakePrim(self, number) for number in range(self._counts[OWNER_PRIM])]

    def vertices(self):
        if self._prim_points is None:
            return [
                _FakeVertex(self, number, None, number, min(number, self._counts[OWNER_POINT] - 1))
                for number in range(self._counts[OWNER_VERTEX])
            ]
        return [vertex for prim in self.prims() for vertex in prim.vertices()]

    def addAttrib(self, attrib_type: str, name: str, default):
        tuple_size = len(default) if isinstance(default, tuple) else 1
        self._attribs[attrib_type][name] = _FakeAttrib(
            name=name,
            tuple_size=tuple_size,
            data_type=self._data_type_for_default(default),
        )
        self._attrib_defaults[attrib_type][name] = default
        if attrib_type == OWNER_DETAIL:
            self._values[OWNER_DETAIL][name] = default
            return
        flat_default = list(default) if isinstance(default, tuple) else [default]
        self._values[attrib_type][name] = flat_default * self._counts[attrib_type]

    def _append_default_values(self, owner: str, count: int) -> None:
        for name, default in self._attrib_defaults[owner].items():
            flat_default = list(default) if isinstance(default, tuple) else [default]
            self._values[owner][name].extend(flat_default * int(count))

    def createPoint(self):
        number = self._counts[OWNER_POINT]
        self._counts[OWNER_POINT] += 1
        self._point_positions.append((0.0, 0.0, 0.0))
        self._append_default_values(OWNER_POINT, 1)
        return _FakePoint(self, number)

    def createPoints(self, positions):
        points = []
        for position in positions:
            point = self.createPoint()
            point.setPosition(position)
            points.append(point)
        return tuple(points)

    def createPolygon(self, *, is_closed=True):
        if self._prim_points is None:
            self._prim_points = [[] for _ in range(self._counts[OWNER_PRIM])]
        number = self._counts[OWNER_PRIM]
        self._counts[OWNER_PRIM] += 1
        self._prim_points.append([])
        self._prim_closed.append(bool(is_closed))
        self._prim_intrinsics.append({})
        self._append_default_values(OWNER_PRIM, 1)
        return _FakePrim(self, number)

    def pointFloatAttribValues(self, name: str):
        return list(self._values[OWNER_POINT][name])

    def primFloatAttribValues(self, name: str):
        return list(self._values[OWNER_PRIM][name])

    def vertexFloatAttribValues(self, name: str):
        return list(self._values[OWNER_VERTEX][name])

    def pointIntAttribValues(self, name: str):
        return list(self._values[OWNER_POINT][name])

    def primIntAttribValues(self, name: str):
        return list(self._values[OWNER_PRIM][name])

    def vertexIntAttribValues(self, name: str):
        return list(self._values[OWNER_VERTEX][name])

    def pointStringAttribValues(self, name: str):
        return list(self._values[OWNER_POINT][name])

    def primStringAttribValues(self, name: str):
        return list(self._values[OWNER_PRIM][name])

    def vertexStringAttribValues(self, name: str):
        return list(self._values[OWNER_VERTEX][name])

    def setPointFloatAttribValues(self, name: str, values) -> None:
        self._values[OWNER_POINT][name] = list(values)

    def setPrimFloatAttribValues(self, name: str, values) -> None:
        self._values[OWNER_PRIM][name] = list(values)

    def setVertexFloatAttribValues(self, name: str, values) -> None:
        self._values[OWNER_VERTEX][name] = list(values)

    def setPointIntAttribValues(self, name: str, values) -> None:
        self._values[OWNER_POINT][name] = list(values)

    def setPrimIntAttribValues(self, name: str, values) -> None:
        self._values[OWNER_PRIM][name] = list(values)

    def setVertexIntAttribValues(self, name: str, values) -> None:
        self._values[OWNER_VERTEX][name] = list(values)

    def setPointStringAttribValues(self, name: str, values) -> None:
        self._values[OWNER_POINT][name] = list(values)

    def setPrimStringAttribValues(self, name: str, values) -> None:
        self._values[OWNER_PRIM][name] = list(values)

    def setVertexStringAttribValues(self, name: str, values) -> None:
        self._values[OWNER_VERTEX][name] = list(values)

    def attribValue(self, name: str):
        name = _attrib_name(name)
        return self._values[OWNER_DETAIL][name]

    def setGlobalAttribValue(self, name: str, value) -> None:
        self._values[OWNER_DETAIL][name] = value

    def setPrimIntrinsic(self, prim_number: int, name: str, value) -> None:
        self._prim_intrinsics[int(prim_number)][name] = value

    def _element_attrib_value(self, owner: str, index: int, name_or_attrib):
        name = _attrib_name(name_or_attrib)
        attrib = self._attribs[owner][name]
        values = self._values[owner][name]
        tuple_size = attrib.size()
        if tuple_size == 1:
            return values[index]
        start = index * tuple_size
        return tuple(values[start : start + tuple_size])

    def _prim_vertex_start(self, prim_number: int) -> int:
        if self._prim_points is None:
            return 0
        return sum(len(points) for points in self._prim_points[:prim_number])

    @staticmethod
    def _data_type_for_default(default) -> str:
        sample = default[0] if isinstance(default, tuple) else default
        if isinstance(sample, str):
            return OWNER_DETAIL + ".string"
        if isinstance(sample, (bool, int, np.integer)):
            return OWNER_DETAIL + ".int"
        return OWNER_DETAIL + ".float"


class _FakeDataFrame:
    def __init__(self, rows) -> None:
        self._records = [dict(row) for row in rows]
        columns = []
        for record in self._records:
            for key in record:
                if key not in columns:
                    columns.append(key)
        self.columns = columns
        self.shape = (len(self._records), len(columns))

    def to_dict(self, orient: str = "dict"):
        if orient != "records":
            raise ValueError("Only records orient is supported by the test fake")
        return [dict(row) for row in self._records]


@pytest.fixture
def fake_pandas(monkeypatch):
    module = ModuleType("pandas")
    module.DataFrame = _FakeDataFrame
    monkeypatch.setitem(sys.modules, "pandas", module)
    return module


@pytest.fixture(autouse=True)
def _fake_hou(monkeypatch):
    fake_hou = SimpleNamespace(
        attribType=SimpleNamespace(
            Point=OWNER_POINT,
            Prim=OWNER_PRIM,
            Vertex=OWNER_VERTEX,
            Global=OWNER_DETAIL,
        ),
        attribData=SimpleNamespace(
            Float=OWNER_DETAIL + ".float",
            Int=OWNER_DETAIL + ".int",
            String=OWNER_DETAIL + ".string",
        ),
    )
    monkeypatch.setattr(adapter_mod, "_get_hou", lambda: fake_hou)
    return fake_hou


def _make_ctx(io) -> CookContext:
    return CookContext(
        node=SimpleNamespace(path=lambda: "/obj/test"),
        frame=1.0,
        time=0.0,
        dt=1.0 / 24.0,
        substep=0,
        is_solver=False,
        session=SimpleNamespace(log_event=lambda *args, **kwargs: None),
        prof=None,
        geo_in=None,
        geo_out=None,
        io=io,
    )


def test_write_point_matches_generic_owner_api():
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    geo_owner_specific = _FakeGeometry(point_count=2)
    geo_generic = _FakeGeometry(point_count=2)

    io_owner_specific = GeometryIO(geo_owner_specific, geo_owner_specific)
    io_generic = GeometryIO(geo_generic, geo_generic)

    io_owner_specific.write_point("vel", values, create=True)
    io_generic.write(OWNER_POINT, "vel", values, create=True)

    assert geo_owner_specific.findPointAttrib("vel") is not None
    assert geo_owner_specific.findPointAttrib("vel").size() == 2
    assert geo_owner_specific.pointFloatAttribValues("vel") == geo_generic.pointFloatAttribValues("vel")


def test_write_prim_and_detail_preserve_validation():
    geo = _FakeGeometry(prim_count=2)
    io = GeometryIO(geo, geo)

    io.write_prim("id", np.array([10, 20], dtype=np.int32), create=True)
    assert geo.primIntAttribValues("id") == [10, 20]

    with pytest.raises(ValueError, match="expects 2 elements"):
        io.write_prim("id", np.array([10, 20, 30], dtype=np.int32), create=False)

    io.write_detail("strength", np.array([1.5], dtype=np.float32), create=True)
    assert geo.attribValue("strength") == pytest.approx(1.5)
    assert type(geo.attribValue("strength")) is float

    io.write_detail("genus", np.array([1], dtype=np.int32), create=True)
    assert geo.attribValue("genus") == 1
    assert type(geo.attribValue("genus")) is int

    io.write_detail("ids", np.array([1, 2], dtype=np.int32), create=True)
    assert geo.attribValue("ids") == [1, 2]
    assert all(type(value) is int for value in geo.attribValue("ids"))

    with pytest.raises(ValueError, match="expected tuple size 1, got 2"):
        io.write_detail("strength", np.array([1.0, 2.0], dtype=np.float32), create=False)


def test_read_point_and_detail_preserve_shape_semantics():
    geo = _FakeGeometry(point_count=2)
    io = GeometryIO(geo, geo)

    io.write_point("P", np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32), create=True)
    io.write_detail("mass", np.array([2.5], dtype=np.float32), create=True)

    points = io.read_point("P", components=3)
    detail = io.read_detail("mass")

    assert points.shape == (2, 3)
    assert np.allclose(points, np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32))
    assert detail.shape == (1,)
    assert np.allclose(detail, np.array([2.5], dtype=np.float32))


def test_write_wrapper_respects_read_only_io():
    geo = _FakeGeometry(point_count=1)
    io = GeometryIO(geo)
    io.geo_out = None

    with pytest.raises(RuntimeError, match="no output geometry"):
        io.write_point("P", np.array([[0.0, 0.0, 0.0]], dtype=np.float32), create=True)

    with pytest.raises(RuntimeError, match="no output geometry"):
        io.create_point((0.0, 0.0, 0.0))

    with pytest.raises(RuntimeError, match="no output geometry"):
        io.clear_output()


def test_geometry_io_creates_single_and_bulk_points():
    geo = _FakeGeometry()
    io = GeometryIO(geo, geo)

    point = io.create_point((1.0, 2.0, 3.0))
    points = io.create_points(
        np.array(
            [
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float32,
        )
    )

    assert point.number() == 0
    assert point.position() == (1.0, 2.0, 3.0)
    assert [pt.number() for pt in points] == [1, 2]
    assert [pt.position() for pt in points] == [
        (4.0, 5.0, 6.0),
        (7.0, 8.0, 9.0),
    ]
    assert len(geo.points()) == 3

    with pytest.raises(ValueError, match="positions must have shape"):
        io.create_points(np.array([1.0, 2.0, 3.0]))


def test_geometry_io_creates_open_and_closed_polygon_primitives():
    geo = _FakeGeometry()
    io = GeometryIO(geo, geo)
    points = io.create_points(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    open_line = io.create_polygon([points[0], points[1]], closed=False)
    closed_triangle = io.create_polygon([0, 1, 2], closed=True)

    assert open_line.number() == 0
    assert closed_triangle.number() == 1
    assert open_line.isClosed() is False
    assert closed_triangle.isClosed() is True
    assert [vertex.point().number() for vertex in open_line.vertices()] == [0, 1]
    assert [vertex.point().number() for vertex in closed_triangle.vertices()] == [0, 1, 2]
    assert len(geo.vertices()) == 5


def test_geometry_io_creates_multiple_polygons_and_returns_vertices():
    geo = _FakeGeometry()
    io = GeometryIO(geo, geo)
    io.create_points(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    prims = io.create_polygons([[0, 1], [2, 3]], closed=False)
    extra_vertices = io.add_vertices(prims[0], [2, 3])

    assert [prim.number() for prim in prims] == [0, 1]
    assert [prim.isClosed() for prim in prims] == [False, False]
    assert [vertex.point().number() for vertex in prims[0].vertices()] == [0, 1, 2, 3]
    assert [vertex.number() for vertex in extra_vertices] == [4, 5]

    with pytest.raises(ValueError, match="Point number 10 does not exist"):
        io.create_polygon([0, 10], closed=False)


def test_geometry_io_clear_output_removes_topology_and_attributes():
    geo = _FakeGeometry()
    io = GeometryIO(geo, geo)
    io.create_points(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32))
    io.create_polygon([0, 1], closed=False)
    io.write_point("id", np.array([1, 2], dtype=np.int32), create=True)

    io.clear_output()

    assert geo.points() == []
    assert geo.prims() == []
    assert geo.vertices() == []
    assert geo.findPointAttrib("id") is None


def test_geometry_io_to_dataframes_exports_spreadsheet_tables(fake_pandas):
    geo = _FakeGeometry(point_count=3, prim_points=((0, 1, 2), (2, 1, 0)))
    io = GeometryIO(geo, geo)

    io.write_point(
        "P",
        np.array(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
            ],
            dtype=np.float32,
        ),
        create=True,
    )
    io.write_point("label", np.array(["a", "b", "c"]), create=True)
    io.write_prim("uv", np.array([[0.25, 0.5], [0.75, 1.0]], dtype=np.float32), create=True)
    io.write_vertex("weight", np.arange(6, dtype=np.int32), create=True)
    io.write_detail("bbox", np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32), create=True)

    dfs = io.to_dataframes()

    assert set(dfs) == {"points", "primitives", "vertices", "detail"}
    assert dfs["points"].shape == (3, 5)
    assert dfs["primitives"].shape == (2, 5)
    assert dfs["vertices"].shape == (6, 5)
    assert dfs["detail"].shape == (1, 5)

    points = dfs["points"].to_dict("records")
    assert points[0] == {
        "ptnum": 0,
        "P_x": 0.0,
        "P_y": 1.0,
        "P_z": 2.0,
        "label": "a",
    }
    assert points[2]["ptnum"] == 2
    assert points[2]["P_z"] == 8.0

    primitives = dfs["primitives"].to_dict("records")
    assert primitives[0] == {
        "primnum": 0,
        "type": "Polygon",
        "num_vertices": 3,
        "uv_x": 0.25,
        "uv_y": 0.5,
    }

    vertices = dfs["vertices"].to_dict("records")
    assert vertices[3] == {
        "vtxnum": 3,
        "primnum": 1,
        "prim_vtxnum": 0,
        "ptnum": 2,
        "weight": 3,
    }

    detail = dfs["detail"].to_dict("records")
    assert detail == [
        {
            "bbox_x": 1.0,
            "bbox_y": 2.0,
            "bbox_z": 3.0,
            "bbox_w": 4.0,
            "bbox_4": 5.0,
        }
    ]


def test_geometry_io_to_dataframes_includes_requested_prim_intrinsics(fake_pandas):
    geo = _FakeGeometry(point_count=3, prim_points=((0, 1, 2),))
    geo.setPrimIntrinsic(0, "bounds", (1.0, 2.0, 3.0, 4.0, 5.0))
    geo.setPrimIntrinsic(0, "unreadable", RuntimeError("intrinsic read failed"))
    io = GeometryIO(geo, geo)

    without_intrinsics = io.to_dataframes()["primitives"].to_dict("records")[0]
    with_intrinsics = io.to_dataframes(include_prim_intrinsics=True)["primitives"].to_dict(
        "records"
    )[0]

    assert "intrinsic_bounds_x" not in without_intrinsics
    assert with_intrinsics["intrinsic_bounds_x"] == 1.0
    assert with_intrinsics["intrinsic_bounds_w"] == 4.0
    assert with_intrinsics["intrinsic_bounds_4"] == 5.0
    assert not any(key.startswith("intrinsic_unreadable") for key in with_intrinsics)


def test_geometry_io_to_dataframes_missing_pandas_raises_clear_error(monkeypatch):
    monkeypatch.delitem(sys.modules, "pandas", raising=False)
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(RuntimeError, match="Install pandas into the active Houdini Python"):
        GeometryIO(_FakeGeometry()).to_dataframes()


def test_cook_context_output_io_targets_output_geometry():
    geo_in = _FakeGeometry(point_count=1)
    geo_out = _FakeGeometry(point_count=2)
    io = GeometryIO(geo_in, geo_out)
    input_io = GeometryIO(geo_in)
    input_io.geo_out = None
    ctx = _make_ctx(io)
    ctx.geo_in = geo_in
    ctx.geo_out = geo_out
    ctx.io_inputs = (input_io,)

    output_io = ctx.output_io()

    assert output_io.geo_in is geo_out
    assert output_io.geo_out is geo_out
    assert ctx.input_io(0).geo_in is geo_in
    assert ctx.input_io(0).geo_out is None


def test_cook_context_owner_specific_methods_delegate_to_io():
    calls = []

    class _FakeIO:
        def read_point(self, name: str, *, dtype=None, components=None):
            calls.append(("read_point", name, dtype, components))
            return np.ones((1, components or 1), dtype=np.float32)

        def read_prim(self, name: str, *, dtype=None, components=None):
            calls.append(("read_prim", name, dtype, components))
            return np.array([1], dtype=np.int32)

        def read_vertex(self, name: str, *, dtype=None, components=None):
            calls.append(("read_vertex", name, dtype, components))
            return np.array([2], dtype=np.int32)

        def read_detail(self, name: str, *, dtype=None, components=None):
            calls.append(("read_detail", name, dtype, components))
            return np.array([3.0], dtype=np.float32)

        def write_point(self, name: str, values, *, create=True):
            calls.append(("write_point", name, np.asarray(values).shape, create))

        def write_prim(self, name: str, values, *, create=True):
            calls.append(("write_prim", name, np.asarray(values).shape, create))

        def write_vertex(self, name: str, values, *, create=True):
            calls.append(("write_vertex", name, np.asarray(values).shape, create))

        def write_detail(self, name: str, values, *, create=True):
            calls.append(("write_detail", name, np.asarray(values).shape, create))

        def clear_output(self):
            calls.append(("clear_output",))

        def create_point(self, position=None):
            calls.append(("create_point", position))
            return "point"

        def create_points(self, positions):
            calls.append(("create_points", np.asarray(positions).shape))
            return ("point0", "point1")

        def create_polygon(self, points, *, closed=True):
            calls.append(("create_polygon", tuple(points), closed))
            return "prim"

        def create_polygons(self, polygons, *, closed=True):
            normalized = tuple(tuple(points) for points in polygons)
            calls.append(("create_polygons", normalized, closed))
            return ("prim0", "prim1")

        def add_vertices(self, prim, points):
            calls.append(("add_vertices", prim, tuple(points)))
            return ("vertex0", "vertex1")

    ctx = _make_ctx(_FakeIO())

    point_values = ctx.read_point("P", components=3)
    prim_values = ctx.read_prim("id")
    vertex_values = ctx.read_vertex("tag")
    detail_values = ctx.read_detail("mass")
    ctx.write_point("Cd", np.ones((2, 3), dtype=np.float32), create=False)
    ctx.write_prim("v", np.ones((1, 3), dtype=np.float32), create=True)
    ctx.write_vertex("tag", np.array([1], dtype=np.int32), create=True)
    ctx.write_detail("scale", np.array([1.0], dtype=np.float32), create=False)
    ctx.P()
    ctx.set_P(np.ones((1, 3), dtype=np.float32))
    created_point = ctx.create_point((0.0, 0.0, 0.0))
    created_points = ctx.create_points(np.ones((2, 3), dtype=np.float32))
    created_prim = ctx.create_polygon([0, 1], closed=False)
    created_prims = ctx.create_polygons([[0, 1], [1, 0]], closed=True)
    created_vertices = ctx.add_vertices("prim", [0, 1])
    ctx.clear_output()

    assert point_values.shape == (1, 3)
    assert prim_values.shape == (1,)
    assert vertex_values.shape == (1,)
    assert detail_values.shape == (1,)
    assert created_point == "point"
    assert created_points == ("point0", "point1")
    assert created_prim == "prim"
    assert created_prims == ("prim0", "prim1")
    assert created_vertices == ("vertex0", "vertex1")
    assert calls == [
        ("read_point", "P", None, 3),
        ("read_prim", "id", None, None),
        ("read_vertex", "tag", None, None),
        ("read_detail", "mass", None, None),
        ("write_point", "Cd", (2, 3), False),
        ("write_prim", "v", (1, 3), True),
        ("write_vertex", "tag", (1,), True),
        ("write_detail", "scale", (1,), False),
        ("read_point", "P", None, 3),
        ("write_point", "P", (1, 3), True),
        ("create_point", (0.0, 0.0, 0.0)),
        ("create_points", (2, 3)),
        ("create_polygon", (0, 1), False),
        ("create_polygons", ((0, 1), (1, 0)), True),
        ("add_vertices", "prim", (0, 1)),
        ("clear_output",),
    ]
