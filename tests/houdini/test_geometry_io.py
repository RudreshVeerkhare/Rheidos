from __future__ import annotations

from types import SimpleNamespace

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


class _FakeGeometry:
    def __init__(self, *, point_count: int = 0, prim_count: int = 0, vertex_count: int = 0) -> None:
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
        self._values = {
            OWNER_POINT: {},
            OWNER_PRIM: {},
            OWNER_VERTEX: {},
            OWNER_DETAIL: {},
        }

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
        return [object()] * self._counts[OWNER_POINT]

    def prims(self):
        return [object()] * self._counts[OWNER_PRIM]

    def vertices(self):
        return [object()] * self._counts[OWNER_VERTEX]

    def addAttrib(self, attrib_type: str, name: str, default):
        tuple_size = len(default) if isinstance(default, tuple) else 1
        self._attribs[attrib_type][name] = _FakeAttrib(
            name=name,
            tuple_size=tuple_size,
            data_type=self._data_type_for_default(default),
        )
        if attrib_type == OWNER_DETAIL:
            self._values[OWNER_DETAIL][name] = default
            return
        flat_default = list(default) if isinstance(default, tuple) else [default]
        self._values[attrib_type][name] = flat_default * self._counts[attrib_type]

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
        return self._values[OWNER_DETAIL][name]

    def setGlobalAttribValue(self, name: str, value) -> None:
        self._values[OWNER_DETAIL][name] = value

    @staticmethod
    def _data_type_for_default(default) -> str:
        sample = default[0] if isinstance(default, tuple) else default
        if isinstance(sample, str):
            return OWNER_DETAIL + ".string"
        if isinstance(sample, (bool, int, np.integer)):
            return OWNER_DETAIL + ".int"
        return OWNER_DETAIL + ".float"


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
    assert geo.attribValue("strength") == np.float32(1.5)

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

    assert point_values.shape == (1, 3)
    assert prim_values.shape == (1,)
    assert vertex_values.shape == (1,)
    assert detail_values.shape == (1,)
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
    ]
