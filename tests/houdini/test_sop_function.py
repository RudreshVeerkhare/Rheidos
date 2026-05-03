from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from rheidos.compute import World
from rheidos.houdini.sop import (
    CallGeo,
    CtxInputGeo,
    SopCall,
    SopFunctionModule,
    SopFunctionSetupError,
    SopVerbRunner,
    SopVerbUnavailableError,
    points_np_to_geo,
    tri_mesh_np_to_geo,
)


class _FakeGeometry:
    def __init__(self, name: str = "geo") -> None:
        self.name = name
        self.created_points = []
        self.polygons = []
        self.float_string_writes = []
        self.increment_all_calls = 0
        self.frozen_from = None
        self.executed_inputs = None

    def createPoints(self, positions):
        self.created_points.extend(positions)
        return tuple(range(len(positions)))

    def createPolygons(self, polygons):
        self.polygons.extend(polygons)
        return tuple(range(len(polygons)))

    def setPointFloatAttribValuesFromString(self, name, values, float_type):
        self.float_string_writes.append((name, values, float_type))

    def incrementAllDataIds(self):
        self.increment_all_calls += 1

    def freeze(self):
        frozen = _FakeGeometry(f"{self.name}_frozen")
        frozen.frozen_from = self
        return frozen


class _FakeVerb:
    def __init__(self, *, min_inputs: int = 0) -> None:
        self._min_inputs = min_inputs
        self.load_calls = []
        self.load_at_time_calls = []
        self.set_parm_calls = []
        self.execute_calls = []
        self.execute_at_time_calls = []

    def minNumInputs(self):
        return self._min_inputs

    def loadParmsFromNode(self, node):
        self.load_calls.append(node)

    def loadParmsFromNodeAtTime(self, node, time):
        self.load_at_time_calls.append((node, time))

    def setParms(self, parms):
        self.set_parm_calls.append(dict(parms))

    def execute(self, dest_geo, inputs):
        dest_geo.executed_inputs = list(inputs)
        self.execute_calls.append((dest_geo, list(inputs)))

    def executeAtTime(self, dest_geo, inputs, time, add_time_dep):
        dest_geo.executed_inputs = list(inputs)
        self.execute_at_time_calls.append(
            (dest_geo, list(inputs), time, add_time_dep)
        )


class _FakeNode:
    def __init__(
        self,
        path: str,
        *,
        verb: _FakeVerb | None = None,
        has_verb: bool = True,
    ) -> None:
        self._path = path
        self._verb = verb
        self._has_verb = has_verb

    def path(self):
        return self._path

    def hasVerb(self):
        return self._has_verb

    def verb(self):
        return self._verb


class _FakeCtx:
    def __init__(self, *geos) -> None:
        self._geos = tuple(geos)
        self.input_geo_calls = []

    def input_geo(self, index: int):
        self.input_geo_calls.append(index)
        return self._geos[index]


@pytest.fixture
def fake_hou(monkeypatch):
    hou = ModuleType("hou")
    hou.numericData = SimpleNamespace(
        Float32="Float32",
        Float64="Float64",
        Int8="Int8",
        Int16="Int16",
        Int32="Int32",
        Int64="Int64",
    )
    hou._nodes = {}
    hou._created_geos = []

    def _node(path):
        return hou._nodes.get(path)

    def _geometry():
        geo = _FakeGeometry(f"created_{len(hou._created_geos)}")
        hou._created_geos.append(geo)
        return geo

    hou.node = _node
    hou.Geometry = _geometry
    monkeypatch.setitem(sys.modules, "hou", hou)
    yield hou
    monkeypatch.delitem(sys.modules, "hou", raising=False)


def test_sop_verb_runner_executes_with_loaded_node_parms(fake_hou) -> None:
    verb = _FakeVerb(min_inputs=1)
    node = _FakeNode("/obj/geo1/project_wrangle", verb=verb)
    fake_hou._nodes[node.path()] = node
    input_geo = _FakeGeometry("query")

    runner = SopVerbRunner(node.path())
    out_geo = runner.execute([input_geo], parms={"class": 2})

    assert verb.load_calls == [node]
    assert verb.set_parm_calls == [{"class": 2}]
    assert verb.execute_calls[0][0] is out_geo
    assert verb.execute_calls[0][1] == [input_geo]


def test_sop_verb_runner_uses_time_specific_execution(fake_hou) -> None:
    verb = _FakeVerb(min_inputs=0)
    node = _FakeNode("/obj/geo1/time_wrangle", verb=verb)
    fake_hou._nodes[node.path()] = node

    runner = SopVerbRunner(node.path())
    out_geo = runner.execute([], time=1.25, add_time_dep=True)

    assert verb.load_at_time_calls == [(node, 1.25)]
    assert verb.execute_at_time_calls == [(out_geo, [], 1.25, True)]


def test_sop_verb_runner_reports_missing_or_unavailable_verbs(fake_hou) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        SopVerbRunner("/obj/missing")

    fake_hou._nodes["/obj/noverb"] = _FakeNode(
        "/obj/noverb",
        verb=None,
        has_verb=False,
    )
    with pytest.raises(SopVerbUnavailableError, match="does not expose"):
        SopVerbRunner("/obj/noverb")


def test_sop_function_module_composes_call_and_context_inputs(fake_hou) -> None:
    class Projector(SopFunctionModule):
        NAME = "Projector"
        SOP_INPUTS = {
            0: CallGeo("query"),
            1: CtxInputGeo(0),
        }

    verb = _FakeVerb(min_inputs=2)
    node = _FakeNode("/obj/geo1/project_wrangle", verb=verb)
    fake_hou._nodes[node.path()] = node
    mesh_geo = _FakeGeometry("mesh")
    query_one = _FakeGeometry("query_one")
    query_two = _FakeGeometry("query_two")
    ctx = _FakeCtx(mesh_geo)

    module = World().require(Projector, node_path=node.path())
    module.setup(ctx)
    module.run(query=query_one)
    module.run(query=query_two)

    assert verb.execute_calls[0][1] == [query_one, mesh_geo]
    assert verb.execute_calls[1][1] == [query_two, mesh_geo]
    assert ctx.input_geo_calls == [0]


def test_ctx_input_geo_requires_setup_for_context_backed_inputs(fake_hou) -> None:
    class Projector(SopFunctionModule):
        NAME = "Projector"
        SOP_INPUTS = {0: CtxInputGeo(0)}

    verb = _FakeVerb(min_inputs=1)
    node = _FakeNode("/obj/geo1/project_wrangle", verb=verb)
    fake_hou._nodes[node.path()] = node

    module = World().require(Projector, node_path=node.path())
    with pytest.raises(SopFunctionSetupError, match="setup"):
        module.run()


def test_ctx_input_geo_session_cache_survives_later_setup_calls(fake_hou) -> None:
    class CookCached(SopFunctionModule):
        NAME = "CookCached"
        SOP_INPUTS = {0: CtxInputGeo(0, cache="cook")}

    class SessionCached(SopFunctionModule):
        NAME = "SessionCached"
        SOP_INPUTS = {0: CtxInputGeo(0, cache="session")}

    cook_verb = _FakeVerb(min_inputs=1)
    session_verb = _FakeVerb(min_inputs=1)
    cook_node = _FakeNode("/obj/cook_cached", verb=cook_verb)
    session_node = _FakeNode("/obj/session_cached", verb=session_verb)
    fake_hou._nodes[cook_node.path()] = cook_node
    fake_hou._nodes[session_node.path()] = session_node

    mesh_one = _FakeGeometry("mesh_one")
    mesh_two = _FakeGeometry("mesh_two")
    world = World()

    cook_cached = world.require(CookCached, node_path=cook_node.path())
    cook_cached.setup(_FakeCtx(mesh_one)).run()
    cook_cached.setup(_FakeCtx(mesh_two)).run()

    session_cached = world.require(SessionCached, node_path=session_node.path())
    session_cached.setup(_FakeCtx(mesh_one)).run()
    session_cached.setup(_FakeCtx(mesh_two)).run()

    assert cook_verb.execute_calls[0][1] == [mesh_one]
    assert cook_verb.execute_calls[1][1] == [mesh_two]
    assert session_verb.execute_calls[0][1] == [mesh_one]
    assert session_verb.execute_calls[1][1] == [mesh_one]


def test_ctx_input_geo_freeze_is_explicit(fake_hou) -> None:
    class FreezingProjector(SopFunctionModule):
        NAME = "FreezingProjector"
        SOP_INPUTS = {0: CtxInputGeo(0, freeze=True)}

    verb = _FakeVerb(min_inputs=1)
    node = _FakeNode("/obj/freezing_projector", verb=verb)
    fake_hou._nodes[node.path()] = node
    mesh_geo = _FakeGeometry("mesh")

    module = World().require(FreezingProjector, node_path=node.path())
    module.setup(_FakeCtx(mesh_geo)).run()

    bound_geo = verb.execute_calls[0][1][0]
    assert bound_geo is not mesh_geo
    assert bound_geo.frozen_from is mesh_geo


def test_sop_function_module_hooks_can_wrap_call_flow(fake_hou) -> None:
    class Hooked(SopFunctionModule):
        NAME = "Hooked"

        def preprocess(self, call: SopCall) -> SopCall:
            call.values["generated"] = _FakeGeometry("generated")
            call.meta["preprocessed"] = True
            return call

        def sop_inputs(self, call: SopCall):
            return [call.values["generated"]]

        def postprocess(self, out_geo, meta):
            return out_geo, dict(meta)

    verb = _FakeVerb(min_inputs=1)
    node = _FakeNode("/obj/hooked", verb=verb)
    fake_hou._nodes[node.path()] = node

    out_geo, meta = World().require(Hooked, node_path=node.path()).run()

    assert out_geo is verb.execute_calls[0][0]
    assert verb.execute_calls[0][1][0].name == "generated"
    assert meta == {"preprocessed": True}


def test_points_np_to_geo_uses_binary_point_position_write(fake_hou) -> None:
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)

    geo = points_np_to_geo(points)

    assert len(geo.created_points) == 2
    assert geo.float_string_writes[0][0] == "P"
    assert geo.float_string_writes[0][1].dtype == np.float64
    assert geo.float_string_writes[0][2] == "Float64"
    assert geo.increment_all_calls == 1


def test_tri_mesh_np_to_geo_adds_polygons_and_increments_data_ids(fake_hou) -> None:
    geo = tri_mesh_np_to_geo(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0, 1, 2]],
    )

    assert geo.polygons == [(0, 1, 2)]
    assert geo.increment_all_calls == 2


def test_points_np_to_geo_rejects_invalid_point_shapes(fake_hou) -> None:
    with pytest.raises(ValueError, match="shape"):
        points_np_to_geo(np.array([1.0, 2.0, 3.0]))
