import importlib.util
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROFILER_DIR = ROOT / "rheidos" / "compute" / "profiler"
COMPUTE_DIR = ROOT / "rheidos" / "compute"


def _ensure_pkg(name: str, path: pathlib.Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if spec.loader is None:
        raise RuntimeError(f"Missing loader for {name}")
    spec.loader.exec_module(module)
    return module


_ensure_pkg("rheidos", ROOT / "rheidos")
_ensure_pkg("rheidos.compute", ROOT / "rheidos" / "compute")
_ensure_pkg("rheidos.compute.profiler", PROFILER_DIR)

ids = _load_module("rheidos.compute.profiler.ids", PROFILER_DIR / "ids.py")
core = _load_module("rheidos.compute.profiler.core", PROFILER_DIR / "core.py")
runtime = _load_module("rheidos.compute.profiler.runtime", PROFILER_DIR / "runtime.py")
registry = _load_module("rheidos.compute.registry", COMPUTE_DIR / "registry.py")
summary_store = _load_module(
    "rheidos.compute.profiler.summary_store", PROFILER_DIR / "summary_store.py"
)

Profiler = core.Profiler
ProfilerConfig = core.ProfilerConfig
set_current_profiler = runtime.set_current_profiler
reset_current_profiler = runtime.reset_current_profiler
ProducerBase = registry.ProducerBase
Registry = registry.Registry
SummaryStore = summary_store.SummaryStore


class ProducerA(ProducerBase):
    outputs = ("res.a",)

    def compute(self, reg: Registry) -> None:
        reg.commit("res.a")


class ProducerB(ProducerBase):
    outputs = ("res.b",)

    def compute(self, reg: Registry) -> None:
        reg.read("res.a")
        reg.commit("res.b")


class ProfilerTraceTest(unittest.TestCase):
    def test_stable_ids(self) -> None:
        pid1 = ids.PRODUCER_IDS.intern("demo.producer")
        pid2 = ids.PRODUCER_IDS.intern("demo.producer")
        rid1 = ids.RESOURCE_IDS.intern("demo.resource")
        rid2 = ids.RESOURCE_IDS.intern("demo.resource")
        self.assertEqual(pid1, pid2)
        self.assertEqual(rid1, rid2)
        self.assertEqual(ids.PRODUCER_IDS.name_for(pid1), "demo.producer")
        self.assertEqual(ids.RESOURCE_IDS.name_for(rid1), "demo.resource")

    def test_exec_tree_records_nested_producers(self) -> None:
        prof = Profiler(ProfilerConfig(enabled=True))
        token = set_current_profiler(prof)
        try:
            reg = Registry()
            prod_a = ProducerA()
            prod_b = ProducerB()
            reg.declare("res.a", producer=prod_a)
            reg.declare("res.b", producer=prod_b)
            prof.next_cook_index()
            reg.ensure("res.b")
        finally:
            reset_current_profiler(token)
        tree = prof.snapshot_exec_tree()
        nodes = tree["nodes"]
        self.assertEqual(len(nodes), 2)
        root = nodes[0]
        child = nodes[1]
        self.assertIsNone(root["parent"])
        self.assertEqual(child["parent"], root["id"])
        self.assertGreaterEqual(root["inclusive_ms"], child["inclusive_ms"])
        self.assertEqual(root["class_name"], "ProducerB")
        self.assertEqual(child["class_name"], "ProducerA")

    def test_resource_reads_record_edges(self) -> None:
        prof = Profiler(ProfilerConfig(enabled=True))
        token = set_current_profiler(prof)
        try:
            reg = Registry()
            prod_a = ProducerA()
            prod_b = ProducerB()
            reg.declare("res.a", producer=prod_a)
            reg.declare("res.b", producer=prod_b)
            prof.next_cook_index()
            reg.ensure("res.b")
        finally:
            reset_current_profiler(token)
        dag = prof.snapshot_dag(mode="observed")
        edges = {(row["source"], row["target"]) for row in dag["edges"]}
        self.assertIn((prod_b.profiler_id(), prod_a.profiler_id()), edges)
        nodes_by_id = {node["id"]: node for node in dag["nodes"]}
        self.assertEqual(
            nodes_by_id[prod_a.profiler_id()]["class_name"], "ProducerA"
        )
        self.assertEqual(
            nodes_by_id[prod_b.profiler_id()]["class_name"], "ProducerB"
        )

    def test_node_details_without_exec_tree(self) -> None:
        store = SummaryStore()
        prof = Profiler(ProfilerConfig(enabled=True), summary_store=store)
        cook_id = prof.next_cook_index()
        prof.record_value("producer", "compute", "demo.producer", 2.0)
        producer_id = ids.PRODUCER_IDS.intern("demo.producer")
        details = prof.snapshot_node_details(producer_id)
        self.assertIsNotNone(details)
        self.assertEqual(details["id"], producer_id)
        metrics = details.get("metrics")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["last_update_id"], cook_id)

    def test_edges_recorded_without_overhead(self) -> None:
        store = SummaryStore()
        prof = Profiler(
            ProfilerConfig(enabled=True, overhead_enabled=False), summary_store=store
        )
        token = set_current_profiler(prof)
        try:
            reg = Registry()
            prod_a = ProducerA()
            prod_b = ProducerB()
            reg.declare("res.a", producer=prod_a)
            reg.declare("res.b", producer=prod_b)
            prof.next_cook_index()
            reg.ensure("res.b")
            prof.next_cook_index()
        finally:
            reset_current_profiler(token)
        snap = store.snapshot_compact()
        self.assertGreater(snap["edges_recorded"], 0)


if __name__ == "__main__":
    unittest.main()
