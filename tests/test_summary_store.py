import importlib.util
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROFILER_DIR = ROOT / "rheidos" / "compute" / "profiler"


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

core = _load_module("rheidos.compute.profiler.core", PROFILER_DIR / "core.py")
summary_store = _load_module(
    "rheidos.compute.profiler.summary_store", PROFILER_DIR / "summary_store.py"
)

Profiler = core.Profiler
ProfilerConfig = core.ProfilerConfig
SummaryStore = summary_store.SummaryStore


class SummaryStoreTest(unittest.TestCase):
    def test_snapshot_compact_records_producer(self) -> None:
        store = SummaryStore()
        prof = Profiler(ProfilerConfig(enabled=True), summary_store=store)

        cook_id = prof.next_cook_index()
        prof.record_value("producer", "compute", "demo.producer", 2.0)

        snap = store.snapshot_compact()
        self.assertEqual(snap["cook_id"], cook_id)
        self.assertEqual(len(snap["rows"]), 1)
        row = snap["rows"][0]
        self.assertEqual(row["id"], "demo.producer")
        self.assertEqual(row["last_update"], cook_id)
        self.assertAlmostEqual(row["last_ms"], 2.0, places=3)
        self.assertEqual(row["full_name"], "demo.producer")
        self.assertEqual(row["class_name"], "producer")

    def test_kernel_and_wall_metrics(self) -> None:
        store = SummaryStore()
        prof = Profiler(ProfilerConfig(enabled=True), summary_store=store)

        prof.next_cook_index()
        prof.record_value("producer", "compute", "demo.producer", 2.0)
        prof.record_value("taichi", "producer_kernel_ms", "demo.producer", 1.0)
        prof.record_value("taichi", "producer_overhead_ms", "demo.producer", 0.5)
        prof.record_value("cook", "cook_total", None, 4.0)
        prof.record_value("taichi", "kernel_total", None, 1.0)

        snap = store.snapshot_compact()
        row = snap["rows"][0]
        self.assertAlmostEqual(row["kernel_ms"], 1.0, places=3)
        self.assertAlmostEqual(row["overhead_est_ms"], 0.5, places=3)
        self.assertAlmostEqual(row["kernel_frac"], 0.5, places=3)
        self.assertAlmostEqual(snap["wall_ms"], 4.0, places=3)
        self.assertAlmostEqual(snap["kernel_ms"], 1.0, places=3)
        self.assertAlmostEqual(snap["kernel_fraction"], 0.25, places=3)

    def test_producer_details_snapshot(self) -> None:
        store = SummaryStore()
        store.update_producer_details(
            "demo.producer",
            last_update_id=7,
            inputs=[{"id": "res.in", "version": 2}],
            outputs=[{"id": "res.out", "version": 3}],
            staleness_reason="",
        )
        details = store.snapshot_producer_details("demo.producer")
        self.assertIsNotNone(details)
        self.assertEqual(details["last_update"], 7)
        self.assertEqual(details["inputs"][0]["id"], "res.in")

    def test_child_span_tracking(self) -> None:
        store = SummaryStore()
        prof = Profiler(ProfilerConfig(enabled=True), summary_store=store)
        prof.next_cook_index()
        _, root_run, root_is_root = prof._span_enter(
            "producer", "compute", "demo.producer"
        )
        _, child_run, child_is_root = prof._span_enter("python", "child_a", None)
        prof._span_exit("python", "child_a", None, 1_500_000, child_run, child_is_root)
        prof._span_exit(
            "producer", "compute", "demo.producer", 3_000_000, root_run, root_is_root
        )
        details = store.snapshot_producer_details("demo.producer")
        self.assertIsNotNone(details)
        self.assertEqual(details["top_child_spans"][0]["name"], "python/child_a")

    def test_category_summary(self) -> None:
        store = SummaryStore()
        prof = Profiler(ProfilerConfig(enabled=True), summary_store=store)
        prof.record_value("solver", "step", None, 1.25)
        snap = store.snapshot_compact()
        self.assertIn("solver", snap["categories"])
        row = snap["categories"]["solver"][0]
        self.assertEqual(row["name"], "step")
        self.assertAlmostEqual(row["last_ms"], 1.25, places=3)


if __name__ == "__main__":
    unittest.main()
