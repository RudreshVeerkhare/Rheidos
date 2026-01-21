import http.client
import importlib.util
import json
import pathlib
import sys
import tempfile
import time
import types
import unittest
from urllib.parse import urlparse

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
ids = _load_module("rheidos.compute.profiler.ids", PROFILER_DIR / "ids.py")
summary_store = _load_module(
    "rheidos.compute.profiler.summary_store", PROFILER_DIR / "summary_store.py"
)
summary_server = _load_module(
    "rheidos.compute.profiler.summary_server", PROFILER_DIR / "summary_server.py"
)

SummaryStore = summary_store.SummaryStore
SummaryWriter = summary_server.SummaryWriter
SummaryWriterConfig = summary_server.SummaryWriterConfig
SummaryServer = summary_server.SummaryServer
SummaryServerConfig = summary_server.SummaryServerConfig
Profiler = core.Profiler
ProfilerConfig = core.ProfilerConfig


class SummaryServerTest(unittest.TestCase):
    def test_summary_writer_outputs_latest(self) -> None:
        store = SummaryStore()
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SummaryWriter(store, SummaryWriterConfig(logdir=tmpdir, export_hz=50))
            writer.start()
            time.sleep(0.05)
            writer.stop()
            out_path = pathlib.Path(tmpdir) / "latest.json"
            self.assertTrue(out_path.exists())
            data = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("tick", data)

    def test_summary_server_api(self) -> None:
        store = SummaryStore()
        prof = Profiler(ProfilerConfig(enabled=True), summary_store=store)
        prof.next_cook_index()
        _, run, is_root = prof._span_enter("producer", "compute", "demo.producer")
        prof.record_resource_read(
            resource_id=ids.RESOURCE_IDS.intern("res.a"),
            producer_id=ids.PRODUCER_IDS.intern("demo.upstream"),
        )
        prof._span_exit("producer", "compute", "demo.producer", 2_000_000, run, is_root)
        server = SummaryServer(
            store,
            SummaryServerConfig(host="127.0.0.1", port=0),
            trace_provider=prof,
        )
        static_dir = str(ROOT / "rheidos" / "compute" / "profiler" / "ui")
        server.start(static_dir=static_dir)
        try:
            self.assertIsNotNone(server.url)
            parsed = urlparse(server.url)
            time.sleep(0.02)
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=2)
            conn.request("GET", "/api/summary")
            resp = conn.getresponse()
            body = resp.read()
            self.assertEqual(resp.status, 200)
            payload = json.loads(body.decode("utf-8"))
            self.assertIn("rows", payload)
            self.assertIn("categories", payload)
            conn.request("GET", "/dag")
            resp = conn.getresponse()
            body = resp.read()
            self.assertEqual(resp.status, 200)
            self.assertIn(b"<!doctype html>", body.lower())
            conn.request("GET", "/tables?pid=1")
            resp = conn.getresponse()
            body = resp.read()
            self.assertEqual(resp.status, 200)
            self.assertIn(b"<!doctype html>", body.lower())
            producer_id = ids.PRODUCER_IDS.intern("demo.producer")
            conn.request("GET", "/api/dag")
            resp = conn.getresponse()
            body = resp.read()
            self.assertEqual(resp.status, 200)
            payload = json.loads(body.decode("utf-8"))
            self.assertIn("nodes", payload)
            nodes_by_id = {node["id"]: node for node in payload["nodes"]}
            self.assertIn(producer_id, nodes_by_id)
            self.assertIn("class_name", nodes_by_id[producer_id])
            self.assertIn("full_name", nodes_by_id[producer_id])
            conn.request("GET", "/api/metrics")
            resp = conn.getresponse()
            body = resp.read()
            self.assertEqual(resp.status, 200)
            payload = json.loads(body.decode("utf-8"))
            self.assertIn("rows", payload)
            self.assertIn("class_name", payload["rows"][0])
            conn.request("GET", "/api/exec_tree")
            resp = conn.getresponse()
            body = resp.read()
            self.assertEqual(resp.status, 200)
            payload = json.loads(body.decode("utf-8"))
            self.assertIn("nodes", payload)
            self.assertIn("class_name", payload["nodes"][0])
            conn.request("GET", f"/api/node/{producer_id}")
            resp = conn.getresponse()
            body = resp.read()
            self.assertEqual(resp.status, 200)
            payload = json.loads(body.decode("utf-8"))
            self.assertEqual(payload["id"], producer_id)
            self.assertIn("class_name", payload)
            store.update_producer_details(
                "demo.producer",
                last_update_id=1,
                inputs=[{"id": "res.in", "version": 0}],
                outputs=[{"id": "res.out", "version": 1}],
            )
            conn.request("GET", "/api/producer/demo.producer")
            resp = conn.getresponse()
            body = resp.read()
            self.assertEqual(resp.status, 200)
            payload = json.loads(body.decode("utf-8"))
            self.assertEqual(payload["id"], "demo.producer")
            self.assertIn("class_name", payload)
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
