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
        server = SummaryServer(store, SummaryServerConfig(host="127.0.0.1", port=0))
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
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
