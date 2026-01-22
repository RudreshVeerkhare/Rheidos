from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
import threading
import time
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

from .summary_store import SummaryStore


@dataclass(frozen=True)
class SummaryWriterConfig:
    logdir: str
    export_hz: float = 1.0
    filename: str = "latest.json"


class SummaryWriter:
    def __init__(self, store: SummaryStore, cfg: SummaryWriterConfig) -> None:
        self.store = store
        self.cfg = cfg
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="rheidos-summary-writer", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        os.makedirs(self.cfg.logdir, exist_ok=True)
        period = 1.0 / max(1e-6, self.cfg.export_hz)
        out_path = os.path.join(self.cfg.logdir, self.cfg.filename)
        tmp_path = out_path + ".tmp"
        while not self._stop.is_set():
            time.sleep(period)
            snap = self.store.snapshot_compact()
            try:
                with open(tmp_path, "w", encoding="utf-8") as handle:
                    json.dump(snap, handle)
                os.replace(tmp_path, out_path)
            except Exception:
                continue


@dataclass(frozen=True)
class SummaryServerConfig:
    host: str = "127.0.0.1"
    port: int = 6007
    ws_hz: float = 8.0


_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _ws_accept_key(key: str) -> str:
    digest = hashlib.sha1((key + _WS_GUID).encode("utf-8")).digest()
    return base64.b64encode(digest).decode("utf-8")


def _ws_send_text(sock, payload: str) -> bool:
    data = payload.encode("utf-8")
    header = bytearray()
    header.append(0x81)
    length = len(data)
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header.extend(length.to_bytes(2, "big"))
    else:
        header.append(127)
        header.extend(length.to_bytes(8, "big"))
    try:
        sock.sendall(header + data)
    except Exception:
        return False
    return True


def _stringify_id(value: object) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _stringify_dag_snapshot(dag: Optional[dict]) -> Optional[dict]:
    if dag is None:
        return None
    nodes = []
    for node in dag.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        node_copy = dict(node)
        if "id" in node_copy:
            node_copy["id"] = _stringify_id(node_copy["id"])
        nodes.append(node_copy)
    edges = []
    for edge in dag.get("edges", []) or []:
        if not isinstance(edge, dict):
            continue
        edge_copy = dict(edge)
        if "source" in edge_copy:
            edge_copy["source"] = _stringify_id(edge_copy["source"])
        if "target" in edge_copy:
            edge_copy["target"] = _stringify_id(edge_copy["target"])
        edges.append(edge_copy)
    dag_copy = dict(dag)
    dag_copy["nodes"] = nodes
    dag_copy["edges"] = edges
    return dag_copy


def _stringify_metrics_snapshot(metrics: Optional[dict]) -> Optional[dict]:
    if metrics is None:
        return None
    rows = []
    for row in metrics.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        row_copy = dict(row)
        if "id" in row_copy:
            row_copy["id"] = _stringify_id(row_copy["id"])
        rows.append(row_copy)
    metrics_copy = dict(metrics)
    metrics_copy["rows"] = rows
    return metrics_copy


def _stringify_exec_tree_snapshot(exec_tree: Optional[dict]) -> Optional[dict]:
    if exec_tree is None:
        return None
    nodes = []
    for node in exec_tree.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        node_copy = dict(node)
        if "producer_id" in node_copy:
            node_copy["producer_id"] = _stringify_id(node_copy["producer_id"])
        nodes.append(node_copy)
    exec_copy = dict(exec_tree)
    exec_copy["nodes"] = nodes
    return exec_copy


def _stringify_exec_subtree(node: object) -> object:
    if isinstance(node, dict):
        node_copy = dict(node)
        if "producer_id" in node_copy:
            node_copy["producer_id"] = _stringify_id(node_copy["producer_id"])
        children = node_copy.get("children")
        if isinstance(children, list):
            node_copy["children"] = [_stringify_exec_subtree(child) for child in children]
        return node_copy
    if isinstance(node, list):
        return [_stringify_exec_subtree(child) for child in node]
    return node


def _stringify_node_details(details: Optional[dict]) -> Optional[dict]:
    if details is None:
        return None
    details_copy = dict(details)
    if "id" in details_copy:
        details_copy["id"] = _stringify_id(details_copy["id"])
    metrics = details_copy.get("metrics")
    if isinstance(metrics, dict) and "id" in metrics:
        metrics_copy = dict(metrics)
        metrics_copy["id"] = _stringify_id(metrics_copy["id"])
        details_copy["metrics"] = metrics_copy
    subtree = details_copy.get("last_exec_subtree")
    if subtree is not None:
        details_copy["last_exec_subtree"] = _stringify_exec_subtree(subtree)
    return details_copy


class _SummaryRequestHandler(BaseHTTPRequestHandler):
    server: "SummaryHTTPServer"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/ws" and self.headers.get("Upgrade", "").lower() == "websocket":
            key = self.headers.get("Sec-WebSocket-Key")
            if not key:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            qs = parse_qs(parsed.query or "")
            mode = (qs.get("mode") or ["union"])[0]
            if mode not in ("union", "observed"):
                mode = "union"
            accept_key = _ws_accept_key(key)
            self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept_key)
            self.end_headers()
            snap = self.server.summary_store.snapshot_compact()
            summary_server = getattr(self.server, "summary_server", None)
            if summary_server is not None:
                payload, dag_key = summary_server._build_ws_payload(
                    snap, dag_mode=mode, last_dag_key=None, force_full=True
                )
            else:
                payload = dict(snap)
                payload["full"] = True
                payload["changed_ids"] = [row.get("id") for row in snap.get("rows", [])]
                payload["dag_mode"] = mode
                dag_key = None
            _ws_send_text(self.connection, json.dumps(payload))
            self.server.register_ws(self.connection, mode=mode, last_dag_key=dag_key)
            self.close_connection = False
            return
        if parsed.path == "/api/summary":
            self._send_json(self.server.summary_store.snapshot_compact())
            return
        if parsed.path == "/api/dag":
            trace = getattr(self.server, "trace_provider", None)
            if trace is None or not hasattr(trace, "snapshot_dag"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            qs = parse_qs(parsed.query or "")
            mode = (qs.get("mode") or ["union"])[0]
            payload = _stringify_dag_snapshot(trace.snapshot_dag(mode=mode))
            self._send_json(payload)
            return
        if parsed.path == "/api/metrics":
            trace = getattr(self.server, "trace_provider", None)
            if trace is None or not hasattr(trace, "snapshot_metrics"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = _stringify_metrics_snapshot(trace.snapshot_metrics())
            self._send_json(payload)
            return
        if parsed.path == "/api/exec_tree":
            trace = getattr(self.server, "trace_provider", None)
            if trace is None or not hasattr(trace, "snapshot_exec_tree"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            qs = parse_qs(parsed.query or "")
            cook_id = qs.get("cook_id", [None])[0]
            try:
                cook_val = int(cook_id) if cook_id is not None else None
            except ValueError:
                cook_val = None
            payload = _stringify_exec_tree_snapshot(
                trace.snapshot_exec_tree(cook_id=cook_val)
            )
            self._send_json(payload)
            return
        if parsed.path.startswith("/api/node/"):
            trace = getattr(self.server, "trace_provider", None)
            if trace is None or not hasattr(trace, "snapshot_node_details"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            node_id = unquote(parsed.path[len("/api/node/") :])
            if not node_id:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                producer_id = int(node_id)
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            details = _stringify_node_details(trace.snapshot_node_details(producer_id))
            if details is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_json(details)
            return
        if parsed.path.startswith("/api/producer/"):
            producer_id = unquote(parsed.path[len("/api/producer/") :])
            if not producer_id:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            details = self.server.summary_store.snapshot_producer_details(producer_id)
            if details is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_json(details)
            return

        normalized_path = parsed.path.lstrip("/") or "index.html"
        safe_path = os.path.normpath(normalized_path)
        if safe_path.startswith(".."):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if self._send_static_if_exists(safe_path):
            return
        # SPA fallback for client-side routes.
        self._send_static("index.html")

    def log_message(self, fmt: str, *args) -> None:  # pragma: no cover - noisy
        return

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static_if_exists(self, name: str) -> bool:
        path = os.path.join(self.server.static_dir, name)
        if not os.path.isfile(path):
            return False
        return self._send_static(name)

    def _send_static(self, name: str) -> bool:
        path = os.path.join(self.server.static_dir, name)
        if not os.path.isfile(path):
            self.send_error(HTTPStatus.NOT_FOUND)
            return False
        ctype, _ = mimetypes.guess_type(path)
        ctype = ctype or "application/octet-stream"
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except Exception:
            self.send_error(HTTPStatus.NOT_FOUND)
            return False
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        return True


class SummaryHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        handler_cls: type[_SummaryRequestHandler],
        *,
        summary_store: SummaryStore,
        static_dir: str,
        trace_provider: Optional[object] = None,
        summary_server: Optional["SummaryServer"] = None,
    ) -> None:
        super().__init__(address, handler_cls)
        self.summary_store = summary_store
        self.static_dir = static_dir
        self.trace_provider = trace_provider
        self.summary_server = summary_server
        self.ws_clients: dict = {}
        self.ws_lock = threading.Lock()

    def register_ws(self, sock, *, mode: str = "union", last_dag_key=None) -> None:
        with self.ws_lock:
            self.ws_clients[sock] = {
                "mode": mode,
                "last_dag_key": last_dag_key,
            }

    def unregister_ws(self, sock) -> None:
        with self.ws_lock:
            if sock in self.ws_clients:
                self.ws_clients.pop(sock, None)


class SummaryServer:
    def __init__(
        self,
        store: SummaryStore,
        cfg: SummaryServerConfig,
        *,
        trace_provider: Optional[object] = None,
    ) -> None:
        self.store = store
        self.cfg = cfg
        self.trace_provider = trace_provider
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[SummaryHTTPServer] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_stop = threading.Event()
        self._last_snapshot: Optional[dict] = None
        self._last_exec_tree_cook: Optional[int] = None
        self._ws_state_lock = threading.Lock()
        self.url: Optional[str] = None

    def start(self, *, static_dir: str) -> None:
        if self._thread is not None:
            return
        server = None
        try:
            server = SummaryHTTPServer(
                (self.cfg.host, self.cfg.port),
                _SummaryRequestHandler,
                summary_store=self.store,
                static_dir=static_dir,
                trace_provider=self.trace_provider,
                summary_server=self,
            )
        except OSError:
            if self.cfg.port != 0:
                server = SummaryHTTPServer(
                    (self.cfg.host, 0),
                    _SummaryRequestHandler,
                    summary_store=self.store,
                    static_dir=static_dir,
                    trace_provider=self.trace_provider,
                    summary_server=self,
                )
            else:
                raise
        host, port = server.server_address
        self.url = f"http://{host}:{port}/"
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever, name="rheidos-summary-server", daemon=True
        )
        self._thread.start()
        self._start_ws_thread()

    def stop(self) -> None:
        self._ws_stop.set()
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=1.0)
        self._ws_thread = None
        if self._server is not None:
            with self._server.ws_lock:
                clients = list(self._server.ws_clients.keys())
                self._server.ws_clients.clear()
            for sock in clients:
                try:
                    sock.close()
                except Exception:
                    pass
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._server = None
        self._thread = None
        self.url = None

    def _start_ws_thread(self) -> None:
        self._ws_stop.clear()
        self._ws_thread = threading.Thread(
            target=self._ws_loop, name="rheidos-summary-ws", daemon=True
        )
        self._ws_thread.start()

    def _ws_loop(self) -> None:
        period = 1.0 / max(1e-6, self.cfg.ws_hz)
        while not self._ws_stop.is_set():
            time.sleep(period)
            server = self._server
            if server is None:
                continue
            with server.ws_lock:
                clients = list(server.ws_clients.items())
            if not clients:
                continue
            snap = self.store.snapshot_compact()
            for sock, info in clients:
                payload, dag_key = self._build_ws_payload(
                    snap,
                    dag_mode=info.get("mode", "union"),
                    last_dag_key=info.get("last_dag_key"),
                    force_full=False,
                )
                payload_str = json.dumps(payload)
                if not _ws_send_text(sock, payload_str):
                    server.unregister_ws(sock)
                    continue
                if dag_key != info.get("last_dag_key"):
                    with server.ws_lock:
                        if sock in server.ws_clients:
                            server.ws_clients[sock]["last_dag_key"] = dag_key

    def _build_ws_payload(
        self,
        snap: dict,
        *,
        dag_mode: str,
        last_dag_key,
        force_full: bool,
    ) -> tuple[dict, Optional[tuple]]:
        payload = dict(snap)
        payload["full"] = True
        payload["changed_ids"] = [row.get("id") for row in snap.get("rows", [])]
        payload["dag_mode"] = dag_mode
        trace = self.trace_provider
        if trace is not None and hasattr(trace, "snapshot_metrics"):
            metrics = _stringify_metrics_snapshot(trace.snapshot_metrics())
            if metrics is not None:
                metrics["full"] = True
                payload["metrics"] = metrics
        cook_id = snap.get("cook_id")
        dag_version = snap.get("dag_version")
        dag_key = last_dag_key
        if trace is not None and hasattr(trace, "snapshot_exec_tree"):
            with self._ws_state_lock:
                should_send_tree = (
                    force_full or cook_id != self._last_exec_tree_cook
                )
                if should_send_tree:
                    self._last_exec_tree_cook = cook_id
            if should_send_tree:
                payload["exec_tree"] = _stringify_exec_tree_snapshot(
                    trace.snapshot_exec_tree(cook_id=cook_id)
                )
        if trace is not None and hasattr(trace, "snapshot_dag"):
            with self._ws_state_lock:
                if dag_mode == "observed":
                    desired_key = ("observed", cook_id)
                else:
                    desired_key = ("union", dag_version)
                should_send_dag = force_full or desired_key != last_dag_key
            if should_send_dag:
                payload["dag"] = _stringify_dag_snapshot(
                    trace.snapshot_dag(mode=dag_mode)
                )
                payload["dag_mode"] = dag_mode
                dag_key = desired_key
        return payload, dag_key

    def _compute_delta(self, snap: dict) -> dict:
        prev = self._last_snapshot
        self._last_snapshot = snap
        if prev is None or len(prev.get("rows", [])) != len(snap.get("rows", [])):
            payload = dict(snap)
            payload["full"] = True
            return payload
        prev_map = {row.get("id"): row.get("last_update") for row in prev.get("rows", [])}
        changed_rows = []
        changed_ids = []
        for row in snap.get("rows", []):
            if prev_map.get(row.get("id")) != row.get("last_update"):
                changed_rows.append(row)
                changed_ids.append(row.get("id"))
        return {
            "tick": snap.get("tick"),
            "cook_id": snap.get("cook_id"),
            "frame": snap.get("frame"),
            "substep": snap.get("substep"),
            "dropped_events": snap.get("dropped_events"),
            "wall_ms": snap.get("wall_ms"),
            "kernel_ms": snap.get("kernel_ms"),
            "kernel_fraction": snap.get("kernel_fraction"),
            "categories": snap.get("categories", {}),
            "rows": changed_rows,
            "changed_ids": changed_ids,
            "full": False,
        }
