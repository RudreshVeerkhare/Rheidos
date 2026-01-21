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


class _SummaryRequestHandler(BaseHTTPRequestHandler):
    server: "SummaryHTTPServer"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/ws" and self.headers.get("Upgrade", "").lower() == "websocket":
            key = self.headers.get("Sec-WebSocket-Key")
            if not key:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            accept_key = _ws_accept_key(key)
            self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept_key)
            self.end_headers()
            snap = self.server.summary_store.snapshot_compact()
            full_payload = dict(snap)
            full_payload["full"] = True
            _ws_send_text(self.connection, json.dumps(full_payload))
            self.server.register_ws(self.connection)
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
            payload = trace.snapshot_dag(mode=mode)
            self._send_json(payload)
            return
        if parsed.path == "/api/metrics":
            trace = getattr(self.server, "trace_provider", None)
            if trace is None or not hasattr(trace, "snapshot_metrics"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = trace.snapshot_metrics()
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
            payload = trace.snapshot_exec_tree(cook_id=cook_val)
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
            details = trace.snapshot_node_details(producer_id)
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

        normalized_path = parsed.path.rstrip("/") or "/"
        if normalized_path in ("/", "/index.html", "/dag", "/tables"):
            self._send_static("index.html")
            return
        if parsed.path in ("/app.js", "/style.css"):
            self._send_static(parsed.path.lstrip("/"))
            return

        self.send_error(HTTPStatus.NOT_FOUND)

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

    def _send_static(self, name: str) -> None:
        path = os.path.join(self.server.static_dir, name)
        if not os.path.isfile(path):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        ctype, _ = mimetypes.guess_type(path)
        ctype = ctype or "application/octet-stream"
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except Exception:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


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
    ) -> None:
        super().__init__(address, handler_cls)
        self.summary_store = summary_store
        self.static_dir = static_dir
        self.trace_provider = trace_provider
        self.ws_clients: set = set()
        self.ws_lock = threading.Lock()

    def register_ws(self, sock) -> None:
        with self.ws_lock:
            self.ws_clients.add(sock)

    def unregister_ws(self, sock) -> None:
        with self.ws_lock:
            if sock in self.ws_clients:
                self.ws_clients.remove(sock)


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
            )
        except OSError:
            if self.cfg.port != 0:
                server = SummaryHTTPServer(
                    (self.cfg.host, 0),
                    _SummaryRequestHandler,
                    summary_store=self.store,
                    static_dir=static_dir,
                    trace_provider=self.trace_provider,
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
                clients = list(self._server.ws_clients)
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
            snap = self.store.snapshot_compact()
            payload = self._compute_delta(snap)
            payload_str = json.dumps(payload)
            with server.ws_lock:
                clients = list(server.ws_clients)
            for sock in clients:
                if not _ws_send_text(sock, payload_str):
                    server.unregister_ws(sock)

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
