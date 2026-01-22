import { useCallback, useEffect, useRef, useState } from "react";
import type {
  DagSnapshot,
  ExecTreeSnapshot,
  MetricsSnapshot,
  SummarySnapshot,
  WsPayload,
} from "../types";

export type StreamStatus = {
  ws: "idle" | "open" | "closed";
  error: string | null;
};

const DEFAULT_POLL_HZ = 4;

function buildSummarySnapshot(payload: WsPayload): SummarySnapshot {
  return {
    tick: payload.tick,
    cook_id: payload.cook_id,
    frame: payload.frame,
    substep: payload.substep,
    dropped_events: payload.dropped_events,
    profiler_overhead_us: payload.profiler_overhead_us,
    edges_recorded: payload.edges_recorded,
    wall_ms: payload.wall_ms,
    kernel_ms: payload.kernel_ms,
    kernel_fraction: payload.kernel_fraction,
    dag_version: payload.dag_version,
    categories: payload.categories,
    rows: payload.rows,
  };
}

export function useProfilerStream(params: {
  live: boolean;
  uiHz: number;
  dagMode: "union" | "observed";
}) {
  const { live, uiHz, dagMode } = params;
  const [summary, setSummary] = useState<SummarySnapshot | null>(null);
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null);
  const [dag, setDag] = useState<DagSnapshot | null>(null);
  const [execTree, setExecTree] = useState<ExecTreeSnapshot | null>(null);
  const [status, setStatus] = useState<StreamStatus>({ ws: "idle", error: null });

  const wsRef = useRef<WebSocket | null>(null);
  const wsBuffer = useRef<WsPayload | null>(null);
  const reconnectTimer = useRef<number | null>(null);
  const backoffMs = useRef(500);
  const connectRef = useRef<() => void>(() => {});
  const dagModeRef = useRef(dagMode);
  const lastCookId = useRef<number | null>(null);
  const lastDagVersion = useRef<number | null>(null);
  const pending = useRef({
    summary: false,
    metrics: false,
    dag: false,
    execTree: false,
  });

  useEffect(() => {
    dagModeRef.current = dagMode;
  }, [dagMode]);

  const getWsUrl = useCallback(() => {
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    const mode = dagModeRef.current;
    return `${scheme}://${window.location.host}/ws?mode=${encodeURIComponent(mode)}`;
  }, []);

  const disconnectWs = useCallback(() => {
    if (reconnectTimer.current) {
      window.clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
    if (wsRef.current) {
      try {
        wsRef.current.close(1000);
      } catch (err) {
        // ignore close errors
      }
    }
    wsRef.current = null;
    wsBuffer.current = null;
    setStatus((prev) => ({ ...prev, ws: "closed" }));
  }, []);

  const scheduleReconnect = useCallback(() => {
    if (reconnectTimer.current || !live) {
      return;
    }
    const delay = Math.min(15000, backoffMs.current);
    reconnectTimer.current = window.setTimeout(() => {
      reconnectTimer.current = null;
      backoffMs.current = Math.min(15000, backoffMs.current * 2);
      connectRef.current();
    }, delay);
  }, [live]);

  const connectWs = useCallback(() => {
    if (!live) {
      return;
    }
    disconnectWs();
    const socket = new WebSocket(getWsUrl());
    wsRef.current = socket;
    setStatus((prev) => ({ ...prev, ws: "idle", error: null }));

    socket.addEventListener("open", () => {
      backoffMs.current = 500;
      setStatus((prev) => ({ ...prev, ws: "open" }));
    });

    socket.addEventListener("message", (event) => {
      try {
        wsBuffer.current = JSON.parse(event.data) as WsPayload;
      } catch (err) {
        setStatus((prev) => ({ ...prev, error: "WS payload parse failed" }));
      }
    });

    socket.addEventListener("close", () => {
      setStatus((prev) => ({ ...prev, ws: "closed" }));
      if (live) {
        scheduleReconnect();
      }
    });

    socket.addEventListener("error", () => {
      try {
        socket.close();
      } catch (err) {
        // ignore socket close errors
      }
    });
  }, [disconnectWs, getWsUrl, live, scheduleReconnect]);

  useEffect(() => {
    connectRef.current = connectWs;
  }, [connectWs]);

  const fetchJson = useCallback(async <T,>(url: string): Promise<T | null> => {
    try {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) {
        return null;
      }
      return (await response.json()) as T;
    } catch (err) {
      return null;
    }
  }, []);

  const fetchDag = useCallback(async () => {
    if (pending.current.dag || !live) {
      return;
    }
    pending.current.dag = true;
    try {
      const mode = dagModeRef.current;
      const snap = await fetchJson<DagSnapshot>(
        `/api/dag?mode=${encodeURIComponent(mode)}`
      );
      if (snap) {
        setDag(snap);
      }
    } finally {
      pending.current.dag = false;
    }
  }, [fetchJson, live]);

  const fetchExecTree = useCallback(
    async (cookId: number | null) => {
      if (pending.current.execTree || !live) {
        return;
      }
      pending.current.execTree = true;
      try {
        const url =
          cookId === null || cookId === undefined
            ? "/api/exec_tree"
            : `/api/exec_tree?cook_id=${encodeURIComponent(cookId)}`;
        const snap = await fetchJson<ExecTreeSnapshot>(url);
        if (snap) {
          setExecTree(snap);
        }
      } finally {
        pending.current.execTree = false;
      }
    },
    [fetchJson, live]
  );

  const fetchMetrics = useCallback(async () => {
    if (pending.current.metrics || !live) {
      return;
    }
    pending.current.metrics = true;
    try {
      const snap = await fetchJson<MetricsSnapshot>("/api/metrics");
      if (snap) {
        setMetrics(snap);
        if (snap.cook_id !== undefined && snap.cook_id !== null) {
          if (lastCookId.current !== snap.cook_id) {
            lastCookId.current = snap.cook_id;
            fetchExecTree(snap.cook_id);
          }
        }
      }
    } finally {
      pending.current.metrics = false;
    }
  }, [fetchExecTree, fetchJson, live]);

  const fetchSummary = useCallback(async () => {
    if (pending.current.summary || !live) {
      return;
    }
    pending.current.summary = true;
    try {
      const snap = await fetchJson<SummarySnapshot>("/api/summary");
      if (snap) {
        setSummary(snap);
        if (dagModeRef.current === "union") {
          if (snap.dag_version !== undefined && snap.dag_version !== null) {
            if (lastDagVersion.current !== snap.dag_version) {
              lastDagVersion.current = snap.dag_version;
              fetchDag();
            }
          }
        } else if (snap.cook_id !== undefined && snap.cook_id !== null) {
          if (lastCookId.current !== snap.cook_id) {
            lastCookId.current = snap.cook_id;
            fetchDag();
          }
        }
      }
    } finally {
      pending.current.summary = false;
    }
  }, [fetchDag, fetchJson, live]);

  const applyPayload = useCallback(
    (payload: WsPayload) => {
      if (!payload) {
        return;
      }
      const summarySnap = buildSummarySnapshot(payload);
      setSummary(summarySnap);

      const mode = payload.dag_mode || dagModeRef.current;
      if (payload.dag && mode === dagModeRef.current) {
        setDag(payload.dag);
        if (payload.dag.dag_version !== undefined) {
          lastDagVersion.current = payload.dag.dag_version ?? null;
        } else if (payload.dag_version !== undefined) {
          lastDagVersion.current = payload.dag_version ?? null;
        }
      } else if (mode === dagModeRef.current) {
        if (dagModeRef.current === "union") {
          if (
            payload.dag_version !== undefined &&
            payload.dag_version !== lastDagVersion.current
          ) {
            lastDagVersion.current = payload.dag_version ?? null;
            fetchDag();
          }
        } else if (payload.cook_id !== undefined && payload.cook_id !== null) {
          if (lastCookId.current !== payload.cook_id) {
            lastCookId.current = payload.cook_id;
            fetchDag();
          }
        }
      }

      if (payload.exec_tree) {
        setExecTree(payload.exec_tree);
      }

      if (payload.metrics) {
        setMetrics(payload.metrics);
        if (payload.metrics.cook_id !== undefined) {
          if (lastCookId.current !== payload.metrics.cook_id) {
            lastCookId.current = payload.metrics.cook_id ?? null;
            if (!payload.exec_tree) {
              fetchExecTree(payload.metrics.cook_id ?? null);
            }
          }
        }
      }
    },
    [fetchDag, fetchExecTree]
  );

  useEffect(() => {
    if (!live) {
      disconnectWs();
      return;
    }
    lastDagVersion.current = null;
    lastCookId.current = null;
    setDag(null);
    connectWs();
    return () => disconnectWs();
  }, [connectWs, dagMode, disconnectWs, live]);

  useEffect(() => {
    if (!live) {
      return;
    }
    const hz = uiHz > 0 ? uiHz : DEFAULT_POLL_HZ;
    const interval = window.setInterval(() => {
      if (status.ws === "open") {
        if (wsBuffer.current) {
          const payload = wsBuffer.current;
          wsBuffer.current = null;
          applyPayload(payload);
        }
      } else {
        fetchSummary();
        fetchMetrics();
      }
    }, 1000 / Math.max(1, hz));

    const initialTimeout = window.setTimeout(() => {
      if (status.ws !== "open") {
        fetchSummary();
        fetchMetrics();
      }
    }, 10);

    return () => {
      window.clearInterval(interval);
      window.clearTimeout(initialTimeout);
    };
  }, [applyPayload, fetchMetrics, fetchSummary, live, status.ws, uiHz]);

  return {
    summary,
    metrics,
    dag,
    execTree,
    status,
  };
}
