import { useEffect, useMemo, useState } from "react";
import { computePercentiles, computeRollup } from "./lib/rollup";
import { useProfilerStream } from "./hooks/useProfilerStream";
import DagPage from "./components/DagPage";
import MetricsStrip from "./components/MetricsStrip";
import TablesPage from "./components/TablesPage";
import type { DagSnapshot, MetricsRow, SummaryRow } from "./types";

function parseRoute() {
  const path = window.location.pathname.replace(/\/+$/, "") || "/dag";
  const page = path === "/tables" ? "tables" : "dag";
  const params = new URLSearchParams(window.location.search || "");
  const mode = params.get("mode");
  const dagMode = mode === "observed" ? "observed" : "union";
  const pid = params.get("pid");
  let selectedId: string | null = null;
  if (pid) {
    selectedId = pid;
  }
  return { page, dagMode, selectedId } as const;
}

function buildUrl(
  page: "dag" | "tables",
  dagMode: "union" | "observed",
  selectedId: string | null
) {
  if (page === "tables") {
    return "/tables";
  }
  const params = new URLSearchParams();
  if (dagMode) {
    params.set("mode", dagMode);
  }
  if (selectedId !== null) {
    params.set("pid", String(selectedId));
  }
  const query = params.toString();
  return `/dag${query ? `?${query}` : ""}`;
}

function inferClassName(fullName: string) {
  if (!fullName) {
    return "";
  }
  const parts = fullName.split(".");
  return parts[parts.length - 1] || fullName;
}

export default function App() {
  const initialRoute = parseRoute();
  const [page, setPage] = useState<"dag" | "tables">(initialRoute.page);
  const [dagMode, setDagMode] = useState<"union" | "observed">(
    initialRoute.dagMode
  );
  const [selectedId, setSelectedId] = useState<string | null>(
    initialRoute.selectedId
  );
  const [live, setLive] = useState(true);
  const [uiHz, setUiHz] = useState(6);

  const { summary, metrics, dag, execTree, status } = useProfilerStream({
    live,
    uiHz,
    dagMode,
  });

  useEffect(() => {
    const onPop = () => {
      const nextRoute = parseRoute();
      setPage(nextRoute.page);
      setDagMode(nextRoute.dagMode);
      setSelectedId(nextRoute.selectedId);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  useEffect(() => {
    const next = buildUrl(page, dagMode, selectedId);
    if (window.location.pathname + window.location.search !== next) {
      window.history.replaceState(null, "", next);
    }
  }, [dagMode, page, selectedId]);

  const summaryRows = summary?.rows ?? [];
  const metricsRows = metrics?.rows ?? [];

  const summaryByName = useMemo(() => {
    const map = new Map<string, SummaryRow>();
    summaryRows.forEach((row) => map.set(row.name, row));
    return map;
  }, [summaryRows]);

  const metricsById = useMemo(() => {
    const map = new Map<string, MetricsRow>();
    metricsRows.forEach((row) => map.set(row.id, row));
    return map;
  }, [metricsRows]);

  const metricsByName = useMemo(() => {
    const map = new Map<string, MetricsRow>();
    metricsRows.forEach((row) => map.set(row.name, row));
    return map;
  }, [metricsRows]);

  const rollup = useMemo(() => {
    return computeRollup(execTree?.nodes ?? []);
  }, [execTree]);

  const percentiles = useMemo(() => {
    return computePercentiles(rollup);
  }, [rollup]);

  const classCounts = useMemo(() => {
    const map = new Map<string, number>();
    const nodes = (dag as DagSnapshot | null)?.nodes || [];
    nodes.forEach((node) => {
      const name = node.full_name || node.name || String(node.id);
      const className = node.class_name || inferClassName(name);
      map.set(className, (map.get(className) || 0) + 1);
    });
    return map;
  }, [dag]);

  const currentCookId = metrics?.cook_id ?? summary?.cook_id ?? null;

  return (
    <div className={`app-shell page-${page}`}>
      <header className="topbar">
        <div className="brand">
          <span className="brand-title">Rheidos Profiler</span>
          <span className="brand-sub">Streaming DAG + runtime trace</span>
        </div>
        <nav className="nav">
          <button
            className={`nav-link ${page === "dag" ? "active" : ""}`}
            type="button"
            onClick={() => {
              setPage("dag");
              window.history.pushState(null, "", buildUrl("dag", dagMode, selectedId));
            }}
          >
            DAG
          </button>
          <button
            className={`nav-link ${page === "tables" ? "active" : ""}`}
            type="button"
            onClick={() => {
              setPage("tables");
              window.history.pushState(null, "", buildUrl("tables", dagMode, null));
            }}
          >
            Tables
          </button>
        </nav>
        <div className="topbar-controls">
          <label className="field inline">
            <span>Rate</span>
            <select
              value={uiHz}
              onChange={(event) => setUiHz(Number(event.target.value) || 6)}
            >
              <option value={2}>2 hz</option>
              <option value={4}>4 hz</option>
              <option value={6}>6 hz</option>
              <option value={10}>10 hz</option>
            </select>
          </label>
          <button
            className={`btn ${live ? "" : "ghost"}`}
            type="button"
            onClick={() => setLive(!live)}
          >
            {live ? "Live" : "Paused"}
          </button>
        </div>
      </header>

      <MetricsStrip summary={summary} metrics={metrics} status={status} live={live} />

      <main className="main">
        {page === "dag" ? (
          <DagPage
            dag={dag}
            metricsById={metricsById}
            summaryByName={summaryByName}
            rollup={rollup}
            percentiles={percentiles}
            classCounts={classCounts}
            currentCookId={currentCookId}
            selectedId={selectedId}
            onSelect={setSelectedId}
            dagMode={dagMode}
            onDagModeChange={setDagMode}
          />
        ) : (
          <TablesPage
            summary={summary}
            metricsByName={metricsByName}
            onSelectProducer={(id) => {
              if (id === null) {
                return;
              }
              setSelectedId(id);
              setPage("dag");
              window.history.pushState(null, "", buildUrl("dag", dagMode, id));
            }}
          />
        )}
      </main>
    </div>
  );
}
