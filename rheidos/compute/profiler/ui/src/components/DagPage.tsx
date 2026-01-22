import { useMemo, useRef, useState } from "react";
import { fuzzyScore } from "../lib/format";
import type { DagSnapshot, MetricsRow, SummaryRow } from "../types";
import type { RollupEntry } from "../lib/rollup";
import DagCanvas, { type DagCanvasHandle } from "./DagCanvas";
import DetailsDrawer from "./DetailsDrawer";
import { useNodeDetails } from "../hooks/useNodeDetails";

type DagPageProps = {
  dag: DagSnapshot | null;
  metricsById: Map<string, MetricsRow>;
  summaryByName: Map<string, SummaryRow>;
  rollup: Map<string, RollupEntry>;
  percentiles: Map<string, number>;
  classCounts: Map<string, number>;
  currentCookId: number | null;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  dagMode: "union" | "observed";
  onDagModeChange: (mode: "union" | "observed") => void;
};

function bestMatchId(dag: DagSnapshot | null, query: string): string | null {
  if (!dag?.nodes || !query.trim()) {
    return null;
  }
  const q = query.toLowerCase();
  let bestId: string | null = null;
  let bestScore = Number.POSITIVE_INFINITY;
  dag.nodes.forEach((node) => {
    const fullName = node.full_name || node.name || String(node.id);
    const className = node.class_name || "";
    const score = Math.min(
      fuzzyScore(fullName.toLowerCase(), q),
      fuzzyScore(className.toLowerCase(), q)
    );
    if (!Number.isFinite(score)) {
      return;
    }
    if (score < bestScore) {
      bestScore = score;
      bestId = node.id;
    }
  });
  return bestId;
}

export default function DagPage({
  dag,
  metricsById,
  summaryByName,
  rollup,
  percentiles,
  classCounts,
  currentCookId,
  selectedId,
  onSelect,
  dagMode,
  onDagModeChange,
}: DagPageProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const canvasRef = useRef<DagCanvasHandle | null>(null);
  const { details, loading } = useNodeDetails(selectedId);

  const matchId = useMemo(
    () => bestMatchId(dag, searchQuery),
    [dag, searchQuery]
  );

  const metrics = selectedId !== null ? metricsById.get(selectedId) || null : null;
  const rollupEntry = selectedId !== null ? rollup.get(selectedId) || null : null;

  return (
    <section className="page page-dag">
      <div className="page-toolbar">
        <div className="toolbar-group">
          <label className="field">
            <span>DAG mode</span>
            <select
              value={dagMode}
              onChange={(event) =>
                onDagModeChange(
                  event.target.value === "observed" ? "observed" : "union"
                )
              }
            >
              <option value="union">Union</option>
              <option value="observed">Observed</option>
            </select>
          </label>
          <label className="field">
            <span>Search</span>
            <input
              type="search"
              placeholder="Find producer or class"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && matchId !== null) {
                  onSelect(matchId);
                  canvasRef.current?.focusNode(matchId);
                }
              }}
            />
          </label>
        </div>
        <div className="toolbar-group">
          <button
            className="btn"
            type="button"
            onClick={() => canvasRef.current?.fitView()}
          >
            Fit view
          </button>
          {matchId !== null && (
            <button
              className="btn ghost"
              type="button"
              onClick={() => {
                onSelect(matchId);
                canvasRef.current?.focusNode(matchId);
              }}
            >
              Focus match
            </button>
          )}
        </div>
      </div>

      <div className="dag-layout">
        <DagCanvas
          ref={canvasRef}
          dag={dag}
          metricsById={metricsById}
          summaryByName={summaryByName}
          rollup={rollup}
          percentiles={percentiles}
          classCounts={classCounts}
          currentCookId={currentCookId}
          selectedId={selectedId}
          onSelect={onSelect}
          searchQuery={searchQuery}
        />
        <DetailsDrawer
          selectedId={selectedId}
          details={details}
          loading={loading}
          metrics={metrics}
          rollup={rollupEntry}
          currentCookId={currentCookId}
          onFocus={() => {
            if (selectedId !== null) {
              canvasRef.current?.focusNode(selectedId);
            }
          }}
          onClear={() => onSelect(null)}
        />
      </div>
    </section>
  );
}
