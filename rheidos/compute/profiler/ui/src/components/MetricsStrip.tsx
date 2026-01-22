import { fmt, fmtPct } from "../lib/format";
import type { MetricsSnapshot, SummarySnapshot } from "../types";
import type { StreamStatus } from "../hooks/useProfilerStream";

type MetricsStripProps = {
  summary: SummarySnapshot | null;
  metrics: MetricsSnapshot | null;
  status: StreamStatus;
  live: boolean;
};

export default function MetricsStrip({
  summary,
  metrics,
  status,
  live,
}: MetricsStripProps) {
  const cookId = metrics?.cook_id ?? summary?.cook_id ?? "-";
  return (
    <section className="metrics-strip">
      <div className="metric-card">
        <span>Cook</span>
        <strong>{cookId}</strong>
      </div>
      <div className="metric-card">
        <span>Dropped</span>
        <strong>{summary?.dropped_events ?? "-"}</strong>
      </div>
      <div className="metric-card">
        <span>Edges</span>
        <strong>{summary?.edges_recorded ?? "-"}</strong>
      </div>
      <div className="metric-card">
        <span>Overhead</span>
        <strong>{fmt(summary?.profiler_overhead_us, 1)} us</strong>
      </div>
      <div className="metric-card">
        <span>Kernel</span>
        <strong>{fmt(summary?.kernel_ms)} ms</strong>
      </div>
      <div className="metric-card">
        <span>Kernel %</span>
        <strong>{fmtPct(summary?.kernel_fraction)}</strong>
      </div>
      <div className={`metric-status ${live ? "live" : "paused"}`}>
        <span>{live ? "Live" : "Paused"}</span>
        <strong>{status.ws === "open" ? "WS" : "HTTP"}</strong>
      </div>
    </section>
  );
}
