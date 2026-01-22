import { fmt, fmtPct } from "../lib/format";
import type { MetricsRow, NodeDetails, SummaryRow } from "../types";
import type { RollupEntry } from "../lib/rollup";

type DetailsDrawerProps = {
  selectedId: string | null;
  details: NodeDetails | null;
  loading: boolean;
  metrics: MetricsRow | SummaryRow | null;
  rollup: RollupEntry | null;
  currentCookId: number | null;
  onFocus: () => void;
  onClear: () => void;
};

function copyToClipboard(text: string) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).catch(() => {});
    return;
  }
  const area = document.createElement("textarea");
  area.value = text;
  document.body.appendChild(area);
  area.select();
  try {
    document.execCommand("copy");
  } catch (err) {
    // ignore copy errors
  }
  document.body.removeChild(area);
}

export default function DetailsDrawer({
  selectedId,
  details,
  loading,
  metrics,
  rollup,
  currentCookId,
  onFocus,
  onClear,
}: DetailsDrawerProps) {
  if (selectedId === null) {
    return (
      <aside className="details-drawer empty">
        <div className="drawer-header">
          <div>
            <h3>Node details</h3>
            <p>Select a producer to inspect.</p>
          </div>
        </div>
      </aside>
    );
  }

  const fullName =
    details?.full_name || details?.name || metrics?.name || String(selectedId);
  const className =
    details?.class_name ||
    metrics?.class_name ||
    (fullName ? fullName.split(".").pop() : "") ||
    String(selectedId);
  const metricSource = details?.metrics || metrics || {};
  const lastUpdate =
    (metricSource as MetricsRow).last_update_id ??
    (metricSource as SummaryRow).last_update ??
    "-";
  const calls = rollup?.calls ?? (metricSource as MetricsRow).calls ?? "-";

  return (
    <aside className="details-drawer">
      <div className="drawer-header">
        <div>
          <h3>{className}</h3>
          <p>{fullName}</p>
        </div>
        <div className="drawer-actions">
          <button className="btn small" type="button" onClick={onFocus}>
            Focus
          </button>
          <button
            className="btn ghost small"
            type="button"
            onClick={() => copyToClipboard(fullName)}
          >
            Copy name
          </button>
          <button className="btn ghost small" type="button" onClick={onClear}>
            Clear
          </button>
        </div>
      </div>

      {loading && <div className="drawer-loading">Loading details...</div>}

      {!loading && (
        <div className="drawer-body">
          <section className="drawer-card">
            <h4>Last cook</h4>
            <div className="stat-grid">
              <div>
                <span>EXCL</span>
                <strong>{fmt(rollup?.excl_ms)}</strong>
              </div>
              <div>
                <span>INCL</span>
                <strong>{fmt(rollup?.incl_ms)}</strong>
              </div>
              <div>
                <span>CALLS</span>
                <strong>{calls}</strong>
              </div>
              <div>
                <span>COOK</span>
                <strong>{currentCookId ?? "-"}</strong>
              </div>
            </div>
          </section>

          <section className="drawer-card">
            <h4>Runtime metrics</h4>
            <div className="stat-grid">
              <div>
                <span>LAST</span>
                <strong>{fmt((metricSource as MetricsRow).last_ms)}</strong>
              </div>
              <div>
                <span>EMA</span>
                <strong>{fmt((metricSource as MetricsRow).ema_ms)}</strong>
              </div>
              <div>
                <span>KERN</span>
                <strong>{fmt((metricSource as MetricsRow).kernel_ms)}</strong>
              </div>
              <div>
                <span>KF</span>
                <strong>{fmtPct((metricSource as MetricsRow).kernel_frac)}</strong>
              </div>
              <div>
                <span>OH</span>
                <strong>{fmt((metricSource as MetricsRow).overhead_est_ms)}</strong>
              </div>
              <div>
                <span>UPDATED</span>
                <strong>{lastUpdate}</strong>
              </div>
            </div>
          </section>

          <section className="drawer-card">
            <h4>Resources + staleness</h4>
            {details?.staleness_reason && (
              <p className="drawer-note">{details.staleness_reason}</p>
            )}
            <div className="drawer-row">
              <span>Reads</span>
              <div className="chip-list">
                {details?.resources_read?.length
                  ? details.resources_read.map((item) => (
                      <span key={item} className="chip">
                        {item}
                      </span>
                    ))
                  : "-"}
              </div>
            </div>
          </section>

          <section className="drawer-card">
            <h4>Inputs / outputs</h4>
            <div className="drawer-row">
              <span>Inputs</span>
              <div className="chip-list">
                {details?.inputs?.length
                  ? details.inputs.map((item) => (
                      <span key={`${item.id}-${item.version}`} className="chip">
                        {item.id}@{item.version}
                      </span>
                    ))
                  : "-"}
              </div>
            </div>
            <div className="drawer-row">
              <span>Outputs</span>
              <div className="chip-list">
                {details?.outputs?.length
                  ? details.outputs.map((item) => (
                      <span key={`${item.id}-${item.version}`} className="chip">
                        {item.id}@{item.version}
                      </span>
                    ))
                  : "-"}
              </div>
            </div>
          </section>
        </div>
      )}
    </aside>
  );
}
