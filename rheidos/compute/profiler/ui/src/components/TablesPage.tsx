import { useMemo, useState } from "react";
import { fmt, fmtPct } from "../lib/format";
import type { MetricsRow, SummarySnapshot } from "../types";

type TablesPageProps = {
  summary: SummarySnapshot | null;
  metricsByName: Map<string, MetricsRow>;
  onSelectProducer: (id: string | null) => void;
};

type SortKey =
  | "executed"
  | "name"
  | "ema_ms"
  | "last_ms"
  | "kernel_ms"
  | "kernel_frac"
  | "overhead_est_ms"
  | "calls"
  | "last_update";

const columns: Array<{ key: SortKey; label: string }> = [
  { key: "executed", label: "Exec" },
  { key: "name", label: "Name" },
  { key: "ema_ms", label: "EMA ms" },
  { key: "last_ms", label: "Last ms" },
  { key: "kernel_ms", label: "Kernel" },
  { key: "kernel_frac", label: "Kernel %" },
  { key: "overhead_est_ms", label: "Overhead" },
  { key: "calls", label: "Calls" },
  { key: "last_update", label: "Last update" },
];

export default function TablesPage({
  summary,
  metricsByName,
  onSelectProducer,
}: TablesPageProps) {
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("ema_ms");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const rows = summary?.rows || [];
  const filteredRows = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) {
      return rows.slice();
    }
    return rows.filter((row) => String(row.name).toLowerCase().includes(q));
  }, [rows, search]);

  const sortedRows = useMemo(() => {
    const data = filteredRows.slice();
    data.sort((a, b) => {
      const metricsA = metricsByName.get(a.name);
      const metricsB = metricsByName.get(b.name);
      const executedA = metricsA?.executed_this_cook ? 1 : 0;
      const executedB = metricsB?.executed_this_cook ? 1 : 0;

      const valueFor = (row: typeof a, metrics: MetricsRow | undefined) => {
        switch (sortKey) {
          case "executed":
            return metrics?.executed_this_cook ? 1 : 0;
          case "name":
            return String(row.name || "");
          case "ema_ms":
            return row.ema_ms ?? 0;
          case "last_ms":
            return row.last_ms ?? 0;
          case "kernel_ms":
            return row.kernel_ms ?? 0;
          case "kernel_frac":
            return row.kernel_frac ?? 0;
          case "overhead_est_ms":
            return row.overhead_est_ms ?? 0;
          case "calls":
            return row.calls ?? 0;
          case "last_update":
            return row.last_update ?? 0;
          default:
            return 0;
        }
      };

      const left = valueFor(a, metricsA);
      const right = valueFor(b, metricsB);

      if (typeof left === "string" || typeof right === "string") {
        const res = String(left).localeCompare(String(right));
        return sortDir === "asc" ? res : -res;
      }

      if (sortKey === "executed") {
        if (executedA !== executedB) {
          return sortDir === "asc" ? executedA - executedB : executedB - executedA;
        }
      }

      return sortDir === "asc" ? left - right : right - left;
    });
    return data;
  }, [filteredRows, metricsByName, sortDir, sortKey]);

  const categories = summary?.categories || {};
  const categoryKeys = Object.keys(categories).sort();

  return (
    <section className="page page-tables">
      <div className="page-toolbar">
        <label className="field">
          <span>Search</span>
          <input
            type="search"
            placeholder="Filter producers or categories"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </label>
      </div>

      <div className="tables-layout">
        <div className="table-card">
          <div className="table-header">
            <h3>Producers</h3>
            <p>{rows.length} producers</p>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  {columns.map((col) => {
                    const active = col.key === sortKey;
                    const dir = active ? (sortDir === "asc" ? "▲" : "▼") : "";
                    return (
                      <th
                        key={col.key}
                        onClick={() => {
                          if (sortKey === col.key) {
                            setSortDir(sortDir === "asc" ? "desc" : "asc");
                          } else {
                            setSortKey(col.key);
                            setSortDir("desc");
                          }
                        }}
                      >
                        {col.label} {dir}
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {sortedRows.map((row) => {
                  const metrics = metricsByName.get(row.name);
                  const executed = metrics?.executed_this_cook;
                  return (
                    <tr
                      key={row.name}
                      onClick={() =>
                        onSelectProducer(metrics?.id ?? null)
                      }
                    >
                      <td>
                        <span
                          className={`exec-dot ${executed ? "active" : ""}`}
                        />
                      </td>
                      <td>{row.name}</td>
                      <td>{fmt(row.ema_ms)}</td>
                      <td>{fmt(row.last_ms)}</td>
                      <td>{fmt(row.kernel_ms)}</td>
                      <td>{fmtPct(row.kernel_frac)}</td>
                      <td>{fmt(row.overhead_est_ms)}</td>
                      <td>{row.calls ?? 0}</td>
                      <td>{row.last_update ?? "-"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        <div className="category-stack">
          {categoryKeys.map((key) => {
            const rowsForCat = categories[key] || [];
            const filtered = rowsForCat.filter((row) => {
              if (!search.trim()) {
                return true;
              }
              const q = search.trim().toLowerCase();
              return (
                row.name.toLowerCase().includes(q) ||
                String(row.producer || "").toLowerCase().includes(q)
              );
            });
            if (!filtered.length) {
              return null;
            }
            const showProducer = filtered.some((row) => row.producer);
            return (
              <section key={key} className="category-card">
                <div className="table-header">
                  <h3>{key}</h3>
                  <p>{filtered.length} spans</p>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>EMA ms</th>
                        <th>Last ms</th>
                        <th>Calls</th>
                        {showProducer && <th>Producer</th>}
                      </tr>
                    </thead>
                    <tbody>
                      {filtered.map((row) => (
                        <tr key={row.id}>
                          <td>{row.name}</td>
                          <td>{fmt(row.ema_ms)}</td>
                          <td>{fmt(row.last_ms)}</td>
                          <td>{row.calls ?? 0}</td>
                          {showProducer && <td>{row.producer || "-"}</td>}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            );
          })}
        </div>
      </div>
    </section>
  );
}
