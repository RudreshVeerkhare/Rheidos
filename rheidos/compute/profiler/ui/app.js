const summaryBody = document.getElementById("summary-rows");
const summaryHead = document.getElementById("summary-head");
const categoryPanels = document.getElementById("category-panels");
const cookId = document.getElementById("cook-id");
const frameId = document.getElementById("frame-id");
const substepId = document.getElementById("substep-id");
const tickPill = document.getElementById("tick-pill");
const searchInput = document.getElementById("search-input");
const kernelToggle = document.getElementById("kernel-toggle");

const state = {
  rows: [],
  rowsById: new Map(),
  categories: {},
  lastUpdateById: new Map(),
  changedIds: new Set(),
  expanded: new Set(),
  detailsCache: new Map(),
  pendingDetails: new Set(),
  pollTimer: null,
};

function fmt(value) {
  if (value === null || value === undefined) {
    return "-";
  }
  if (typeof value === "number") {
    return value.toFixed(3);
  }
  return value;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmtPct(value) {
  if (value === null || value === undefined) {
    return "-";
  }
  if (typeof value !== "number") {
    return value;
  }
  return `${(value * 100).toFixed(1)}%`;
}

function renderHeader() {
  const columns = ["Name", "EMA ms", "Last ms"];
  if (kernelToggle.checked) {
    columns.push("Kernel frac", "Overhead ms");
  }
  columns.push("Last update");
  summaryHead.innerHTML = columns.map((label) => `<th>${label}</th>`).join("");
}

function renderRows(rows) {
  summaryBody.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    if (state.changedIds.has(row.id)) {
      tr.classList.add("row-updated");
    }
    tr.addEventListener("click", () => toggleRow(row.id));
    const cells = [
      `<td>${escapeHtml(row.name)}</td>`,
      `<td>${fmt(row.ema_ms)}</td>`,
      `<td>${fmt(row.last_ms)}</td>`,
    ];
    if (kernelToggle.checked) {
      cells.push(`<td>${fmtPct(row.kernel_frac)}</td>`);
      cells.push(`<td>${fmt(row.overhead_est_ms)}</td>`);
    }
    cells.push(`<td style="text-align:right">${row.last_update ?? "-"}</td>`);
    tr.innerHTML = `
      ${cells.join("")}
    `;
    summaryBody.appendChild(tr);

    if (state.expanded.has(row.id)) {
      const detailsRow = document.createElement("tr");
      detailsRow.classList.add("details-row");
      const details = renderDetails(row.id);
      const colspan = kernelToggle.checked ? 6 : 4;
      detailsRow.innerHTML = `<td colspan="${colspan}">${details}</td>`;
      summaryBody.appendChild(detailsRow);
    }
  });
}

function renderDetails(id) {
  const details = state.detailsCache.get(id);
  if (!details) {
    return `<div class="details">Loading details...</div>`;
  }
  const inputs = (details.inputs || [])
    .map((item) => `${item.id}@${item.version}`)
    .join(", ");
  const outputs = (details.outputs || [])
    .map((item) => `${item.id}@${item.version}`)
    .join(", ");
  const children = (details.top_child_spans || [])
    .map(
      (span) =>
        `${span.name} (${fmt(span.last_ms)} ms / ${fmt(span.ema_ms)} ms)`
    )
    .join("<br />");

  return `
    <div class="details">
      <div class="details-grid">
        <div><strong>Last update:</strong> ${details.last_update ?? "-"}</div>
        ${
          details.staleness_reason
            ? `<div><strong>Staleness:</strong> ${details.staleness_reason}</div>`
            : ""
        }
      </div>
      <div class="details-grid">
        <div><strong>Inputs:</strong> ${inputs || "-"}</div>
        <div><strong>Outputs:</strong> ${outputs || "-"}</div>
      </div>
      <div class="details-grid">
        <div><strong>Top child spans:</strong></div>
        <div>${children || "-"}</div>
      </div>
    </div>
  `;
}

function toggleRow(id) {
  if (state.expanded.has(id)) {
    state.expanded.delete(id);
    render();
    return;
  }
  state.expanded.add(id);
  fetchDetails(id);
  render();
}

async function fetchDetails(id) {
  if (state.pendingDetails.has(id)) {
    return;
  }
  state.pendingDetails.add(id);
  try {
    const response = await fetch(`/api/producer/${encodeURIComponent(id)}`, {
      cache: "no-store",
    });
    if (!response.ok) {
      return;
    }
    const details = await response.json();
    state.detailsCache.set(id, details);
    render();
  } catch (err) {
    console.warn("details fetch failed", err);
  } finally {
    state.pendingDetails.delete(id);
  }
}

function render() {
  const query = (searchInput.value || "").toLowerCase();
  const filtered = (state.rows || []).filter((row) => {
    if (!query) {
      return true;
    }
    return String(row.name || "").toLowerCase().includes(query);
  });
  filtered.sort((a, b) => (b.ema_ms || 0) - (a.ema_ms || 0));
  renderHeader();
  renderRows(filtered);
  renderCategories(query);
}

function renderCategories(query) {
  categoryPanels.innerHTML = "";
  const categories = state.categories || {};
  Object.keys(categories)
    .sort()
    .forEach((cat) => {
      const rows = categories[cat] || [];
      const filtered = rows.filter((row) => {
        if (!query) {
          return true;
        }
        const name = String(row.name || "").toLowerCase();
        const prod = String(row.producer || "").toLowerCase();
        return name.includes(query) || prod.includes(query);
      });
      if (!filtered.length) {
        return;
      }
      filtered.sort((a, b) => (b.ema_ms || 0) - (a.ema_ms || 0));
      const showProducer = filtered.some((row) => row.producer);
      const headerCells = [
        "<th>Name</th>",
        "<th>EMA ms</th>",
        "<th>Last ms</th>",
        "<th>Calls</th>",
      ];
      if (showProducer) {
        headerCells.push("<th>Producer</th>");
      }
      const bodyRows = filtered
        .map((row) => {
          const cells = [
            `<td>${escapeHtml(row.name)}</td>`,
            `<td>${fmt(row.ema_ms)}</td>`,
            `<td>${fmt(row.last_ms)}</td>`,
            `<td>${row.calls ?? 0}</td>`,
          ];
          if (showProducer) {
            cells.push(
              `<td>${row.producer ? escapeHtml(row.producer) : "-"}</td>`
            );
          }
          return `<tr>${cells.join("")}</tr>`;
        })
        .join("");
      const section = document.createElement("section");
      section.className = "panel category-panel";
      section.innerHTML = `
        <div class="panel-header">
          <h3>${escapeHtml(cat)}</h3>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>${headerCells.join("")}</tr>
            </thead>
            <tbody>
              ${bodyRows}
            </tbody>
          </table>
        </div>
      `;
      categoryPanels.appendChild(section);
    });
}

async function fetchSummary() {
  const response = await fetch("/api/summary", { cache: "no-store" });
  if (!response.ok) {
    return null;
  }
  return response.json();
}

function applySnapshot(snap) {
  if (!snap) {
    return;
  }
  cookId.textContent = snap.cook_id ?? "-";
  frameId.textContent = snap.frame ?? "-";
  substepId.textContent = snap.substep ?? "-";
  tickPill.textContent = `tick: ${snap.tick ?? 0}`;

  if (snap.full) {
    state.rowsById.clear();
  }
  const incomingRows = Array.isArray(snap.rows) ? snap.rows : [];
  incomingRows.forEach((row) => {
    state.rowsById.set(row.id, row);
  });
  const changed = new Set();
  if (Array.isArray(snap.changed_ids)) {
    snap.changed_ids.forEach((id) => changed.add(id));
    incomingRows.forEach((row) => {
      state.lastUpdateById.set(row.id, row.last_update);
    });
  } else {
    incomingRows.forEach((row) => {
      const prev = state.lastUpdateById.get(row.id);
      if (prev !== undefined && prev !== row.last_update) {
        changed.add(row.id);
      }
      state.lastUpdateById.set(row.id, row.last_update);
    });
  }
  state.changedIds = changed;
  state.rows = Array.from(state.rowsById.values());
  if (snap.categories) {
    state.categories = snap.categories;
  }
  render();
}

async function refresh() {
  try {
    const snap = await fetchSummary();
    if (snap) {
      applySnapshot(snap);
    }
  } catch (err) {
    console.warn("summary refresh failed", err);
  }
}

function startPolling() {
  if (state.pollTimer !== null) {
    return;
  }
  state.pollTimer = setInterval(refresh, 500);
}

function stopPolling() {
  if (state.pollTimer !== null) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function connectWebSocket() {
  if (!("WebSocket" in window)) {
    startPolling();
    return;
  }
  const wsProtocol = location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${wsProtocol}://${location.host}/ws`;
  const socket = new WebSocket(wsUrl);
  socket.addEventListener("open", () => stopPolling());
  socket.addEventListener("message", (event) => {
    try {
      const snap = JSON.parse(event.data);
      applySnapshot(snap);
    } catch (err) {
      console.warn("ws payload error", err);
    }
  });
  socket.addEventListener("close", () => startPolling());
  socket.addEventListener("error", () => startPolling());
}

refresh();
startPolling();
connectWebSocket();

searchInput.addEventListener("input", () => render());
kernelToggle.addEventListener("change", () => render());
