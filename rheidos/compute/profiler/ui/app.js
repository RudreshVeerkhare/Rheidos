const appRoot = document.querySelector(".app");
const navLinks = Array.from(document.querySelectorAll("[data-nav]"));
const pageDag = document.getElementById("page-dag");
const pageTables = document.getElementById("page-tables");
const dagContainer = document.getElementById("dag-container");
const dagModeSelect = document.getElementById("dag-mode");
const dagSearch = document.getElementById("dag-search");
const fitViewBtn = document.getElementById("fit-view");
const liveToggle = document.getElementById("live-toggle");
const uiHzSelect = document.getElementById("ui-hz");
const cookIndicator = document.getElementById("cook-indicator");
const healthDropped = document.getElementById("health-dropped");
const healthEdges = document.getElementById("health-edges");
const healthOverhead = document.getElementById("health-overhead");
const nodeDetails = document.getElementById("node-details");
const producerHead = document.getElementById("producer-head");
const producerBody = document.getElementById("producer-body");
const categoryPanels = document.getElementById("category-panels");
const tableSearch = document.getElementById("table-search");

const state = {
  live: true,
  uiHz: 4,
  uiTimer: null,
  pending: {
    metrics: false,
    summary: false,
    dag: false,
    execTree: false,
    nodeDetails: null,
  },
  dagMode: "union",
  dagVersion: null,
  dagModeApplied: null,
  dagNodesById: new Map(),
  dagEdges: [],
  metricsById: new Map(),
  rollupById: new Map(),
  rollupPercentiles: new Map(),
  rollupCookId: null,
  selectedProducerId: null,
  nodeDetailsCache: new Map(),
  lastCookId: null,
  lastDagVersionSeen: null,
  executedLastTick: new Set(),
  pulseTimer: null,
  searchQuery: "",
  tableSearch: "",
  tableSort: { key: "ema_ms", dir: "desc" },
  summaryRows: [],
  categories: {},
  summaryByName: new Map(),
  cy: null,
  dagClassCounts: new Map(),
  didInitialFit: false,
  denseThreshold: 600,
  isDense: false,
  isZoomed: true,
};

let dagTooltip = null;

const BASE_NODE_COLOR = "#f8fbff";
const MAGNITUDE_COLOR = "#d9e7ff";

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  if (typeof value === "number") {
    return value.toFixed(digits);
  }
  return value;
}

function fmtPct(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  if (typeof value !== "number") {
    return value;
  }
  return `${(value * 100).toFixed(1)}%`;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function hexToRgb(hex) {
  const normalized = hex.replace("#", "");
  const value = parseInt(normalized, 16);
  return [
    (value >> 16) & 255,
    (value >> 8) & 255,
    value & 255,
  ];
}

function rgbToHex(rgb) {
  return `#${rgb.map((c) => c.toString(16).padStart(2, "0")).join("")}`;
}

function lerpColor(a, b, t) {
  const aRgb = hexToRgb(a);
  const bRgb = hexToRgb(b);
  const out = aRgb.map((val, idx) => Math.round(lerp(val, bRgb[idx], t)));
  return rgbToHex(out);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function parseLocation() {
  const path = location.pathname.replace(/\/+$/, "") || "/dag";
  const page = path === "/tables" ? "tables" : "dag";
  const params = new URLSearchParams(location.search || "");
  return { page, params };
}

function buildUrl(page) {
  const params = new URLSearchParams();
  if (page === "dag") {
    if (state.dagMode) {
      params.set("mode", state.dagMode);
    }
    if (state.selectedProducerId !== null) {
      params.set("pid", String(state.selectedProducerId));
    }
  }
  const query = params.toString();
  return `/${page}${query ? `?${query}` : ""}`;
}

function syncUrl(push = false) {
  const page = state.activePage || "dag";
  const next = buildUrl(page);
  if (location.pathname + location.search === next) {
    return;
  }
  if (push) {
    history.pushState(null, "", next);
  } else {
    history.replaceState(null, "", next);
  }
}

function setActivePage(page) {
  state.activePage = page;
  if (appRoot) {
    appRoot.dataset.page = page;
  }
  if (pageDag) {
    pageDag.classList.toggle("is-active", page === "dag");
  }
  if (pageTables) {
    pageTables.classList.toggle("is-active", page === "tables");
  }
  navLinks.forEach((link) => {
    link.classList.toggle("active", link.dataset.nav === page);
  });
  if (page === "dag" && state.cy) {
    state.cy.resize();
  }
}

function applyRoute() {
  const { page, params } = parseLocation();
  setActivePage(page);
  const mode = params.get("mode");
  if (mode && dagModeSelect) {
    const normalized = mode === "observed" ? "observed" : "union";
    state.dagMode = normalized;
    dagModeSelect.value = normalized;
  }
  const pid = params.get("pid");
  if (pid !== null) {
    const parsed = Number(pid);
    if (!Number.isNaN(parsed)) {
      selectProducer(parsed, { focus: false, sync: false });
    }
  }
  syncUrl();
}

function updateLiveToggle() {
  if (!liveToggle) {
    return;
  }
  liveToggle.textContent = state.live ? "Live" : "Paused";
  liveToggle.classList.toggle("paused", !state.live);
  liveToggle.setAttribute("aria-pressed", state.live ? "true" : "false");
}

function startUiTimer() {
  stopUiTimer();
  const period = Math.max(0.05, 1 / Math.max(1, state.uiHz));
  state.uiTimer = setInterval(runUiTick, period * 1000);
}

function stopUiTimer() {
  if (state.uiTimer !== null) {
    clearInterval(state.uiTimer);
    state.uiTimer = null;
  }
}

function setLive(enabled) {
  state.live = Boolean(enabled);
  updateLiveToggle();
  if (state.live) {
    runUiTick();
    startUiTimer();
  } else {
    stopUiTimer();
  }
}

function setUiHz(value) {
  const next = Number(value);
  if (!Number.isNaN(next) && next > 0) {
    state.uiHz = next;
    if (state.live) {
      startUiTimer();
    }
  }
}

function ensureDagTooltip() {
  if (dagTooltip) {
    return dagTooltip;
  }
  const el = document.createElement("div");
  el.id = "dag-tooltip";
  el.className = "dag-tooltip";
  document.body.appendChild(el);
  dagTooltip = el;
  return el;
}

function buildDagTooltip(node) {
  const id = Number(node.id());
  const meta = state.dagNodesById.get(String(id)) || {};
  const fullName = meta.full_name || String(id);
  const className = meta.class_name || fullName;
  const metrics = state.metricsById.get(id) || {};
  const rollup = state.rollupById.get(id);
  return `
    <div class="tooltip-title">${escapeHtml(className)}</div>
    <div class="tooltip-sub">${escapeHtml(fullName)}</div>
    <div class="tooltip-grid">
      <div>Excl</div><div>${fmt(rollup?.excl_ms)}</div>
      <div>Incl</div><div>${fmt(rollup?.incl_ms)}</div>
      <div>Kernel</div><div>${fmt(metrics.kernel_ms)}</div>
      <div>Overhead</div><div>${fmt(metrics.overhead_est_ms)}</div>
    </div>
  `;
}

function showDagTooltip(node, event) {
  const tooltip = ensureDagTooltip();
  tooltip.innerHTML = buildDagTooltip(node);
  tooltip.classList.add("visible");
  moveDagTooltip(event);
}

function moveDagTooltip(event) {
  if (!dagTooltip || !event || !event.originalEvent) {
    return;
  }
  const evt = event.originalEvent;
  dagTooltip.style.left = `${evt.clientX + 12}px`;
  dagTooltip.style.top = `${evt.clientY + 12}px`;
}

function hideDagTooltip() {
  if (dagTooltip) {
    dagTooltip.classList.remove("visible");
  }
}

function inferClassName(fullName) {
  if (!fullName) {
    return "";
  }
  return String(fullName).split(".").pop() || "";
}

function shortId(value) {
  const text = String(value ?? "");
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = (hash * 31 + text.charCodeAt(i)) | 0;
  }
  const mod = Math.abs(hash) % 1296;
  return mod.toString(36).padStart(2, "0");
}

function labelForNode(id) {
  const meta = state.dagNodesById.get(String(id));
  if (!meta) {
    return String(id);
  }
  const className = meta.class_name || inferClassName(meta.full_name);
  const count = state.dagClassCounts.get(className) || 0;
  if (count > 1) {
    return `${className} #${shortId(meta.full_name)}`;
  }
  return className;
}

function buildNodeLabel(id) {
  const title = labelForNode(id);
  const meta = state.dagNodesById.get(String(id));
  const summaryRow =
    meta && meta.full_name ? state.summaryByName.get(meta.full_name) : null;
  const metrics = summaryRow || state.metricsById.get(id) || {};
  const rollup = state.rollupById.get(id) || {};
  const excl = rollup.excl_ms;
  const ema = metrics.ema_ms;
  const kern = metrics.kernel_ms;
  const overhead = metrics.overhead_est_ms;
  const kernelFrac = metrics.kernel_frac;
  const calls = rollup.calls ?? metrics.calls;
  const cook = metrics.last_update_id ?? metrics.last_update ?? "-";
  const line1 = `EXCL ${fmt(excl)}  EMA ${fmt(ema)}`;
  const line2 = `KERN ${fmt(kern)}  ${
    overhead ? `OH ${fmt(overhead)}` : `KF ${fmtPct(kernelFrac)}`
  }`;
  const line3 = `CALLS ${calls ?? "-"}  CK ${cook}`;
  if (state.isDense && !state.isZoomed) {
    return `${title}\n${line1}\n${line2}`;
  }
  return `${title}\n${line1}\n${line2}\n${line3}`;
}

function ensureDag() {
  if (!dagContainer || !window.cytoscape) {
    return null;
  }
  if (state.cy) {
    return state.cy;
  }
  if (window.cytoscapeDagre) {
    window.cytoscape.use(window.cytoscapeDagre);
  }
  state.cy = window.cytoscape({
    container: dagContainer,
    layout: {
      name: "dagre",
      rankDir: "TB",
      nodeSep: 46,
      edgeSep: 12,
      rankSep: 80,
    },
    selectionType: "single",
    autounselectify: true,
    boxSelectionEnabled: false,
    style: [
      {
        selector: "node",
        style: {
          shape: "round-rectangle",
          width: 248,
          height: 128,
          "background-color": BASE_NODE_COLOR,
          "border-color": "#c9d3e2",
          "border-width": 1,
          "shadow-blur": 10,
          "shadow-opacity": 0.16,
          "shadow-color": "#b7c3d4",
          label: "data(label)",
          "text-opacity": 1,
          "text-outline-opacity": 0.9,
          "background-opacity": 1,
          color: "#1b2330",
          "font-size": 14,
          "font-weight": 600,
          "font-family": "Sora, sans-serif",
          "min-zoomed-font-size": 11,
          "text-outline-width": 0.8,
          "text-outline-color": "#f7f9fc",
          "line-height": 1.25,
          "text-halign": "center",
          "text-valign": "center",
          "text-justification": "left",
          "text-wrap": "wrap",
          "text-max-width": 220,
          "text-margin-x": 0,
          "text-margin-y": 0,
          "transition-property":
            "background-color, border-color, border-width, shadow-opacity, shadow-blur",
          "transition-duration": "350ms",
        },
      },
      {
        selector: "node.is-selected",
        style: {
          width: 300,
          height: 168,
          "overlay-color": "#1f6fb2",
          "overlay-opacity": 0.08,
          "overlay-padding": 6,
        },
      },
      {
        selector: "node.age-0",
        style: {
          "border-width": 2.5,
          "border-color": "#1f6fb2",
          "shadow-color": "#1f6fb2",
          "shadow-opacity": 0.4,
          "shadow-blur": 18,
        },
      },
      {
        selector: "node.age-1-3",
        style: {
          "border-width": 2,
          "border-color": "#1f6fb2",
          "shadow-color": "#1f6fb2",
          "shadow-opacity": 0.3,
          "shadow-blur": 14,
        },
      },
      {
        selector: "node.age-4-10",
        style: {
          "border-width": 1.5,
          "border-color": "#9bb0c8",
          "shadow-color": "#9bb0c8",
          "shadow-opacity": 0.2,
          "shadow-blur": 10,
        },
      },
      {
        selector: "node.age-old",
        style: {
          "border-width": 1,
          "border-color": "#c9d3e2",
          "shadow-opacity": 0.1,
          "shadow-blur": 6,
        },
      },
      {
        selector: "node.executed-now",
        style: {
          "border-color": "#e18643",
          "shadow-color": "#e18643",
          "shadow-opacity": 0.35,
          "shadow-blur": 16,
        },
      },
      {
        selector: "node.pulse",
        style: {
          "border-width": 3,
          "shadow-opacity": 0.8,
          "shadow-blur": 24,
        },
      },
      {
        selector: "node.dimmed",
        style: {
          opacity: 0.18,
        },
      },
      {
        selector: "edge",
        style: {
          width: 1.25,
          "line-color": "#b7c3d4",
          "target-arrow-color": "#b7c3d4",
          "target-arrow-shape": "triangle",
          "curve-style": "bezier",
          opacity: 0.55,
          "transition-property": "line-color, target-arrow-color, width, opacity",
          "transition-duration": "350ms",
        },
      },
      {
        selector: "edge.edge-executed",
        style: {
          width: 2.25,
          "line-color": "#e18643",
          "target-arrow-color": "#e18643",
          opacity: 0.95,
        },
      },
      {
        selector: "edge.dimmed",
        style: {
          opacity: 0.12,
        },
      },
    ],
    elements: [],
  });

  state.cy.on("tap", "node", (evt) => {
    const id = Number(evt.target.id());
    if (!Number.isNaN(id)) {
      selectProducer(id, { focus: true, sync: true });
    }
  });
  state.cy.on("tap", (evt) => {
    if (evt.target === state.cy) {
      clearSelection();
    }
  });
  state.cy.on("mouseover", "node", (evt) => {
    showDagTooltip(evt.target, evt);
  });
  state.cy.on("mousemove", "node", (evt) => {
    moveDagTooltip(evt);
  });
  state.cy.on("mouseout", "node", () => {
    hideDagTooltip();
  });
  state.cy.on("zoom", () => {
    updateDagDensity();
  });
  return state.cy;
}

const DAG_POS_PREFIX = "dag_pos::";
const DAG_LAYOUT_VERSION = "v4-card";

function dagPositionsKey(version, mode) {
  return `${DAG_POS_PREFIX}${DAG_LAYOUT_VERSION}::${mode || "union"}::${version}`;
}

function loadDagPositions(version, mode) {
  if (version === null || version === undefined) {
    return null;
  }
  try {
    const raw = localStorage.getItem(dagPositionsKey(version, mode));
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return null;
    }
    return parsed;
  } catch (err) {
    return null;
  }
}

function saveDagPositions(version, mode, cy) {
  if (version === null || version === undefined || !cy) {
    return;
  }
  const positions = {};
  cy.nodes().forEach((node) => {
    positions[node.id()] = node.position();
  });
  try {
    localStorage.setItem(dagPositionsKey(version, mode), JSON.stringify(positions));
  } catch (err) {
    return;
  }
}

function applyDag(snap) {
  if (!snap || !dagContainer) {
    return;
  }
  const cy = ensureDag();
  if (!cy) {
    return;
  }
  const nextVersion = snap.dag_version ?? 0;
  const nodes = Array.isArray(snap.nodes) ? snap.nodes : [];
  const edges = Array.isArray(snap.edges) ? snap.edges : [];
  const classCounts = new Map();
  nodes.forEach((node) => {
    const fullName = node.full_name || node.name || String(node.id);
    const className = node.class_name || inferClassName(fullName);
    classCounts.set(className, (classCounts.get(className) || 0) + 1);
  });
  state.dagNodesById = new Map();
  state.dagEdges = edges;
  state.dagClassCounts = classCounts;
  nodes.forEach((node) => {
    const fullName = node.full_name || node.name || String(node.id);
    state.dagNodesById.set(String(node.id), {
      id: node.id,
      full_name: fullName,
      class_name: node.class_name || inferClassName(fullName),
    });
  });

  const shouldRebuild =
    state.dagVersion !== nextVersion || state.dagModeApplied !== state.dagMode;

  if (shouldRebuild) {
    const elements = [];
    nodes.forEach((node) => {
      elements.push({
        data: {
          id: String(node.id),
          label: buildNodeLabel(node.id),
        },
      });
    });
    edges.forEach((edge, idx) => {
      elements.push({
        data: {
          id: `${edge.source}-${edge.target}-${idx}`,
          source: String(edge.source),
          target: String(edge.target),
          seen_count: edge.seen_count ?? 0,
          via_resources: edge.via_resources || [],
        },
      });
    });
    const cached = loadDagPositions(nextVersion, state.dagMode);
    cy.batch(() => {
      cy.elements().remove();
      cy.add(elements);
    });
    if (cached) {
      cy.batch(() => {
        cy.nodes().forEach((node) => {
          const pos = cached[node.id()];
          if (pos && typeof pos.x === "number" && typeof pos.y === "number") {
            node.position(pos);
          }
        });
      });
    } else {
      cy.once("layoutstop", () => {
        saveDagPositions(nextVersion, state.dagMode, cy);
        if (!state.didInitialFit) {
          cy.fit();
          state.didInitialFit = true;
        }
      });
      cy
        .layout({
          name: "dagre",
          rankDir: "TB",
          nodeSep: 46,
          edgeSep: 12,
          rankSep: 80,
        })
        .run();
    }
    state.dagVersion = nextVersion;
    state.dagModeApplied = state.dagMode;
    state.nodeDetailsCache.clear();
    if (state.selectedProducerId !== null && nodeDetails) {
      nodeDetails.textContent = "Loading details...";
      fetchNodeDetails(state.selectedProducerId);
    }
  } else {
    cy.batch(() => {
      cy.nodes().forEach((node) => {
        node.data("label", buildNodeLabel(Number(node.id())));
      });
    });
  }
  updateDagMetrics();
  applyDagSearch(state.searchQuery);
  syncDagSelection();
  updateDagDensity();
}

function updateDagDensity() {
  if (!state.cy || !dagContainer) {
    return;
  }
  const dense = state.cy.nodes().length > state.denseThreshold;
  const zoomed = state.cy.zoom() > 0.7;
  const denseChanged = dense !== state.isDense;
  const zoomChanged = zoomed !== state.isZoomed;
  state.isDense = dense;
  state.isZoomed = zoomed;
  dagContainer.classList.toggle("dag-dense", dense);
  dagContainer.classList.toggle("dag-zoomed", zoomed);
  if (denseChanged || zoomChanged) {
    updateDagMetrics();
  }
}

function updateDagMetrics() {
  const cy = state.cy;
  if (!cy) {
    return;
  }
  const rollup = state.rollupById;
  const percentiles = state.rollupPercentiles;
  cy.batch(() => {
    cy.nodes().forEach((node) => {
      const id = Number(node.id());
      const metrics = state.metricsById.get(id) || {};
      const percentile = percentiles.get(id) || 0;
      const bgColor = lerpColor(BASE_NODE_COLOR, MAGNITUDE_COLOR, percentile);
      node.data("label", buildNodeLabel(id));
      node.style("background-color", bgColor);
      node.data("executed_this_cook", metrics.executed_this_cook || false);
    });
  });
}

function applyDagTickUpdate(executedIds) {
  const cy = state.cy;
  if (!cy) {
    return;
  }
  const newlyExecuted = new Set();
  executedIds.forEach((id) => {
    if (!state.executedLastTick.has(id)) {
      newlyExecuted.add(id);
    }
  });
  if (state.pulseTimer) {
    clearTimeout(state.pulseTimer);
    state.pulseTimer = null;
  }
  const cookId = state.lastCookId;
  const rollup = state.rollupById;
  const percentiles = state.rollupPercentiles;
  cy.batch(() => {
    cy.nodes().forEach((node) => {
      const id = Number(node.id());
      const metrics = state.metricsById.get(id) || {};
      const percentile = percentiles.get(id) || 0;
      const bgColor = lerpColor(BASE_NODE_COLOR, MAGNITUDE_COLOR, percentile);
      node.data("label", buildNodeLabel(id));
      node.style("background-color", bgColor);
      node.data("executed_this_cook", metrics.executed_this_cook || false);
      node.removeClass("age-0 age-1-3 age-4-10 age-old");
      const lastUpdate = metrics.last_update_id;
      const age =
        cookId === null || cookId === undefined || lastUpdate === undefined
          ? null
          : cookId - lastUpdate;
      if (age === null || age < 0) {
        node.addClass("age-old");
      } else if (age === 0) {
        node.addClass("age-0");
      } else if (age <= 3) {
        node.addClass("age-1-3");
      } else if (age <= 10) {
        node.addClass("age-4-10");
      } else {
        node.addClass("age-old");
      }
      node.toggleClass("executed-now", executedIds.has(id));
      node.toggleClass("pulse", newlyExecuted.has(id));
      if (rollup.has(id)) {
        node.data("rollup", rollup.get(id));
      }
    });
    cy.edges().removeClass("edge-executed");
    cy.edges().forEach((edge) => {
      const src = Number(edge.source().id());
      const dst = Number(edge.target().id());
      if (executedIds.has(src) && executedIds.has(dst)) {
        edge.addClass("edge-executed");
      }
    });
  });
  state.executedLastTick = executedIds;
  state.pulseTimer = setTimeout(() => {
    if (!state.cy) {
      return;
    }
    state.cy.batch(() => {
      state.cy.nodes().removeClass("pulse");
    });
  }, 450);
}

function applyMetricsSnapshot(snap) {
  if (!snap) {
    return;
  }
  const rows = Array.isArray(snap.rows) ? snap.rows : [];
  const map = new Map();
  const executedIds = new Set();
  rows.forEach((row) => {
    map.set(row.id, row);
    if (row.executed_this_cook) {
      executedIds.add(row.id);
    }
  });
  state.metricsById = map;
  const cookId = snap.cook_id ?? null;
  const cookChanged = cookId !== state.lastCookId;
  state.lastCookId = cookId;
  if (cookIndicator) {
    cookIndicator.textContent = `Cook ${cookId ?? "-"}`;
  }
  applyDagTickUpdate(executedIds);
  if (cookChanged) {
    refreshExecTree(cookId);
  }
  if (state.selectedProducerId !== null) {
    const cached = state.nodeDetailsCache.get(state.selectedProducerId);
    if (cached) {
      renderNodeDetails(cached);
    }
  }
}

function computeRollup(nodes) {
  const rollup = new Map();
  nodes.forEach((node) => {
    const id = node.producer_id;
    if (id === null || id === undefined) {
      return;
    }
    const entry = rollup.get(id) || { excl_ms: 0, incl_ms: 0, calls: 0 };
    entry.excl_ms += node.exclusive_ms || 0;
    entry.incl_ms += node.inclusive_ms || 0;
    entry.calls += 1;
    rollup.set(id, entry);
  });
  return rollup;
}

function percentileForValue(sorted, value) {
  if (!sorted.length) {
    return 0;
  }
  let lo = 0;
  let hi = sorted.length - 1;
  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (sorted[mid] <= value) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  const idx = Math.max(0, hi);
  if (sorted.length === 1) {
    return 1;
  }
  return clamp(idx / (sorted.length - 1), 0, 1);
}

function computePercentiles(rollup) {
  const values = [];
  rollup.forEach((entry) => {
    if (entry.excl_ms > 0) {
      values.push(entry.excl_ms);
    }
  });
  values.sort((a, b) => a - b);
  const percentiles = new Map();
  rollup.forEach((entry, id) => {
    if (entry.excl_ms > 0) {
      percentiles.set(id, percentileForValue(values, entry.excl_ms));
    } else {
      percentiles.set(id, 0);
    }
  });
  return percentiles;
}

function applyExecTreeSnapshot(snap) {
  if (!snap) {
    state.rollupById = new Map();
    state.rollupPercentiles = new Map();
    return;
  }
  const nodes = Array.isArray(snap.nodes) ? snap.nodes : [];
  state.rollupById = computeRollup(nodes);
  state.rollupPercentiles = computePercentiles(state.rollupById);
  state.rollupCookId = snap.cook_id ?? null;
  updateDagMetrics();
  if (state.selectedProducerId !== null) {
    const cached = state.nodeDetailsCache.get(state.selectedProducerId);
    if (cached) {
      renderNodeDetails(cached);
    }
  }
}

function applySummarySnapshot(snap) {
  if (!snap) {
    return;
  }
  if (healthDropped) {
    healthDropped.textContent = snap.dropped_events ?? "-";
  }
  if (healthEdges) {
    healthEdges.textContent = snap.edges_recorded ?? "-";
  }
  if (healthOverhead) {
    healthOverhead.textContent = fmt(snap.profiler_overhead_us, 1);
  }
  state.summaryRows = Array.isArray(snap.rows) ? snap.rows : [];
  state.summaryByName = new Map();
  state.summaryRows.forEach((row) => {
    if (row && row.name) {
      state.summaryByName.set(row.name, row);
    }
  });
  state.categories = snap.categories || {};
  if (snap.dag_version !== undefined && snap.dag_version !== null) {
    if (state.lastDagVersionSeen !== snap.dag_version) {
      state.lastDagVersionSeen = snap.dag_version;
      refreshDag();
    }
  }
  if (state.activePage === "tables") {
    renderTables();
  }
  if (state.selectedProducerId !== null) {
    const cached = state.nodeDetailsCache.get(state.selectedProducerId);
    if (cached) {
      renderNodeDetails(cached);
    }
  }
}

async function refreshDag() {
  if (state.pending.dag || !state.live) {
    return;
  }
  const mode = state.dagMode || "union";
  state.pending.dag = true;
  try {
    const response = await fetch(`/api/dag?mode=${encodeURIComponent(mode)}`, {
      cache: "no-store",
    });
    if (!response.ok) {
      return;
    }
    if (!state.live) {
      return;
    }
    const snap = await response.json();
    requestAnimationFrame(() => applyDag(snap));
  } catch (err) {
    console.warn("dag refresh failed", err);
  } finally {
    state.pending.dag = false;
  }
}

async function refreshMetrics() {
  if (state.pending.metrics || !state.live) {
    return;
  }
  state.pending.metrics = true;
  try {
    const response = await fetch("/api/metrics", { cache: "no-store" });
    if (!response.ok) {
      return;
    }
    if (!state.live) {
      return;
    }
    const snap = await response.json();
    requestAnimationFrame(() => applyMetricsSnapshot(snap));
  } catch (err) {
    console.warn("metrics refresh failed", err);
  } finally {
    state.pending.metrics = false;
  }
}

async function refreshSummary() {
  if (state.pending.summary || !state.live) {
    return;
  }
  state.pending.summary = true;
  try {
    const response = await fetch("/api/summary", { cache: "no-store" });
    if (!response.ok) {
      return;
    }
    if (!state.live) {
      return;
    }
    const snap = await response.json();
    requestAnimationFrame(() => applySummarySnapshot(snap));
  } catch (err) {
    console.warn("summary refresh failed", err);
  } finally {
    state.pending.summary = false;
  }
}

async function refreshExecTree(cookId) {
  if (state.pending.execTree || !state.live) {
    return;
  }
  state.pending.execTree = true;
  try {
    const url = cookId === null || cookId === undefined
      ? "/api/exec_tree"
      : `/api/exec_tree?cook_id=${encodeURIComponent(cookId)}`;
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      return;
    }
    if (!state.live) {
      return;
    }
    const snap = await response.json();
    requestAnimationFrame(() => applyExecTreeSnapshot(snap));
  } catch (err) {
    console.warn("exec tree refresh failed", err);
  } finally {
    state.pending.execTree = false;
  }
}

function runUiTick() {
  refreshMetrics();
  refreshSummary();
}

function applyDagSearch(query) {
  state.searchQuery = query || "";
  const cy = state.cy;
  if (!cy) {
    return;
  }
  const q = state.searchQuery.trim().toLowerCase();
  cy.batch(() => {
    if (!q) {
      cy.nodes().removeClass("dimmed");
      cy.edges().removeClass("dimmed");
      return;
    }
    cy.nodes().forEach((node) => {
      const id = Number(node.id());
      const meta = state.dagNodesById.get(String(id)) || {};
      const fullName = String(meta.full_name || "");
      const className = String(meta.class_name || "");
      const match =
        fuzzyMatch(fullName.toLowerCase(), q) ||
        fuzzyMatch(className.toLowerCase(), q);
      node.toggleClass("dimmed", !match);
    });
    cy.edges().forEach((edge) => {
      const dimmed =
        edge.source().hasClass("dimmed") || edge.target().hasClass("dimmed");
      edge.toggleClass("dimmed", dimmed);
    });
  });
  applySelectionStyles();
}

function fuzzyMatch(text, query) {
  if (!query) {
    return true;
  }
  let t = 0;
  let q = 0;
  while (t < text.length && q < query.length) {
    if (text[t] === query[q]) {
      q += 1;
    }
    t += 1;
  }
  return q === query.length;
}

function bestFuzzyMatch(nodes, query) {
  let best = null;
  let bestScore = Infinity;
  nodes.forEach((node) => {
    const id = Number(node.id());
    const meta = state.dagNodesById.get(String(id)) || {};
    const fullName = String(meta.full_name || "").toLowerCase();
    const className = String(meta.class_name || "").toLowerCase();
    const score = fuzzyScore(fullName, query);
    const classScore = fuzzyScore(className, query);
    const minScore = Math.min(score, classScore);
    if (minScore < bestScore) {
      bestScore = minScore;
      best = node;
    }
  });
  return best;
}

function fuzzyScore(text, query) {
  if (!query) {
    return 0;
  }
  let score = 0;
  let t = 0;
  let q = 0;
  while (t < text.length && q < query.length) {
    if (text[t] === query[q]) {
      q += 1;
    } else {
      score += 1;
    }
    t += 1;
  }
  if (q < query.length) {
    return Infinity;
  }
  return score + (text.length - t);
}

function focusBestSearchResult() {
  const cy = state.cy;
  if (!cy) {
    return;
  }
  const query = state.searchQuery.trim().toLowerCase();
  if (!query) {
    return;
  }
  const matches = cy.nodes().filter((node) => !node.hasClass("dimmed"));
  if (!matches.length) {
    return;
  }
  const best = bestFuzzyMatch(matches, query) || matches[0];
  if (best) {
    cy.fit(best, 120);
    const id = Number(best.id());
    if (!Number.isNaN(id)) {
      selectProducer(id, { focus: false, sync: true });
    }
  }
}

function applySelectionStyles() {
  const cy = state.cy;
  if (!cy) {
    return;
  }
  const selection = state.selectedProducerId;
  const hasSearch = state.searchQuery.trim().length > 0;
  cy.batch(() => {
    cy.nodes().removeClass("is-selected");
    if (selection === null) {
      if (!hasSearch) {
        cy.nodes().removeClass("dimmed");
        cy.edges().removeClass("dimmed");
      }
      return;
    }
    const selectedNode = cy.$id(String(selection));
    if (!selectedNode) {
      return;
    }
    selectedNode.addClass("is-selected");
    const neighbors = selectedNode.neighborhood().nodes();
    const keep = new Set([selectedNode.id(), ...neighbors.map((n) => n.id())]);
    cy.nodes().forEach((node) => {
      const id = node.id();
      const shouldDim = !keep.has(id) || (hasSearch && node.hasClass("dimmed"));
      node.toggleClass("dimmed", shouldDim);
    });
    cy.edges().forEach((edge) => {
      const dimmed =
        edge.source().hasClass("dimmed") || edge.target().hasClass("dimmed");
      edge.toggleClass("dimmed", dimmed);
    });
  });
}

function syncDagSelection() {
  const cy = state.cy;
  if (!cy || state.selectedProducerId === null) {
    return;
  }
  const node = cy.$id(String(state.selectedProducerId));
  if (node) {
    node.addClass("is-selected");
  }
}

function clearSelection() {
  state.selectedProducerId = null;
  renderNodeDetails(null);
  applySelectionStyles();
  syncUrl();
}

function selectProducer(id, { focus = false, sync = false, push = false } = {}) {
  state.selectedProducerId = id;
  const cached = state.nodeDetailsCache.get(id);
  if (cached) {
    renderNodeDetails(cached);
  } else if (nodeDetails) {
    nodeDetails.textContent = "Loading details...";
  }
  fetchNodeDetails(id);
  if (focus && state.cy) {
    const node = state.cy.$id(String(id));
    if (node) {
      state.cy.center(node);
    }
  }
  applySelectionStyles();
  if (sync) {
    syncUrl(push);
  }
}

async function fetchNodeDetails(id) {
  if (state.pending.nodeDetails) {
    state.pending.nodeDetails.abort();
  }
  const controller = new AbortController();
  state.pending.nodeDetails = controller;
  try {
    const response = await fetch(`/api/node/${encodeURIComponent(id)}`, {
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok) {
      if (state.selectedProducerId === id && nodeDetails) {
        nodeDetails.textContent = "Details unavailable.";
      }
      return;
    }
    const details = await response.json();
    state.nodeDetailsCache.set(id, details);
    if (state.selectedProducerId === id) {
      renderNodeDetails(details);
    }
  } catch (err) {
    if (err && err.name === "AbortError") {
      return;
    }
    if (state.selectedProducerId === id && nodeDetails) {
      nodeDetails.textContent = "Details unavailable.";
    }
    console.warn("node details fetch failed", err);
  } finally {
    if (state.pending.nodeDetails === controller) {
      state.pending.nodeDetails = null;
    }
  }
}

function renderNodeDetails(details) {
  if (!nodeDetails) {
    return;
  }
  if (!details) {
    nodeDetails.textContent = "Select a producer to inspect.";
    return;
  }
  const id = details.id;
  const fullName = details.full_name || details.name || String(id);
  const className = details.class_name || inferClassName(fullName);
  const summaryRow = state.summaryByName.get(fullName);
  const metrics =
    summaryRow || state.metricsById.get(id) || details.metrics || {};
  const rollup = state.rollupById.get(id) || {};
  const cook = state.lastCookId ?? "-";
  const resources = details.resources_read || [];
  const inputs = details.inputs || [];
  const outputs = details.outputs || [];
  const staleness = details.staleness_reason || "";
  const lastMs = metrics.last_ms ?? "-";
  const emaMs = metrics.ema_ms ?? "-";
  const lastUpdate =
    metrics.last_update ?? metrics.last_update_id ?? "-";
  const renderList = (items) =>
    items.length
      ? items.map((item) => `<span class="chip">${escapeHtml(item)}</span>`).join("")
      : "<span class=\"chip\">-</span>";
  const renderIO = (items) =>
    items.length
      ? items
          .map((item) => `<span class="chip">${escapeHtml(item.id)}@${escapeHtml(item.version)}</span>`)
          .join("")
      : "<span class=\"chip\">-</span>";

  nodeDetails.innerHTML = `
    <div class="details-header">
      <div>
        <div class="details-title">${escapeHtml(className)}</div>
        <div class="details-sub">${escapeHtml(fullName)}</div>
      </div>
      <div class="details-actions">
        <button class="btn small" type="button" data-focus="true">Focus in DAG</button>
        <button class="btn small" type="button" data-copy="${escapeHtml(fullName)}">Copy name</button>
      </div>
    </div>
    <div class="details-card">
      <h4>Last cook</h4>
      <div class="details-grid">
        <div><span>EXCL</span><strong>${fmt(rollup.excl_ms)}</strong></div>
        <div><span>INCL</span><strong>${fmt(rollup.incl_ms)}</strong></div>
        <div><span>CALLS</span><strong>${rollup.calls ?? "-"}</strong></div>
        <div><span>COOK</span><strong>${cook}</strong></div>
      </div>
    </div>
    <div class="details-card">
      <h4>Runtime metrics</h4>
      <div class="details-grid">
        <div><span>LAST</span><strong>${fmt(lastMs)}</strong></div>
        <div><span>EMA</span><strong>${fmt(emaMs)}</strong></div>
        <div><span>KERN</span><strong>${fmt(metrics.kernel_ms)}</strong></div>
        <div><span>KF</span><strong>${fmtPct(metrics.kernel_frac)}</strong></div>
        <div><span>OH</span><strong>${fmt(metrics.overhead_est_ms)}</strong></div>
        <div><span>CALLS</span><strong>${metrics.calls ?? "-"}</strong></div>
        <div><span>UPDATED</span><strong>${lastUpdate}</strong></div>
      </div>
    </div>
    <div class="details-card">
      <h4>Resources + staleness</h4>
      ${staleness ? `<p class="details-note">${escapeHtml(staleness)}</p>` : ""}
      <div class="details-row">
        <span>Reads</span>
        <div class="chip-list">${renderList(resources)}</div>
      </div>
    </div>
    <div class="details-card">
      <h4>Inputs / outputs</h4>
      <div class="details-row">
        <span>Inputs</span>
        <div class="chip-list">${renderIO(inputs)}</div>
      </div>
      <div class="details-row">
        <span>Outputs</span>
        <div class="chip-list">${renderIO(outputs)}</div>
      </div>
    </div>
  `;
}

function renderTables() {
  if (!producerHead || !producerBody) {
    return;
  }
  const executedLookup = new Map();
  state.metricsById.forEach((row) => {
    executedLookup.set(row.name, row.executed_this_cook);
  });
  const sortKey = state.tableSort.key;
  const sortDir = state.tableSort.dir;
  const rows = state.summaryRows.slice();
  const query = state.tableSearch.trim().toLowerCase();
  const filtered = query
    ? rows.filter((row) => String(row.name || "").toLowerCase().includes(query))
    : rows;
  const sortValue = (row, key) => {
    if (key === "executed") {
      return executedLookup.get(row.name) ? 1 : 0;
    }
    if (key === "name") {
      return String(row.name || "");
    }
    return row[key] ?? 0;
  };
  filtered.sort((a, b) => {
    const aVal = sortValue(a, sortKey);
    const bVal = sortValue(b, sortKey);
    if (typeof aVal === "string" || typeof bVal === "string") {
      const left = String(aVal);
      const right = String(bVal);
      return sortDir === "asc"
        ? left.localeCompare(right)
        : right.localeCompare(left);
    }
    if (sortDir === "asc") {
      return aVal - bVal;
    }
    return bVal - aVal;
  });
  const columns = [
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
  producerHead.innerHTML = columns
    .map((col) => {
      const active = col.key === sortKey;
      const dir = active ? (sortDir === "asc" ? "▲" : "▼") : "";
      return `<th data-sort="${col.key}">${escapeHtml(col.label)} ${dir}</th>`;
    })
    .join("");
  producerBody.innerHTML = filtered
    .map((row) => {
      const executed = executedLookup.get(row.name);
      const execDot = `<span class="exec-dot ${executed ? "active" : ""}"></span>`;
      return `
        <tr data-producer-name="${escapeHtml(row.name)}">
          <td class="exec-cell">${execDot}</td>
          <td>${escapeHtml(row.name)}</td>
          <td>${fmt(row.ema_ms)}</td>
          <td>${fmt(row.last_ms)}</td>
          <td>${fmt(row.kernel_ms)}</td>
          <td>${fmtPct(row.kernel_frac)}</td>
          <td>${fmt(row.overhead_est_ms)}</td>
          <td>${row.calls ?? 0}</td>
          <td>${row.last_update ?? "-"}</td>
        </tr>
      `;
    })
    .join("");
  renderCategories(query);
}

function renderCategories(query) {
  if (!categoryPanels) {
    return;
  }
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
      section.className = "category-panel";
      section.innerHTML = `
        <h3>${escapeHtml(cat)}</h3>
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

if (dagSearch) {
  dagSearch.addEventListener("input", () => {
    applyDagSearch(dagSearch.value || "");
  });
  dagSearch.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      focusBestSearchResult();
    }
  });
}

if (fitViewBtn) {
  fitViewBtn.addEventListener("click", () => {
    if (state.cy) {
      state.cy.fit();
    }
  });
}

if (dagModeSelect) {
  state.dagMode = dagModeSelect.value || "union";
  dagModeSelect.addEventListener("change", () => {
    state.dagMode = dagModeSelect.value || "union";
    refreshDag();
    syncUrl();
  });
}

if (liveToggle) {
  liveToggle.addEventListener("click", () => {
    setLive(!state.live);
  });
}

if (uiHzSelect) {
  state.uiHz = Number(uiHzSelect.value) || 4;
  uiHzSelect.addEventListener("change", () => {
    setUiHz(uiHzSelect.value);
  });
}

navLinks.forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    const page = link.dataset.nav || "dag";
    setActivePage(page);
    syncUrl(true);
    if (page === "tables") {
      renderTables();
    }
  });
});

if (producerHead) {
  producerHead.addEventListener("click", (event) => {
    const th = event.target.closest("th[data-sort]");
    if (!th) {
      return;
    }
    const key = th.dataset.sort;
    if (!key) {
      return;
    }
    if (state.tableSort.key === key) {
      state.tableSort.dir = state.tableSort.dir === "asc" ? "desc" : "asc";
    } else {
      state.tableSort = { key, dir: "desc" };
    }
    renderTables();
  });
}

if (producerBody) {
  producerBody.addEventListener("click", (event) => {
    const row = event.target.closest("tr[data-producer-name]");
    if (!row) {
      return;
    }
    const name = row.dataset.producerName;
    if (!name) {
      return;
    }
    const match = Array.from(state.metricsById.values()).find(
      (rowData) => rowData.name === name
    );
    if (match) {
      setActivePage("dag");
      selectProducer(match.id, { focus: true, sync: true, push: true });
    }
  });
}

if (tableSearch) {
  tableSearch.addEventListener("input", () => {
    state.tableSearch = tableSearch.value || "";
    renderTables();
  });
}

if (nodeDetails) {
  nodeDetails.addEventListener("click", (event) => {
    const focusBtn = event.target.closest("[data-focus]");
    if (focusBtn && state.cy && state.selectedProducerId !== null) {
      const node = state.cy.$id(String(state.selectedProducerId));
      if (node) {
        state.cy.fit(node, 120);
      }
      return;
    }
    const copyBtn = event.target.closest("[data-copy]");
    if (copyBtn) {
      const text = copyBtn.getAttribute("data-copy");
      if (text) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).catch(() => {});
        } else {
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
      }
    }
  });
}

window.addEventListener("popstate", () => {
  applyRoute();
});

applyRoute();
updateLiveToggle();
refreshDag();
runUiTick();
startUiTimer();
