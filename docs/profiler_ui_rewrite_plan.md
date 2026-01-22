# Profiler UI React Rebuild Plan

## Goals

- Replace polling with the `/ws` stream for summary/metrics/exec-tree/dag.
- Modernize the UI with a cohesive visual system and faster interactions.
- Keep feature parity with current DAG explorer + tables + details drawer.
- Scale to large graphs (hundreds to thousands of nodes) with stable layout.

## Stack + Tooling

- React 18 + TypeScript for UI structure and safer data modeling.
- Vite for fast local builds and static asset output.
- React Flow (`@xyflow/react`) + ELK (`elkjs`) for DAG rendering and layout.
- TanStack Table + react-virtual for large tables without DOM thrash.
- CSS modules or vanilla CSS with design tokens (no CSS-in-JS required).

## Data Contract

- WebSocket `/ws` becomes the primary transport.
- UI expects messages with:
  - Summary snapshot (top-level) each tick.
  - `metrics` snapshot each tick.
  - `exec_tree` only when cook changes.
  - `dag` only when `dag_version` changes (union by default).
- Keep HTTP endpoints for on-demand fetch (node details, observed DAG mode).

## Component Map

- `AppShell`: navigation + status + live toggle + update rate.
- `DagPage`:
  - `DagToolbar`: mode switch, search, fit, view options.
  - `DagCanvas`: React Flow graph + layout cache + selection handling.
  - `DagInspector`: right drawer for node details + quick actions.
- `TablesPage`:
  - `ProducerTable`: sortable, virtualized table.
  - `CategoryPanels`: expandable rollups with search.
- `MetricsStrip`: global health + cook/frame + overhead indicators.

## DAG Rendering Plan (React Flow + ELK)

- Use React Flow nodes/edges with ELK layout for DAG-specific spacing.
- Maintain a layout cache keyed by `dag_version` and `mode`.
- Use a "stable positions first" strategy to avoid jarring relayouts:
  - Reuse cached positions when the node set is unchanged.
  - Animate transitions when the layout changes.
- Use a compact node design with row-based metrics (EXCL/EMA/KERN/CK).
- Provide "focus selection" and "neighbor highlight" interactions.

## Visual Direction

- Typography: "Space Grotesk" for headings, "IBM Plex Sans" for body.
- Palette: warm sand background + deep slate text + teal/copper accents.
- Layout: asymmetric split panels, prominent DAG stage, stacked cards.
- Motion: page-load fade + slide; DAG layout transitions; drawer reveal.
- Use CSS variables for all colors, sizing, and durations.

## Implementation Phases

1. Baseline React shell + WS client
   - Vite + React app served from `rheidos/compute/profiler/ui`.
   - WS client hooks with shared state store (Context or Zustand).
2. DAG view
   - React Flow canvas + ELK layout + search + selection drawer.
3. Tables view
   - Virtualized producer table + category panels.
4. Polish + parity
   - URL routing, live toggle, per-node actions, copy, focus in DAG.
5. Performance sweep
   - Memoization, layout cache, large-graph tests.

## Migration Plan

- Keep legacy UI live while React UI ships under a `?ui=react` flag.
- Verify WS payload compatibility + fallbacks.
- Remove legacy UI once React UI is stable and tested.

## Open Questions

- Should observed DAG mode be pushed via WS or fetched on-demand?
- Is there interest in a third page for execution tree visualization?
- Any constraints on adding a small JS build step (Vite output)?
