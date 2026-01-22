import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  type Edge,
  MarkerType,
  Position,
  type Node,
  type NodeTypes,
  type ReactFlowInstance,
} from "@xyflow/react";
import React, {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import { fmt, fmtPct, fuzzyMatch, shortId } from "../lib/format";
import { layoutDag } from "../lib/layout";
import type { DagSnapshot, MetricsRow, SummaryRow } from "../types";
import type { RollupEntry } from "../lib/rollup";
import DagNode, { type DagNodeData } from "./DagNode";
import ParallelEdge from "./ParallelEdge";

export type DagCanvasHandle = {
  fitView: () => void;
  focusNode: (id: string) => void;
};

type DagCanvasProps = {
  dag: DagSnapshot | null;
  metricsById: Map<string, MetricsRow>;
  summaryByName: Map<string, SummaryRow>;
  rollup: Map<string, RollupEntry>;
  percentiles: Map<string, number>;
  classCounts: Map<string, number>;
  currentCookId: number | null;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  searchQuery: string;
};

const nodeTypes: NodeTypes = {
  dagNode: DagNode,
};

const edgeTypes = {
  parallel: ParallelEdge,
};

const EDGE_MARKER = { type: MarkerType.ArrowClosed, color: "#1f8a70" };

function inferClassName(fullName: string) {
  if (!fullName) {
    return "";
  }
  const parts = fullName.split(".");
  return parts[parts.length - 1] || fullName;
}

function buildNodeLabel(
  fullName: string,
  className: string,
  classCounts: Map<string, number>
) {
  const count = classCounts.get(className) || 0;
  if (count > 1) {
    return `${className} #${shortId(fullName)}`;
  }
  return className;
}

const DagCanvas = forwardRef<DagCanvasHandle, DagCanvasProps>(
  (
    {
      dag,
      metricsById,
      summaryByName,
      rollup,
      percentiles,
      classCounts,
      currentCookId,
      selectedId,
      onSelect,
      searchQuery,
    },
    ref
  ) => {
    const [nodes, setNodes] = useState<Node<DagNodeData>[]>([]);
    const [edges, setEdges] = useState<Edge[]>([]);
    const flowRef = useRef<ReactFlowInstance<Node<DagNodeData>, Edge> | null>(
      null
    );
    const layoutKeyRef = useRef<string>("");
    const layoutToken = useRef(0);
    const [compact, setCompact] = useState(false);

    const normalizedQuery = searchQuery.trim().toLowerCase();

    const matchedIds = useMemo(() => {
      if (!normalizedQuery || !dag?.nodes) {
        return new Set<string>();
      }
      const matches = new Set<string>();
      dag.nodes.forEach((node) => {
        const fullName = node.full_name || node.name || String(node.id);
        const className = node.class_name || inferClassName(fullName);
        const match =
          fuzzyMatch(fullName.toLowerCase(), normalizedQuery) ||
          fuzzyMatch(className.toLowerCase(), normalizedQuery);
        if (match) {
          matches.add(String(node.id));
        }
      });
      return matches;
    }, [dag?.nodes, normalizedQuery]);

    useImperativeHandle(
      ref,
      () => ({
        fitView: () => {
          if (flowRef.current) {
            flowRef.current.fitView({ padding: 0.25, duration: 300 });
          }
        },
        focusNode: (id: string) => {
          if (!flowRef.current) {
            return;
          }
          const node = flowRef.current.getNode(String(id));
          if (node) {
            flowRef.current.fitView({
              nodes: [node],
              padding: 0.35,
              duration: 400,
            });
          }
        },
      }),
      []
    );

    const buildNodes = useCallback(
      (positions: Map<string, { x: number; y: number }>) => {
        if (!dag?.nodes) {
          return [] as Node<DagNodeData>[];
        }

        return dag.nodes.map((node) => {
          const fullName = node.full_name || node.name || String(node.id);
          const className = node.class_name || inferClassName(fullName);
          const label = buildNodeLabel(fullName, className, classCounts);
          const metrics =
            metricsById.get(node.id) || summaryByName.get(fullName) || {};
          const roll = rollup.get(node.id);
          const percentile = percentiles.get(node.id) || 0;
          const lastUpdate =
            (metrics as MetricsRow).last_update_id ??
            (metrics as SummaryRow).last_update;
          const calls = roll?.calls ?? (metrics as MetricsRow).calls;
          const line1 = `EXCL ${fmt(roll?.excl_ms)}  EMA ${fmt(
            (metrics as MetricsRow).ema_ms
          )}`;
          const line2 = `KERN ${fmt(
            (metrics as MetricsRow).kernel_ms
          )}  ${
            (metrics as MetricsRow).overhead_est_ms
              ? `OH ${fmt((metrics as MetricsRow).overhead_est_ms)}`
              : `KF ${fmtPct((metrics as MetricsRow).kernel_frac)}`
          }`;
          const line3 = `CALLS ${calls ?? "-"}  CK ${lastUpdate ?? "-"}`;
          const lines = compact ? [line1, line2] : [line1, line2, line3];

          const executed = Boolean((metrics as MetricsRow).executed_this_cook);
          const age =
            currentCookId !== null &&
            lastUpdate !== undefined &&
            lastUpdate !== null
              ? currentCookId - Number(lastUpdate)
              : null;
          let ageClass = "age-old";
          if (age === 0) {
            ageClass = "age-0";
          } else if (age !== null && age <= 3) {
            ageClass = "age-1-3";
          } else if (age !== null && age <= 10) {
            ageClass = "age-4-10";
          }

          const dimmed =
            normalizedQuery.length > 0 && !matchedIds.has(String(node.id));

          const classNameParts = [
            "dag-node",
            executed ? "executed" : "",
            dimmed ? "dimmed" : "",
            ageClass,
          ].filter(Boolean);

          const borderAlpha = 0.2 + percentile * 0.6;
          const shadowAlpha = 0.08 + percentile * 0.25;

          return {
            id: String(node.id),
            type: "dagNode",
            position: positions.get(String(node.id)) || { x: 0, y: 0 },
            sourcePosition: Position.Bottom,
            targetPosition: Position.Top,
            data: {
              title: label || fullName,
              fullName,
              lines,
              compact,
            },
            className: classNameParts.join(" "),
            selected: selectedId === node.id,
            style: {
              backgroundColor: "#fffaf2",
              borderColor: `rgba(31, 138, 112, ${borderAlpha})`,
              boxShadow: `0 10px 24px rgba(31, 138, 112, ${shadowAlpha})`,
            },
          } as Node<DagNodeData>;
        });
      },
      [
        classCounts,
        compact,
        dag?.nodes,
        matchedIds,
        metricsById,
        currentCookId,
        normalizedQuery,
        percentiles,
        rollup,
        selectedId,
        summaryByName,
      ]
    );

    useEffect(() => {
      if (!dag?.nodes || !dag.edges) {
        setNodes([]);
        setEdges([]);
        return;
      }
      const edgeList = dag.edges ?? [];
      const maxSeen = edgeList.reduce(
        (acc, edge) => Math.max(acc, edge.seen_count ?? 1),
        1
      );
      const pairCounts = new Map<string, number>();
      edgeList.forEach((edge) => {
        const key = `${edge.source}->${edge.target}`;
        pairCounts.set(key, (pairCounts.get(key) || 0) + 1);
      });
      const pairOffsets = new Map<string, number>();
      const key = `${dag.dag_version ?? "na"}-${dag.nodes.length}-${
        edgeList.length
      }`;
      if (layoutKeyRef.current === key && nodes.length) {
        return;
      }
      layoutKeyRef.current = key;
      const token = (layoutToken.current += 1);
      layoutDag(dag.nodes, edgeList).then((positions) => {
        if (token !== layoutToken.current) {
          return;
        }
        const nextNodes = buildNodes(positions);
        const nextEdges = edgeList.map((edge, idx) => {
          const key = `${edge.source}->${edge.target}`;
          const offset = pairOffsets.get(key) || 0;
          pairOffsets.set(key, offset + 1);
          const parallelCount = pairCounts.get(key) || 1;
          const ratio = maxSeen > 0 ? (edge.seen_count ?? 1) / maxSeen : 0.4;
          const opacity = Math.max(0.18, Math.min(1, 0.18 + 0.82 * ratio));
          return {
            id: `e-${edge.source}-${edge.target}-${idx}`,
            source: String(edge.source),
            target: String(edge.target),
            type: "parallel",
            markerEnd: EDGE_MARKER,
            data: {
              parallelIndex: offset,
              parallelCount,
              opacity,
            },
          };
        });
        setNodes(nextNodes);
        setEdges(nextEdges);
      });
    }, [buildNodes, dag?.dag_version, dag?.edges, dag?.nodes, nodes.length]);

    useEffect(() => {
      if (!dag?.nodes || !dag.edges) {
        return;
      }
      setNodes((prev) => {
        if (!prev.length) {
          return prev;
        }
        const positions = new Map<string, { x: number; y: number }>();
        prev.forEach((node) => {
          positions.set(node.id, node.position);
        });
        return buildNodes(positions);
      });
    }, [buildNodes, dag?.nodes, metricsById, percentiles, rollup, selectedId]);

    useEffect(() => {
      if (!dag?.edges) {
        return;
      }
      const executedIds = new Set<string>();
      metricsById.forEach((row) => {
        if (row.executed_this_cook) {
          executedIds.add(row.id);
        }
      });
      setEdges((prev) =>
        prev.map((edge) => {
          const executed =
            executedIds.has(String(edge.source)) &&
            executedIds.has(String(edge.target));
          return {
            ...edge,
            data: {
              ...(edge.data || {}),
              executed,
            },
          };
        })
      );
    }, [dag?.edges, metricsById]);

    return (
      <div className="dag-canvas">
        <ReactFlow<Node<DagNodeData>, Edge>
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          nodesDraggable
          nodesConnectable={false}
          elementsSelectable
          fitView
          fitViewOptions={{ padding: 0.2 }}
          onInit={(instance) => {
            flowRef.current = instance;
          }}
          onPaneClick={() => onSelect(null)}
          onNodeClick={(_, node) => onSelect(node.id)}
          onMove={(_, viewport) => {
            const dense = (dag?.nodes?.length || 0) > 500;
            setCompact(dense && viewport.zoom < 0.7);
          }}
          nodeExtent={[
            [-2000, -2000],
            [8000, 8000],
          ]}
        >
          <Background color="rgba(33, 50, 80, 0.08)" gap={28} />
          <Controls showInteractive={false} position="bottom-left" />
          <MiniMap
            pannable
            zoomable
            nodeColor={() => "#2f5c4a"}
            maskColor="rgba(10, 14, 20, 0.12)"
          />
        </ReactFlow>
        <div className="dag-legend">
          <span className="legend-chip">New</span>
          <span className="legend-chip">Warm</span>
          <span className="legend-chip">Cold</span>
        </div>
        <div className="dag-size">
          {dag?.nodes?.length ?? 0} nodes / {dag?.edges?.length ?? 0} edges
        </div>
      </div>
    );
  }
);

DagCanvas.displayName = "DagCanvas";

export default DagCanvas;
