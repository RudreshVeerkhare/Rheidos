import ELK from "elkjs/lib/elk.bundled.js";
import type { DagEdge, DagNode } from "../types";

const elk = new ELK();

const NODE_WIDTH = 220;
const NODE_HEIGHT = 96;

export async function layoutDag(
  nodes: DagNode[],
  edges: DagEdge[]
): Promise<Map<string, { x: number; y: number }>> {
  const graph = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": "DOWN",
      "elk.layered.spacing.nodeNodeBetweenLayers": "90",
      "elk.spacing.nodeNode": "60",
      "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
      "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
      "elk.layered.nodePlacement.favorStraightEdges": "true",
      "elk.layered.edgeRouting": "ORTHOGONAL",
      "elk.layered.spacing.edgeEdgeBetweenLayers": "24",
      "elk.spacing.edgeNode": "24",
      "elk.layered.mergeEdges": "false",
    },
    children: nodes.map((node) => ({
      id: String(node.id),
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
    })),
    edges: edges.map((edge, idx) => ({
      id: `e-${edge.source}-${edge.target}-${idx}`,
      sources: [String(edge.source)],
      targets: [String(edge.target)],
    })),
  };

  const layout = await elk.layout(graph);
  const positions = new Map<string, { x: number; y: number }>();
  (layout.children || []).forEach((child) => {
    positions.set(String(child.id), {
      x: child.x ?? 0,
      y: child.y ?? 0,
    });
  });
  return positions;
}

export function nodeSize() {
  return { width: NODE_WIDTH, height: NODE_HEIGHT };
}
