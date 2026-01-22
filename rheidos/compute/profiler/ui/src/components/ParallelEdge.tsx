import { getBezierPath, type EdgeProps } from "@xyflow/react";

const BASE_COLOR = "rgba(33, 50, 80, 1)";
const ACTIVE_COLOR = "rgba(31, 138, 112, 1)";

export default function ParallelEdge(props: EdgeProps) {
  const {
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    markerEnd,
    data,
  } = props;

  const parallelIndex = typeof data?.parallelIndex === "number" ? data.parallelIndex : 0;
  const parallelCount = typeof data?.parallelCount === "number" ? data.parallelCount : 1;
  const opacity = typeof data?.opacity === "number" ? data.opacity : 0.4;
  const executed = Boolean(data?.executed);

  const spread = 14;
  const offset = (parallelIndex - (parallelCount - 1) / 2) * spread;
  const dx = targetX - sourceX;
  const dy = targetY - sourceY;
  const length = Math.max(1, Math.hypot(dx, dy));
  const nx = -dy / length;
  const ny = dx / length;

  const sx = sourceX + nx * offset;
  const sy = sourceY + ny * offset;
  const tx = targetX + nx * offset;
  const ty = targetY + ny * offset;

  const [edgePath] = getBezierPath({
    sourceX: sx,
    sourceY: sy,
    targetX: tx,
    targetY: ty,
    sourcePosition,
    targetPosition,
    curvature: 0.3,
  });

  const stroke = executed ? ACTIVE_COLOR : BASE_COLOR;
  const strokeOpacity = executed ? Math.min(1, opacity + 0.2) : opacity;
  const strokeWidth = executed ? 2.2 : 1.4;

  return (
    <path
      id={id}
      d={edgePath}
      fill="none"
      stroke={stroke}
      strokeOpacity={strokeOpacity}
      strokeWidth={strokeWidth}
      markerEnd={markerEnd}
    />
  );
}
