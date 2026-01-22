import { Handle, Position, type NodeProps } from "@xyflow/react";

export type DagNodeData = {
  title: string;
  fullName: string;
  lines: string[];
  compact: boolean;
};

export default function DagNode({ data }: NodeProps) {
  const nodeData = data as DagNodeData;
  return (
    <div className="dag-node-body" title={nodeData.fullName}>
      <Handle type="target" position={Position.Top} className="dag-handle" />
      <Handle type="source" position={Position.Bottom} className="dag-handle" />
      <div className="dag-node-title">{nodeData.title}</div>
      <div className={`dag-node-lines ${nodeData.compact ? "compact" : ""}`}>
        {nodeData.lines.map((line, idx) => (
          <div key={`${nodeData.fullName}-${idx}`} className="dag-node-line">
            {line}
          </div>
        ))}
      </div>
    </div>
  );
}
