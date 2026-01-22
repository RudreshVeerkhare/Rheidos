export type CategoryRow = {
  id: string;
  name: string;
  producer?: string | null;
  ema_ms?: number;
  last_ms?: number;
  calls?: number;
};

export type SummaryRow = {
  id: string;
  name: string;
  full_name?: string;
  class_name?: string;
  ema_ms?: number;
  last_ms?: number;
  last_update?: number;
  calls?: number;
  kernel_ms?: number;
  kernel_frac?: number;
  overhead_est_ms?: number;
};

export type SummarySnapshot = {
  tick?: number;
  cook_id?: number;
  frame?: number;
  substep?: number;
  dropped_events?: number;
  profiler_overhead_us?: number;
  edges_recorded?: number;
  wall_ms?: number;
  kernel_ms?: number;
  kernel_fraction?: number;
  dag_version?: number;
  categories?: Record<string, CategoryRow[]>;
  rows?: SummaryRow[];
};

export type MetricsRow = {
  id: string;
  name: string;
  full_name?: string;
  class_name?: string;
  ema_ms?: number;
  last_ms?: number;
  last_update_id?: number;
  calls?: number;
  kernel_ms?: number;
  kernel_frac?: number;
  overhead_est_ms?: number;
  executed_this_cook?: boolean;
};

export type MetricsSnapshot = {
  cook_id?: number;
  frame?: number;
  substep?: number;
  rows?: MetricsRow[];
  full?: boolean;
};

export type DagNode = {
  id: string;
  name: string;
  full_name?: string;
  class_name?: string;
};

export type DagEdge = {
  source: string;
  target: string;
  seen_count?: number;
  via_resources?: string[];
};

export type DagSnapshot = {
  cook_id?: number;
  dag_version?: number;
  nodes?: DagNode[];
  edges?: DagEdge[];
};

export type ExecTreeNode = {
  producer_id: string;
  inclusive_ms?: number;
  exclusive_ms?: number;
};

export type ExecTreeSnapshot = {
  cook_id?: number;
  nodes?: ExecTreeNode[];
};

export type NodeDetails = {
  id: string;
  name?: string;
  full_name?: string;
  class_name?: string;
  metrics?: MetricsRow;
  inputs?: Array<{ id: string; version: string | number }>;
  outputs?: Array<{ id: string; version: string | number }>;
  staleness_reason?: string;
  resources_read?: string[];
  last_exec_subtree?: unknown;
};

export type WsPayload = SummarySnapshot & {
  full?: boolean;
  changed_ids?: Array<string | number>;
  metrics?: MetricsSnapshot;
  exec_tree?: ExecTreeSnapshot;
  dag?: DagSnapshot;
  dag_mode?: string;
};
