import type { ExecTreeNode } from "../types";

export type RollupEntry = {
  excl_ms: number;
  incl_ms: number;
  calls: number;
};

export function computeRollup(nodes: ExecTreeNode[]): Map<string, RollupEntry> {
  const rollup = new Map<string, RollupEntry>();
  nodes.forEach((node) => {
    if (node.producer_id === null || node.producer_id === undefined) {
      return;
    }
    const current = rollup.get(node.producer_id) || {
      excl_ms: 0,
      incl_ms: 0,
      calls: 0,
    };
    current.excl_ms += node.exclusive_ms || 0;
    current.incl_ms += node.inclusive_ms || 0;
    current.calls += 1;
    rollup.set(node.producer_id, current);
  });
  return rollup;
}

function percentileForValue(sorted: number[], value: number): number {
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
  return Math.min(1, Math.max(0, idx / (sorted.length - 1)));
}

export function computePercentiles(
  rollup: Map<string, RollupEntry>
): Map<string, number> {
  const values: number[] = [];
  rollup.forEach((entry) => {
    if (entry.excl_ms > 0) {
      values.push(entry.excl_ms);
    }
  });
  values.sort((a, b) => a - b);
  const percentiles = new Map<string, number>();
  rollup.forEach((entry, id) => {
    if (entry.excl_ms > 0) {
      percentiles.set(id, percentileForValue(values, entry.excl_ms));
    } else {
      percentiles.set(id, 0);
    }
  });
  return percentiles;
}
