export function fmt(value: number | string | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  if (typeof value === "number") {
    return value.toFixed(digits);
  }
  return String(value);
}

export function fmtPct(value: number | string | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  if (typeof value !== "number") {
    return String(value);
  }
  return `${(value * 100).toFixed(1)}%`;
}

export function shortId(text: string): string {
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = (hash * 31 + text.charCodeAt(i)) | 0;
  }
  const mod = Math.abs(hash) % 1296;
  return mod.toString(36).padStart(2, "0");
}

export function fuzzyMatch(text: string, query: string): boolean {
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

export function fuzzyScore(text: string, query: string): number {
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
    return Number.POSITIVE_INFINITY;
  }
  return score + (text.length - t);
}
