export function fmtCoverage(v) {
  if (v === null || v === undefined) return "--";
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(2) : "--";
}

export function fmtTime(ms) {
  if (!ms) return "--";
  const d = new Date(ms);
  return Number.isNaN(d.getTime()) ? "--" : d.toLocaleString();
}

export function fmtNumber(v, digits = 2) {
  if (v === null || v === undefined) return "--";
  const n = Number(v);
  if (!Number.isFinite(n)) return "--";
  const d = Math.max(0, Math.min(8, Number.isFinite(digits) ? Math.floor(digits) : 2));
  return n.toFixed(d);
}
