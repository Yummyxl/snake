const DEFAULT_API_BASE = "";

function apiBase() {
  const base = import.meta.env.VITE_API_BASE || DEFAULT_API_BASE;
  return String(base).replace(/\/$/, "");
}

export async function fetchRollout({ stageId, phase, source, rolloutId, signal }) {
  const qs = new URLSearchParams({ stage_id: String(stageId), phase: String(phase), source: String(source) });
  const res = await fetch(`${apiBase()}/api/rollouts/${encodeURIComponent(rolloutId)}?${qs.toString()}`, { signal });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  if (!data || typeof data !== "object") throw new Error("invalid response");
  if (!data.ok) throw new Error(data.error || "read rollout failed");
  return data;
}
