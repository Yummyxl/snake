const DEFAULT_API_BASE = "http://127.0.0.1:8000";

export async function fetchStages(signal) {
  const apiBase = import.meta.env.VITE_API_BASE || DEFAULT_API_BASE;
  const res = await fetch(`${apiBase}/api/stages`, { signal });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return Array.isArray(data) ? data : [];
}

export async function fetchStageDetail(stageId, signal) {
  const apiBase = import.meta.env.VITE_API_BASE || DEFAULT_API_BASE;
  const res = await fetch(`${apiBase}/api/stages/${encodeURIComponent(stageId)}`, { signal });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return data && typeof data === "object" ? data : null;
}

export async function resetStage(stageId) {
  const apiBase = import.meta.env.VITE_API_BASE || DEFAULT_API_BASE;
  const res = await fetch(`${apiBase}/api/stages/${encodeURIComponent(stageId)}/reset`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  if (!data || typeof data !== "object") throw new Error("invalid response");
  if (!data.ok) throw new Error(data.error || "reset failed");
  return data;
}

export async function startStageBc(stageId) {
  const apiBase = import.meta.env.VITE_API_BASE || DEFAULT_API_BASE;
  const res = await fetch(`${apiBase}/api/stages/${encodeURIComponent(stageId)}/bc/start`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  if (!data || typeof data !== "object") throw new Error("invalid response");
  if (!data.ok) throw new Error(data.error || "start bc failed");
  return data;
}

export async function stopStageBc(stageId) {
  const apiBase = import.meta.env.VITE_API_BASE || DEFAULT_API_BASE;
  const res = await fetch(`${apiBase}/api/stages/${encodeURIComponent(stageId)}/bc/stop`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  if (!data || typeof data !== "object") throw new Error("invalid response");
  if (!data.ok) throw new Error(data.error || "stop bc failed");
  return data;
}

export async function resumeStageBc(stageId) {
  const apiBase = import.meta.env.VITE_API_BASE || DEFAULT_API_BASE;
  const res = await fetch(`${apiBase}/api/stages/${encodeURIComponent(stageId)}/bc/resume`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  if (!data || typeof data !== "object") throw new Error("invalid response");
  if (!data.ok) throw new Error(data.error || "resume bc failed");
  return data;
}

export async function completeStageBc(stageId) {
  const apiBase = import.meta.env.VITE_API_BASE || DEFAULT_API_BASE;
  const res = await fetch(`${apiBase}/api/stages/${encodeURIComponent(stageId)}/bc/complete`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  if (!data || typeof data !== "object") throw new Error("invalid response");
  if (!data.ok) throw new Error(data.error || "complete bc failed");
  return data;
}

export async function startStagePpo(stageId) {
  const apiBase = import.meta.env.VITE_API_BASE || DEFAULT_API_BASE;
  const res = await fetch(`${apiBase}/api/stages/${encodeURIComponent(stageId)}/ppo/start`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  if (!data || typeof data !== "object") throw new Error("invalid response");
  if (!data.ok) throw new Error(data.error || "start ppo failed");
  return data;
}

export async function stopStagePpo(stageId) {
  const apiBase = import.meta.env.VITE_API_BASE || DEFAULT_API_BASE;
  const res = await fetch(`${apiBase}/api/stages/${encodeURIComponent(stageId)}/ppo/stop`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  if (!data || typeof data !== "object") throw new Error("invalid response");
  if (!data.ok) throw new Error(data.error || "stop ppo failed");
  return data;
}

export async function resumeStagePpo(stageId) {
  const apiBase = import.meta.env.VITE_API_BASE || DEFAULT_API_BASE;
  const res = await fetch(`${apiBase}/api/stages/${encodeURIComponent(stageId)}/ppo/resume`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  if (!data || typeof data !== "object") throw new Error("invalid response");
  if (!data.ok) throw new Error(data.error || "resume ppo failed");
  return data;
}

export async function completeStagePpo(stageId) {
  const apiBase = import.meta.env.VITE_API_BASE || DEFAULT_API_BASE;
  const res = await fetch(`${apiBase}/api/stages/${encodeURIComponent(stageId)}/ppo/complete`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  if (!data || typeof data !== "object") throw new Error("invalid response");
  if (!data.ok) throw new Error(data.error || "complete ppo failed");
  return data;
}
