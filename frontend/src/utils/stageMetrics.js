export function pickMetricsPhase(d) {
  if (!d) return null;
  if (d.bc_status === "not_started" && d.ppo_status === "not_started") return "bc";
  if (d.bc_status === "running") return "bc";
  if (d.ppo_status === "running") return "ppo";
  const p = d.last_eval?.phase;
  return p === "bc" || p === "ppo" ? p : null;
}

export function pickLatestPhase(d) {
  if (!d) return null;
  if (d.ppo_status && d.ppo_status !== "not_started") return "ppo";
  if (d.bc_status && d.bc_status !== "not_started") return "bc";
  const p = d.last_eval?.phase;
  return p === "bc" || p === "ppo" ? p : null;
}

export function metricOptions(phase) {
  if (phase === "ppo") return ["ppo_loss", "reward_mean"];
  if (phase === "bc") return ["bc_loss"];
  return ["coverage"];
}

export function defaultMetrics(phase) {
  return phase === "ppo" ? ["ppo_loss", "reward_mean"] : ["bc_loss"];
}
