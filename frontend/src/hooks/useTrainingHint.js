import { useMemo } from "react";

export function useTrainingHint(items) {
  return useMemo(() => {
    const s = items.find((x) => x.bc_status === "running" || x.ppo_status === "running");
    if (!s) return null;
    return { stage_id: s.stage_id, phase: s.bc_status === "running" ? "bc" : "ppo" };
  }, [items]);
}

