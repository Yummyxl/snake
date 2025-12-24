import { useMemo } from "react";
import { getActionStates, getStartSlot, isPrevStageCompleted } from "../utils/stageControls.js";
import { deriveStageStatus } from "../utils/stageStatus.js";
import { sortEvalRollouts } from "../utils/rollouts.js";
import { pickLatestPhase, pickMetricsPhase } from "../utils/stageMetrics.js";

export function useStageDetailVm(item) {
  const stageStatus = useMemo(() => deriveStageStatus(item?.bc_status, item?.ppo_status), [item]);
  const start = useMemo(() => getStartSlot(item), [item]);
  const actions = useMemo(() => getActionStates(item), [item]);
  const metricsPhase = useMemo(() => pickMetricsPhase(item), [item]);
  const rolloutsPhase = useMemo(() => pickLatestPhase(item), [item]);
  const rollouts = useMemo(() => {
    const sorted = sortEvalRollouts(item?.eval_rollouts);
    if (!rolloutsPhase) return [];
    return sorted.filter((r) => r?.phase === rolloutsPhase);
  }, [item, rolloutsPhase]);
  const probe = item?.probe;
  const trainingStale = Boolean(probe?.stale_running);
  const trainingEffective = probe?.effective_training || null;
  const training = trainingStale ? null : trainingEffective || (item?.bc_status === "running" ? "bc" : item?.ppo_status === "running" ? "ppo" : null);
  const workerError = probe?.bc?.runtime?.last_error || probe?.ppo?.runtime?.last_error || null;
  const prevOk = isPrevStageCompleted(item?.prev_stage);
  return { stageStatus, start, actions, metricsPhase, rolloutsPhase, rollouts, training, trainingStale, workerError, prevOk };
}
