function isCompletedPhase(s) {
  return (s || "not_started") === "completed";
}

export function isPrevStageCompleted(prevStage) {
  if (!prevStage) return true;
  return isCompletedPhase(prevStage.bc_status) && isCompletedPhase(prevStage.ppo_status);
}

export function getStartSlot(detail) {
  if (!detail) return { kind: null };
  const bc = detail.bc_status || "not_started";
  const ppo = detail.ppo_status || "not_started";
  if (bc === "not_started" && ppo === "not_started") {
    const ok = isPrevStageCompleted(detail.prev_stage);
    return { kind: "bc", enabled: ok, reason: ok ? null : "需要先完成前置 Stage（BC+PPO）" };
  }
  if (bc === "completed" && ppo === "not_started") {
    const ok = Boolean(detail.has_bc_best_checkpoint);
    return { kind: "ppo", enabled: ok, reason: ok ? null : "缺少 BC best checkpoint" };
  }
  return { kind: null };
}

export function getActionStates(detail) {
  if (!detail) return { phase: null, stop: false, resume: false, complete: false, init: false };
  const bc = detail.bc_status || "not_started";
  const ppo = detail.ppo_status || "not_started";
  const stageCompleted = bc === "completed" && ppo === "completed";
  const anyRunning = bc === "running" || ppo === "running";
  const phase = ppo !== "not_started" ? "ppo" : bc !== "not_started" ? "bc" : null;
  const status = phase === "ppo" ? ppo : phase === "bc" ? bc : "not_started";
  const alive = phase === "bc" ? Boolean(detail?.probe?.bc?.alive) : phase === "ppo" ? Boolean(detail?.probe?.ppo?.alive) : false;
  const ppoBestOk = phase !== "ppo" || Boolean(detail.has_ppo_best_checkpoint);
  const bcBestOk = phase !== "bc" || Boolean(detail.has_bc_best_checkpoint);
  const canComplete = status === "paused" && !alive && bcBestOk && ppoBestOk;
  const canResume = status === "paused" && !alive;
  return { phase, stop: status === "running", resume: canResume, complete: canComplete, init: !anyRunning && !stageCompleted };
}
