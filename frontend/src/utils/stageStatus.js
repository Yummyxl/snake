export function deriveStageStatus(bcStatus, ppoStatus) {
  const bc = bcStatus || "not_started";
  const ppo = ppoStatus || "not_started";
  if (bc === "not_started" && ppo === "not_started") return "not_started";
  if (bc === "completed" && ppo === "completed") return "completed";
  return "running";
}

export function stageStatusText(s) {
  if (s === "running") return "进行中";
  if (s === "completed") return "已完成";
  return "未开始";
}

export function stageStatusTip(s) {
  if (s === "running") return "进行中：BC/PPO 训练中或处于中间阶段";
  if (s === "completed") return "已完成：BC 与 PPO 均已完成";
  return "未开始：可开始 BC 训练（受前置 Stage 约束）";
}

export function phaseStatusText(s) {
  if (s === "running") return "进行中";
  if (s === "paused") return "已停止";
  if (s === "completed") return "已完成";
  return "未开始";
}

export function pillClassForStatus(s) {
  if (s === "running") return "pill pill--blue";
  if (s === "completed") return "pill pill--green";
  if (s === "paused") return "pill pill--amber";
  return "pill pill--gray";
}

