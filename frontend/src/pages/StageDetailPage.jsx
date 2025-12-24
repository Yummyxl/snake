import { useParams } from "react-router-dom";
import StageDetailView from "../components/StageDetailView.jsx";
import RolloutPlayerModal from "../components/RolloutPlayerModal.jsx";
import { useStageDetailData } from "../hooks/useStageDetailData.js";
import { useStageDetailVm } from "../hooks/useStageDetailVm.js";
import { completeStageBc, completeStagePpo, resetStage, resumeStageBc, resumeStagePpo, startStageBc, startStagePpo, stopStageBc, stopStagePpo } from "../api/stages.js";
import { useUi } from "../hooks/useUi.js";
import { fetchRollout } from "../api/rollouts.js";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

async function runStartBc({ id, item, retry, ui }) {
  if (!item) { ui.toast("数据未加载，稍后再试", "error"); return; }
  try {
    await startStageBc(id);
    ui.toast("启动 BC 成功", "success");
    retry();
  } catch (e) {
    ui.toast(`启动 BC 失败：${e instanceof Error ? e.message : String(e)}`, "error");
  }
}

async function runStartPpo({ start, ui }) {
  const reason = start?.reason || "条件不满足";
  ui.toast(`无法开始 PPO：${reason}`, "error");
}

async function runStartPpoOk({ id, retry, ui }) {
  try {
    await startStagePpo(id);
    ui.toast("启动 PPO 成功", "success");
    retry();
  } catch (e) {
    ui.toast(`启动 PPO 失败：${e instanceof Error ? e.message : String(e)}`, "error");
  }
}

async function runStopBc({ id, retry, ui }) {
  try {
    const res = await stopStageBc(id);
    const phase = res?.phase;
    const reason = res?.reason;
    if (!phase) ui.toast("当前未在训练", "info");
    else if (reason === "signal_sent") ui.toast(`已停止 ${phase.toUpperCase()}（已发送停止信号）`, "success");
    else ui.toast(`已修复 ${phase.toUpperCase()}：未发现进程，已标记为暂停`, "success");
    retry();
  } catch (e) {
    ui.toast(`停止 BC 失败：${e instanceof Error ? e.message : String(e)}`, "error");
  }
}

async function runStopPpo({ id, retry, ui }) {
  try {
    const res = await stopStagePpo(id);
    const phase = res?.phase;
    const reason = res?.reason;
    if (!phase) ui.toast("当前未在训练", "info");
    else if (reason === "signal_sent") ui.toast(`已停止 ${phase.toUpperCase()}（已发送停止信号）`, "success");
    else ui.toast(`已修复 ${phase.toUpperCase()}：未发现进程，已标记为暂停`, "success");
    retry();
  } catch (e) {
    ui.toast(`停止 PPO 失败：${e instanceof Error ? e.message : String(e)}`, "error");
  }
}

async function runResumeBc({ id, retry, ui }) {
  try {
    await resumeStageBc(id);
    ui.toast("恢复 BC 成功", "success");
    retry();
  } catch (e) {
    ui.toast(`恢复 BC 失败：${e instanceof Error ? e.message : String(e)}`, "error");
  }
}

async function runResumePpo({ id, retry, ui }) {
  try {
    await resumeStagePpo(id);
    ui.toast("恢复 PPO 成功", "success");
    retry();
  } catch (e) {
    ui.toast(`恢复 PPO 失败：${e instanceof Error ? e.message : String(e)}`, "error");
  }
}

async function runCompleteBc({ id, retry, ui }) {
  try {
    await completeStageBc(id);
    ui.toast("完成 BC 成功", "success");
    retry();
  } catch (e) {
    ui.toast(`完成 BC 失败：${e instanceof Error ? e.message : String(e)}`, "error");
  }
}

async function runCompletePpo({ id, retry, ui }) {
  try {
    await completeStagePpo(id);
    ui.toast("完成 PPO 成功", "success");
    retry();
  } catch (e) {
    ui.toast(`完成 PPO 失败：${e instanceof Error ? e.message : String(e)}`, "error");
  }
}

async function runReset({ id, item, retry, ui }) {
  if (!item) { ui.toast("数据未加载，稍后再试", "error"); return; }
  const bc = item?.bc_status;
  const ppo = item?.ppo_status;
  if (bc === "running" || ppo === "running") { ui.toast("无法初始化：训练中不可初始化（reset）", "error"); return; }
  if (bc === "completed" && ppo === "completed") { ui.toast("无法初始化：已完成 Stage 不可初始化（reset）", "error"); return; }
  const target = ppo !== "not_started" ? "ppo" : bc !== "not_started" ? "bc" : null;
  const scopeText = target === "ppo" ? "PPO 阶段" : target === "bc" ? "BC 阶段" : "全部";
  const ok = await ui.confirm({ title: "确认初始化", message: `确认初始化 Stage ${id}？这会清空该 Stage 的${scopeText}历史数据。`, confirmText: "初始化", cancelText: "取消" });
  if (!ok) { ui.toast("已取消初始化", "info"); return; }
  try {
    await resetStage(id);
    ui.toast("初始化成功", "success");
    retry();
  } catch (e) {
    ui.toast(`初始化失败：${e instanceof Error ? e.message : String(e)}`, "error");
  }
}

async function handleAction(label, ctx) {
  if (label === "开始 BC") {
    if (ctx.start?.kind !== "bc") { ctx.ui.toast("当前不可开始 BC", "error"); return; }
    if (!ctx.start?.enabled) { ctx.ui.toast(`无法开始 BC：${ctx.start?.reason || "条件不满足"}`, "error"); return; }
    return runStartBc(ctx);
  }
  if (label === "开始 PPO") {
    if (ctx.start?.kind !== "ppo") { ctx.ui.toast("当前不可开始 PPO", "error"); return; }
    if (!ctx.start?.enabled) return runStartPpo(ctx);
    return runStartPpoOk(ctx);
  }
  if (label === "停止 BC") return runStopBc(ctx);
  if (label === "恢复 BC") return runResumeBc(ctx);
  if (label === "完成 BC") return runCompleteBc(ctx);
  if (label === "停止 PPO") return runStopPpo(ctx);
  if (label === "恢复 PPO") return runResumePpo(ctx);
  if (label === "完成 PPO") return runCompletePpo(ctx);
  if (label === "初始化") return runReset(ctx);
  ctx.ui.toast(`${label}：暂未实现`, "info");
}

export default function StageDetailPage() {
  const { id } = useParams();
  const { item, error, loading, retry } = useStageDetailData(id);
  const vm = useStageDetailVm(item);
  const ui = useUi();

  const onAction = (label) => handleAction(label, { id, item, retry, ui, start: vm.start });

  const [playerOpen, setPlayerOpen] = useState(false);
  const [playerLoading, setPlayerLoading] = useState(false);
  const [playerError, setPlayerError] = useState(null);
  const [playerData, setPlayerData] = useState(null);
  const [playerTitle, setPlayerTitle] = useState("");
  const abortRef = useRef(null);

  const stageSize = item?.size;
  const stageId = useMemo(() => (id ? Number(id) : null), [id]);

  const closePlayer = useCallback(() => {
    abortRef.current?.abort?.();
    abortRef.current = null;
    setPlayerOpen(false);
    setPlayerLoading(false);
    setPlayerError(null);
    setPlayerData(null);
  }, []);

  useEffect(() => closePlayer(), [id, closePlayer]);

  const onPlayRollout = useCallback(async (r) => {
    if (!stageId || !r?.rollout_id || !r?.phase) { ui.toast("无法播放：缺少必要信息", "error"); return; }
    setPlayerOpen(true);
    setPlayerLoading(true);
    setPlayerError(null);
    setPlayerData(null);
    const source = r?.source || "eval";
    setPlayerTitle(`${source.toUpperCase()} / ${String(r.phase).toUpperCase()} / ${r.rollout_id}`);
    abortRef.current?.abort?.();
    const ac = new AbortController();
    abortRef.current = ac;
    try {
      const res = await fetchRollout({ stageId, phase: r.phase, source, rolloutId: r.rollout_id, signal: ac.signal });
      setPlayerData(res);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setPlayerError(msg);
      ui.toast(`加载回放失败：${msg}`, "error");
    } finally {
      setPlayerLoading(false);
    }
  }, [stageId, ui]);

  return (
    <>
      <StageDetailView
        id={id}
        item={item}
        error={error}
        retry={retry}
        loading={loading}
        vm={vm}
        onAction={onAction}
        onCheckpoint={() => ui.toast("Checkpoint 列表：暂未实现", "info")}
        onPlayRollout={onPlayRollout}
      />
      <RolloutPlayerModal
        open={playerOpen}
        title={playerTitle}
        size={stageSize}
        data={playerData}
        loading={playerLoading}
        error={playerError}
        onClose={closePlayer}
      />
    </>
  );
}
