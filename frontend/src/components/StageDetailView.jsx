import { Link } from "react-router-dom";
import Banner from "./Banner.jsx";
import StageDetailControlBar from "./StageDetailControlBar.jsx";
import StageDetailMetricsCard from "./StageDetailMetricsCard.jsx";
import StageDetailRolloutsCard from "./StageDetailRolloutsCard.jsx";
import StageDetailStatusCards from "./StageDetailStatusCards.jsx";

export default function StageDetailView({ id, item, error, retry, loading, vm, onAction, onCheckpoint, onPlayRollout }) {
  const staleAction = item?.bc_status === "running" ? "停止 BC" : item?.ppo_status === "running" ? "停止 PPO" : vm?.actions?.phase === "ppo" ? "停止 PPO" : "停止 BC";
  return (
    <div className="page">
      {error ? <Banner kind="error" action={<button className="btn" onClick={retry} type="button">重试</button>}>后端不可用：{error}</Banner> : null}
      {vm.trainingStale ? <Banner kind="error" action={<button className="btn btn--danger" onClick={() => onAction(staleAction)} type="button">修复为暂停</button>}>状态异常：显示训练中，但未发现训练进程（可能后端崩溃/重启）</Banner> : null}
      {!vm.trainingStale && !vm.training && vm.workerError ? <Banner kind="error">训练进程异常退出：{vm.workerError}</Banner> : null}
      {vm.training ? <Banner kind="info">正在训练：Stage {item?.stage_id} / {vm.training}</Banner> : null}
      <header className="header"><Link className="link-inline" to="/">← 返回 Stage 列表</Link><div className="title">Stage {id}</div><span /></header>
      <StageDetailControlBar id={id} item={item} stageStatus={vm.stageStatus} start={vm.start} actions={vm.actions} prevOk={vm.prevOk} onAction={onAction} onCheckpoint={onCheckpoint} />
      <StageDetailStatusCards item={item} stageStatus={vm.stageStatus} />
      <StageDetailMetricsCard loading={loading} metricsPhase={vm.metricsPhase} metrics={item?.metrics} />
      <StageDetailRolloutsCard phase={vm.rolloutsPhase} rollouts={vm.rollouts} onPlay={onPlayRollout} />
    </div>
  );
}
