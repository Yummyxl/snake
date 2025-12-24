import { Link } from "react-router-dom";
import Banner from "./Banner.jsx";
import StageDetailControlBar from "./StageDetailControlBar.jsx";
import StageDetailMetricsCard from "./StageDetailMetricsCard.jsx";
import StageDetailRolloutsCard from "./StageDetailRolloutsCard.jsx";
import StageDetailStatusCards from "./StageDetailStatusCards.jsx";
import { fmtTime } from "../utils/format.js";
import { phaseStatusText, pillClassForStatus } from "../utils/stageStatus.js";

export default function StageDetailView({ id, item, error, retry, loading, vm, onAction, onCheckpoint, onPlayRollout }) {
  const staleAction = item?.bc_status === "running" ? "停止 BC" : item?.ppo_status === "running" ? "停止 PPO" : vm?.actions?.phase === "ppo" ? "停止 PPO" : "停止 BC";
  const phase = vm?.training || vm?.actions?.phase || item?.current_phase || null;
  const running = Boolean(vm?.training || item?.bc_status === "running" || item?.ppo_status === "running");
  return (
    <div className="page">
      {error ? <Banner kind="error" action={<button className="btn" onClick={retry} type="button">重试</button>}>后端不可用：{error}</Banner> : null}
      {vm.trainingStale ? <Banner kind="error" action={<button className="btn btn--danger" onClick={() => onAction(staleAction)} type="button">修复为暂停</button>}>状态异常：显示训练中，但未发现训练进程（可能后端崩溃/重启）</Banner> : null}
      {!vm.trainingStale && !vm.training && vm.workerError ? <Banner kind="error">训练进程异常退出：{vm.workerError}</Banner> : null}
      <div className="pageHeader">
        <div className="pageHeader__left">
          <div className="breadcrumbs">
            <Link className="breadcrumbs__link" to="/">Stages</Link>
            <span className="breadcrumbs__sep">/</span>
            <span className="breadcrumbs__current">Stage {id}</span>
          </div>
          <div className="pageTitleRow">
            <div className="pageTitle">Stage {id} / {phase ? String(phase).toUpperCase() : "--"}</div>
            <div className={`statusDot ${running ? "statusDot--running" : ""}`} aria-label={running ? "Running" : "Stopped"} />
          </div>
          <div className="pageMetaRow">
            <span className="mono">phase: {item?.current_phase ?? "--"}</span>
            <span className={`${pillClassForStatus(item?.bc_status)} pill--sm`}>BC {phaseStatusText(item?.bc_status)}</span>
            <span className={`${pillClassForStatus(item?.ppo_status)} pill--sm`}>PPO {phaseStatusText(item?.ppo_status)}</span>
            <span className="mono">updated: {fmtTime(item?.updated_at_ms)}</span>
            {item?.prev_stage ? (
              <span className={`mono ${vm?.prevOk ? "subtle" : "dangerText"}`}>
                prev: Stage {item.prev_stage.stage_id} (BC {phaseStatusText(item.prev_stage.bc_status)} / PPO {phaseStatusText(item.prev_stage.ppo_status)})
              </span>
            ) : (
              <span className="mono subtle">prev: --</span>
            )}
          </div>
        </div>
        <StageDetailControlBar
          id={id}
          item={item}
          start={vm.start}
          actions={vm.actions}
          prevOk={vm.prevOk}
          onAction={onAction}
          onCheckpoint={onCheckpoint}
        />
      </div>
      <StageDetailStatusCards item={item} stageStatus={vm.stageStatus} />
      <StageDetailMetricsCard loading={loading} metricsPhase={vm.metricsPhase} metrics={item?.metrics} />
      <StageDetailRolloutsCard phase={vm.rolloutsPhase} rollouts={vm.rollouts} onPlay={onPlayRollout} />
    </div>
  );
}
