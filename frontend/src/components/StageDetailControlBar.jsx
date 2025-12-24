import { phaseStatusText, pillClassForStatus, stageStatusText, stageStatusTip } from "../utils/stageStatus.js";

export default function StageDetailControlBar({ id, item, stageStatus, start, actions, prevOk, onAction, onCheckpoint }) {
  const initDisabled = !actions.init;
  const startDisabled = Boolean(start.kind && !start.enabled);
  const phase = actions?.phase;
  const stopLabel = phase === "ppo" ? "停止 PPO" : "停止 BC";
  const resumeLabel = phase === "ppo" ? "恢复 PPO" : "恢复 BC";
  const completeLabel = phase === "ppo" ? "完成 PPO" : "完成 BC";
  return (
    <div className="card controlBar">
      <div className="controlBar__left">
        <div className="controlBar__line">
          <span className={pillClassForStatus(stageStatus)} title={stageStatusTip(stageStatus)}>{stageStatusText(stageStatus)}</span>
          <span className="mono subtle">phase: {item?.current_phase ?? "--"}</span>
          <span className={`${pillClassForStatus(item?.bc_status)} pill--sm`}>BC {phaseStatusText(item?.bc_status)}</span>
          <span className={`${pillClassForStatus(item?.ppo_status)} pill--sm`}>PPO {phaseStatusText(item?.ppo_status)}</span>
        </div>
        {item?.prev_stage ? (
          <div className="controlBar__subtle">
            前置 Stage {item.prev_stage.stage_id}：BC {phaseStatusText(item.prev_stage.bc_status)} / PPO {phaseStatusText(item.prev_stage.ppo_status)}
            {!prevOk ? <span className="dangerText">（未完成）</span> : null}
          </div>
        ) : <div className="controlBar__subtle">前置 Stage：无</div>}
      </div>
      <div className="controlBar__right">
        {start.kind ? (
          <button
            className={`btn btn--primary ${startDisabled ? "btn--softDisabled" : ""}`}
            type="button"
            aria-disabled={startDisabled}
            title={start.reason || ""}
            onClick={() => onAction(start.kind === "bc" ? "开始 BC" : "开始 PPO")}
          >
            {start.kind === "bc" ? "开始 BC" : "开始 PPO"}
          </button>
        ) : null}
        {phase ? (
          <>
            <button className="btn" type="button" disabled={!actions.stop} onClick={() => onAction(stopLabel)}>{stopLabel}</button>
            <button className="btn" type="button" disabled={!actions.resume} onClick={() => onAction(resumeLabel)}>{resumeLabel}</button>
            <button className="btn" type="button" disabled={!actions.complete} onClick={() => onAction(completeLabel)}>{completeLabel}</button>
          </>
        ) : null}
        <button
          className={`btn btn--danger ${initDisabled ? "btn--softDisabled" : ""}`}
          type="button"
          aria-disabled={initDisabled}
          onClick={() => onAction("初始化")}
        >
          初始化
        </button>
        <button className="btn" type="button" onClick={() => onCheckpoint(id)}>Checkpoint 列表</button>
      </div>
    </div>
  );
}
