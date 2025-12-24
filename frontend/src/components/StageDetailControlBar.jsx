import { IconList, IconRefresh, IconStop } from "./Icons.jsx";

export default function StageDetailControlBar({ id, item, start, actions, prevOk, onAction, onCheckpoint }) {
  const initDisabled = !actions.init;
  const startDisabled = Boolean(start.kind && !start.enabled);
  const phase = actions?.phase;
  const stopLabel = phase === "ppo" ? "停止 PPO" : "停止 BC";
  const resumeLabel = phase === "ppo" ? "恢复 PPO" : "恢复 BC";
  const completeLabel = phase === "ppo" ? "完成 PPO" : "完成 BC";
  const startLabel = start.kind === "bc" ? "开始 BC" : start.kind === "ppo" ? "开始 PPO" : null;
  return (
    <div className="actionBar">
      <button className="iconBtn" type="button" onClick={() => onCheckpoint(id)} title="Checkpoint 列表" aria-label="Checkpoint 列表">
        <IconList />
      </button>
      <button
        className={`iconBtn ${initDisabled ? "btn--softDisabled" : ""}`}
        type="button"
        aria-disabled={initDisabled}
        disabled={initDisabled}
        onClick={() => onAction("初始化")}
        title={item?.prev_stage && !prevOk ? "前置 Stage 未完成" : "初始化 Stage（清空数据）"}
      >
        <IconRefresh />
      </button>
      {phase ? (
        <>
          <button className={`btn ${!actions.resume ? "btn--softDisabled" : ""}`} type="button" disabled={!actions.resume} onClick={() => onAction(resumeLabel)}>
            {resumeLabel}
          </button>
          <button className={`btn ${!actions.complete ? "btn--softDisabled" : ""}`} type="button" disabled={!actions.complete} onClick={() => onAction(completeLabel)}>
            {completeLabel}
          </button>
          <button
            className={`btn btn--solidDanger ${!actions.stop ? "btn--softDisabled" : ""}`}
            type="button"
            disabled={!actions.stop}
            onClick={() => onAction(stopLabel)}
          >
            <span style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
              <IconStop />
              {stopLabel}
            </span>
          </button>
        </>
      ) : null}
      {!phase && start.kind && startLabel ? (
        <button
          className={`btn btn--solid ${startDisabled ? "btn--softDisabled" : ""}`}
          type="button"
          disabled={startDisabled}
          title={start.reason || ""}
          onClick={() => onAction(startLabel)}
        >
          {startLabel}
        </button>
      ) : null}
    </div>
  );
}
