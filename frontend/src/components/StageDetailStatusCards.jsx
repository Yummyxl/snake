import { fmtCoverage, fmtTime } from "../utils/format.js";
import { phaseStatusText, pillClassForStatus, stageStatusText, stageStatusTip } from "../utils/stageStatus.js";

function Card({ title, status, tip, rows }) {
  return (
    <div className="card">
      <div className="cardHead">
        <div className="cardTitle">{title}</div>
        <div className={pillClassForStatus(status)} title={tip}>{title === "Stage" ? stageStatusText(status) : phaseStatusText(status)}</div>
      </div>
      <div className="cardBody">{rows.map((r) => (<div className="kv" key={r.k}><div className="kv__k">{r.k}</div><div className={r.mono ? "kv__v mono" : "kv__v"}>{r.v}</div></div>))}</div>
    </div>
  );
}

export default function StageDetailStatusCards({ item, stageStatus }) {
  const last = item?.last_eval;
  return (
    <div className="grid3">
      <Card title="Stage" status={stageStatus} tip={stageStatusTip(stageStatus)} rows={[{ k: "尺寸", v: item ? `${item.size}x${item.size}` : "--", mono: true }, { k: "更新时间", v: fmtTime(item?.updated_at_ms), mono: true }, { k: "最近 Eval 覆盖率", v: fmtCoverage(item?.last_eval_coverage), mono: true }]} />
      <Card title="BC" status={item?.bc_status} rows={[{ k: "Episode", v: item?.bc_episode ?? 0, mono: true }, { k: "最近 Eval 覆盖率", v: fmtCoverage(last?.phase === "bc" ? last?.coverage : null), mono: true }]} />
      <Card title="PPO" status={item?.ppo_status} rows={[{ k: "Episode", v: item?.ppo_episode ?? 0, mono: true }, { k: "最近 Eval 覆盖率", v: fmtCoverage(last?.phase === "ppo" ? last?.coverage : null), mono: true }]} />
    </div>
  );
}

