import { fmtCoverage, fmtTime } from "../utils/format.js";
import { phaseStatusText, pillClassForStatus, stageStatusText, stageStatusTip } from "../utils/stageStatus.js";

function KvRow({ label, value, mono }) {
  return (
    <div className="kv">
      <div className="kv__k">{label}</div>
      <div className={mono ? "kv__v mono num" : "kv__v"}>{value}</div>
    </div>
  );
}

function MetricCard({ label, value, pill, rows }) {
  return (
    <div className="card metricCard">
      <div className="metricCard__top">
        <div className="metricCard__label">{label}</div>
        {pill}
      </div>
      <div className="metricCard__value mono num">{value}</div>
      <div className="metricCard__subgrid">{rows}</div>
    </div>
  );
}

export default function StageDetailStatusCards({ item, stageStatus }) {
  const last = item?.last_eval;
  return (
    <div className="grid3">
      <MetricCard
        label="Eval Coverage"
        value={fmtCoverage(item?.last_eval_coverage)}
        pill={<span className={`${pillClassForStatus(stageStatus)} pill--sm`} title={stageStatusTip(stageStatus)}>{stageStatusText(stageStatus)}</span>}
        rows={(
          <>
            <KvRow label="Size" value={item ? `${item.size}x${item.size}` : "--"} mono />
            <KvRow label="Updated" value={fmtTime(item?.updated_at_ms)} mono />
          </>
        )}
      />
      <MetricCard
        label="BC Episode"
        value={String(item?.bc_episode ?? 0)}
        pill={<span className={`${pillClassForStatus(item?.bc_status)} pill--sm`}>BC {phaseStatusText(item?.bc_status)}</span>}
        rows={(
          <>
            <KvRow label="Last Eval" value={fmtCoverage(last?.phase === "bc" ? last?.coverage : null)} mono />
            <KvRow label="Status" value={phaseStatusText(item?.bc_status)} />
          </>
        )}
      />
      <MetricCard
        label="PPO Episode"
        value={String(item?.ppo_episode ?? 0)}
        pill={<span className={`${pillClassForStatus(item?.ppo_status)} pill--sm`}>PPO {phaseStatusText(item?.ppo_status)}</span>}
        rows={(
          <>
            <KvRow label="Last Eval" value={fmtCoverage(last?.phase === "ppo" ? last?.coverage : null)} mono />
            <KvRow label="Status" value={phaseStatusText(item?.ppo_status)} />
          </>
        )}
      />
    </div>
  );
}
