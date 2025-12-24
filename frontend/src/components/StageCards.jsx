import { Link } from "react-router-dom";
import { fmtCoverage, fmtTime } from "../utils/format.js";
import { deriveStageStatus, phaseStatusText, pillClassForStatus, stageStatusText, stageStatusTip } from "../utils/stageStatus.js";

function CardRow({ label, value, mono }) {
  return (
    <div className="kv">
      <div className="kv__k">{label}</div>
      <div className={mono ? "kv__v mono" : "kv__v"}>{value}</div>
    </div>
  );
}

function CoverageBar({ value }) {
  const n = Number(value);
  const v = Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : null;
  return (
    <div className="coverageBar" title={v === null ? "暂无 Eval 覆盖率" : `覆盖率 ${fmtCoverage(v)}`}>
      <div className="coverageBar__track" aria-hidden="true">
        <div className="coverageBar__fill" style={{ width: v === null ? "0%" : `${Math.round(v * 100)}%` }} />
      </div>
      <div className="coverageBar__text mono">{fmtCoverage(v)}</div>
    </div>
  );
}

function PhasePills({ bcStatus, ppoStatus }) {
  return (
    <div className="phasePills">
      <span className={`${pillClassForStatus(bcStatus)} pill--sm`} title={`BC：${phaseStatusText(bcStatus)}`}>
        BC {phaseStatusText(bcStatus)}
      </span>
      <span className={`${pillClassForStatus(ppoStatus)} pill--sm`} title={`PPO：${phaseStatusText(ppoStatus)}`}>
        PPO {phaseStatusText(ppoStatus)}
      </span>
    </div>
  );
}

function StageCard({ s }) {
  const stageStatus = deriveStageStatus(s.bc_status, s.ppo_status);
  return (
    <Link className={`stageCard stageCard--${stageStatus}`} to={`/stages/${s.stage_id}`}>
      <div className="stageCard__top">
        <div className="stageCard__title">
          Stage {s.stage_id} <span className="muted">({s.size}x{s.size})</span>
        </div>
        <div className={pillClassForStatus(stageStatus)} title={stageStatusTip(stageStatus)}>
          {stageStatusText(stageStatus)}
        </div>
      </div>
      <div className="stageCard__body">
        <CardRow label="Phase" value={s.current_phase ?? "--"} mono />
        <CardRow label="阶段状态" value={<PhasePills bcStatus={s.bc_status} ppoStatus={s.ppo_status} />} />
        <CardRow
          label="Episode"
          value={`BC ${s.bc_episode} / PPO ${s.ppo_episode} / Total ${s.total_episode}`}
          mono
        />
        <CardRow label="Eval 覆盖率" value={<CoverageBar value={s.last_eval_coverage} />} />
        <CardRow label="更新时间" value={fmtTime(s.updated_at_ms)} mono />
      </div>
    </Link>
  );
}

function SkeletonStageCard({ i }) {
  return (
    <div className="stageCard stageCard--skeleton" role="listitem" aria-label={`加载中 ${i + 1}`}>
      <div className="stageCard__top">
        <div className="skeleton skeleton--title" />
        <div className="skeleton skeleton--pill" />
      </div>
      <div className="stageCard__body">
        <div className="skeleton skeleton--row" />
        <div className="skeleton skeleton--row" />
        <div className="skeleton skeleton--row" />
        <div className="skeleton skeleton--row" />
        <div className="skeleton skeleton--row" />
      </div>
    </div>
  );
}

export default function StageCards({ items, loading }) {
  if (loading && items.length === 0) {
    return (
      <div className="cardsRow" role="list" aria-label="Stage 列表加载中">
        {Array.from({ length: 6 }).map((_, i) => <SkeletonStageCard key={String(i)} i={i} />)}
      </div>
    );
  }
  if (items.length === 0) return <div className="empty">暂无 Stage，请初始化</div>;
  return (
    <div className="cardsRow" role="list">
      {items.map((s) => (
        <StageCard key={String(s.stage_id)} s={s} />
      ))}
    </div>
  );
}
