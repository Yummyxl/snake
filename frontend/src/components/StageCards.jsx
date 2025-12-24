import { Link } from "react-router-dom";
import { fmtCoverage, fmtTime } from "../utils/format.js";
import { deriveStageStatus, phaseStatusText, pillClassForStatus, stageStatusText, stageStatusTip } from "../utils/stageStatus.js";

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

function Stat({ label, value, sub }) {
  return (
    <div className="stageStat">
      <div className="stageStat__label">{label}</div>
      <div className="stageStat__value">{value}</div>
      {sub ? <div className="stageStat__sub">{sub}</div> : null}
    </div>
  );
}

function StageCard({ s }) {
  const stageStatus = deriveStageStatus(s.bc_status, s.ppo_status);
  return (
    <Link className={`stageCard stageCard--${stageStatus}`} to={`/stages/${s.stage_id}`}>
      <div className="stageCard__header">
        <div className="stageCard__headerLeft">
          <div className="stageCard__title">Stage {s.stage_id}</div>
          <div className="stageCard__subtitle mono">
            {s.size}x{s.size} · phase: {s.current_phase ?? "--"}
          </div>
        </div>
        <div className={pillClassForStatus(stageStatus)} title={stageStatusTip(stageStatus)}>
          {stageStatusText(stageStatus)}
        </div>
      </div>
      <div className="stageCard__stats">
        <Stat
          label="Episodes"
          value={<span className="mono num">{s.total_episode ?? 0}</span>}
          sub={<span className="mono subtle">BC {s.bc_episode ?? 0} / PPO {s.ppo_episode ?? 0}</span>}
        />
        <Stat label="Eval Coverage" value={<CoverageBar value={s.last_eval_coverage} />} />
      </div>
      <div className="stageCard__footer">
        <span className="mono subtle">{fmtTime(s.updated_at_ms)}</span>
        <PhasePills bcStatus={s.bc_status} ppoStatus={s.ppo_status} />
      </div>
    </Link>
  );
}

function SkeletonStageCard({ i }) {
  return (
    <div className="stageCard stageCard--skeleton" role="listitem" aria-label={`加载中 ${i + 1}`}>
      <div className="stageCard__header">
        <div className="skeleton skeleton--title" />
        <div className="skeleton skeleton--pill" />
      </div>
      <div className="stageCard__stats">
        <div className="skeleton skeleton--row" />
        <div className="skeleton skeleton--row" />
      </div>
      <div className="stageCard__footer">
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
