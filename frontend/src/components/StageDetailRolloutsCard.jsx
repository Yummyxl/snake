import { fmtCoverage, fmtNumber, fmtTime } from "../utils/format.js";
import { IconPlay } from "./Icons.jsx";

export default function StageDetailRolloutsCard({ phase, rollouts, onPlay }) {
  return (
    <div className="card">
      <div className="cardHead">
        <div className="cardTitle">Eval Rollouts</div>
        <div className="subtle mono">phase: {phase ?? "--"} · sorted: best → newest</div>
      </div>
      {rollouts.length === 0 ? (
        <div className="empty">暂无评估回放</div>
      ) : (
        <table className="table table--center">
          <colgroup>
            <col style={{ width: 140 }} />
            <col style={{ width: 90 }} />
            <col style={{ width: 90 }} />
            <col style={{ width: 110 }} />
            <col style={{ width: 110 }} />
            <col style={{ width: 180 }} />
            <col style={{ width: 72 }} />
          </colgroup>
          <thead><tr><th>ID</th><th>coverage</th><th>steps</th><th>length_max</th><th>reward</th><th>时间</th><th>操作</th></tr></thead>
          <tbody>
            {rollouts.map((r) => (
              <tr key={`${r.phase}:${r.rollout_id}`} className={r.is_best ? "bestRow" : ""}>
                <td className="mono">
                  <span className="idCell">
                    <span>{r.rollout_id}</span>
                    {r.is_best ? <span className="bestTag">best</span> : null}
                  </span>
                </td>
                <td className="mono">{fmtCoverage(r.coverage)}</td>
                <td className="mono">{r.steps ?? "--"}</td>
                <td className="mono">{r.length_max ?? "--"}</td>
                <td className={`mono ${Number(r.reward_total) > 0 ? "rewardPos" : Number(r.reward_total) < 0 ? "rewardNeg" : "rewardZero"}`}>{fmtNumber(r.reward_total, 2)}</td>
                <td className="mono">{fmtTime(r.created_at_ms)}</td>
                <td>
                  <button className="iconBtn" type="button" onClick={() => onPlay(r)} aria-label="Play" title="Play">
                    <IconPlay />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
