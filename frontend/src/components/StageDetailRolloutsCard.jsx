import { fmtCoverage, fmtNumber, fmtTime } from "../utils/format.js";

export default function StageDetailRolloutsCard({ phase, rollouts, onPlay }) {
  return (
    <div className="card">
      <div className="cardHead"><div className="cardTitle">Eval Rollouts</div><div className="subtle mono">阶段：{phase ?? "--"} / best 优先，其余按时间倒序</div></div>
      {rollouts.length === 0 ? (
        <div className="empty">暂无评估回放</div>
      ) : (
        <table className="table table--center">
          <colgroup>
            <col style={{ width: 72 }} />
            <col style={{ width: 64 }} />
            <col style={{ width: 120 }} />
            <col style={{ width: 90 }} />
            <col style={{ width: 90 }} />
            <col style={{ width: 110 }} />
            <col style={{ width: 110 }} />
            <col style={{ width: 180 }} />
            <col style={{ width: 90 }} />
          </colgroup>
          <thead><tr><th>标记</th><th>phase</th><th>rollout_id</th><th>coverage</th><th>steps</th><th>length_max</th><th>reward</th><th>时间</th><th>操作</th></tr></thead>
          <tbody>
            {rollouts.map((r) => (
              <tr key={`${r.phase}:${r.rollout_id}`}>
                <td>{r.is_best ? <span className="pill pill--green pill--sm">Best</span> : ""}</td>
                <td className="mono">{r.phase}</td><td className="mono">{r.rollout_id}</td>
                <td className="mono">{fmtCoverage(r.coverage)}</td><td className="mono">{r.steps ?? "--"}</td><td className="mono">{r.length_max ?? "--"}</td>
                <td className="mono">{fmtNumber(r.reward_total, 2)}</td>
                <td className="mono">{fmtTime(r.created_at_ms)}</td><td><button className="btn" type="button" onClick={() => onPlay(r)}>播放</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
