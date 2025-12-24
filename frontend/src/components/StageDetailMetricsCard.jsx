import { useEffect, useMemo, useState } from "react";

const COLORS = {
  bc_loss: "#60a5fa",
  ppo_loss: "#f97316",
  reward_mean: "#22c55e",
};

function bounds(series, key) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const it of series) {
    const v = Number(it?.[key]);
    if (!Number.isFinite(v)) continue;
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
  if (lo === hi) return { lo: lo - 1, hi: hi + 1 };
  const pad = (hi - lo) * 0.05;
  return { lo: lo - pad, hi: hi + pad };
}

function fmtTick(key, v) {
  if (!Number.isFinite(v)) return "--";
  if (key === "reward_mean") return v.toFixed(2);
  return v.toFixed(4);
}

function toPath(series, key, b, w, h, padL, padR, padT, padB) {
  const n = series.length;
  if (n < 2) return [];
  const segs = [];
  let cur = [];
  const xAt = (i) => padL + (i * (w - padL - padR)) / (n - 1);
  const yAt = (v) => padT + (1 - (v - b.lo) / (b.hi - b.lo)) * (h - padT - padB);
  for (let i = 0; i < n; i += 1) {
    const v = Number(series[i]?.[key]);
    if (!Number.isFinite(v)) { if (cur.length) { segs.push(cur); cur = []; } continue; }
    cur.push([xAt(i), yAt(v)]);
  }
  if (cur.length) segs.push(cur);
  return segs.map((pts) => `M ${pts[0][0].toFixed(2)} ${pts[0][1].toFixed(2)} ${pts.slice(1).map(([x, y]) => `L ${x.toFixed(2)} ${y.toFixed(2)}`).join(" ")}`);
}

function pickEpisodes(series) {
  const eps = series.map((it, i) => (Number.isFinite(Number(it?.episode)) ? Number(it.episode) : i + 1));
  if (!eps.length) return { first: 0, mid: 0, last: 0 };
  return { first: eps[0], mid: eps[Math.floor((eps.length - 1) / 2)], last: eps[eps.length - 1] };
}

function MetricChart({ series, metricKey }) {
  const w = 680;
  const h = 220;
  const padL = 54;
  const padR = 14;
  const padT = 14;
  const padB = 30;
  const b = bounds(series, metricKey);
  if (!series?.length || !b) return <div className="chartEmpty">暂无指标数据</div>;
  const color = COLORS[metricKey] || "#0f172a";
  const paths = toPath(series, metricKey, b, w, h, padL, padR, padT, padB);
  const y0 = h - padB;
  const x0 = padL;
  const ticks = [
    { t: 0, v: b.lo },
    { t: 0.5, v: (b.lo + b.hi) / 2 },
    { t: 1, v: b.hi },
  ];
  const eps = pickEpisodes(series);
  return (
    <div className="chartWrap">
      <svg className="chartSvg" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`${metricKey} chart`}>
        <rect x="0" y="0" width={w} height={h} fill="#ffffff" />
        <rect x={padL} y={padT} width={w - padL - padR} height={h - padT - padB} fill="#f8fafc" />
        {ticks.map(({ t, v }) => (
          <line key={t} x1={padL} x2={w - padR} y1={padT + (1 - t) * (h - padT - padB)} y2={padT + (1 - t) * (h - padT - padB)} stroke="#e2e8f0" strokeWidth="1" />
        ))}
        <line x1={x0} x2={w - padR} y1={y0} y2={y0} stroke="#cbd5e1" strokeWidth="1.5" />
        <line x1={x0} x2={x0} y1={padT} y2={y0} stroke="#cbd5e1" strokeWidth="1.5" />
        {ticks.map(({ t, v }) => (
          <text key={`yt:${t}`} x={x0 - 8} y={padT + (1 - t) * (h - padT - padB) + 4} textAnchor="end" fontSize="11" fill="#64748b">{fmtTick(metricKey, v)}</text>
        ))}
        <text x={x0} y={h - 8} textAnchor="start" fontSize="11" fill="#64748b">{String(eps.first)}</text>
        <text x={(x0 + (w - padR)) / 2} y={h - 8} textAnchor="middle" fontSize="11" fill="#64748b">{String(eps.mid)}</text>
        <text x={w - padR} y={h - 8} textAnchor="end" fontSize="11" fill="#64748b">{String(eps.last)}</text>
        {paths.map((d) => (
          <path key={d} d={d} fill="none" stroke={color} strokeWidth="2" />
        ))}
      </svg>
    </div>
  );
}

function PpoCombinedChart({ series }) {
  const w = 680;
  const h = 240;
  const padL = 54;
  const padR = 54;
  const padT = 14;
  const padB = 30;
  const bLoss = bounds(series, "ppo_loss");
  const bReward = bounds(series, "reward_mean");
  if (!series?.length || !bLoss || !bReward) return <div className="chartEmpty">暂无指标数据</div>;
  const lossPaths = toPath(series, "ppo_loss", bLoss, w, h, padL, padR, padT, padB);
  const rewardPaths = toPath(series, "reward_mean", bReward, w, h, padL, padR, padT, padB);
  const y0 = h - padB;
  const xL = padL;
  const xR = w - padR;
  const ticks = [0, 0.5, 1];
  const eps = pickEpisodes(series);
  return (
    <div className="chartWrap">
      <svg className="chartSvg" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="ppo chart">
        <rect x="0" y="0" width={w} height={h} fill="#ffffff" />
        <rect x={padL} y={padT} width={w - padL - padR} height={h - padT - padB} fill="#f8fafc" />
        {ticks.map((t) => (
          <line key={t} x1={padL} x2={w - padR} y1={padT + (1 - t) * (h - padT - padB)} y2={padT + (1 - t) * (h - padT - padB)} stroke="#e2e8f0" strokeWidth="1" />
        ))}
        <line x1={xL} x2={w - padR} y1={y0} y2={y0} stroke="#cbd5e1" strokeWidth="1.5" />
        <line x1={xL} x2={xL} y1={padT} y2={y0} stroke="#cbd5e1" strokeWidth="1.5" />
        <line x1={xR} x2={xR} y1={padT} y2={y0} stroke="#cbd5e1" strokeWidth="1.5" />
        {ticks.map((t) => {
          const y = padT + (1 - t) * (h - padT - padB);
          const vLoss = bLoss.lo + (bLoss.hi - bLoss.lo) * t;
          const vReward = bReward.lo + (bReward.hi - bReward.lo) * t;
          return (
            <g key={t}>
              <text x={xL - 8} y={y + 4} textAnchor="end" fontSize="11" fill={COLORS.ppo_loss}>{fmtTick("ppo_loss", vLoss)}</text>
              <text x={xR + 8} y={y + 4} textAnchor="start" fontSize="11" fill={COLORS.reward_mean}>{fmtTick("reward_mean", vReward)}</text>
            </g>
          );
        })}
        <text x={xL} y={h - 8} textAnchor="start" fontSize="11" fill="#64748b">{String(eps.first)}</text>
        <text x={(xL + xR) / 2} y={h - 8} textAnchor="middle" fontSize="11" fill="#64748b">{String(eps.mid)}</text>
        <text x={xR} y={h - 8} textAnchor="end" fontSize="11" fill="#64748b">{String(eps.last)}</text>
        {lossPaths.map((d) => <path key={`l:${d}`} d={d} fill="none" stroke={COLORS.ppo_loss} strokeWidth="2" />)}
        {rewardPaths.map((d) => <path key={`r:${d}`} d={d} fill="none" stroke={COLORS.reward_mean} strokeWidth="2" />)}
        <g transform={`translate(${padL + 10} ${padT + 10})`}>
          <rect x="0" y="0" width="10" height="10" fill={COLORS.ppo_loss} />
          <text x="14" y="10" fontSize="12" fill="#0f172a">ppo_loss</text>
          <rect x="94" y="0" width="10" height="10" fill={COLORS.reward_mean} />
          <text x="108" y="10" fontSize="12" fill="#0f172a">reward_mean</text>
        </g>
      </svg>
    </div>
  );
}

function PhaseTab({ value, active, onPick }) {
  const cls = value === "ppo" ? "pill pill--amber pill--sm" : "pill pill--blue pill--sm";
  return (
    <button className={`pillBtn ${cls} ${active ? "pillBtn--active" : ""}`} type="button" onClick={() => onPick(value)}>
      {value.toUpperCase()}
    </button>
  );
}

function availablePhases(metrics, hintPhase) {
  const hasBc = Array.isArray(metrics?.bc) && metrics.bc.length > 0;
  const hasPpo = Array.isArray(metrics?.ppo) && metrics.ppo.length > 0;
  const out = [];
  if (hintPhase === "ppo" || hasPpo) out.push("ppo");
  if (hintPhase === "bc" || hasBc) out.push("bc");
  if (out.length === 2) return out;
  if (hasBc && !out.includes("bc")) out.push("bc");
  if (hasPpo && !out.includes("ppo")) out.push("ppo");
  return out;
}

export default function StageDetailMetricsCard({ loading, metricsPhase, metrics }) {
  const phases = useMemo(() => availablePhases(metrics, metricsPhase), [metrics, metricsPhase]);
  const [selected, setSelected] = useState(metricsPhase);
  const [touched, setTouched] = useState(false);

  useEffect(() => { if (!touched) setSelected(metricsPhase); }, [metricsPhase, touched]);
  useEffect(() => {
    if (selected && phases.includes(selected)) return;
    setSelected(phases[0] || metricsPhase || null);
  }, [metricsPhase, phases, selected]);

  const phase = selected || metricsPhase;
  const series = phase === "bc" ? (metrics?.bc || []) : phase === "ppo" ? (metrics?.ppo || []) : [];
  const showCharts = phase === "bc" || phase === "ppo";
  const emptyText = loading ? "加载中…" : showCharts ? "暂无指标数据" : phase ? "暂无指标数据（尚未接入）" : "暂无训练指标";
  return (
    <div className="card chartCard">
      <div className="cardHead">
        <div className="cardTitle">指标（大图）</div>
        <div className="toolbar__right">
          {phases.length > 1 ? phases.map((p) => <PhaseTab key={p} value={p} active={p === phase} onPick={(v) => { setTouched(true); setSelected(v); }} />) : null}
          <div className="subtle mono">阶段：{phase ?? "--"}</div>
        </div>
      </div>
      {showCharts && series.length ? (
        phase === "ppo" ? <PpoCombinedChart series={series} /> : <MetricChart series={series} metricKey="bc_loss" />
      ) : <div className="chartEmpty">{emptyText}</div>}
    </div>
  );
}
