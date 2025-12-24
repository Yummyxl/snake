import { useMemo } from "react";
import { defaultMetrics, metricOptions } from "../utils/stageMetrics.js";

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

function toPath(series, key, w, h, padL, padR, padT, padB) {
  const b = bounds(series, key);
  if (!b) return [];
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
  const h = 180;
  const padL = 54;
  const padR = 14;
  const padT = 14;
  const padB = 30;
  const b = bounds(series, metricKey);
  if (!series?.length || !b) return <div className="chartEmpty">暂无指标数据</div>;
  const color = COLORS[metricKey] || "#0f172a";
  const paths = toPath(series, metricKey, w, h, padL, padR, padT, padB);
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

export default function StageDetailMetricsCard({ loading, metricsPhase, metrics }) {
  const options = useMemo(() => metricOptions(metricsPhase), [metricsPhase]);
  const series = metricsPhase === "bc" ? (metrics?.bc || []) : metricsPhase === "ppo" ? (metrics?.ppo || []) : [];
  const keys = useMemo(() => defaultMetrics(metricsPhase).filter((k) => options.includes(k)), [metricsPhase, options]);
  const showCharts = metricsPhase === "bc" || metricsPhase === "ppo";
  const emptyText = loading ? "加载中…" : showCharts ? "暂无指标数据" : metricsPhase ? "暂无指标数据（尚未接入）" : "暂无训练指标";
  return (
    <div className="card chartCard">
      <div className="cardHead"><div className="cardTitle">指标（大图）</div><div className="subtle mono">阶段：{metricsPhase ?? "--"}</div></div>
      {showCharts && series.length ? keys.map((k) => <MetricChart key={k} series={series} metricKey={k} />) : <div className="chartEmpty">{emptyText}</div>}
    </div>
  );
}
