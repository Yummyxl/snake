import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmtNumber } from "../utils/format.js";

const COLORS = {
  bc_loss: "#f59e0b",
  ppo_loss: "#f43f5e",
  reward_mean: "#06b6d4",
};

function normalizeSeries(series) {
  if (!Array.isArray(series)) return [];
  return series.map((it, i) => ({
    x: Number.isFinite(Number(it?.episode)) ? Number(it.episode) : i + 1,
    bc_loss: Number(it?.bc_loss),
    ppo_loss: Number(it?.ppo_loss),
    reward_mean: Number(it?.reward_mean),
  }));
}

function axisTick(digits) {
  return (v) => fmtNumber(v, digits);
}

function TooltipRow({ color, label, value }) {
  return (
    <div className="chartTooltip__row">
      <span className="chartTooltip__key">
        <span className="chartTooltip__dot" style={{ background: color }} aria-hidden="true" />
        {label}
      </span>
      <span className="chartTooltip__value mono num">{value}</span>
    </div>
  );
}

function MetricsTooltip({ active, payload, label, mode }) {
  if (!active || !payload?.length) return null;
  const byKey = new Map(payload.map((p) => [p?.dataKey, p]));
  const loss = byKey.get("ppo_loss")?.value;
  const reward = byKey.get("reward_mean")?.value;
  const bc = byKey.get("bc_loss")?.value;
  return (
    <div className="chartTooltip">
      <div className="chartTooltip__title mono">step: {label ?? "--"}</div>
      {mode === "ppo" ? (
        <>
          <TooltipRow color={COLORS.reward_mean} label="Reward" value={fmtNumber(reward, 2)} />
          <TooltipRow color={COLORS.ppo_loss} label="Loss" value={fmtNumber(loss, 4)} />
        </>
      ) : (
        <TooltipRow color={COLORS.bc_loss} label="BC Loss" value={fmtNumber(bc, 4)} />
      )}
    </div>
  );
}

function ChartShell({ ariaLabel, children }) {
  return (
    <div className="chartRecharts" role="img" aria-label={ariaLabel}>
      <ResponsiveContainer width="100%" height="100%">{children}</ResponsiveContainer>
    </div>
  );
}

function ChartGrid() {
  return <CartesianGrid vertical={false} stroke="#e5e7eb" strokeDasharray="3 3" strokeOpacity={0.55} />;
}

function ChartXAxis() {
  return <XAxis dataKey="x" axisLine={false} tickLine={false} tick={{ fill: "#71717a", fontSize: 11 }} minTickGap={28} />;
}

function ChartTooltip({ mode }) {
  return <Tooltip content={<MetricsTooltip mode={mode} />} cursor={{ stroke: "#e5e7eb", strokeDasharray: "3 3", strokeOpacity: 0.7 }} />;
}

function PpoDefs() {
  return (
    <defs>
      <linearGradient id="rewardFill" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor={COLORS.reward_mean} stopOpacity={0.22} />
        <stop offset="100%" stopColor={COLORS.reward_mean} stopOpacity={0} />
      </linearGradient>
      <linearGradient id="lossFill" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor={COLORS.ppo_loss} stopOpacity={0.22} />
        <stop offset="100%" stopColor={COLORS.ppo_loss} stopOpacity={0} />
      </linearGradient>
    </defs>
  );
}

function PpoAxes() {
  return (
    <>
      <YAxis yAxisId="left" axisLine={false} tickLine={false} tick={{ fill: COLORS.ppo_loss, fontSize: 11 }} width={46} domain={["auto", "auto"]} tickFormatter={axisTick(2)} />
      <YAxis yAxisId="right" orientation="right" axisLine={false} tickLine={false} tick={{ fill: COLORS.reward_mean, fontSize: 11 }} width={46} domain={["auto", "auto"]} tickFormatter={axisTick(2)} />
    </>
  );
}

function PpoAreas() {
  return (
    <>
      <Area type="monotone" yAxisId="right" dataKey="reward_mean" name="Reward" stroke={COLORS.reward_mean} strokeWidth={3} fill="url(#rewardFill)" dot={false} activeDot={{ r: 4, fill: COLORS.reward_mean, strokeWidth: 0 }} isAnimationActive={false} />
      <Area type="monotone" yAxisId="left" dataKey="ppo_loss" name="Loss" stroke={COLORS.ppo_loss} strokeWidth={3} fill="url(#lossFill)" dot={false} activeDot={{ r: 4, fill: COLORS.ppo_loss, strokeWidth: 0 }} isAnimationActive={false} />
    </>
  );
}

function PpoAreaChartView({ data }) {
  return (
    <ChartShell ariaLabel="PPO metrics chart">
      <AreaChart data={data} margin={{ top: 10, right: 28, left: 10, bottom: 6 }}>
        <PpoDefs />
        <ChartGrid />
        <ChartXAxis />
        <PpoAxes />
        <ChartTooltip mode="ppo" />
        <PpoAreas />
      </AreaChart>
    </ChartShell>
  );
}

function BcDefs() {
  return (
    <defs>
      <linearGradient id="bcFill" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor={COLORS.bc_loss} stopOpacity={0.22} />
        <stop offset="100%" stopColor={COLORS.bc_loss} stopOpacity={0} />
      </linearGradient>
    </defs>
  );
}

function BcAxis() {
  return <YAxis axisLine={false} tickLine={false} tick={{ fill: COLORS.bc_loss, fontSize: 11 }} width={46} domain={["auto", "auto"]} tickFormatter={axisTick(2)} />;
}

function BcArea() {
  return <Area type="monotone" dataKey="bc_loss" name="BC Loss" stroke={COLORS.bc_loss} strokeWidth={3} fill="url(#bcFill)" dot={false} activeDot={{ r: 4, fill: COLORS.bc_loss, strokeWidth: 0 }} isAnimationActive={false} />;
}

function BcAreaChartView({ data }) {
  return (
    <ChartShell ariaLabel="BC metrics chart">
      <AreaChart data={data} margin={{ top: 10, right: 18, left: 10, bottom: 6 }}>
        <BcDefs />
        <ChartGrid />
        <ChartXAxis />
        <BcAxis />
        <ChartTooltip mode="bc" />
        <BcArea />
      </AreaChart>
    </ChartShell>
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

function usePhaseSelection(metrics, metricsPhase) {
  const phases = useMemo(() => availablePhases(metrics, metricsPhase), [metrics, metricsPhase]);
  const [selected, setSelected] = useState(metricsPhase);
  const [touched, setTouched] = useState(false);

  useEffect(() => { if (!touched) setSelected(metricsPhase); }, [metricsPhase, touched]);
  useEffect(() => {
    if (selected && phases.includes(selected)) return;
    setSelected(phases[0] || metricsPhase || null);
  }, [metricsPhase, phases, selected]);

  const phase = selected || metricsPhase;
  const onPick = (v) => { setTouched(true); setSelected(v); };
  return { phases, phase, onPick };
}

function MetricsLegend({ phase }) {
  if (phase === "ppo") {
    return (
      <div className="legend" aria-label="legend">
        <span className="legendItem"><span className="legendSwatch" style={{ background: COLORS.reward_mean }} />Reward</span>
        <span className="legendItem"><span className="legendSwatch" style={{ background: COLORS.ppo_loss }} />Loss</span>
      </div>
    );
  }
  if (phase === "bc") {
    return (
      <div className="legend" aria-label="legend">
        <span className="legendItem"><span className="legendSwatch" style={{ background: COLORS.bc_loss }} />BC Loss</span>
      </div>
    );
  }
  return null;
}

function MetricsBody({ phase, loading, metrics }) {
  const rawSeries = phase === "bc" ? (metrics?.bc || []) : phase === "ppo" ? (metrics?.ppo || []) : [];
  const data = useMemo(() => normalizeSeries(rawSeries), [rawSeries]);
  const showCharts = phase === "bc" || phase === "ppo";
  const emptyText = loading ? "加载中…" : showCharts ? "暂无指标数据" : phase ? "暂无指标数据（尚未接入）" : "暂无训练指标";
  if (!showCharts || !data.length) return <div className="chartEmpty">{emptyText}</div>;
  return phase === "ppo" ? <PpoAreaChartView data={data} /> : <BcAreaChartView data={data} />;
}

export default function StageDetailMetricsCard({ loading, metricsPhase, metrics }) {
  const { phases, phase, onPick } = usePhaseSelection(metrics, metricsPhase);
  return (
    <div className="card chartCard">
      <div className="cardHead">
        <div className="cardTitle">Training Metrics</div>
        <div className="toolbar__right">
          <MetricsLegend phase={phase} />
          {phases.length > 1 ? phases.map((p) => <PhaseTab key={p} value={p} active={p === phase} onPick={onPick} />) : null}
          <div className="subtle mono">阶段：{phase ?? "--"}</div>
        </div>
      </div>
      <MetricsBody phase={phase} loading={loading} metrics={metrics} />
    </div>
  );
}
