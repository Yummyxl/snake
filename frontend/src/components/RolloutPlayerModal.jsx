import { useEffect, useMemo, useRef, useState } from "react";
import { fmtCoverage } from "../utils/format.js";
import { IconFirst, IconNext, IconPause, IconPlay, IconPrev, IconX } from "./Icons.jsx";
import { drawRolloutFrame } from "./rolloutPlayer/draw.js";

function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n));
}

function useAutoplay({ open, playing, speed, stepsLen, onStop, onNext }) {
  useEffect(() => {
    if (!open || !playing || stepsLen <= 1) return;
    const ms = Math.max(16, Math.floor(200 / (speed || 1)));
    const id = window.setInterval(() => {
      const done = onNext();
      if (done) onStop();
    }, ms);
    return () => window.clearInterval(id);
  }, [open, onNext, onStop, playing, speed, stepsLen]);
}

function useHotkeys({ open, canPlay, onPrev, onNext, onTogglePlay, onClose, onToggleGrid }) {
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e) => {
      if (e.key === "Escape") { onClose(); return; }
      if (!canPlay) return;
      if (e.key === "ArrowLeft") { e.preventDefault(); onPrev(); return; }
      if (e.key === "ArrowRight") { e.preventDefault(); onNext(); return; }
      if (e.key === " " || e.key === "Spacebar") { e.preventDefault(); onTogglePlay(); return; }
      if (e.key === "g" || e.key === "G") { e.preventDefault(); onToggleGrid(); }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [canPlay, onClose, onNext, onPrev, onToggleGrid, onTogglePlay, open]);
}

function usePlaybackState({ open, stepsLen, rolloutId }) {
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(6);
  const [showGrid, setShowGrid] = useState(true);

  useEffect(() => { if (open) { setT(0); setPlaying(false); } }, [open, rolloutId]);
  useAutoplay({
    open,
    playing,
    speed,
    stepsLen,
    onStop: () => setPlaying(false),
    onNext: () => {
      let done = false;
      setT((cur) => { const next = clamp(cur + 1, 0, Math.max(0, stepsLen - 1)); done = next === cur; return next; });
      return done;
    },
  });

  return { t, setT, playing, setPlaying, speed, setSpeed, showGrid, setShowGrid };
}

function HeaderPill({ children }) {
  return <span className="pill pill--gray pill--sm mono num">{children}</span>;
}

function PlayerHeader({ title, stepsLen, coverage, onClose }) {
  return (
    <div className="playerHeader">
      <div className="playerHeader__left">
        <div className="playerHeader__title">{title || "Playback"}</div>
        <div className="playerHeader__meta">
          <HeaderPill>steps: {stepsLen || "--"}</HeaderPill>
          <HeaderPill>coverage_max: {fmtCoverage(coverage)}</HeaderPill>
        </div>
      </div>
      <button className="iconBtn" type="button" onClick={onClose} aria-label="Close" title="Close">
        <IconX />
      </button>
    </div>
  );
}

function InspectorRow({ label, value }) {
  return (
    <div className="kv">
      <div className="kv__k">{label}</div>
      <div className="kv__v mono num">{value}</div>
    </div>
  );
}

function PlayerInspector({ step, t, stepsLen, size, summary }) {
  return (
    <div className="logPanel">
      <div className="logPanel__title">Inspector</div>
      <div className="cardBody" style={{ padding: 0 }}>
        <InspectorRow label="Step" value={`${t + 1}/${Math.max(1, stepsLen)}`} />
        <InspectorRow label="Dir" value={`${step?.dir ?? "--"}${step?.dir_next ? ` → ${step.dir_next}` : ""}`} />
        <InspectorRow label="Action" value={step?.action ?? "--"} />
        <InspectorRow label="Ate" value={String(Boolean(step?.info?.ate))} />
        <InspectorRow label="Collision" value={step?.info?.collision ?? "--"} />
        <InspectorRow label="Done" value={String(Boolean(step?.done))} />
        <InspectorRow label="Size" value={size ?? "--"} />
        <InspectorRow label="Max Length" value={summary?.length_max ?? summary?.snake_length_max ?? "--"} />
      </div>
    </div>
  );
}

function PlayerDock({ stepsLen, t, setT, canPlay, playing, setPlaying, speed, setSpeed, showGrid, setShowGrid, size }) {
  return (
    <div className="playerDock" aria-label="player controls">
      <div className="scrubRow">
        <input className="scrubBar" type="range" min={0} max={Math.max(0, stepsLen - 1)} value={t} onChange={(e) => { setPlaying(false); setT(Number(e.target.value)); }} disabled={!canPlay} />
        <div className="mono num">{t + 1}/{Math.max(1, stepsLen)}</div>
      </div>
      <div className="playerDock__row">
        <div className="playerDock__left">
          <label className="subtle">Speed</label>
          <select className="btn" value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            <option value={0.5}>0.5x</option><option value={1}>1x</option><option value={2}>2x</option><option value={4}>4x</option><option value={6}>6x</option><option value={8}>8x</option>
          </select>
          <button className={`btn ${showGrid ? "btn--primary" : ""}`} type="button" onClick={() => setShowGrid((v) => !v)}>Grid</button>
        </div>
        <div className="playerDock__center">
          <button className="iconBtn iconBtn--lg" type="button" onClick={() => { setPlaying(false); setT(0); }} disabled={!canPlay} aria-label="First">
            <IconFirst width="18" height="18" />
          </button>
          <button className="iconBtn iconBtn--lg" type="button" onClick={() => { setPlaying(false); setT((x) => clamp(x - 1, 0, Math.max(0, stepsLen - 1))); }} disabled={!canPlay} aria-label="Prev">
            <IconPrev width="18" height="18" />
          </button>
          <button className="iconBtn iconBtn--lg" type="button" onClick={() => setPlaying((v) => !v)} disabled={!canPlay} aria-label={playing ? "Pause" : "Play"}>
            {playing ? <IconPause width="18" height="18" /> : <IconPlay width="18" height="18" />}
          </button>
          <button className="iconBtn iconBtn--lg" type="button" onClick={() => { setPlaying(false); setT((x) => clamp(x + 1, 0, Math.max(0, stepsLen - 1))); }} disabled={!canPlay} aria-label="Next">
            <IconNext width="18" height="18" />
          </button>
        </div>
        <div className="playerDock__right mono subtle">size: {size ?? "--"}</div>
      </div>
    </div>
  );
}

function useRolloutData(data) {
  const steps = useMemo(() => (Array.isArray(data?.rollout?.steps) ? data.rollout.steps : []), [data]);
  const summary = data?.rollout?.summary || null;
  const coverage = summary?.coverage_max ?? summary?.coverage;
  return { steps, summary, coverage };
}

function useCanvasDraw({ open, canvasRef, size, step, showGrid }) {
  useEffect(() => {
    if (!open) return;
    drawRolloutFrame({ canvas: canvasRef.current, size: Number(size), step, showGrid });
  }, [open, showGrid, size, step, canvasRef]);
}

function PlayerStage({ canvasRef, stepsLen, t, step, error, loading }) {
  return (
    <div className="playerStage">
      <div className="playerCanvasWrap">
        <canvas ref={canvasRef} width={560} height={560} className="playerCanvas" />
      </div>
      {error ? <div className="dangerText">加载失败：{error}</div> : null}
      {loading ? <div className="subtle">加载中...</div> : null}
      {!loading && !error && stepsLen === 0 ? <div className="subtle">该 rollout 未包含 steps，无法播放</div> : null}
    </div>
  );
}

function RolloutPlayerDialog({ open, title, size, data, loading, error, onClose }) {
  const { steps, summary, coverage } = useRolloutData(data);
  const pb = usePlaybackState({ open, stepsLen: steps.length, rolloutId: data?.rollout_id });
  const canvasRef = useRef(null);
  const step = steps[pb.t] || null;
  useCanvasDraw({ open, canvasRef, size, step, showGrid: pb.showGrid });
  useHotkeys({
    open,
    canPlay: steps.length > 1,
    onClose,
    onPrev: () => { pb.setPlaying(false); pb.setT((x) => clamp(x - 1, 0, Math.max(0, steps.length - 1))); },
    onNext: () => { pb.setPlaying(false); pb.setT((x) => clamp(x + 1, 0, Math.max(0, steps.length - 1))); },
    onTogglePlay: () => pb.setPlaying((v) => !v),
    onToggleGrid: () => pb.setShowGrid((v) => !v),
  });

  return (
    <div className="modalOverlay modalOverlay--center" role="dialog" aria-modal="true" aria-label="回放">
      <div className="modal modal--player">
        <PlayerHeader title={title} stepsLen={steps.length} coverage={coverage} onClose={onClose} />
        <div className="playerLayout">
          <PlayerStage canvasRef={canvasRef} stepsLen={steps.length} t={pb.t} step={step} error={error} loading={loading} />
          <div className="playerSide"><PlayerInspector step={step} t={pb.t} stepsLen={steps.length} size={size} summary={summary} /></div>
          <PlayerDock stepsLen={steps.length} t={pb.t} setT={pb.setT} canPlay={steps.length > 1} playing={pb.playing} setPlaying={pb.setPlaying} speed={pb.speed} setSpeed={pb.setSpeed} showGrid={pb.showGrid} setShowGrid={pb.setShowGrid} size={size} />
        </div>
      </div>
    </div>
  );
}

export default function RolloutPlayerModal({ open, title, size, data, loading, error, onClose }) {
  if (!open) return null;
  return <RolloutPlayerDialog open={open} title={title} size={size} data={data} loading={loading} error={error} onClose={onClose} />;
}
