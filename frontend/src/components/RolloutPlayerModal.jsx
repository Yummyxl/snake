import { useEffect, useMemo, useRef, useState } from "react";
import { fmtCoverage } from "../utils/format.js";

function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n));
}

function clearCanvas(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, w, h);
}

function calcBoard(size, w, h) {
  const cell = Math.floor(Math.min(w, h) / size);
  const ox = Math.floor((w - cell * size) / 2);
  const oy = Math.floor((h - cell * size) / 2);
  return { cell, ox, oy };
}

function drawBoard(ctx, size, b) {
  ctx.fillStyle = "#1e293b";
  ctx.fillRect(b.ox, b.oy, b.cell * size, b.cell * size);
}

function drawGrid(ctx, size, b) {
  if (b.cell < 6) return;
  ctx.strokeStyle = "rgba(226, 232, 240, 0.10)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= size; i += 1) {
    const x = b.ox + i * b.cell;
    ctx.moveTo(x, b.oy);
    ctx.lineTo(x, b.oy + size * b.cell);
    const y = b.oy + i * b.cell;
    ctx.moveTo(b.ox, y);
    ctx.lineTo(b.ox + size * b.cell, y);
  }
  ctx.stroke();
}

function drawFood(ctx, b, food) {
  if (!food || food.length !== 2) return;
  const c = cellCenter(b, food);
  const r = Math.max(3, Math.floor(b.cell * 0.34));
  ctx.fillStyle = "#f59e0b";
  ctx.beginPath();
  ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.75)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(c.x, c.y, Math.max(2, r - 1.5), 0, Math.PI * 2);
  ctx.stroke();
}

function cellCenter(b, p) {
  const x = b.ox + p[0] * b.cell + b.cell / 2;
  const y = b.oy + p[1] * b.cell + b.cell / 2;
  return { x, y };
}

function drawSnakePath(ctx, b, snake) {
  if (!Array.isArray(snake) || snake.length < 2) return;
  const head = snake[0];
  const tail = snake[snake.length - 1];
  const a = cellCenter(b, head);
  const z = cellCenter(b, tail);
  const grad = ctx.createLinearGradient(a.x, a.y, z.x, z.y);
  grad.addColorStop(0, "#60a5fa");
  grad.addColorStop(1, "#16a34a");
  ctx.strokeStyle = grad;
  ctx.lineWidth = Math.max(3, Math.floor(b.cell * 0.62));
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  for (let i = 1; i < snake.length; i += 1) {
    const p = snake[i];
    if (!Array.isArray(p) || p.length !== 2) continue;
    const c = cellCenter(b, p);
    ctx.lineTo(c.x, c.y);
  }
  ctx.stroke();
}

function drawSnakeMarkers(ctx, b, snake) {
  if (!Array.isArray(snake) || snake.length === 0) return;
  const head = snake[0];
  if (!Array.isArray(head) || head.length !== 2) return;
  const c = cellCenter(b, head);
  const r = Math.max(3, Math.floor(b.cell * 0.36));
  ctx.fillStyle = "#60a5fa";
  ctx.beginPath();
  ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(c.x, c.y, Math.max(2, r - 2), 0, Math.PI * 2);
  ctx.stroke();
}

function drawSnake(ctx, b, snake) {
  if (!Array.isArray(snake)) return;
  ctx.globalAlpha = 0.18;
  ctx.fillStyle = "#16a34a";
  for (let i = 0; i < snake.length; i += 1) {
    const p = snake[i];
    if (!Array.isArray(p) || p.length !== 2) continue;
    ctx.fillRect(b.ox + p[0] * b.cell, b.oy + p[1] * b.cell, b.cell, b.cell);
  }
  ctx.globalAlpha = 1;
  drawSnakePath(ctx, b, snake);
  drawSnakeMarkers(ctx, b, snake);
}

function drawFrame(canvas, size, step) {
  if (!canvas || !step || !Number.isFinite(size) || size <= 0) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  clearCanvas(ctx, w, h);
  const b = calcBoard(size, w, h);
  drawBoard(ctx, size, b);
  drawGrid(ctx, size, b);
  drawFood(ctx, b, Array.isArray(step?.food) ? step.food : null);
  drawSnake(ctx, b, step?.snake);
}

export default function RolloutPlayerModal({ open, title, size, data, loading, error, onClose }) {
  const steps = useMemo(() => (data?.rollout?.steps && Array.isArray(data.rollout.steps) ? data.rollout.steps : []), [data]);
  const summary = data?.rollout?.summary || null;
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(6);
  const canvasRef = useRef(null);
  const step = steps[t] || null;

  useEffect(() => {
    if (!open) return;
    setT(0);
    setPlaying(false);
  }, [open, data?.rollout_id]);

  useEffect(() => {
    if (!open) return;
    drawFrame(canvasRef.current, Number(size), step);
  }, [open, size, step]);

  useEffect(() => {
    if (!open || !playing) return;
    if (steps.length <= 1) return;
    const ms = Math.max(16, Math.floor(200 / (speed || 1)));
    const id = window.setInterval(() => {
      setT((cur) => {
        const next = clamp(cur + 1, 0, Math.max(0, steps.length - 1));
        if (next === cur) setPlaying(false);
        return next;
      });
    }, ms);
    return () => window.clearInterval(id);
  }, [open, playing, speed, steps.length]);

  if (!open) return null;
  return (
    <div className="modalOverlay modalOverlay--top" role="dialog" aria-modal="true" aria-label="回放">
      <div className="modal modal--wide">
        <div className="modal__title">{title || "回放"}</div>
        <div className="playerMeta">
          <div className="mono">steps: {steps.length || "--"} / coverage_max: {fmtCoverage(summary?.coverage_max ?? summary?.coverage)}</div>
          <div className="mono subtle">
            dir: {step?.dir ?? "--"}
            {step?.dir_next ? ` → ${step.dir_next}` : ""}
            {" / "}
            action: {step?.action ?? "--"} / done: {String(Boolean(step?.done))}
          </div>
          {error ? <div className="dangerText">加载失败：{error}</div> : null}
          {loading ? <div className="subtle">加载中...</div> : null}
          {!loading && !error && steps.length === 0 ? <div className="subtle">该 rollout 未包含 steps，无法播放</div> : null}
        </div>
        <div className="playerBody">
          <canvas ref={canvasRef} width={520} height={520} className="playerCanvas" />
          <div className="playerControls">
            <div className="playerRow">
              <button className="btn" type="button" onClick={() => setPlaying((v) => !v)} disabled={steps.length <= 1}>
                {playing ? "暂停" : "播放"}
              </button>
              <button className="btn" type="button" onClick={() => { setPlaying(false); setT(0); }} disabled={steps.length <= 1}>
                首帧
              </button>
              <button className="btn" type="button" onClick={() => setT((x) => clamp(x - 1, 0, Math.max(0, steps.length - 1)))} disabled={steps.length <= 1}>
                上一帧
              </button>
              <button className="btn" type="button" onClick={() => setT((x) => clamp(x + 1, 0, Math.max(0, steps.length - 1)))} disabled={steps.length <= 1}>
                下一帧
              </button>
            </div>
            <div className="playerRow">
              <input
                className="playerRange"
                type="range"
                min={0}
                max={Math.max(0, steps.length - 1)}
                value={t}
                onChange={(e) => { setPlaying(false); setT(Number(e.target.value)); }}
                disabled={steps.length <= 1}
              />
              <div className="mono">{t + 1}/{Math.max(1, steps.length)}</div>
            </div>
            <div className="playerRow">
              <label className="subtle">倍速</label>
              <select className="btn" value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
                <option value={0.5}>0.5x</option>
                <option value={1}>1x</option>
                <option value={2}>2x</option>
                <option value={4}>4x</option>
                <option value={6}>6x</option>
                <option value={8}>8x</option>
              </select>
              <span className="subtle mono">size: {size ?? "--"}</span>
            </div>
          </div>
        </div>
        <div className="modal__actions">
          <button className="btn" type="button" onClick={onClose}>关闭</button>
        </div>
      </div>
    </div>
  );
}
