const THEME = {
  bg: "#0b1220",
  board: "#0f1a2b",
  grid: "rgba(226, 232, 240, 0.10)",
  food: "#f59e0b",
  head: "#06b6d4",
  tail: "#8b5cf6",
  body: "rgba(99, 102, 241, 0.22)",
  stroke: "rgba(255,255,255,0.75)",
};

function clearCanvas(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = THEME.bg;
  ctx.fillRect(0, 0, w, h);
}

function calcBoard(size, w, h) {
  const cell = Math.floor(Math.min(w, h) / size);
  const ox = Math.floor((w - cell * size) / 2);
  const oy = Math.floor((h - cell * size) / 2);
  return { cell, ox, oy };
}

function cellCenter(b, p) {
  const x = b.ox + p[0] * b.cell + b.cell / 2;
  const y = b.oy + p[1] * b.cell + b.cell / 2;
  return { x, y };
}

function drawBoard(ctx, size, b) {
  ctx.fillStyle = THEME.board;
  ctx.fillRect(b.ox, b.oy, b.cell * size, b.cell * size);
}

function drawGrid(ctx, size, b) {
  if (b.cell < 6) return;
  ctx.strokeStyle = THEME.grid;
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
  if (!Array.isArray(food) || food.length !== 2) return;
  const c = cellCenter(b, food);
  const r = Math.max(3, Math.floor(b.cell * 0.34));
  ctx.fillStyle = THEME.food;
  ctx.beginPath();
  ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = THEME.stroke;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(c.x, c.y, Math.max(2, r - 1.5), 0, Math.PI * 2);
  ctx.stroke();
}

function drawBodyBlocks(ctx, b, snake) {
  if (!Array.isArray(snake)) return;
  ctx.globalAlpha = 1;
  ctx.fillStyle = THEME.body;
  for (const p of snake) {
    if (!Array.isArray(p) || p.length !== 2) continue;
    ctx.fillRect(b.ox + p[0] * b.cell, b.oy + p[1] * b.cell, b.cell, b.cell);
  }
}

function snakeGradient(ctx, a, z) {
  const grad = ctx.createLinearGradient(a.x, a.y, z.x, z.y);
  grad.addColorStop(0, THEME.head);
  grad.addColorStop(1, THEME.tail);
  return grad;
}

function drawSnakePath(ctx, b, snake) {
  if (!Array.isArray(snake) || snake.length < 2) return;
  const a = cellCenter(b, snake[0]);
  const z = cellCenter(b, snake[snake.length - 1]);
  ctx.strokeStyle = snakeGradient(ctx, a, z);
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

function drawSnakeHead(ctx, b, head) {
  if (!Array.isArray(head) || head.length !== 2) return;
  const c = cellCenter(b, head);
  const r = Math.max(3, Math.floor(b.cell * 0.36));
  ctx.fillStyle = THEME.head;
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
  if (!Array.isArray(snake) || snake.length === 0) return;
  drawBodyBlocks(ctx, b, snake);
  drawSnakePath(ctx, b, snake);
  drawSnakeHead(ctx, b, snake[0]);
}

export function drawRolloutFrame({ canvas, size, step, showGrid }) {
  if (!canvas || !step || !Number.isFinite(size) || size <= 0) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  clearCanvas(ctx, w, h);
  const b = calcBoard(size, w, h);
  drawBoard(ctx, size, b);
  if (showGrid) drawGrid(ctx, size, b);
  drawFood(ctx, b, step?.food);
  drawSnake(ctx, b, step?.snake);
}

