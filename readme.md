# chichi

贪吃蛇训练与离线回放的单机系统：后端（FastAPI）负责训练/阶段管理与数据落盘，前端（React + Vite）提供控制台与回放 UI。

- PRD：`prd/prd_full.md`
- 开发规范：`AGENTS.md`（通用）、`backend/AGENTS.md`（后端）、`frontend/AGENTS.md`（前端/CSS）

## 目录结构

```
.
├── backend/           # FastAPI + 训练/worker
├── frontend/          # React 控制台（Vite）
├── datas/             # 本地数据（stage/rollout/checkpoint/logs）
├── scripts/           # 一键启动/停止/造数据
└── prd/               # 产品与研发说明
```

## 环境要求

- `uv`（后端依赖与运行必须使用 `uv` 包装）
- Node.js（建议 18+，用于 `npm`）

## 快速开始（一键）

在项目根目录：

```bash
./scripts/start
```

启动后：

- 前端：`http://127.0.0.1:5173`
- 后端健康检查：`http://127.0.0.1:8000/api/health`
- 日志：`datas/logs/backend.log`、`datas/logs/frontend.log`

停止服务：

```bash
./scripts/stop
```

## 手动启动（调试用）

### 后端

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 前端

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

## 配置

### 端口与地址

`./scripts/start` 支持用环境变量覆盖默认地址/端口：

- `BACKEND_HOST`（默认 `127.0.0.1`）
- `BACKEND_PORT`（默认 `8000`）
- `FRONTEND_HOST`（默认 `127.0.0.1`）
- `FRONTEND_PORT`（默认 `5173`）

示例：

```bash
BACKEND_PORT=8001 FRONTEND_PORT=5174 ./scripts/start
```

### 前端 API Base

前端默认请求 `http://127.0.0.1:8000`，可通过 `frontend/.env.local` 覆盖：

```bash
VITE_API_BASE=http://127.0.0.1:8000
```

### 数据目录

默认数据目录为 `datas/`，可通过 `DATAS_DIR` 覆盖（后端与 worker 共用）：

```bash
DATAS_DIR=/abs/path/to/datas ./scripts/start
```

## 常用脚本

- 造演示数据：`./scripts/mock_data`（输出到 `datas/stages/`）
- 一键启动：`./scripts/start`
- 一键停止：`./scripts/stop`

## API（最小集合）

- `GET /api/health`
- `GET /api/stages`
- `GET /api/stages/{stage_id}`
- `GET /api/stages/{stage_id}/reset`
- `GET /api/stages/{stage_id}/bc/start|resume|stop|complete`
- `GET /api/stages/{stage_id}/ppo/start|resume|stop|complete`
- `GET /api/rollouts/{rollout_id}?stage_id=...&phase=...&source=...`

## 常见问题

- 端口被占用：先运行 `./scripts/stop`，或修改 `BACKEND_PORT/FRONTEND_PORT` 后重试。
- 依赖安装慢：确保网络可用；后端依赖缓存目录由 `UV_CACHE_DIR` 控制（默认 `.uv-cache/`）。
