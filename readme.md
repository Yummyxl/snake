# chichi

贪吃蛇训练与离线回放的单机系统：后端（FastAPI）负责阶段管理、训练进程编排、数据落盘；前端（React + Vite）提供训练控制台、指标与评估回放播放。

- PRD：`prd/prd_full.md`
- 开发规范：`AGENTS.md`（通用）、`backend/AGENTS.md`（后端）、`frontend/AGENTS.md`（前端/CSS）

---

## 业务能力（你能用它做什么）

### 训练管理（Stage / Phase）

- **Stage 管理**：按 `stage_id` 管理一组训练数据与状态（默认 10/20/30）。
- **两阶段训练**：
  - `BC`（Behavior Cloning）：从“教师生成的轨迹”采样监督数据，训练策略网络。
  - `PPO`（Proximal Policy Optimization）：在 Gym 环境中强化学习训练。
- **控制操作**：开始 / 停止 / 恢复 / 完成 / 初始化（reset）。
- **课程继承**：Stage 20/30 的 BC 启动前，会校验前置 Stage 的 `PPO best checkpoint` 可用，并将其作为权重继承来源（见“权重继承”）。

### 指标与评估

- **指标**：每个训练 episode 追加写入 `metrics/episodes.jsonl`（BC: `bc_loss`；PPO: `ppo_loss`、`reward_mean`）。
- **评估回放（Eval Rollouts）**：
  - 每个训练 round 结束会生成 eval rollouts，并写入索引（`evals/index.json`）。
  - `best` / `latest` 两套视图：`best` 由 `coverage` 优先、`step_count` 次之挑选。
- **回放播放**：前端按 `rollout_id + stage_id + phase + source` 拉取轨迹 JSON 并在画布中渲染蛇身、食物与网格。

---

## 核心概念（术语对齐）

- **Stage**：一个训练关卡，当前实现用 `stage_id` 同时表达棋盘尺寸（`size = stage_id`），数据落盘在 `datas/stages/<stage_id>/`。
- **Phase**：`bc` 或 `ppo`（每个 Stage 各自拥有两套目录与状态）。
- **Episode**：
  - BC：一次“监督训练 episode”（由采样+若干次梯度更新构成），对应 `state.json: bc_episode`。
  - PPO：一次 `model.learn(...)` 调用结束后记为一个 episode，对应 `state.json: ppo_episode`。
- **Round（实现上的一个循环）**：
  - BC：`采集 train_rollouts → 训练 N 个 episode → eval + checkpoint`。
  - PPO：`训练 N 个 episode → eval + checkpoint`。
- **Rollout（轨迹）**：
  - `train_rollouts`：BC 训练使用的“教师轨迹”（本项目用启发式遍历生成）。
  - `eval`：用当前模型 rollout 生成的评估轨迹（用于比较与回放）。
  - `manual`：预留目录与索引（读取 API 支持；生成流程当前未提供独立 API）。
- **Checkpoint**：权重快照（`checkpoints/latest/*.pt` 与 `checkpoints/best/*.pt`）。

---

## 指标定义（如何理解 coverage / reward）

- `length_max`：一次 rollout 中达到的最大蛇身长度。
- `coverage`/`coverage_max`：用 `length_max / (size^2)` 归一化后的覆盖率（0~1）。
- `reward_total`：
  - 仅在 PPO eval 中统计（`include_reward=True`），来自 `backend/app/sim/snake_env.py` 的 reward 累加。
  - BC eval 虽然也会 rollout，但默认不记录 reward（`reward_total=0.0`），主要关注 coverage。

---

## 环境要求

- 后端：`uv`（并遵守项目约束：涉及 Python 后端的运行命令必须使用 `uv` 包装），Python 版本要求见 `backend/pyproject.toml`（`>=3.11`）。
- 前端：Node.js（建议 18+）与 `npm`。
- 训练硬件：支持 CPU 运行；如 macOS 且 PyTorch 支持 MPS，会自动使用 MPS（见 `backend/app/ml/checkpoints.py: pick_device()`）。

---

## 技术架构（系统如何工作）

### 组件划分

```
React Console (frontend)
  └── HTTP fetch
      └── FastAPI (backend/app/main.py)
            ├── services/           # 业务编排（Stage 状态机 / 进程管理 / 聚合输出）
            ├── external/           # 子进程训练启动/探测/停止（uv run python -m ...）
            ├── workers/            # BC/PPO 训练主循环 + eval + 指标写入 + checkpoint
            ├── sim/                # 贪吃蛇环境（eval env / gym env / teacher rollout 生成）
            └── data/               # JSON/JSONL 落盘与索引读取
```

### 为什么训练要用“子进程 worker”

- 训练会长期占用 CPU/GPU，且需要随时停止；将训练放入独立进程可以：
  - 避免阻塞 API 服务（FastAPI 仍可响应 UI 轮询）。
  - 用 PID / runtime 心跳做可观测性与“僵尸进程修复”。
  - 通过 `SIGTERM` 实现可控停止，并在停止时落盘 eval + checkpoint（便于回放与复现）。

### 进程与运行时文件（.run）

在 `DATAS_DIR/.run/` 下会出现两类运行文件：

- **脚本级 pid**：`backend.pid`、`frontend.pid`（`./scripts/start` 写入）。
- **训练 worker pid/runtime**：
  - PID：`bc_<stage>.pid`、`ppo_<stage>.pid`
  - Runtime：`bc_<stage>.runtime.json`、`ppo_<stage>.runtime.json`（写入 `status/run_id/heartbeat_at_ms/...`）

API 会在 `GET /api/stages/{id}` 的 `probe` 字段中暴露精简后的 runtime，用于前端展示和诊断。

---

## 训练流程（深入）

### BC（Behavior Cloning）

1. **采集训练轨迹（train_rollouts）**
   - 使用 `backend/app/sim/snake_rollout_gen.py` 的启发式遍历生成轨迹。
   - 目标是收集 `BC_TRAIN_ROLLOUT_COUNT` 条“达标”的轨迹：以 `coverage_max >= BC_TRAIN_MIN_ROLLOUT_COVERAGE` 判定。
   - 产物：
     - `datas/stages/<id>/bc/train_rollouts/rollouts/rollout_<collect_id>-<attempt>.json`
     - `datas/stages/<id>/bc/train_rollouts/summary.json`（汇总 coverage/step_count/accepted/rejected）

2. **监督训练（N 个 episode）**
   - 每个 episode 内会做若干次 `cross_entropy(logits, action_id)` 更新。
   - 输入编码：`backend/app/ml/encoding.py` 将网格状态编码为张量；动作空间固定为 `["L","S","R"]`（相对转向）。
   - 指标追加：`datas/stages/<id>/bc/metrics/episodes.jsonl`

3. **评估 + 保存 checkpoint（在 round 结束或 stop 时触发）**
   - eval 生成使用 `backend/app/sim/snake_env.py` 环境；动作选择为 `argmax(logits)`。
   - eval 的落盘与索引写入由 `backend/app/workers/worker_eval_rollouts.py` 完成，写到：
     - `datas/stages/<id>/bc/evals/latest/eval_<eval_id>/summary.json`
     - `datas/stages/<id>/bc/evals/latest/eval_<eval_id>/rollouts/rollout_<rollout_id>.json`
     - `datas/stages/<id>/bc/evals/index.json`（`best/latest` 列表）
   - checkpoint 写到：
     - `datas/stages/<id>/bc/checkpoints/latest/bc_latest_<eval_id>.pt`
     - `datas/stages/<id>/bc/checkpoints/best/bc_best_<eval_id>.pt`（若成为 best）
     - `datas/stages/<id>/bc/checkpoints/index.json`

补充：BC 的 eval **不是每个 episode 都生成**，而是在“round 结束”统一生成一次；默认 `BC_EPISODES_PER_TRAIN=5`，所以通常要到 episode=5 才会看到第一轮 eval（或你点击“停止”触发一次 stop-eval）。

### PPO（Stable-Baselines3 PPO）

1. **Gym 环境训练**
   - 环境：`backend/app/sim/snake_gym_env.py`（Gymnasium wrapper）。
   - 算法：SB3 `PPO`（见 `backend/app/workers/ppo_worker_runner.py`）。
   - 每个 episode 结束会记录 `ppo_loss`、`reward_mean` 等，写入 `datas/stages/<id>/ppo/metrics/episodes.jsonl`。

2. **评估 + checkpoint**
   - eval 同样生成 rollout 文件并写入 `datas/stages/<id>/ppo/evals/...`，区别是 `include_reward=True`，汇总里会含 `reward_total`。
   - PPO checkpoint 存储会把 SB3 policy 导出到统一的 `cnn_vit_v1` state（便于与 BC 共享权重结构）。

### 权重继承（课程学习的落地方式）

- Stage 20/30 启动 BC：继承前置 Stage 的 `PPO best checkpoint`（模式 `inherit_prev_ppo_best`）。
- 启动 PPO：继承当前 Stage 的 `BC best checkpoint`（模式 `inherit_bc_best`）。
- 恢复（resume）：从当前 Stage 对应 phase 的 `latest checkpoint` 加载权重（模式 `resume_weights_only`）。

上述模式会写入 runtime 的 `init_plan`，便于定位“为什么这次从哪个 checkpoint 起步”。

---

## 状态机与约束（为什么按钮会置灰）

Stage 的核心状态在 `datas/stages/<id>/state.json`：

- `bc_status / ppo_status`：`not_started | running | paused | completed`
- `bc_episode / ppo_episode`：episode 计数
- `last_eval / last_eval_coverage`：最近一次 eval 摘要（用于 Stage 列表的“最新覆盖率”）

约束规则（来自 `backend/app/services/stages_service.py`）：

- **开始 BC**：仅当 `BC/PPO 都未开始`；Stage 20/30 还要求前置 Stage `BC+PPO` 已 completed，且前置 Stage 存在 `PPO best checkpoint`。
- **开始 PPO**：要求 `BC=completed` 且存在 `BC best checkpoint`，并且 `PPO=not_started`。
- **恢复 BC**：要求 `BC=paused` 且 `PPO != running`。
- **恢复 PPO**：要求 `PPO=paused` 且 `BC=completed` 且 `BC != running`。
- **停止**：优先停止当前 running 的 phase；若发现无进程但状态仍是 running，会“修复状态”为 paused。
- **完成（complete）**：仅允许 `paused → completed`，且要求对应 phase 的 best checkpoint 文件真实存在。
- **初始化（reset）**：训练中不可 reset；已完全完成（BC+PPO 都 completed）不可 reset；若 PPO 已开始则只 reset PPO 阶段，否则 reset BC 阶段，否则 reset 整个 Stage。

---

## 数据落盘与格式（如何复现与对接）

### Stage 目录结构

```
datas/stages/<stage_id>/
├── stage.json                 # Stage 元信息（stage_id/size）
├── state.json                 # Stage 状态机
├── bc/
│   ├── metrics/episodes.jsonl
│   ├── train_rollouts/
│   │   ├── index.json
│   │   ├── summary.json
│   │   └── rollouts/rollout_*.json
│   ├── evals/
│   │   ├── index.json         # {"best":[...],"latest":[...]}
│   │   ├── latest/eval_<id>/
│   │   │   ├── summary.json
│   │   │   └── rollouts/rollout_<rollout_id>.json
│   │   └── best/eval_<id>/... # 仅保留当前 best 的那一轮
│   ├── checkpoints/
│   │   ├── index.json         # {"best":[...],"latest":[...]}
│   │   ├── latest/*.pt
│   │   └── best/*.pt
│   └── manual/                # 预留：手动生成的回放（索引与目录已就绪）
└── ppo/                        # 同 bc（目录结构一致）
```

### Rollout 文件（eval / train）核心字段

`GET /api/rollouts/{rollout_id}` 返回的 `rollout` 结构统一为：

- `meta`
  - `phase`: `bc|ppo`
  - `kind`: `train` 或 `eval`
  - `coord`: `xy`（坐标为 `[x,y]`）
  - `action_space`: `relative_lsr`，`action_map`: `["L","S","R"]`
  - `step_state`: 固定 `pre_action`（后端会强校验，不满足将拒绝返回）
- `summary`
  - `coverage/coverage_max`、`length_max`、`steps`、（PPO eval 可能含）`reward_total`
- `steps[]`：每步快照（核心字段：`snake`、`food`、`dir`、`action`、`done`、`snake_next`、`food_next`、`dir_next`）

---

## 快速开始（一键）

在项目根目录：

```bash
./scripts/start
```

启动后：

- 前端（本机）：`http://127.0.0.1:5173`
- 后端健康检查（本机）：`http://127.0.0.1:8000/api/health`
- 局域网访问：用启动脚本输出的 `lan:` 链接（`http://<本机IP>:5173`）
- 日志：`datas/logs/backend.log`、`datas/logs/frontend.log`、`datas/logs/bc_<stage>.log`、`datas/logs/ppo_<stage>.log`

停止服务：

```bash
./scripts/stop
```

---

## 手动启动（调试用）

### 后端（必须使用 uv）

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 前端

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

---

## 配置（环境变量）

### 基础运行参数（脚本/服务）

`./scripts/start` 支持用环境变量覆盖默认地址/端口：

- `BACKEND_HOST`（默认 `0.0.0.0`）
- `BACKEND_PORT`（默认 `8000`）
- `FRONTEND_HOST`（默认 `0.0.0.0`）
- `FRONTEND_PORT`（默认 `5173`）

`./scripts/start` 默认会对外监听（`0.0.0.0`），其他机器通过 `http://<本机IP>:<端口>` 访问即可；如果你当前只监听在 `127.0.0.1`，可显式改为：

```bash
BACKEND_HOST=0.0.0.0 FRONTEND_HOST=0.0.0.0 ./scripts/start
```

并用 `http://<本机IP>:<端口>/api/health` 访问（`127.0.0.1` 只在本机可用）；若仍无法访问，再检查系统防火墙/云安全组/路由器端口转发是否放行对应端口。

示例：

```bash
BACKEND_PORT=8001 FRONTEND_PORT=5174 ./scripts/start
```

如需仅本机可访问：

```bash
BACKEND_HOST=127.0.0.1 FRONTEND_HOST=127.0.0.1 ./scripts/start
```

后端通用参数（见 `backend/app/config.py`）：

- `DATAS_DIR`：数据目录（默认 `<repo>/datas`）
- `CHICHI_API_BASE`：后端对外地址（默认 `http://127.0.0.1:8000`，用于 worker runtime/链接）
- `CHICHI_HEALTH_URL`：worker 监听健康检查的 URL（默认 `http://127.0.0.1:8000/api/health`）
- `CHICHI_TORCH_DEVICE`：强制训练设备（`cpu|mps|cuda`；默认自动选择）
- `UV_CACHE_DIR`：uv 缓存目录（默认 `<repo>/.uv-cache`）
- `WORKER_ACTION`/`BC_ACTION`：训练动作（`start|resume`，默认 `start`）

前端 API Base：

- 默认使用同源 `/api`（通过 Vite dev server 代理转发到后端）。
- 如需指定其他后端地址，可设置 `VITE_API_BASE=http://<后端IP>:8000`

### BC 训练参数（`backend/app/config.py: bc_worker_cfg()`）

- `BC_EPISODES_PER_TRAIN`（默认 `5`）：每个 round 的训练 episode 数（决定 eval 的出现频率）。
- `BC_LR`（默认 `3e-4`）
- `BC_BATCH_SIZE`（默认 `1024`）
- `BC_TRAIN_ROLLOUT_COUNT`（默认 `50`）：每轮采集达标 teacher rollout 条数。
- `BC_TRAIN_MIN_ROLLOUT_COVERAGE`（默认 `0.70`）
- `BC_TRAIN_ROLLOUT_MAX_STEPS`（默认 `0`，代表用默认 `size^2 * 400`）
- `BC_TRAIN_MAX_ATTEMPTS`（默认 `100000`）
- `BC_EVAL_MAX_STEPS`（默认 `0`，代表用默认 `size^2 * 40`）
- `LATEST_KEEP`（默认 `10`）：latest checkpoints 保留数量。
- `METRICS_KEEP`（默认 `200`）：metrics 文件保留最近 N 行（防止无限增长）。

### PPO 训练参数（`backend/app/config.py: ppo_worker_cfg()`）

- `PPO_EPISODES_PER_TRAIN`（默认 `1`）
- `PPO_LR`（默认 `2.5e-4`）
- `PPO_GAMMA`（默认 `0.99`）
- `PPO_GAE_LAMBDA`（默认 `0.95`）
- `PPO_CLIP`（默认 `0.2`）
- `PPO_EPOCHS`（默认 `1`）
- `PPO_MINIBATCH_SIZE`（默认 `512`）
- `PPO_VF_COEF`（默认 `0.5`）
- `PPO_ENT_COEF`（默认 `0.01`）
- `PPO_MAX_GRAD_NORM`（默认 `0.5`）
- `PPO_ROLLOUT_STEPS`（默认 `20 * 100 * 6`）
- `PPO_ROLLOUT_MAX_STEPS`（默认 `0`，代表用默认 `size^2 * 8`）
- `PPO_EVAL_MAX_STEPS`（默认 `0`，代表用默认 `size^2 * 40`）
- `LATEST_KEEP` / `METRICS_KEEP`：同上

---

## 常用脚本

- 造演示数据（清空并重建 stages 骨架）：`./scripts/mock_data`（输出到 `datas/stages/`）
- 一键启动：`./scripts/start`
- 一键停止：`./scripts/stop`

---

## API 参考（对接/调试）

- `GET /api/health`：健康检查
- `GET /api/stages`：Stage 列表（精简字段）
- `GET /api/stages/{stage_id}`：Stage 详情（含 metrics / eval_rollouts / probe）
- `GET /api/stages/{stage_id}/reset`：初始化（按规则 reset 整个 Stage 或某一 phase）
- `GET /api/stages/{stage_id}/bc/start|resume|stop|complete`
- `GET /api/stages/{stage_id}/ppo/start|resume|stop|complete`
- `GET /api/rollouts/{rollout_id}?stage_id=...&phase=...&source=...`：读取回放（`source=eval|manual`）

---

## 前端说明（页面与交互）

- 路由：
  - `/`：Stage 列表（每秒轮询，展示 running hint）
  - `/stages/:id`：Stage 详情（按钮受状态机约束）
- 回放播放器：`frontend/src/components/rolloutPlayer/*`，按 step 渲染画布（蛇头/身体/食物/网格），支持播放/暂停与进度控制。

---

## 常见问题（排障）

- **BC 已跑到 Episode 3 但没有 eval**：正常；BC eval 在 round 结束才生成（默认 `BC_EPISODES_PER_TRAIN=5`），或点击“停止”触发一次 stop-eval。
- **端口被占用**：先运行 `./scripts/stop`，或修改 `BACKEND_PORT/FRONTEND_PORT` 后重试。
- **Stage 显示 running 但训练进程不在**：`GET /api/stages/{id}` 的 `probe.stale_running=true` 表示状态与进程不一致；可点击“停止”让后端修复状态为 paused，或执行 reset。
- **依赖安装慢/重复下载**：设置 `UV_CACHE_DIR` 到稳定磁盘路径；默认使用 `<repo>/.uv-cache/`。

---

## 已知限制 / 未实现

- 前端“Checkpoint 列表”入口目前仅占位（UI 会提示“暂未实现”）。
- `manual rollouts` 目录与读取 API 已具备，但缺少“生成 manual rollouts”的独立 API。
