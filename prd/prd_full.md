# 贪吃蛇训练与播放系统 PRD（评审版）

> 基于 `prd.md` 扩写，补全可研发细节。若信息不足处已标记“假设/待确认”。

## 1. 背景与目标（业务问题、成功指标）
### 1.1 背景
- 现有训练与评估过程缺少统一管理与可视化，难以复现与对比。
- 训练数据、checkpoint、rollout 分散在本地文件中，回放与分析成本高。
- 需要一个单机可用的训练+回放系统，支持课程学习与阶段切换。

### 1.2 业务目标
- 提供可控的训练管理流程（BC/PPO），支持停止/恢复/完成/初始化。
- 支持 eval/manual rollout 离线回放，并展示关键指标。
- 支持 Stage 间继承，形成课程学习流程。

### 1.3 成功指标（必须可度量）
- 训练管理闭环：开始/停止/恢复/完成/初始化操作成功率 100%。
- 回放成功率：eval/manual rollout 回放成功率 100%。
- 指标完整性：BC/PPO 每个 episode 必须写入指标 JSONL，缺失率 0%。
- 单机性能（假设/待确认：M2 Pro 或等效设备）：Stage 30 启动 PPO 后 5 分钟内产生首个 eval rollout。

## 2. 范围与非范围
### 2.1 范围
- 前端 React 管理控制台。
- 后端 Python（可 FastAPI），本地文件存储。
- 训练流程：BC + PPO。
- 离线回放 eval/manual rollout。
- 单机运行，使用 uv 管理环境。

### 2.2 非范围
- 分布式训练与云端训练。
- 云端存储与多设备同步。
- 实时直播训练画面。
- 训练算法创新（仅实现指定流程与结构）。
- 多用户/权限体系（明确不支持）。

## 3. 用户画像与使用场景
### 3.1 用户画像
- 算法工程师：关注训练过程、指标曲线、checkpoint 选择。
- 研究员/学生：需要复现训练与评估回放。
- QA/开发：验证流程完整性与稳定性。

### 3.2 使用场景
- 启动 BC 训练并观察指标，决定是否切换到 PPO。
- 选择 checkpoint 生成 10 条 rollouts 并离线回放。
- Stage 20/30 继承低阶 Stage checkpoint 进行课程学习。

## 4. 用户故事与关键流程（文字流程 + 关键页面/步骤说明）
### 4.1 用户故事
- 作为算法工程师，我必须能在 Stage 10 启动 BC 训练，并在覆盖率达标后切换 PPO。
- 作为研究员，我应当能选择任意 checkpoint 生成 10 条 rollouts 并回放。
- 作为 QA，我必须验证 Stage 状态切换与文件生成符合规则。

### 4.2 关键流程（文字流程）
1) 进入 Stage 列表页 → 查看 Stage 状态
2) 进入 Stage 详情 → 若“开始 BC”可用（前置 Stage 已完成）则点击开始 → 若未初始化则自动完成初始化 → BC 训练启动
3) 训练中每秒刷新 → 观察指标/覆盖率
4) 点击“停止” → 保存 checkpoint + eval rollout，并将当前阶段置为 paused
5) 点击“完成（BC 阶段）” → BC 阶段标记为 completed
6) 若“开始 PPO”可用（BC 已完成且存在 BC best checkpoint）→ 点击“开始 PPO” → PPO 训练 → 生成 eval rollout
7) 点击 rollout 播放 → 查看回放与指标摘要

### 4.3 端到端例子（必须）
- 用户进入 Stage 列表 → 进入 Stage 10 详情 → 点击“开始 BC”（若可用且未开始则自动初始化）→ 指标曲线每秒刷新 → 覆盖率达到 0.7 后点击“停止” → 系统保存最新 checkpoint + eval rollout → 点击“完成（BC 阶段）” → 点击“开始 PPO”（需 BC 已完成且存在 BC best checkpoint）→ PPO 训练 → 生成 eval rollout → 用户点击播放按钮 → 在播放组件查看覆盖率/步数/轨迹。

## 5. 功能需求（模块 → 子功能 → 规则/边界 → 交互）

### 5.1 全局框架与导航
- 子功能：页面导航、全局状态提示、错误提示。
- 规则/边界：
  - 必须提供“Stage 列表”入口。
  - 应当提供“Checkpoint 列表”入口，入口内提供 Stage 选择器，默认显示当前 Stage。
  - 训练中必须有明显的全局“正在训练”提示。
- 交互：
  - 发生全局错误（如后端不可用）时，必须在页面顶部显示错误条并提供“重试”。

### 5.2 Stage 列表页
- 子功能：展示 Stage 列表、状态、关键指标、进入详情。
- 规则/边界：
  - 默认显示 Stage 10/20/30；若 `index.json` 中存在其他 Stage，应当追加展示。
- 列表字段必须包含：Stage ID/尺寸、状态、当前 Phase、最近 eval 覆盖率、最后更新时间。
  - 最近 eval 覆盖率取 BC/PPO 中最新一次 eval 结果；若 PPO 未开始则取 BC；若两者均无 eval 则显示 “--”。
  - 最后更新时间取最近一次 eval 的时间戳；无 eval 时取 Stage 最后状态变更时间。
  - 必须每 1s 轮询刷新数据；连续 3 次失败应提示“后端不可用”。
- 状态枚举：
  - BC/PPO 阶段状态（后端）：not_started/running/paused/completed；前端展示：未开始/进行中/已暂停/已完成。
  - Stage 展示状态（聚合计算，无单独字段）：not_started/running/completed；前端展示：未开始/进行中/已完成。
  - Stage 状态聚合规则见 7.2。
  - 当 BC/PPO 均为 not_started 时，Stage 展示为“未开始”（可在 UI 辅助文案中提示“未训练”）。
  - 列表页应展示 BC/PPO 的阶段状态（用于解释 Stage=进行中但可能处于 paused 或中间态）。
- 交互：
  - 点击 Stage 行或“详情”按钮进入 Stage 详情页。
  - 鼠标悬停状态字段显示状态解释（如“已暂停：可恢复或删除”）。
  - 列表为空时应提示“暂无 Stage，请初始化”。

### 5.3 Stage 详情页
- 子功能：展示 Stage 状态、训练控制、指标图、eval rollout 列表。
- 规则/边界：
  - 详情必须展示三张状态卡：Stage 状态、BC 状态、PPO 状态。
  - Stage 状态不单独存储，按 7.2 聚合规则展示。
  - 必须展示 BC/PPO 的累计 episode 数量与最近一次 eval 覆盖率。
- 顶部控制区必须包含 Start Slot（开始 BC/开始 PPO 二选一）+ 操作按钮（停止/恢复/完成/初始化）。
- “停止/恢复/完成”只针对“当前可操作阶段”展示：仅当该阶段状态为 running/paused 时展示；当 BC 已完成且 PPO 未开始时，不应再展示 BC 的停止/恢复/完成按钮。
  - Start Slot 同一时刻只展示一个按钮：
    - 开始 BC：仅当 BC=not_started 且 PPO=not_started 时展示；可用性受“前置 Stage 完成”限制（见 7.3）。
    - 开始 PPO：仅当 BC=completed 且 PPO=not_started 时展示；且必须存在 BC best checkpoint 才可用（见 7.3）。
  - 操作按钮（停止/恢复/完成）必须按“当前阶段”展示与生效：
    - 当前阶段定义：若 PPO!=not_started 则为 PPO，否则若 BC!=not_started 则为 BC，否则为空。
    - 当前阶段=BC：仅展示“停止 BC / 恢复 BC / 完成 BC”。
    - 当前阶段=PPO：仅展示“停止 PPO / 恢复 PPO / 完成 PPO”。
    - 其他阶段的按钮不展示（避免误操作）；开始按钮仍按 Start Slot 规则展示。
  - 指标图仅展示一个“大图”（单区域）：
    - 若 BC 或 PPO 正在训练（status=running），展示对应阶段指标。
    - 若无正在训练阶段，则展示最近一次 eval 所在阶段的指标。
    - 若无训练且无 eval，展示空图（空态文案“暂无训练指标”）。
- 指标序列由 `GET /api/stages/{stage_id}` 的 `metrics` 字段提供（本期不单独拆 metrics 接口）。
  - eval rollout 列表必须 `source=eval`，由 best 与 latest 组成，且每条记录带 `is_best` 字段。
  - 每个 phase 的 best 最多 BEST_KEEP（默认 1）条；latest 最多 LATEST_KEEP（默认 10）条；Stage 详情页合并展示 BC/PPO 两个 phase。
- 交互（训练控制）：
  - “开始 BC”：仅当 BC=not_started 且 PPO=not_started 时展示；仅在“前置 Stage 已完成”时可点击；触发后若目录未初始化则自动初始化。
  - “开始 PPO”：仅当 BC=completed 且 PPO=not_started 时展示；仅在存在 BC best checkpoint 时可点击。
  - “停止”：仅当 BC 或 PPO 任一为 running 可点击；点击后将当前 running 的阶段置为 paused，并保存 checkpoint + eval rollouts + 指标。
  - “恢复”：仅当 BC 或 PPO 任一为 paused 可点击；恢复最近一次 running 的阶段并重新采集训练 rollouts。
  - “完成”：仅当 BC 或 PPO 任一为 paused 可点击；将最近一次 paused 的阶段标记为 completed（必须先停止再完成，不允许 running 直接完成）。
    - 完成 BC 额外约束：必须存在 `BC best .pt checkpoint`（用于开始 PPO）。
    - 完成 PPO 额外约束：必须存在 `PPO best .pt checkpoint`（用于后续 Stage 继承）。
  - “初始化”：按钮触发 `GET /api/stages/{stage_id}/reset`，前端必须先做状态拦截与二次确认（见 8.2）。
  - 当用户点击“初始化”但命中拦截条件（训练中/已完成）时，前端必须弹出 toast 提示“无法初始化”及原因；禁止使用浏览器原生弹窗。
- 交互（指标图）：
  - 默认展示最近 200 个 episode，不支持切换范围。
  - 鼠标悬停显示 episode 级别指标（loss、coverage 等）。
  - 指标区仅展示曲线大图，不展示额外的“latest xxx”文本行（避免出现两个同名指标，如两个 `bc_loss`）。
  - 指标不提供“按指标选择器”；但当 PPO 已开始时，指标区必须支持在 BC/PPO 两个阶段间切换查看历史指标。
  - BC 指标固定展示 `bc_loss`；PPO 指标固定展示 `ppo_loss` + `reward_mean`，且两条线必须画在同一个图中。
  - 图表应尽量放大（例如增大图高），优先保证可读性。
- 交互（rollout 列表）：
  - 列表项必须展示：rollout_id、coverage、steps、length_max、reward_total、phase、生成时间。
  - Stage 详情页 rollout 列表仅展示“当前阶段（最新阶段）”的 eval rollouts：
    - 当前阶段定义：若 PPO 已开始（PPO!=not_started）则为 PPO，否则若 BC 已开始（BC!=not_started）则为 BC，否则为空。
  - best 项必须在 UI 中有明显标识（例如“Best”标签）；列表排序：best 在最前，其余按生成时间倒序排列。
  - 点击“播放”进入 Rollout 播放组件。（可后续实现）
  - 列表为空时显示“暂无评估回放”。

### 5.4 Checkpoint 列表页
- 子功能：展示 best/latest checkpoint；生成 manual rollouts；展示手动 rollouts。
- 规则/边界：
  - 必须区分 best/latest 列表；best 仅保留 1 条。
  - 每个 checkpoint 列表项必须显示：id、phase、episode、覆盖率、生成时间。
  - 生成 manual rollouts 数量固定为 MANUAL_ROLLOUTS（默认 10），不支持配置（明确）。
  - manual rollouts 列表必须 `source=manual`。
- 交互：
  - 选择某 checkpoint → 点击“生成 rollouts” → 发起生成任务。
  - 若该 checkpoint 已存在 manual rollouts，则生成操作会覆盖旧数据，必须二次确认。
  - 生成期间按钮置灰并显示加载态；完成后自动刷新列表；失败需提示原因并允许重试。
  - 生成完成后，列表展示覆盖率/步数/长度摘要。
  - 必须支持 Stage 与 phase（BC/PPO）切换查看对应 checkpoint。

### 5.5 Rollout 播放组件
- 子功能：播放/暂停/重播/倍速/缩放、轨迹渲染、摘要信息。
- 规则/边界：
  - 播放读取接口必须带 `phase` 与 `source` 参数。
  - 必须支持 JSON 轨迹完整回放，包含 head/body/food/dir。
  - 需显示覆盖率、长度、步数摘要。
  - 地图渲染必须依据 meta 中坐标系与方向定义：coord_origin/top-left、row_dir/down、col_dir/right、dir_order；缺失时使用默认值。
  - 格子编号必须展示：按行优先（row-major）从 0 到 N^2-1；若 meta 定义了坐标系方向则以其为准。
  - 单步数据缺失时必须中断回放并提示“轨迹数据不完整”。
- 交互：
  - 控制区：播放/暂停、重播、速度（0.25x/0.5x/1x/2x/4x）。
  - 时间轴：可拖动到任意 step。
  - 画布：支持缩放（0.75x/1x/1.5x/2x）。
  - 必须显示坐标/格子编号与蛇头方向箭头。
  - 地图图例必须明确区分：蛇头/身体/食物/空格（至少 4 种颜色或标识）。
  - 当前步必须高亮蛇头与食物位置；身体段按先后顺序渲染。
  - 步进显示：在控制区显示当前 step、总步数、当前 action、reward、done 状态。
  - 到达 done=true 的 step 时自动停止，并保留最后一帧。
  - 键盘快捷键可以支持（可选）：空格播放/暂停，左右箭头步进。
  - 回放失败时必须显示错误信息并允许重试。

### 5.6 训练管理与控制
- 子功能：开始/停止/恢复/完成/初始化。
- 规则/边界：
  - 必须人为触发，不支持自动启动。
  - 暂停必须立即保存当前状态，不等待 episode 结束。
  - 恢复后必须清理训练 rollouts 并重新采集，不复用旧 batch。
  - 全局仅允许一个 Stage 处于训练中（定义：任一 Stage 的 BC/PPO 状态为 running），冲突时必须拒绝并提示。
- 交互：
  - 所有不可用按钮必须置灰，并显示原因提示。
  - 初始化必须弹确认框，明确会清空历史数据。
- 按钮效果（必须与状态机一致）：
  - 开始 BC：若未初始化则先初始化目录；启动 BC 训练并进入 rollout 采集；BC=running。
  - 开始 PPO：启动 PPO 训练并进入 rollout 采集；PPO=running。
  - 停止：立即停止训练并保存 checkpoint + eval rollouts + 指标；将正在运行的阶段置为 paused（Stage 展示状态由 7.2 聚合规则计算）。
  - 恢复：恢复最近一次运行阶段并重新采集训练 rollouts；将该阶段置为 running。
  - 完成：仅允许 paused → completed（必须先停止再完成，不允许 running 直接完成）；Stage 展示状态由 7.2 聚合规则计算。
  - 初始化（reset）：若 Stage 已进入某阶段，则仅清空“当前阶段”的历史数据并重建该阶段骨架；若两阶段均未开始，则清空全 Stage 并重建标准骨架（见 8.2）。

### 5.7 启动与继承流程
- 子功能：继承低阶 Stage checkpoint。
- 规则/边界：
  - Stage 20/30 训练时必须允许选择继承来源（best/latest/不继承）。
  - 继承参数获取必须复用 checkpoint 列表接口。
- 交互：
  - Stage 20/30 在点击“开始 BC”时弹窗选择继承来源；Stage 10 不弹窗直接开始。
  - 默认继承策略（单机版可先内置）：Stage 20/30 开始 BC 时默认继承前置 Stage 的 PPO best checkpoint（而非 BC best）；由于 BC/PPO 模型参数结构不同，必须在训练 worker 内做参数适配（缺失参数按初始化策略处理）。

### 5.8 存储目录初始化
- 子功能：目录创建、index.json 维护。
- 规则/边界：
  - 服务启动时必须自动创建 `data/` 与默认 Stage 目录。
  - `index.json` 缺失时必须按默认尺寸重建。
  - 新增 Stage 首次访问时应当触发目录创建。

### 5.9 全局硬性要求
- 单个方法/函数的非空非注释行数不得超过 30 行，超出必须拆分。
- 交付时必须提供检查脚本，作为 CI/本地校验的一部分。
- 必须提供一键生成演示数据脚本：`scripts/mock_data`，用于清空并重建 `datas/stages/` 下的 mock 数据，便于 UI/接口联调。

## 6. 训练流程与规则
### 6.1 课程学习与继承
- Stage 20/30 训练必须提供继承选择：best/latest/不继承。
- 继承范围：backbone + heads + 位置编码参数；优化器状态与 PPO 统计量必须重置。
- 继承仅允许在 Stage 启动时选择；启动后不允许更改继承来源。

### 6.2 BC 阶段训练循环
- 每轮训练流程：清理 `bc/train_rollouts/` → 采集 teacher rollout（数量=train_rollout_count，默认 10；按单条 `coverage_max >= min_rollout_coverage` 准入）→ 训练 episodes_per_train（默认 4）→ eval（固定 5 条）→ 保存 checkpoint + eval rollouts + 指标。
- rollout 最大步数：train_rollout_max_steps 与 eval_max_steps 默认均为 `2048`（可通过环境变量覆盖）。
- 停止条件：人工根据 eval 覆盖率决定停止或切换到 PPO。
- 暂停：必须立即保存状态，不等待 episode 结束。
- 恢复：episode 计数延续；恢复后必须重新采集训练 rollouts，不复用暂停前 batch。
- 停止（用户点击“停止 BC”）：
  - 后端发送停止信号后，阶段状态在 worker 落盘（eval+checkpoint）并退出后才由 running → paused（避免“UI 显示已暂停但进程仍在运行”的错觉）。
  - 停止过程中（进程仍存活）必须禁止恢复/完成。
- 训练 rollouts 采集准入：按“单个 rollout 覆盖率达标”判断，而不是平均值；仅当该 rollout 的 `coverage_max >= min_rollout_coverage` 才可进入训练 batch，不达标的 rollout 直接丢弃并继续重采样直到满足数量或达到 max_attempts。
- BC 训练 rollouts（每轮采集前清空，仅保留当前 batch）：
  - `datas/stages/{id}/bc/train_rollouts/summary.json`
  - `datas/stages/{id}/bc/train_rollouts/index.json`
  - `datas/stages/{id}/bc/train_rollouts/rollouts/rollout_{rollout_id}.json`
  - `collect_id` 命名：由 episode 生成，9 位补零（如 `000000123`）；`rollout_id = {collect_id}-{k}`（k 从 1 开始）。
  - steps 坐标：统一 `[x,y]`（x=列，y=行）；动作空间：相对动作 `L/S/R`（左转/直行/右转）；必须在每步保存 `dir` 以便解释相对动作。
  - BC 阶段不产出 reward；rollout steps 中不写 `reward` 字段（或写 null），保留给 PPO 使用。

### 6.3 PPO 阶段训练循环
- PPO 使用 stable-baselines3（SB3）实现：不落盘 `train_rollouts`（使用 SB3 rollout buffer），每轮 `learn(total_timesteps=rollout_steps)` 后 eval（固定 5 条）并保存 checkpoint+指标。
- 单环境自采样（不做多环境并行）；episode 最大步数与 eval 最大步数默认均为 `2048`（可通过环境变量覆盖）。
- 停止条件：人工根据 eval 覆盖率决定停止。
- 暂停/恢复规则同 BC。

### 6.4 Best 选择规则
- Eval best（用于回放列表的 `evals/best`）：以“单条 eval rollout”作为候选进行比较（而不是 agg summary）。
- 比较规则：coverage 高优先；coverage 相同取 steps 更少；仍相同取 created_at_ms 更新。
- Checkpoint best（用于 `checkpoints/best`）：仍以 eval summary（本次 eval 的 summary.json）作为候选进行比较，规则同上。
- best 与 latest 分别维护列表，best 仅保留 1 条，不占用 latest 名额。
- PPO 起点：必须继承同尺寸 BC 的 best checkpoint。

### 6.5 产出与保留策略
- Checkpoint：latest 保留 LATEST_KEEP（默认 10）条 + best 保留 BEST_KEEP（默认 1）条；超出时删除最旧 latest。
- Eval rollouts：latest 保留 LATEST_KEEP（默认 10）条 + best 保留 BEST_KEEP（默认 1）条；超出时删除最旧 latest。
- Manual rollouts：每次生成覆盖该 checkpoint 对应目录；manual index 仅保留 latest LATEST_KEEP（默认 10）条 + best BEST_KEEP（默认 1）条。
- 训练 rollouts（BC/PPO）：每轮采集前清空，仅保留当前 batch。
- 读写一致性（强制）：所有“高频读取”的文件（`evals/index.json`、`checkpoints/index.json`、`*.pt`、`summary.json`）写入必须使用临时文件/目录 + 原子 replace/rename，避免出现 index 指向不存在/空目录/半写文件。

### 6.6 覆盖率定义与 PPO 奖励函数
- 覆盖率定义：coverage = snake_length / (size^2)；snake_length 包含蛇头在内的当前长度。
- 目标：鼓励吃食物并最终“吃完整张地图”（snake_length == size^2），同时避免不进食的循环策略。
- PPO 奖励（每步，示意）：
  - 吃到食物：`+1`
  - 撞墙/撞身：`-10`
  - 距离 shaping：`+0.02 * clip(dist_delta, -2, 2)`（dist_delta 为「离食物曼哈顿距离」的减少量）
  - 饥饿惩罚（可配）：超过 `hunger_grace_steps` 未进食后，每步追加 `-(hunger_budget / max_steps)`（将总惩罚预算均摊到 episode）
  - 终局调整（可配，done 时）：
    - 若完成（覆盖率=1.0）：`+completion_bonus`
    - 若未完成（包含 truncated 超时/死亡）：`-terminal_incomplete_beta * (1 - coverage)`
- 说明：覆盖率与长度严格一致；覆盖率主要用于评估与 best 选择，奖励用于引导行为与避免局部最优。
- 奖励配置（环境变量，默认值与 `backend/app/config.py` 同步）：
  - `SNAKE_HUNGER_BUDGET=2.0`
  - `SNAKE_HUNGER_GRACE_STEPS=50`
  - `SNAKE_TERMINAL_INCOMPLETE_BETA=50.0`
  - `SNAKE_COMPLETION_BONUS=50.0`

### 6.7 全局常量（固定不可配）
- BEST_KEEP=1（best 列表保留数量）
- LATEST_KEEP=10（latest 列表保留数量，适用于 checkpoint/eval/manual rollouts）
- EVAL_ROLLOUTS=5（每次 eval 生成 rollout 数量）
- MANUAL_ROLLOUTS=10（手动生成 rollout 数量）

### 6.8 训练配置默认值（与 `backend/app/config.py` 同步）
- BC（环境变量 → 默认值）：
  - `BC_EPISODES_PER_TRAIN=4`
  - `BC_LR=3e-4`
  - `BC_BATCH_SIZE=2048`
  - `BC_TRAIN_ROLLOUT_COUNT=10`
  - `BC_TRAIN_MIN_ROLLOUT_COVERAGE=0.70`
  - `BC_TRAIN_ROLLOUT_MAX_STEPS=2048`
  - `BC_EVAL_MAX_STEPS=2048`
- PPO（环境变量 → 默认值）：
  - `PPO_EPISODES_PER_TRAIN=1`
  - `PPO_LR=1e-4`
  - `PPO_CLIP=0.1`
  - `PPO_EPOCHS=4`
  - `PPO_MINIBATCH_SIZE=2048`
  - `PPO_ROLLOUT_STEPS=8192`
  - `PPO_ROLLOUT_MAX_STEPS=2048`
  - `PPO_EVAL_MAX_STEPS=2048`
  - 继承与稳定性：`PPO_SHARE_FEATURES_EXTRACTOR=0`、`PPO_FREEZE_POLICY_ROUNDS=1`、`PPO_FREEZE_BN=1`、`PPO_TARGET_KL=0.02`
  - value warmup：`PPO_VALUE_WARMUP_STEPS=20000`、`PPO_VALUE_WARMUP_EPOCHS=2`、`PPO_VALUE_WARMUP_BATCH=1024`、`PPO_VALUE_WARMUP_LR=1e-4`、`PPO_VALUE_WARMUP_MAX_STEPS=5000`

## 7. 数据与模型（核心实体/字段/状态机）
### 7.1 核心实体
- Stage：stage_id、size、current_phase、last_eval_coverage。
  - Stage 状态不单独存储，通过 BC/PPO 状态聚合展示（见 7.2）。
  - last_eval_coverage 取 BC/PPO 中最新一次 eval 结果，未产生 eval 时为空。
- Phase：phase_id、type（bc/ppo）、status、episode_count。
- Checkpoint：id、stage_id、phase、episode、coverage、step_count、is_best、created_at、path。
  - episode/coverage/step_count 取该 checkpoint 对应 eval summary（与 best 选择规则一致）。
- Rollout：id、stage_id、phase、source（eval/manual）、coverage、steps、length_max、file_path。
- Metrics：stage_id、phase、episode、timestamp_ms、loss/acc/kl 等。

### 7.2 状态机
- Stage 状态不单独维护，按 BC/PPO 聚合得到：
  - not_started：BC=not_started 且 PPO=not_started
  - completed：BC=completed 且 PPO=completed
  - running：除以上两种情况外（包含 paused/running/混合）
- BC：not_started → running → paused → completed。
- PPO：not_started → running → paused → completed。
  - 后端与存储使用英文枚举；前端文案映射：not_started=未开始，running=进行中，paused=已暂停，completed=已完成（当前不在 UI 触发）。

### 7.3 状态切换触发条件
- 开始 BC：BC 由 not_started → running；仅当 BC=not_started 且 PPO=not_started；且必须满足“前置 Stage 已完成”（若存在前置 Stage）。
- 暂停：若 BC/PPO 为 running，则该阶段置为 paused。
- 恢复：恢复最近一次运行的阶段（BC 或 PPO）为 running。
- 完成：仅当 BC/PPO 为 paused 时，允许将该阶段由 paused → completed（必须先暂停再完成）。
  - 完成 BC 额外约束：必须确认训练进程已退出（不处于 stopping），且存在 `BC best .pt checkpoint`（否则无法开始 PPO）。
- 开始 PPO：仅当 BC=completed 且存在 BC best checkpoint 时，PPO 由 not_started → running。
- 删除/初始化：Stage/BC/PPO 全部重置为 not_started，episode 计数与训练数据清空。
- Stage 状态聚合规则见 7.2（不再单独存储 stage_status）。

### 7.4 核心算法伪代码（BC/PPO 训练循环）
- BC 训练循环：
```text
init_stage(stage_id)
if inherit_checkpoint_selected:
  load_checkpoint(inherit_checkpoint)  # 继承参数，重置优化器状态
if resume_from_pause:
  load_checkpoint(latest_checkpoint)
while user_not_stop:
  clear(bc/train_rollouts)             # 清理上一轮训练数据
  init_training_buffers(phase="bc")
  rollouts = collect_bc_rollouts(count=rollout_count)
  for i in range(episodes_per_train):  # eval 频率由 episodes_per_train 定义
    train_bc(rollouts)
    log_bc_metrics(episode=i)
  eval_result = eval_policy(phase="bc", rollouts=EVAL_ROLLOUTS)
  save_checkpoint(phase="bc", metrics=eval_result.summary)  # checkpoint best 以 summary 评估
  update_best_latest(phase="bc", rule=best_rule)
  save_eval_rollouts(eval_result)
```
- PPO 训练循环：
```text
init_stage(stage_id)
ensure_bc_best_checkpoint()
load_checkpoint(phase="bc", type="best")  # PPO 起点
if resume_from_pause:
  load_checkpoint(latest_checkpoint)
while user_not_stop:
  sb3_learn(total_timesteps=rollout_steps)  # SB3 on-policy rollout buffer（不落盘 train_rollouts）
  log_ppo_metrics()
  eval_result = eval_policy(phase="ppo", rollouts=EVAL_ROLLOUTS)
  save_checkpoint(phase="ppo", metrics=eval_result.summary)
  update_best_latest(phase="ppo", rule=best_rule)
  save_eval_rollouts(eval_result)
```

### 7.5 数据文件 JSON Schema（核心文件）
- `datas/index.json`（Stage 列表）
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["stages"],
  "properties": {
    "stages": {
      "type": "array",
      "items": { "type": "integer", "minimum": 2 }
    }
  }
}
```
- `datas/stages/{size}/stage.json`（Stage 元信息）

#### 训练 Rollout（当前实现：仅 BC 采集；PPO 使用 SB3 rollout buffer 不落盘）
- `datas/stages/{id}/bc/train_rollouts/rollouts/rollout_{rollout_id}.json`
  - 顶层结构：`{meta, summary, steps}`
  - `meta` 必须包含：`stage_id`、`phase`、`kind=train`、`coord=xy`、`action_space=relative_lsr`、`step_state=pre_action`、`collect_id`、`rollout_id`、`created_at_ms`、`max_steps`
  - `summary` 推荐字段：`coverage_max`、`snake_length_max`、`steps`、`terminated`、`truncated`
  - `steps[t]` 语义（关键）：记录 **执行 action 之前** 的观测 `state_t`，并保存该步采取的 `action_t`（用于 BC 监督），`done` 表示 **执行 action 之后** 是否终止。
  - `steps[t]` 必须包含：`t`、`snake`（head 在 index=0）、`food`、`dir`、`action`、`done`；`info` 可选扩展（如 `ate/collision`）
  - 为了回放更直观，推荐额外写入：`snake_next`、`food_next`、`dir_next`（执行 action 之后的状态快照；最后一步无 next 时可缺省）
- `datas/stages/{id}/{phase}/train_rollouts/summary.json`
  - 必须包含：本轮 `collect_id`、采集 target（rollout_count/min_rollout_coverage/max_steps/max_attempts）、采集结果（accepted/rejected/coverage_min/coverage_max/step_count）、rollouts 列表（id/coverage/step_count/path）
- `datas/stages/{id}/{phase}/train_rollouts/index.json`
  - 仅维护 latest：`{collect_id, episode_start, accepted, rejected, created_at_ms, path}`

#### Rollout 生成（参考实现）
- 为保证回放/调试一致性，演示/占位 rollout 的生成需满足基本贪吃蛇规则：
  - 地图边界不可穿越；移动必须为相邻格。
  - 食物必须生成在空格（不与蛇身重叠）；吃到食物蛇身增长 1。
  - 训练/评估 rollout 允许采用“路径/循环”策略生成可控覆盖率，并在安全前提下做近路：
    - 偶数 size：使用 Hamiltonian cycle（闭环），可无限前进。
    - 奇数 size：使用蛇形 path（非闭环），走到尽头则 done。
    - Hamiltonian + shortcut：以 Hamiltonian cycle 作为安全兜底；每步仅允许在 `L/S/R` 三个相对动作候选中选择“更接近 food 的近路”，且必须满足不穿越 tail 的安全约束，否则回退为沿 cycle 前进。
- `dir` 为当前朝向；动作空间使用相对动作 `L/S/R`，每步的 `action` 由 `cur_dir -> next_dir` 推导得到，保证回放与训练一致。
- Eval rollout（真实环境 + 当前 policy，自采样）：
  - Eval 必须在“真实 Snake 环境”中运行，按当前 policy 每步选 `L/S/R` 并推进环境，直到撞墙/撞身或达到 max_steps。
  - 默认 eval max_steps：`2048`；训练采集 max_steps 也为 `2048`。
  - Eval 使用确定性 seed：`seed = eval_seed(stage_id, phase, eval_id, k)`（k=1..5），确保可复现与 best 判定稳定。
  - 真实环境初始化：蛇身为直线，长度固定为 2；初始化位置与朝向为“随机采样（由 seed 决定，可复现）”，且保证 `S`（直行）第一步不撞墙/不撞身。
  - 统一要求：BC teacher rollout 生成器、Eval 真实环境、以及后续 PPO rollout 采集，蛇身初始化长度均固定为 2；初始化分布保持一致（随机但可复现）。
  - 非闭环 path（奇数 size）初始化必须保证蛇身连续且不跨端点“回绕”（避免出现首尾不相邻导致方向无法定义）。
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["stage_id", "size", "created_at_ms"],
  "properties": {
    "stage_id": { "type": "integer" },
    "size": { "type": "integer", "minimum": 2 },
    "created_at_ms": { "type": "integer" }
  }
}
```
- `datas/stages/{size}/state.json`（Stage 状态）
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["bc_status", "ppo_status", "bc_episode", "ppo_episode", "updated_at_ms"],
  "properties": {
    "bc_status": { "type": "string", "enum": ["not_started", "running", "paused", "completed"] },
    "ppo_status": { "type": "string", "enum": ["not_started", "running", "paused", "completed"] },
    "bc_episode": { "type": "integer", "minimum": 0 },
    "ppo_episode": { "type": "integer", "minimum": 0 },
    "last_eval_coverage": { "type": ["number", "null"] },
    "updated_at_ms": { "type": "integer" }
  }
}
```
- `datas/stages/{size}/{phase}/checkpoints/index.json`（checkpoint 列表）
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["best", "latest"],
  "properties": {
    "best": { "type": "array", "items": { "$ref": "#/definitions/checkpoint" }, "maxItems": 1 },
    "latest": { "type": "array", "items": { "$ref": "#/definitions/checkpoint" }, "maxItems": 10 }
  },
  "definitions": {
    "checkpoint": {
      "type": "object",
      "required": ["id", "episode", "coverage", "step_count", "created_at_ms", "path", "is_best"],
      "properties": {
        "id": { "type": "string" },
        "episode": { "type": "integer", "minimum": 0 },
        "coverage": { "type": "number", "minimum": 0, "maximum": 1 },
        "step_count": { "type": "integer", "minimum": 0 },
        "created_at_ms": { "type": "integer" },
        "path": { "type": "string" },
        "is_best": { "type": "boolean" }
      }
    }
  }
}
```
- 说明：best.maxItems=BEST_KEEP，latest.maxItems=LATEST_KEEP。
- `datas/stages/{size}/{phase}/checkpoints/latest/{checkpoint_id}.pt`（latest checkpoint 文件）
- `datas/stages/{size}/{phase}/checkpoints/best/{checkpoint_id}.pt`（best checkpoint 文件）
- `datas/stages/{size}/{phase}/metrics/episodes.jsonl`（单行指标；前端默认展示：BC=bc_loss；PPO=ppo_loss+reward_mean）
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["stage_id", "phase", "episode", "timestamp_ms"],
  "properties": {
    "stage_id": { "type": "integer" },
    "phase": { "type": "string", "enum": ["bc", "ppo"] },
    "episode": { "type": "integer", "minimum": 0 },
    "timestamp_ms": { "type": "integer" },
    "step_count": { "type": "integer", "minimum": 0 },
    "reward_mean": { "type": "number" },
    "bc_loss": { "type": "number" },
    "ppo_loss": { "type": "number" },
    "kl": { "type": "number" },
    "entropy": { "type": "number" }
  }
}
```
- `datas/stages/{size}/{phase}/evals/latest/eval_{id}/summary.json`（评估摘要，latest）
- `datas/stages/{size}/{phase}/evals/best/eval_{id}/summary.json`（评估摘要，best）
  - 说明：best 的判定以 `evals/index.json` 的 best/latest 列表为准；`summary.json` 仅存储评估结果本身，不额外标记“是否 best”。
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["coverage", "length_max", "steps", "reward_total"],
  "properties": {
    "coverage": { "type": "number", "minimum": 0, "maximum": 1 },
    "length_max": { "type": "integer", "minimum": 1 },
    "steps": { "type": "integer", "minimum": 0 },
    "reward_total": { "type": "number" }
  }
}
```
- `datas/stages/{size}/{phase}/evals/latest/eval_{id}/rollouts/rollout_{id}-{k}.json`（单条 rollout，k=1..5）
- `datas/stages/{size}/{phase}/evals/best/eval_{id}/rollouts/rollout_{id}-{k}.json`（单条 rollout：best 仅保留 1 条，k 为被选中的那条）
  - 说明：manual rollout 的目录结构与 eval 完全一致：`manual/latest/...`、`manual/best/...`，文件命名规则相同（rollout_{id}-{k}.json）。
  - 说明：evals/index.json 的每个 item.path 建议直接指向单条 rollout 文件（而非 summary.json），后端从 rollout 文件中读取 summary 字段用于列表展示。
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["meta", "summary", "steps"],
  "properties": {
    "meta": {
      "type": "object",
      "required": ["schema_version", "stage_id", "size", "phase", "kind", "coord", "action_space", "action_map", "step_state", "rollout_id", "seed", "created_at_ms", "max_steps"],
      "properties": {
        "schema_version": { "type": "integer" },
        "stage_id": { "type": "integer" },
        "size": { "type": "integer" },
        "phase": { "type": "string", "enum": ["bc", "ppo"] },
        "seed": { "type": "integer" },
        "kind": { "type": "string", "enum": ["train", "eval", "manual"] },
        "coord": { "type": "string", "enum": ["xy"] },
        "action_space": { "type": "string", "enum": ["relative_lsr"] },
        "action_map": { "type": "array", "items": { "type": "string" }, "minItems": 3, "maxItems": 3 },
        "step_state": { "type": "string", "enum": ["pre_action"] },
        "collect_id": { "type": "string" },
        "eval_id": { "type": "string" },
        "rollout_id": { "type": "string" },
        "created_at_ms": { "type": "integer" },
        "max_steps": { "type": "integer", "minimum": 1 }
      }
    },
    "summary": { "$ref": "#/definitions/summary" },
    "steps": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["t", "snake", "food", "dir", "action", "done"],
        "properties": {
          "t": { "type": "integer", "minimum": 0 },
          "snake": { "type": "array" },
          "dir": { "type": "string" },
          "food": { "type": "array", "items": { "type": "integer" }, "minItems": 2, "maxItems": 2 },
          "action": { "type": "string", "enum": ["L", "S", "R"] },
          "done": { "type": "boolean" },
          "info": { "type": "object" },
          "snake_next": { "type": "array" },
          "food_next": { "type": "array" },
          "dir_next": { "type": "string" },
          "reward": { "type": "number" }
        }
      }
    }
  },
  "definitions": {
    "summary": {
      "type": "object",
      "required": ["coverage", "coverage_max", "length_max", "steps", "reward_total"],
      "properties": {
        "coverage": { "type": "number", "minimum": 0, "maximum": 1 },
        "coverage_max": { "type": "number", "minimum": 0, "maximum": 1 },
        "length_max": { "type": "integer", "minimum": 1 },
        "steps": { "type": "integer", "minimum": 0 },
        "reward_total": { "type": "number" },
        "terminated": { "type": "boolean" },
        "truncated": { "type": "boolean" },
        "created_at_ms": { "type": "integer" }
      }
    }
  }
}
```

### 7.6 实体关系图（ERD）
```text
Stage (1) ────< Phase (2: BC/PPO)
Phase (1) ────< Checkpoint (best/latest)
Phase (1) ────< Metrics (episodes.jsonl)
Checkpoint (1) ────< Rollout (eval/manual)
Stage (1) ────< Rollout (via phase + source)
```

### 7.7 默认模型结构（CNN → ViT，跨 Stage10/20/30 同构可继承）
> 目的：BC/PPO 使用同构网络；Stage10/20/30 仅输入 size 不同，支持 strict=True 直接继承参数。

#### 7.7.1 输入编码（grid）
- 输入张量：`X ∈ R^{B×C×N×N}`，`N=size`，默认 `C=11`：
  - 空间 one-hot（8 通道）：`head`（1）、`food`（1）、`occupied`（1）、`body_order`（1）、`dir_onehot`（4，广播到 N×N）。
  - 全局标量广播（3 通道，均广播到 N×N）：
    - `time_left = clamp((max_steps - steps) / max_steps, 0, 1)`
    - `hunger = clamp(max(0, steps_since_last_eat - hunger_grace_steps) / max_steps, 0, 1)`
    - `coverage_norm = snake_length / (size^2)`
- action 空间：离散 3（相对动作 `L/S/R`）。

#### 7.7.2 CNN backbone（局部特征）
- 默认结构（推荐）：
  - `Conv3×3 (C→64)` + ResBlock×2（64）
  - `Conv3×3 stride=2 (64→128)` + ResBlock×2（128）
  - `Conv1×1 (128→D)`，默认 `D=256`
- 输出：`F ∈ R^{B×D×H×W}`，默认总 stride=2，因此 `H=W=N/2`。

#### 7.7.3 Token 化与 2D sin-cos positional encoding（跨尺寸）
- Token 化：按 row-major 将 `(H,W)` 展平成 `T=H·W` 个 token：
  - `tokens ∈ R^{B×T×D}`，`t = i·W + j`
- Positional encoding：使用 2D sin-cos（无可训练参数），对每个 token 的 `(i,j)` 动态生成 `PE(i,j) ∈ R^D` 并与 token 相加。
  - 约束：`D % 4 == 0`。

#### 7.7.4 ViT encoder（全局建模）
- TransformerEncoder：默认 `L=6` 层，`heads=8`，`mlp_ratio=4`（FFN=4D）。
- 聚合：默认 mean-pool（不强制 CLS token；若引入 CLS，Stage10/20/30 必须一致）。

#### 7.7.5 Heads（BC/PPO 同构）
- `policy head`：`Linear(D→3)`（输出 logits）
- `value head`：`Linear(D→1)`（PPO 使用；BC 阶段不参与 loss，但参数随 checkpoint 一起保存用于继承）

#### 7.7.6 Stage10/20/30 token 网格（仅形状差异）
- Stage10（N=10）：`H=W=5`，`T=25`
- Stage20（N=20）：`H=W=10`，`T=100`
- Stage30（N=30）：`H=W=15`，`T=225`

#### 7.7.7 继承与加载规则（BC↔PPO，PPO↔BC）
- 默认 strict=True 全量加载（同构网络）：
  - BC→PPO：PPO 起点加载同尺寸 BC best checkpoint（只继承策略相关权重：features_extractor + action head；value head 零初始化后 warmup）。
  - PPO→BC：Stage20/30 默认继承前置 Stage 的 PPO best checkpoint。
- 若未来结构演进导致 key 不匹配：允许 strict=False，缺失参数按初始化策略处理，并记录 missing/unexpected keys 以便排查。

### 7.8 训练器与优化器约束
- Optimizer：仅允许使用 `AdamW`（BC/PPO 统一），禁止 `Adam`/SGD 等其他优化器。

## 8. 接口与系统对接（若未知，先给接口占位与字段建议）
> 假设：后端采用 REST API；前端每 1s 轮询。

### 8.1 接口占位
- `GET /api/stages`
  - 响应：[{stage_id, size, current_phase, bc_status, ppo_status, bc_episode, ppo_episode, last_eval_coverage, updated_at_ms}]
  - 说明：不返回 stage.status 字段；Stage 展示状态由前端按 7.2 聚合规则计算。
- `GET /api/stages/{stage_id}`
  - 响应：{stage_id, size, current_phase, bc_status, ppo_status, bc_episode, ppo_episode, last_eval, last_eval_coverage, has_bc_best_checkpoint, has_ppo_best_checkpoint, prev_stage, eval_rollouts, metrics, probe}
  - metrics：用于详情页展示指标；本期默认展示：BC=bc_loss；PPO=ppo_loss+reward_mean（返回 `{bc:[...], ppo:[...]}`）。
  - eval_rollouts：合并自 `bc/evals/index.json` 与 `ppo/evals/index.json` 的 best/latest（去重后合并），每条必须带 `is_best`，供前端排序与标识。
  - probe：训练进程探测结果（不走缓存），用于识别“状态显示 running 但进程已不在”的异常情况。
- `GET /api/stages/{stage_id}/bc/start`
- `GET /api/stages/{stage_id}/bc/resume`
- `GET /api/stages/{stage_id}/bc/stop`
- `GET /api/stages/{stage_id}/bc/complete`
- `GET /api/stages/{stage_id}/ppo/start`
- `GET /api/stages/{stage_id}/ppo/resume`
- `GET /api/stages/{stage_id}/ppo/stop`
- `GET /api/stages/{stage_id}/ppo/complete`
- `POST /api/stages/{stage_id}/pause`
- `POST /api/stages/{stage_id}/resume`
- `POST /api/stages/{stage_id}/complete`
- `GET /api/stages/{stage_id}/reset`
- `GET /api/checkpoints?stage_id=&phase=&type=best|latest`
- `POST /api/checkpoints/{id}/rollouts`
  - body：{count: 10, source: "manual"}
- `GET /api/rollouts?stage_id=&phase=&source=eval|manual`
- `GET /api/rollouts/{rollout_id}?stage_id=&phase=&source=eval|manual`
  - 说明：用于播放弹窗拉取完整 steps；当前不支持 `source=train`
  - 响应：`{ok: true, rollout: {meta, summary, steps}}`，失败：`{ok: false, error}`

### 8.2 字段与约束
- 所有写操作必须返回最新状态与更新时间。
- `GET /api/stages/{stage_id}/bc/start` / `GET /api/stages/{stage_id}/ppo/start`：不需要请求体，仅依赖 `stage_id`；后端只修改 `state.json` 中必要字段（如 `current_phase`、`*_status`、`updated_at_ms`、`last_status_change_at_ms`）。
- `GET /api/stages/{stage_id}` 的 probe 必须实时反映训练进程存活状态：`effective_training` 仅当对应 phase status==running 且进程存活时才成立；若 status==running 但进程不存活，必须标记 `stale_running=true` 供前端提示。
- `GET /api/stages/{stage_id}` 的 probe 必须携带 worker 运行态信息（来自 `datas/.run/*runtime.json`）：包含 `status/heartbeat_at_ms/last_error`；前端用于展示“异常退出原因”。
- 训练 worker 必须在每个“大训练循环”结束时探测后端存活：`GET http://127.0.0.1:8000/api/health`；若不可达，则将对应 phase 状态修复为 paused 并退出进程。
- rollout 播放接口必须带 phase 与 source。
- checkpoint 列表应返回 is_best 字段，便于前端区分。
- checkpoint 列表必须返回 episode、coverage、step_count，满足列表展示与 best 判定。
- `GET /api/stages/{stage_id}` 的 eval_rollouts 必须包含：rollout_id、phase、source（固定为 eval）、episode、coverage、steps、length_max、reward_total、created_at_ms、is_best。
- `eval_id` / `checkpoint_id` 的命名必须由 episode 生成：使用 9 位补零（如 `000000123`）。
- 数值类型约定：
  - `episode`、`step_count`、`created_at_ms`、`updated_at_ms`、`last_status_change_at_ms` 语义为 64-bit 整数（后端 Python `int`；前端 `number` 在该范围内可精确表示）。
  - `eval_id`、`checkpoint_id`、`rollout_id` 均为字符串（带补零与分隔符，禁止当作整数参与运算）。
- evals/index.json 的 best/latest 语义：
  - latest：仅保留“最近一次 eval 产生的全部 rollouts”（固定 10 条，id 形如 `{eval_id}-{k}`，k=1..10）。
  - best：仅保留“截至当前所有 eval rollouts 中的最优 rollout”（固定 1 条），best rollout 文件需落盘在 `evals/best/...`。
- metrics 字段：返回 items 必须按 episode（或 timestamp_ms）升序，便于前端直接追加渲染。

#### Stage Reset（初始化）约束（必须）
- `GET /api/stages/{stage_id}/reset` 必须执行：
  - 若 `ppo_status!=not_started`：删除 `datas/stages/{stage_id}/ppo/` 并重建 PPO 骨架（保留 BC 历史数据）。
  - 否则若 `bc_status!=not_started`：删除 `datas/stages/{stage_id}/bc/` 并重建 BC 骨架（保留 PPO 历史数据）。
  - 否则：删除 `datas/stages/{stage_id}/` 并重建为“标准骨架”。
- `GET /api/stages/{stage_id}/reset` 必须拒绝以下状态：
  - 若 `bc_status==running` 或 `ppo_status==running`：禁止 reset（训练中不可删除）。
  - 若 `bc_status==completed` 且 `ppo_status==completed`：禁止 reset（已完成不可删除）。
- 前端在触发 reset 前必须做拦截：
  - 若命中禁止条件：弹窗提示原因并阻止请求。
  - 若允许：必须二次确认（提示会清空“当前阶段”或“全部”的历史数据）。
  - 无论请求成功/失败，前端都必须提示结果（成功提示 + 自动刷新；失败提示错误原因）。
  - 提示组件要求：使用系统内置 toast + 自定义确认弹窗（modal），禁止使用 `window.alert/window.confirm`。

## 9. 权限与安全/合规
- 不支持多用户/权限体系（明确）。
- 必须限制训练并发：全局只允许一个 Stage 训练（定义同 5.6）。
- 文件访问必须限定在 `data/` 子目录内，禁止路径穿越。
- 后端不使用 Stage 详情内存缓存；读接口直接从 `datas/` 读取，确保前端轮询实时生效。

## 10. 性能与稳定性要求
- 训练设备必须自动识别并选择：优先级 CUDA > MPS > CPU；若首选不可用必须自动降级。
- 前端轮询间隔：1s；失败 3 次需提示错误。
- Rollout 播放加载：首帧 <= 1s（本地文件）。
- 写盘必须异步，避免阻塞训练。
- 服务启动必须在 5s 内完成目录初始化。

## 11. 埋点与指标（核心指标、事件、属性）
### 11.1 核心指标
- 训练启动成功率。
- eval/manual rollout 生成成功率。
- BC/PPO 覆盖率均值与最大值。
- checkpoint 生成数量。

### 11.2 事件建议
- stage_start（stage_id, phase）
- stage_pause（stage_id, phase）
- stage_resume（stage_id, phase）
- checkpoint_generate（checkpoint_id, phase）
- rollout_play（rollout_id, source）
- rollout_generate_manual（checkpoint_id, count）

## 12. 异常与边界情况
- 状态不合法操作必须返回错误提示（如 running 阶段不能删除）。
- checkpoint 文件缺失/损坏：禁止生成 rollouts 并提示。
- rollout 文件缺失：播放组件提示错误并允许重试。
- 后端不可用：前端提示并暂停轮询。
- manual rollouts 数量不足 10：必须提示原因。

## 13. 验收标准（可测试、可度量）
- Stage 列表 1s 刷新，状态与覆盖率展示正确。
- Stage 详情页能展示 BC/PPO 状态与指标曲线。
- 训练控制：开始/停止/恢复/完成/初始化均符合状态约束。
- eval/manual rollout 可播放，覆盖率/步数/长度正确显示。
- 生成 manual rollouts 后列表展示 10 条。
- CI/本地检查脚本能检测函数超 30 行并报错。
- 启动时自动选择训练设备（CUDA > MPS > CPU），可通过日志确认选择结果。

## 14. 里程碑与发布计划
- M1（P0）：训练流程 + 指标保存 + eval rollout 回放。
- M2（P1）：Checkpoint 列表 + manual rollouts + 播放增强。
- M3（P2）：UI 优化 + 扩展 Stage 支持。

## 15. 风险与依赖
- 依赖 stable-baselines3 与自定义 policy 的兼容性风险。
- 本地写盘速度影响训练流畅性。
- 前端轮询可能导致 UI 卡顿，需要优化渲染。

## 16. 待确认问题清单
- 暂无。

## 17. 技术栈与版本约束
- Frontend: React 18+，Vite 5+（图表使用轻量 SVG 实现）。
- Backend: Python 3.11+，FastAPI 0.100+，uvicorn 0.23+。
- ML: PyTorch 2.0+，stable-baselines3（SB3）2.0+ + Gymnasium。
- 环境管理：uv（必须），禁止直接调用 python/pip。
- 兼容性要求：macOS（MPS）与 Linux（CUDA）优先；Windows 可选（待确认）。

## 18. 项目目录结构
- `/frontend`
  - `/src/pages`（Stage 列表、Stage 详情、Checkpoint、Rollout 播放）
  - `/src/components`（通用 UI、图表、播放组件）
  - `/src/hooks`（轮询/状态管理/播放器控制）
  - `/src/api`（后端请求封装）
- `/backend`
  - `/app/api`（路由与接口定义）
  - `/app/services`（训练、存储、回放、指标服务）
  - `/app/models`（数据结构/Schema/状态机）
- `/datas`
  - `/stages`（按 Stage 保存训练数据）
- `/scripts`（检查脚本/运行辅助脚本）
- 前端路由：
  - `/` → Stage 列表页（5.2）
  - `/stages/:id` → Stage 详情页（5.3）
  - `/checkpoints` → Checkpoint 列表页（5.4）
  - `/rollouts/:id` → Rollout 播放（5.5，可为页面或 modal）

## 19. 配置文件规范
- 主配置入口：`backend/app/config.py`（集中读取环境变量并提供默认值）。
- 必须支持环境变量覆盖（如 `PPO_ROLLOUT_STEPS`、`BC_TRAIN_ROLLOUT_MAX_STEPS`、`SNAKE_HUNGER_BUDGET`）。
- 配置加载后按“只读”使用（其他模块禁止直接读 env，需通过 config 函数获取）。

## 20. 错误码规范
- 4001：状态冲突（如 running 时删除/重复启动）。
- 4002：文件缺失（checkpoint/rollout/metrics 不存在）。
- 4003：训练中禁止操作（如强制删除/并发训练）。
- 4004：参数非法（如未知阶段/未知 source）。
- 5001：训练异常（内部训练进程失败）。
  - 统一错误响应格式：
```json
{
  "error": {
    "code": 4001,
    "message": "状态冲突：Stage 正在训练中，无法删除"
  }
}
```

## 21. 测试策略
- 后端：pytest，核心服务覆盖率 ≥ 80%。
- 前端：vitest + react-testing-library，关键流程 E2E（建议 Playwright）。
- CI：检查脚本 + 单元测试 + lint。
