from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SnakeEnvCfg:
    size: int
    seed: int
    max_steps: int


def bc_worker_cfg() -> dict[str, Any]:
    """
    BC worker 配置（集中在此处，其他模块禁止直接读 env）。

    round 流程：
    1) 采集 train_rollout_count 条「达标」teacher rollout
    2) 训练 episodes_per_train 个 episode（episode 内 update 次数≈总 steps / batch_size）
    3) eval 固定生成 10 条 rollout
    4) 保存 checkpoint（latest + best）
    """
    return {
        # 每轮训练的 episode 次数（一次 episode 会跑若干个 parameter update）
        "episodes_per_train": _env_int("BC_EPISODES_PER_TRAIN", 1),
        # checkpoints/latest 保留条数（最旧的会被删除；best 单独保留1条）
        "latest_keep": _env_int("LATEST_KEEP", 10),
        # metrics/episodes.jsonl 只保留最近 N 行（避免文件无限增长）
        "metrics_keep": _env_int("METRICS_KEEP", 200),
        # AdamW 学习率（BC）
        "lr": _env_float("BC_LR", 3e-4),
        # BC 监督学习 batch size：每次 update 采样多少个 step 样本
        "batch_size": _env_int("BC_BATCH_SIZE", 1024),
        # 每轮采集需要「接受」的 teacher rollout 条数（不达标的会丢弃并继续采样）
        "train_rollout_count": _env_int("BC_TRAIN_ROLLOUT_COUNT", 1),
        # 单条 rollout 的最低覆盖率门槛（按 rollout.summary.coverage_max 判定）
        "train_min_rollout_coverage": _env_float("BC_TRAIN_MIN_ROLLOUT_COVERAGE", 0.70),
        # 单条 rollout 的最大步数（0 表示使用默认：size^2 * 400）
        "train_rollout_max_steps": _env_int("BC_TRAIN_ROLLOUT_MAX_STEPS", 0),
        # 为凑够 train_rollout_count 条达标 rollout，最多尝试生成的次数（超出则采集失败报错）
        "train_max_attempts": _env_int("BC_TRAIN_MAX_ATTEMPTS", 100000),
        # eval 每次固定 10 条（PRD 固定不可配；这里保留字段用于 worker 调用）
        "eval_rollouts": 10,
        # eval rollout 的最大步数（0 表示使用默认：size^2 * 40；训练采集的 max_steps 仍是 size^2 * 400）
        "eval_max_steps": _env_int("BC_EVAL_MAX_STEPS", 0),
    }


def ppo_worker_cfg() -> dict[str, Any]:
    """
    PPO worker 配置（集中在此处，其他模块禁止直接读 env）。
    """
    return {
        # 每轮训练的 learn 次数（每次 learn=采样 rollout_steps 个 step + 做一次 PPO 更新）
        "episodes_per_train": _env_int("PPO_EPISODES_PER_TRAIN", 1),
        # checkpoints/latest 保留条数（最旧的会被删除；best 单独保留1条）
        "latest_keep": _env_int("LATEST_KEEP", 10),
        # metrics/episodes.jsonl 只保留最近 N 行（避免文件无限增长）
        "metrics_keep": _env_int("METRICS_KEEP", 200),
        # Adam 学习率（PPO）
        "lr": _env_float("PPO_LR", 2.5e-4),
        # 折扣因子
        "gamma": _env_float("PPO_GAMMA", 0.99),
        # GAE lambda
        "gae_lambda": _env_float("PPO_GAE_LAMBDA", 0.95),
        # PPO clip range（重要：避免 policy update 过大）
        "clip": _env_float("PPO_CLIP", 0.2),
        # 每次 learn 对同一批 rollout 数据重复优化的 epoch 次数（SB3: n_epochs）
        "ppo_epochs": _env_int("PPO_EPOCHS", 1),
        # PPO minibatch size（SB3: batch_size）
        "minibatch_size": _env_int("PPO_MINIBATCH_SIZE", 512),
        # value loss 系数
        "vf_coef": _env_float("PPO_VF_COEF", 0.5),
        # entropy bonus 系数（鼓励探索；太大可能发散）
        "ent_coef": _env_float("PPO_ENT_COEF", 0.01),
        # 梯度裁剪
        "max_grad_norm": _env_float("PPO_MAX_GRAD_NORM", 0.5),
        # 每次 learn 采样的环境 step 数（SB3: n_steps）；不要设置过大，否则 rollout buffer 占用巨量内存
        "rollout_steps": _env_int("PPO_ROLLOUT_STEPS", 2000),
        # PPO 采样时，单个 episode 的最大步数（0 表示使用默认：size^2 * 8）
        "rollout_max_steps": _env_int("PPO_ROLLOUT_MAX_STEPS", 0),
        # eval 每次固定 10 条（PRD 固定不可配；这里保留字段用于 worker 调用）
        "eval_rollouts": 10,
        # eval rollout 的最大步数（0 表示使用默认：size^2 * 40）
        "eval_max_steps": _env_int("PPO_EVAL_MAX_STEPS", 0),
    }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def datas_dir() -> Path:
    return Path(os.environ.get("DATAS_DIR") or (repo_root() / "datas"))


def api_base() -> str:
    return str(os.environ.get("CHICHI_API_BASE") or "http://127.0.0.1:8000").rstrip("/")


def health_url() -> str:
    return str(os.environ.get("CHICHI_HEALTH_URL") or "http://127.0.0.1:8000/api/health").strip()


def uv_cache_dir() -> Path:
    return Path(os.environ.get("UV_CACHE_DIR") or (repo_root() / ".uv-cache"))


def worker_action() -> str:
    v = str(os.environ.get("WORKER_ACTION") or os.environ.get("BC_ACTION") or "start").strip().lower()
    return v if v in ("start", "resume") else "start"


def bc_action() -> str:
    return worker_action()


def worker_env(datas_dir: Path, api_base: str, *, backend_pid: int, action: str) -> dict[str, str]:
    env = dict(os.environ)
    env["DATAS_DIR"] = str(datas_dir)
    env["CHICHI_API_BASE"] = str(api_base).rstrip("/")
    env["CHICHI_BACKEND_PID"] = str(int(backend_pid))
    env["WORKER_ACTION"] = str(action or "start")
    env["BC_ACTION"] = str(action or "start")
    env.setdefault("UV_CACHE_DIR", str(uv_cache_dir()))
    return env


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key) or default)
    except ValueError:
        return int(default)


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key) or default)
    except ValueError:
        return float(default)
