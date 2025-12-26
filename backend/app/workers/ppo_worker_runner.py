from __future__ import annotations

import os
import random
import signal
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from app.config import SnakeEnvCfg, datas_dir as cfg_datas_dir, health_url as cfg_health_url, ppo_worker_cfg, worker_action
from app.data.runtime_repo import write_runtime
from app.data.stages_repo import read_stage_state, write_stage_state
from app.ml.checkpoints import InitPlan, load_weights, pick_device, select_ppo_init_plan
from app.ml.decoupled_rollout_buffer import DecoupledGAERolloutBuffer
from app.ml.encoding import encode_grid
from app.ml.model import CnnVitActorCritic, ModelCfg
from app.ml.sb3_extractor import CnnVitFeaturesExtractor
from app.sim.snake_env import SnakeEnv, eval_seed
from app.sim.snake_gym_env import SnakeGymEnv
from app.workers.worker_eval_rollouts import EvalResult, write_eval_latest
from app.workers.worker_health import is_backend_healthy
from app.workers.worker_index import pick_best_item, push_latest_items, read_index, write_index
from app.workers.worker_metrics import append_diag_line, append_metrics_line

_STOP = False


@dataclass
class _StopCtl:
    dirty: bool = False
    finalized: bool = False

    def mark_dirty(self) -> None:
        self.dirty = True


class _StopAndStatsCallback(BaseCallback):
    def __init__(self, stop_requested: Callable[[], bool]) -> None:
        super().__init__()
        self._stop_requested = stop_requested
        self.reward_sum = 0.0
        self.step_count = 0
        self.ate_steps = 0
        self.episodes = 0
        self.episode_lens: list[int] = []
        self.end_covs: list[float] = []
        self.end_lengths: list[int] = []
        self.end_wall = 0
        self.end_self = 0
        self.end_truncated = 0
        self.rollout_buffer_stats: dict[str, Any] | None = None
        self._ep_len: list[int] | None = None

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self.reward_sum += float(torch.as_tensor(rewards).sum().item())
            self.step_count += int(torch.as_tensor(rewards).numel())
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        if dones is None:
            return not self._stop_requested()
        d = np.asarray(dones, dtype=bool).reshape(-1)
        if self._ep_len is None:
            self._ep_len = [0 for _ in range(int(d.size))]
        for i in range(int(d.size)):
            self._ep_len[i] += 1
            info = infos[i] if i < len(infos) and isinstance(infos[i], dict) else {}
            if bool(info.get("ate")):
                self.ate_steps += 1
            if bool(d[i]):
                self.episodes += 1
                self.episode_lens.append(int(self._ep_len[i]))
                self._ep_len[i] = 0
                self._add_end_info(info)
        return not self._stop_requested()

    def reward_mean(self) -> float:
        return float(self.reward_sum) / float(max(1, int(self.step_count)))

    def train_summary(self) -> dict[str, Any]:
        lens = np.asarray(self.episode_lens, dtype=np.int64)
        covs = np.asarray(self.end_covs, dtype=np.float32)
        lens_mean = float(lens.mean()) if lens.size else None
        cov_mean = float(covs.mean()) if covs.size else None
        return {
            "steps": int(self.step_count),
            "episodes": int(self.episodes),
            "ate_step_rate": float(self.ate_steps) / float(max(1, int(self.step_count))),
            "ep_len_mean": lens_mean,
            "ep_len_min": int(lens.min()) if lens.size else None,
            "ep_len_max": int(lens.max()) if lens.size else None,
            "end_cov_mean": cov_mean,
            "end_len_mean": float(np.asarray(self.end_lengths, dtype=np.float32).mean()) if self.end_lengths else None,
            "end_wall": int(self.end_wall),
            "end_self": int(self.end_self),
            "end_truncated": int(self.end_truncated),
            "rollout_buffer": self.rollout_buffer_stats or {},
        }

    def _on_rollout_end(self) -> None:
        rb = getattr(self.model, "rollout_buffer", None)
        self.rollout_buffer_stats = _rollout_buffer_stats(rb)

    def _add_end_info(self, info: dict[str, Any]) -> None:
        trunc = bool(info.get("truncated") or info.get("TimeLimit.truncated"))
        self.end_truncated += 1 if trunc else 0
        coll = str(info.get("collision") or "")
        self.end_wall += 1 if coll == "wall" else 0
        self.end_self += 1 if coll == "self" else 0
        cov = info.get("coverage")
        if isinstance(cov, (int, float)):
            self.end_covs.append(float(cov))
        length = info.get("snake_length")
        if isinstance(length, (int, float)):
            self.end_lengths.append(int(length))


def _rollout_buffer_stats(rb: Any) -> dict[str, Any]:
    if rb is None:
        return {}
    out: dict[str, Any] = {}
    for name in ("advantages", "returns", "values", "rewards", "log_probs"):
        arr = getattr(rb, name, None)
        if arr is None:
            continue
        stats = _arr_stats(np.asarray(arr).reshape(-1))
        if stats:
            out[name] = stats
    return out


def _arr_stats(x: np.ndarray) -> dict[str, Any]:
    if x.size <= 0:
        return {}
    x = x.astype(np.float64, copy=False)
    finite = np.isfinite(x)
    out: dict[str, Any] = {"count": int(x.size), "finite": int(finite.sum())}
    if not bool(finite.any()):
        return out
    xf = x[finite]
    out.update({"mean": float(xf.mean()), "std": float(xf.std()), "min": float(xf.min()), "max": float(xf.max())})
    return out


def _make_probe_batch(stage_id: int, size: int, max_steps: int, device: torch.device) -> torch.Tensor:
    rng = random.Random(int(stage_id) * 1009 + 17)
    xs: list[torch.Tensor] = []
    for k in range(1, 6):
        seed = eval_seed(stage_id, "ppo", "probe", int(k))
        env = SnakeEnv(SnakeEnvCfg(size=size, seed=seed, max_steps=max_steps))
        denom = float(max(1, int(env.max_steps)))
        for _t in range(64):
            time_left = float(max(0.0, min(1.0, (float(env.max_steps) - float(env.steps)) / denom)))
            hunger_steps = max(0, int(env.steps_since_last_eat) - int(env.hunger_grace_steps))
            hunger = float(max(0.0, min(1.0, float(hunger_steps) / denom)))
            cov = float(len(env.snake)) / float(max(1, int(size) * int(size)))
            pre = env.snapshot()
            xs.append(encode_grid(size=size, snake=list(pre["snake"]), food=list(pre["food"]), dir_name=str(pre["dir"]), device=device, time_left=time_left, hunger=hunger, coverage_norm=cov))
            res = env.step(rng.choice(["L", "S", "R"]))
            if bool(res.done):
                break
    if not xs:
        raise ValueError("probe batch empty")
    return torch.stack(xs, dim=0)


def _probe_stats(logits: torch.Tensor, values: torch.Tensor) -> dict[str, Any]:
    p = torch.softmax(logits, dim=-1)
    logp = torch.log_softmax(logits, dim=-1)
    ent = -(p * logp).sum(dim=-1)
    a = torch.argmax(logits, dim=-1)
    n = float(max(1, int(a.numel())))
    counts = torch.bincount(a, minlength=3).float() / n
    return {
        "n": int(a.numel()),
        "entropy_mean": float(ent.mean().item()),
        "max_prob_mean": float(p.max(dim=-1).values.mean().item()),
        "act_frac": {"L": float(counts[0].item()), "S": float(counts[1].item()), "R": float(counts[2].item())},
        "value_mean": float(values.mean().item()),
        "value_std": float(values.std(unbiased=False).item()) if int(values.numel()) > 1 else 0.0,
    }


def _batch_std_mean(x: torch.Tensor) -> float:
    if x.numel() <= 0:
        return 0.0
    if int(x.dim()) < 2:
        return 0.0
    x2 = x.float().reshape(int(x.shape[0]), -1)
    if int(x2.shape[0]) <= 1:
        return 0.0
    return float(x2.std(dim=0, unbiased=False).mean().item())


def _probe_feature_stats(sb3_policy: Any, obs: torch.Tensor) -> dict[str, Any]:
    share = bool(getattr(sb3_policy, "share_features_extractor", True))
    features = sb3_policy.extract_features(obs)
    if share:
        return {"share": True, "features_std_mean": _batch_std_mean(features)}
    pi_f, vf_f = features
    return {"share": False, "pi_std_mean": _batch_std_mean(pi_f), "vf_std_mean": _batch_std_mean(vf_f)}


def _kl_logits(old_logits: torch.Tensor, new_logits: torch.Tensor) -> float:
    p_old = torch.softmax(old_logits, dim=-1)
    lp_old = torch.log_softmax(old_logits, dim=-1)
    lp_new = torch.log_softmax(new_logits, dim=-1)
    kl = (p_old * (lp_old - lp_new)).sum(dim=-1)
    return float(kl.mean().item())

class _EvalModel(torch.nn.Module):
    def __init__(self, sb3_policy: Any) -> None:
        super().__init__()
        self.policy = sb3_policy

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.policy.extract_features(x)
        if bool(getattr(self.policy, "share_features_extractor", True)):
            latent_pi, latent_vf = self.policy.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.policy.mlp_extractor.forward_critic(vf_features)
        logits = self.policy.action_net(latent_pi)
        values = self.policy.value_net(latent_vf).squeeze(-1)
        return logits, values


def run_ppo_worker(stage_id: int) -> None:
    _install_signal_handlers()
    datas_dir = cfg_datas_dir()
    health_url = cfg_health_url()
    run_id = _run_id(stage_id)
    write_runtime(datas_dir, "ppo", stage_id, {"status": "running", "pid": os.getpid(), "run_id": run_id})
    _ensure_running_state(datas_dir, stage_id)
    try:
        _loop(datas_dir, health_url, stage_id, ppo_worker_cfg())
    except SystemExit:
        raise
    except Exception as e:
        reason = f"crash: {type(e).__name__}: {e}"
        _pause_and_exit(datas_dir, stage_id, reason, tb=traceback.format_exc())


def _loop(datas_dir: Path, health_url: str, stage_id: int, cfg: dict[str, Any]) -> None:
    size = _stage_size(datas_dir, stage_id)
    device = pick_device()
    action = worker_action()
    env = _make_env(size=size, seed=stage_id, cfg=cfg)
    model = _make_model(env, cfg=cfg, device=device)
    plan, ckpt_episode = _init_weights(model, datas_dir, stage_id, device, action=action)
    write_runtime(datas_dir, "ppo", stage_id, {"device": str(device), "init_plan": plan.to_runtime()})
    _sync_episode_from_ckpt(datas_dir, stage_id, plan, ckpt_episode)

    from_bc_start = plan.ckpt_phase == "bc"
    finetune_from_bc = _ensure_ppo_finetune_from_bc(datas_dir, stage_id, from_bc_start)
    probe_batch = _make_probe_batch(stage_id, size, int(getattr(env, "max_steps", 2048)), device)
    append_diag_line(
        datas_dir,
        stage_id,
        "ppo",
        int(read_stage_state(datas_dir, stage_id).get("ppo_episode") or 0),
        "init",
        cfg,
        payload={"init_plan": plan.to_runtime(), "cfg": _cfg_snapshot(cfg), "probe_n": int(probe_batch.shape[0]), "from_bc_start": bool(from_bc_start), "finetune_from_bc": bool(finetune_from_bc)},
    )
    if from_bc_start:
        info = _warmup_value_network(model, env, cfg=cfg, device=device)
        if info:
            write_runtime(datas_dir, "ppo", stage_id, {"value_warmup": info})
            append_diag_line(datas_dir, stage_id, "ppo", int(read_stage_state(datas_dir, stage_id).get("ppo_episode") or 0), "value_warmup", cfg, payload=info)

    ctl = _StopCtl()
    probe_model = _EvalModel(model.policy).to(device)
    while True:
        _maybe_pause_and_exit(ctl, datas_dir, stage_id, model, cfg, size, device)
        _train_round(ctl, datas_dir, stage_id, model, cfg, size, probe_model=probe_model, probe_batch=probe_batch, from_bc=finetune_from_bc)
        if _STOP:
            _maybe_pause_and_exit(ctl, datas_dir, stage_id, model, cfg, size, device)
        _eval_and_checkpoint(datas_dir, stage_id, size, cfg, model, device, reason="round_end")
        ctl.dirty = False
        _exit_if_backend_unhealthy(datas_dir, stage_id, health_url)


def _ensure_ppo_finetune_from_bc(datas_dir: Path, stage_id: int, from_bc_start: bool) -> bool:
    state = read_stage_state(datas_dir, stage_id)
    if bool(state.get("ppo_finetune_from_bc")):
        return True
    if not bool(from_bc_start):
        return False
    nxt = dict(state)
    nxt["ppo_finetune_from_bc"] = True
    nxt["updated_at_ms"] = int(time.time() * 1000)
    write_stage_state(datas_dir, stage_id, nxt)
    return True


def _warmup_value_network(model: PPO, env: SnakeGymEnv, *, cfg: dict[str, Any], device: torch.device) -> dict[str, Any] | None:
    steps = int(cfg.get("value_warmup_steps") or 0)
    epochs = int(cfg.get("value_warmup_epochs") or 0)
    if steps <= 0 or epochs <= 0:
        return None
    gamma = float(cfg.get("gamma") or 0.99)
    max_ep_steps = int(cfg.get("value_warmup_max_steps") or 5000)
    obs, returns, collect = _collect_value_warmup_dataset(model, env, steps=steps, gamma=gamma, max_ep_steps=max_ep_steps)
    loss = _train_value_net(model, obs, returns, cfg=cfg, device=device)
    return {"steps": int(obs.shape[0]), "epochs": int(epochs), "loss": loss, "collect": collect, "returns": _arr_stats(returns.reshape(-1))}


@dataclass
class _WarmupCollectStats:
    ended: int = 0
    len_sum: int = 0
    len_min: int | None = None
    len_max: int | None = None
    rew_sum: float = 0.0
    end_term: int = 0
    end_trunc: int = 0

    def add(self, ep_steps: int, ep_rewards: list[float], *, terminated: bool, truncated: bool) -> None:
        self.ended += 1
        self.len_sum += int(ep_steps)
        self.len_min = int(ep_steps) if self.len_min is None else min(int(self.len_min), int(ep_steps))
        self.len_max = int(ep_steps) if self.len_max is None else max(int(self.len_max), int(ep_steps))
        self.rew_sum += float(sum(ep_rewards))
        self.end_term += 1 if bool(terminated) else 0
        self.end_trunc += 1 if bool(truncated) else 0

    def to_dict(self) -> dict[str, Any]:
        e = int(self.ended)
        return {
            "episodes": e,
            "ep_len_mean": (float(self.len_sum) / float(e)) if e > 0 else None,
            "ep_len_min": int(self.len_min) if isinstance(self.len_min, int) else None,
            "ep_len_max": int(self.len_max) if isinstance(self.len_max, int) else None,
            "ep_reward_sum_mean": (float(self.rew_sum) / float(e)) if e > 0 else None,
            "ended_terminated": int(self.end_term),
            "ended_truncated": int(self.end_trunc),
        }


def _collect_value_warmup_dataset(
    model: PPO, env: SnakeGymEnv, *, steps: int, gamma: float, max_ep_steps: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    obs_out: list[np.ndarray] = []
    ret_out: list[float] = []
    stats = _WarmupCollectStats()
    _collect_value_warmup_raw(model, env, steps=steps, gamma=gamma, max_ep_steps=max_ep_steps, obs_out=obs_out, ret_out=ret_out, stats=stats)
    obs_arr = np.stack(obs_out[:steps]).astype(np.float32, copy=False)
    ret_arr = np.asarray(ret_out[:steps], dtype=np.float32).reshape(-1, 1)
    return obs_arr, ret_arr, stats.to_dict()


def _collect_value_warmup_raw(
    model: PPO,
    env: SnakeGymEnv,
    *,
    steps: int,
    gamma: float,
    max_ep_steps: int,
    obs_out: list[np.ndarray],
    ret_out: list[float],
    stats: _WarmupCollectStats,
) -> None:
    ep_obs: list[np.ndarray] = []
    ep_rewards: list[float] = []
    ep_steps = 0
    max_ep_steps = int(max(1, min(int(max_ep_steps), int(getattr(env, "max_steps", max_ep_steps)))))
    obs, _info = env.reset(seed=_time_seed())
    while len(obs_out) < int(steps):
        ep_obs.append(obs.copy())
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _info = env.step(action)
        ep_rewards.append(float(reward))
        ep_steps += 1
        if terminated or truncated or ep_steps >= max_ep_steps:
            _append_returns(ep_obs, ep_rewards, gamma, obs_out, ret_out)
            stats.add(ep_steps, ep_rewards, terminated=bool(terminated), truncated=bool(truncated))
            ep_obs, ep_rewards, ep_steps = [], [], 0
            obs, _info = env.reset(seed=_time_seed())
    if ep_rewards:
        _append_returns(ep_obs, ep_rewards, gamma, obs_out, ret_out)


def _time_seed() -> int:
    return int(time.time() * 1000) % 2_000_000_000


def _append_returns(ep_obs: list[np.ndarray], ep_rewards: list[float], gamma: float, obs_out: list[np.ndarray], ret_out: list[float]) -> None:
    g = 0.0
    for obs, reward in zip(reversed(ep_obs), reversed(ep_rewards)):
        g = float(reward) + float(gamma) * g
        obs_out.append(obs)
        ret_out.append(float(g))


def _warmup_value_params(sb3_policy: Any) -> list[torch.nn.Parameter]:
    return [*sb3_policy.mlp_extractor.value_net.parameters(), *sb3_policy.value_net.parameters()]


def _warmup_predict_values(sb3_policy: Any, obs_t: torch.Tensor) -> torch.Tensor:
    share = bool(getattr(sb3_policy, "share_features_extractor", True))
    with torch.no_grad():
        features = sb3_policy.extract_features(obs_t)
    vf_features = features if share else features[1]
    latent_vf = sb3_policy.mlp_extractor.forward_critic(vf_features)
    return sb3_policy.value_net(latent_vf)


def _train_value_net(model: PPO, obs: np.ndarray, returns: np.ndarray, *, cfg: dict[str, Any], device: torch.device) -> float | None:
    epochs = int(cfg.get("value_warmup_epochs") or 0)
    if obs.size == 0 or epochs <= 0:
        return None
    lr = float(cfg.get("value_warmup_lr") or 1e-4)
    batch = int(cfg.get("value_warmup_batch") or 1024)
    idx = np.arange(int(obs.shape[0]))
    optimizer = torch.optim.AdamW(_warmup_value_params(model.policy), lr=lr)
    criterion = nn.MSELoss()
    last_loss: float | None = None
    model.policy.train()
    for _ in range(int(epochs)):
        np.random.shuffle(idx)
        for start in range(0, int(idx.size), int(batch)):
            j = idx[start : start + int(batch)]
            obs_t = torch.from_numpy(obs[j]).float().to(device)
            ret_t = torch.from_numpy(returns[j]).float().to(device)
            pred = _warmup_predict_values(model.policy, obs_t)
            loss = criterion(pred, ret_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
    model.policy.eval()
    return last_loss


def _train_round(
    ctl: _StopCtl,
    datas_dir: Path,
    stage_id: int,
    model: PPO,
    cfg: dict[str, Any],
    size: int,
    *,
    probe_model: torch.nn.Module,
    probe_batch: torch.Tensor,
    from_bc: bool,
) -> None:
    rounds = int(cfg.get("episodes_per_train") or 1)
    if rounds <= 0:
        return
    for _k in range(int(rounds)):
        episode, metrics, diag, dirty = _train_one(datas_dir, stage_id, model, cfg, probe_model, probe_batch, from_bc=from_bc)
        append_metrics_line(datas_dir, stage_id, "ppo", episode, size, cfg, metrics=metrics)
        append_diag_line(datas_dir, stage_id, "ppo", episode, "train_round", cfg, payload=diag)
        _maybe_vf_collapse_fallback(datas_dir, stage_id, model, cfg, episode, diag, from_bc=from_bc)
        if dirty:
            ctl.mark_dirty()
        if _STOP:
            return


def _train_one(
    datas_dir: Path,
    stage_id: int,
    model: PPO,
    cfg: dict[str, Any],
    probe_model: torch.nn.Module,
    probe_batch: torch.Tensor,
    *,
    from_bc: bool,
) -> tuple[int, dict[str, Any], dict[str, Any], bool]:
    with torch.no_grad():
        probe_model.eval()
        pre_logits, pre_values = probe_model(probe_batch)
        pre_feat = _probe_feature_stats(model.policy, probe_batch)
    guard = _apply_finetune_guard(model, datas_dir, stage_id, cfg, from_bc=from_bc)
    cb = _StopAndStatsCallback(stop_requested=lambda: _STOP)
    model.learn(total_timesteps=max(1, int(cfg.get("rollout_steps") or 8192)), reset_num_timesteps=False, callback=cb)
    log = model.logger.name_to_value
    with torch.no_grad():
        probe_model.eval()
        post_logits, post_values = probe_model(probe_batch)
        post_feat = _probe_feature_stats(model.policy, probe_batch)
    episode = _bump_episode(datas_dir, stage_id, cb.step_count)
    metrics = _ppo_metrics(cb, log)
    diag = {
        **guard,
        "probe": {"kl_pre_post": _kl_logits(pre_logits, post_logits), "pre": _probe_stats(pre_logits, pre_values), "post": _probe_stats(post_logits, post_values)},
        "probe_features": {"pre": pre_feat, "post": post_feat},
        "train": cb.train_summary(),
        "sb3": {k: metrics.get(k) for k in ("ppo_loss", "reward_mean", "approx_kl", "clip_fraction", "value_loss", "explained_variance")},
    }
    return episode, metrics, diag, bool(cb.step_count > 0)


def _diag_vf_std(diag: dict[str, Any]) -> float | None:
    feats = diag.get("probe_features")
    if not isinstance(feats, dict):
        return None
    post = feats.get("post")
    if not isinstance(post, dict):
        return None
    v = post.get("vf_std_mean")
    return float(v) if isinstance(v, (int, float)) else None


def _restore_vf_from_pi(sb3_policy: Any) -> bool:
    if bool(getattr(sb3_policy, "share_features_extractor", True)):
        return False
    pi_ex = sb3_policy.pi_features_extractor
    vf_ex = sb3_policy.vf_features_extractor
    try:
        vf_ex.backbone.load_state_dict(pi_ex.backbone.state_dict())
        vf_ex.vit.load_state_dict(pi_ex.vit.state_dict())
    except Exception:
        vf_ex.load_state_dict(pi_ex.state_dict())
    return True


def _write_vf_refreeze_until(datas_dir: Path, stage_id: int, until_episode: int) -> None:
    state = read_stage_state(datas_dir, stage_id)
    nxt = dict(state)
    nxt["ppo_vf_refreeze_until_episode"] = int(until_episode)
    nxt["updated_at_ms"] = int(time.time() * 1000)
    write_stage_state(datas_dir, stage_id, nxt)


def _maybe_vf_collapse_fallback(
    datas_dir: Path,
    stage_id: int,
    model: PPO,
    cfg: dict[str, Any],
    episode: int,
    diag: dict[str, Any],
    *,
    from_bc: bool,
) -> None:
    if not bool(from_bc) or bool(getattr(model.policy, "share_features_extractor", True)):
        return
    thr = float(cfg.get("vf_collapse_feature_std") or 0.0)
    if thr <= 0:
        return
    vf_std = _diag_vf_std(diag)
    if vf_std is None or float(vf_std) >= float(thr):
        return
    restored = _restore_vf_from_pi(model.policy)
    refreeze_rounds = int(cfg.get("vf_collapse_refreeze_rounds") or 0)
    until = int(episode) + int(refreeze_rounds) if restored and refreeze_rounds > 0 else None
    if until is not None:
        _write_vf_refreeze_until(datas_dir, stage_id, until)
    append_diag_line(datas_dir, stage_id, "ppo", int(episode), "vf_fallback", cfg, payload={"vf_std_mean": float(vf_std), "threshold": float(thr), "refreeze_rounds": int(refreeze_rounds), "refreeze_until_episode": int(until) if until is not None else None})


def _vf_schedule(state: dict[str, Any], episode: int, cfg: dict[str, Any], *, from_bc: bool, share_features: bool) -> dict[str, Any]:
    if not bool(from_bc) or bool(share_features):
        return {"vf_stage": "all", "vf_refreeze_until_episode": None}
    refreeze_until = int(state.get("ppo_vf_refreeze_until_episode") or 0)
    if refreeze_until > int(episode):
        return {"vf_stage": "frozen", "vf_refreeze_until_episode": int(refreeze_until)}
    freeze_rounds = int(cfg.get("vf_freeze_rounds") or 0)
    vit_rounds = int(cfg.get("vf_unfreeze_vit_rounds") or 0)
    if freeze_rounds > 0 and int(episode) < int(freeze_rounds):
        return {"vf_stage": "frozen", "vf_refreeze_until_episode": int(refreeze_until) or None}
    if vit_rounds > 0 and int(episode) < int(freeze_rounds) + int(vit_rounds):
        return {"vf_stage": "vit", "vf_refreeze_until_episode": int(refreeze_until) or None}
    return {"vf_stage": "all", "vf_refreeze_until_episode": int(refreeze_until) or None}


def _apply_vf_stage(vf_ex: torch.nn.Module, stage: str) -> None:
    if str(stage) == "frozen":
        _set_trainable(vf_ex, False)
        return
    if str(stage) == "vit" and hasattr(vf_ex, "backbone") and hasattr(vf_ex, "vit"):
        _set_trainable(vf_ex.backbone, False)
        _set_trainable(vf_ex.vit, True)
        return
    _set_trainable(vf_ex, True)


def _apply_finetune_guard(model: PPO, datas_dir: Path, stage_id: int, cfg: dict[str, Any], *, from_bc: bool) -> dict[str, Any]:
    freeze_rounds = int(cfg.get("freeze_policy_rounds") or 0)
    state = read_stage_state(datas_dir, stage_id)
    episode = int(state.get("ppo_episode") or 0)
    freeze_actor = bool(from_bc and freeze_rounds > 0 and episode < freeze_rounds)
    pi_ex, vf_ex = _policy_pi_vf_extractors(model.policy)
    share_features = bool(getattr(model.policy, "share_features_extractor", True))
    vf_guard = _vf_schedule(state, episode, cfg, from_bc=from_bc, share_features=share_features)
    vf_stage = str(vf_guard.get("vf_stage") or "all")
    _apply_vf_stage(vf_ex, vf_stage)
    _set_trainable(model.policy.mlp_extractor.value_net, True)
    _set_trainable(model.policy.value_net, True)
    _set_trainable(pi_ex, not freeze_actor)
    _set_trainable(model.policy.mlp_extractor.policy_net, not freeze_actor)
    _set_trainable(model.policy.action_net, not freeze_actor)
    _set_bn_momentum(pi_ex, 0.0 if freeze_actor else 0.1)
    _set_bn_momentum(vf_ex, 0.0 if vf_stage == "frozen" else 0.1)
    return {"freeze_actor": bool(freeze_actor), "vf_stage": vf_stage, "vf_refreeze_until_episode": vf_guard.get("vf_refreeze_until_episode")}


def _set_trainable(module: torch.nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(trainable)


def _set_bn_momentum(module: torch.nn.Module, momentum: float) -> None:
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = float(momentum)


def _maybe_pause_and_exit(ctl: _StopCtl, datas_dir: Path, stage_id: int, model: PPO, cfg: dict[str, Any], size: int, device: torch.device) -> None:
    if not _STOP:
        return
    _finalize_stop_if_needed(ctl, datas_dir, stage_id, size, cfg, model, device)
    now = int(time.time() * 1000)
    state = read_stage_state(datas_dir, stage_id)
    nxt = dict(state)
    nxt["current_phase"] = "ppo"
    nxt["ppo_status"] = "paused"
    nxt["updated_at_ms"] = now
    nxt["last_status_change_at_ms"] = now
    write_stage_state(datas_dir, stage_id, nxt)
    write_runtime(datas_dir, "ppo", stage_id, {"status": "exited", "exit_code": 0})
    raise SystemExit(0)


def _finalize_stop_if_needed(ctl: _StopCtl, datas_dir: Path, stage_id: int, size: int, cfg: dict[str, Any], model: PPO, device: torch.device) -> None:
    if ctl.finalized:
        return
    state = read_stage_state(datas_dir, stage_id)
    episode = int(state.get("ppo_episode") or 0)
    eval_id = f"{episode:09d}"
    last = state.get("last_eval")
    has_eval = isinstance(last, dict) and str(last.get("phase") or "") == "ppo" and str(last.get("eval_id") or "") == eval_id
    if not ctl.dirty and has_eval:
        return
    ctl.finalized = True
    _eval_and_checkpoint(datas_dir, stage_id, size, cfg, model, device, reason="stop")
    ctl.dirty = False


def _eval_and_checkpoint(datas_dir: Path, stage_id: int, size: int, cfg: dict[str, Any], model: PPO, device: torch.device, *, reason: str) -> None:
    episode = int(read_stage_state(datas_dir, stage_id).get("ppo_episode") or 0)
    eval_id = f"{episode:09d}"
    max_steps_cfg = int(cfg.get("eval_max_steps") or 0)
    max_steps = max_steps_cfg if max_steps_cfg > 0 else int(size * size * 40)
    res = _run_eval_latest(datas_dir, stage_id, device, size, eval_id, max_steps, cfg, model)
    ckpt_id = _save_latest_ckpt(datas_dir, stage_id, size, model, episode, res.summary, int(cfg.get("latest_keep") or 10))
    _maybe_update_best_ckpt(datas_dir, stage_id, size, model, episode, res.summary)
    _write_last_eval(datas_dir, stage_id, eval_id, ckpt_id, res.summary)
    write_runtime(datas_dir, "ppo", stage_id, {"last_eval_id": eval_id, "last_checkpoint_id": ckpt_id, "last_eval_reason": reason})
    _append_eval_diag(datas_dir, stage_id, episode, cfg, reason=reason, eval_id=eval_id, ckpt_id=ckpt_id, res=res)


def _run_eval_latest(
    datas_dir: Path,
    stage_id: int,
    device: torch.device,
    size: int,
    eval_id: str,
    max_steps: int,
    cfg: dict[str, Any],
    model: PPO,
) -> EvalResult:
    eval_model = _EvalModel(model.policy).to(device)
    return write_eval_latest(
        datas_dir=datas_dir,
        stage_id=stage_id,
        phase="ppo",
        eval_id=eval_id,
        model=eval_model,
        device=device,
        size=size,
        count=int(cfg.get("eval_rollouts") or 10),
        max_steps=max_steps,
        include_reward=True,
    )


def _append_eval_diag(
    datas_dir: Path, stage_id: int, episode: int, cfg: dict[str, Any], *, reason: str, eval_id: str, ckpt_id: str, res: EvalResult
) -> None:
    items = res.items if isinstance(res.items, list) else []
    short = sum(1 for it in items if isinstance(it, dict) and int(it.get("step_count") or 0) < 50)
    append_diag_line(
        datas_dir,
        stage_id,
        "ppo",
        episode,
        "eval",
        cfg,
        payload={"reason": str(reason), "eval_id": str(eval_id), "checkpoint_id": str(ckpt_id), "summary": res.summary, "items": items, "short_death_count": int(short), "catastrophic": bool(items and short == len(items))},
    )


def _save_latest_ckpt(datas_dir: Path, stage_id: int, size: int, model: PPO, episode: int, summary: dict[str, Any], keep: int) -> str:
    eval_id = f"{episode:09d}"
    ckpt_id = f"ppo_latest_{eval_id}"
    rel = f"checkpoints/latest/{ckpt_id}.pt"
    now_ms = int(time.time() * 1000)
    _write_ckpt_pt(_ppo_dir(datas_dir, stage_id) / rel, model, stage_id, size, episode, now_ms)
    _update_ckpt_latest_index(datas_dir, stage_id, ckpt_id, rel, episode, summary, now_ms, keep)
    return ckpt_id


def _maybe_update_best_ckpt(datas_dir: Path, stage_id: int, size: int, model: PPO, episode: int, summary: dict[str, Any]) -> None:
    idx_path = _ckpt_index_path(datas_dir, stage_id)
    idx = read_index(idx_path)
    cur = idx.get("best")
    prev = cur[0] if isinstance(cur, list) and cur and isinstance(cur[0], dict) else None
    new_item = _ckpt_item(f"ppo_best_{episode:09d}", episode, summary, int(time.time() * 1000), best=True)
    best = pick_best_item([*([prev] if isinstance(prev, dict) else []), new_item])
    if best and best.get("id") == new_item.get("id"):
        _write_ckpt_best_pt(datas_dir, stage_id, size, model, episode)
        idx["best"] = [new_item]
        write_index(idx_path, idx)


def _write_ckpt_best_pt(datas_dir: Path, stage_id: int, size: int, model: PPO, episode: int) -> None:
    best_id = f"ppo_best_{episode:09d}"
    dst_root = _ppo_dir(datas_dir, stage_id) / "checkpoints" / "best"
    dst_root.mkdir(parents=True, exist_ok=True)
    now_ms = int(time.time() * 1000)
    target = dst_root / f"{best_id}.pt"
    _write_ckpt_pt(target, model, stage_id, size, episode, now_ms)
    for p in dst_root.glob("*.pt"):
        if p.name != target.name:
            p.unlink(missing_ok=True)


def _ckpt_item(ckpt_id: str, episode: int, summary: dict[str, Any], now_ms: int, *, best: bool) -> dict[str, Any]:
    bucket = "best" if best else "latest"
    rel = f"checkpoints/{bucket}/{ckpt_id}.pt"
    return {"id": ckpt_id, "episode": int(episode), "coverage": summary.get("coverage"), "step_count": summary.get("steps"), "created_at_ms": int(now_ms), "path": rel, "is_best": bool(best)}


def _update_ckpt_latest_index(datas_dir: Path, stage_id: int, ckpt_id: str, rel_path: str, episode: int, summary: dict[str, Any], now_ms: int, keep: int) -> None:
    idx_path = _ckpt_index_path(datas_dir, stage_id)
    idx = read_index(idx_path)
    item = {"id": ckpt_id, "episode": int(episode), "coverage": summary.get("coverage"), "step_count": summary.get("steps"), "created_at_ms": int(now_ms), "path": rel_path, "is_best": False}
    latest, pruned = push_latest_items(idx.get("latest"), item, keep)
    idx["latest"] = latest
    _prune_ckpt_latest(_ppo_dir(datas_dir, stage_id), pruned)
    write_index(idx_path, idx)


def _prune_ckpt_latest(phase_dir: Path, pruned: list[dict[str, Any]]) -> None:
    for it in pruned:
        rel = str(it.get("path") or "")
        if rel.startswith("checkpoints/latest/"):
            (phase_dir / rel).unlink(missing_ok=True)


def _write_ckpt_pt(path: Path, model: PPO, stage_id: int, size: int, episode: int, now_ms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp_{os.getpid()}_{int(time.time() * 1000)}")
    state = _export_cnnvit_state(model.policy, device=model.device)
    torch.save(
        {
            "schema_version": 1,
            "arch": "cnn_vit_v1",
            "phase": "ppo",
            "stage_id": stage_id,
            "size": size,
            "episode": int(episode),
            "created_at_ms": int(now_ms),
            "model_state": state,
            "sb3_policy_state": model.policy.state_dict(),
        },
        tmp,
    )
    tmp.replace(path)


def _policy_pi_vf_extractors(sb3_policy: Any) -> tuple[Any, Any]:
    if bool(getattr(sb3_policy, "share_features_extractor", True)):
        ex = sb3_policy.features_extractor
        return ex, ex
    return sb3_policy.pi_features_extractor, sb3_policy.vf_features_extractor


def _export_cnnvit_state(sb3_policy: Any, device: torch.device) -> dict[str, torch.Tensor]:
    tmp = CnnVitActorCritic(ModelCfg()).to(device)
    pi_ex, _vf_ex = _policy_pi_vf_extractors(sb3_policy)
    tmp.backbone.load_state_dict(pi_ex.backbone.state_dict())
    tmp.vit.load_state_dict(pi_ex.vit.state_dict())
    tmp.heads.policy.load_state_dict(sb3_policy.action_net.state_dict())
    tmp.heads.value.load_state_dict(sb3_policy.value_net.state_dict())
    return tmp.state_dict()


def _init_weights(model: PPO, datas_dir: Path, stage_id: int, device: torch.device, *, action: str) -> tuple[InitPlan, int | None]:
    plan = select_ppo_init_plan(datas_dir, stage_id, action)
    if plan.ckpt_phase == "ppo":
        plan2, episode = _load_sb3_policy_state(model, datas_dir, plan, device)
        if episode is not None:
            return plan2, episode
    tmp = CnnVitActorCritic(ModelCfg()).to(device)
    plan2, episode = load_weights(tmp, datas_dir, plan, device)
    # 如果是从BC继承，需要��初始化value网络（BC的value权重是随机的）
    from_bc = plan.ckpt_phase == "bc"
    _apply_cnnvit_weights(model.policy, tmp, from_bc=from_bc)
    return plan2, episode


def _load_sb3_policy_state(model: PPO, datas_dir: Path, plan: InitPlan, device: torch.device) -> tuple[InitPlan, int | None]:
    if plan.ckpt_stage_id is None or not plan.ckpt_path:
        return plan, None
    ckpt = datas_dir / "stages" / str(plan.ckpt_stage_id) / "ppo" / str(plan.ckpt_path)
    obj = torch.load(ckpt, map_location=device)
    if not isinstance(obj, dict) or "sb3_policy_state" not in obj:
        return plan, None
    sd = obj.get("sb3_policy_state")
    if not isinstance(sd, dict):
        return plan, None
    strict = True
    try:
        info = model.policy.load_state_dict(sd, strict=strict)
    except RuntimeError:
        info = model.policy.load_state_dict(sd, strict=False)
        strict = False
    missing = list(getattr(info, "missing_keys", []) or [])
    unexpected = list(getattr(info, "unexpected_keys", []) or [])
    mode = f"{plan.mode}_non_strict" if not strict else plan.mode
    episode = int(obj.get("episode") or 0) if isinstance(obj.get("episode"), (int, float)) else None
    out = InitPlan(**{**plan.__dict__, "mode": mode, "missing_keys": missing or None, "unexpected_keys": unexpected or None})
    return out, episode


def _apply_cnnvit_weights(sb3_policy: Any, tmp: CnnVitActorCritic, *, from_bc: bool = False) -> None:
    pi_ex, vf_ex = _policy_pi_vf_extractors(sb3_policy)
    for ex in (pi_ex, vf_ex):
        ex.backbone.load_state_dict(tmp.backbone.state_dict())
        ex.vit.load_state_dict(tmp.vit.state_dict())
    sb3_policy.action_net.load_state_dict(tmp.heads.policy.state_dict())
    # 如果是从BC继承，value网络是随机权重，需要零初始化以避免破坏已学习的policy
    if from_bc:
        _zero_init_value_net(sb3_policy.value_net)
    else:
        sb3_policy.value_net.load_state_dict(tmp.heads.value.state_dict())


def _zero_init_value_net(value_net: torch.nn.Module) -> None:
    """将value网络输出层零初始化，使初始value估计接近0（无偏见起点）"""
    for param in value_net.parameters():
        nn.init.zeros_(param)
    if hasattr(value_net, 'bias') and value_net.bias is not None:
        nn.init.zeros_(value_net.bias)


def _sync_episode_from_ckpt(datas_dir: Path, stage_id: int, plan: InitPlan, episode: int | None) -> None:
    if not str(getattr(plan, "mode", "")).startswith("resume_weights_only"):
        return
    if not isinstance(episode, int) or episode <= 0:
        raise ValueError("resume checkpoint episode 无效")
    state = read_stage_state(datas_dir, stage_id)
    nxt = dict(state)
    nxt["ppo_episode"] = episode
    nxt["updated_at_ms"] = int(time.time() * 1000)
    write_stage_state(datas_dir, stage_id, nxt)


def _bump_episode(datas_dir: Path, stage_id: int, step_count: int) -> int:
    state = read_stage_state(datas_dir, stage_id)
    now = int(time.time() * 1000)
    episode = int(state.get("ppo_episode") or 0) + 1
    nxt = dict(state)
    nxt["ppo_episode"] = episode
    nxt["updated_at_ms"] = now
    write_stage_state(datas_dir, stage_id, nxt)
    write_runtime(datas_dir, "ppo", stage_id, {"last_step_count": int(step_count)})
    return episode


def _write_last_eval(datas_dir: Path, stage_id: int, eval_id: str, ckpt_id: str, summary: dict[str, Any]) -> None:
    state = read_stage_state(datas_dir, stage_id)
    now = int(time.time() * 1000)
    nxt = dict(state)
    nxt["last_eval"] = {"phase": "ppo", "eval_id": eval_id, "checkpoint_id": ckpt_id, "coverage": summary.get("coverage"), "length_max": summary.get("length_max"), "steps": summary.get("steps"), "reward_total": summary.get("reward_total"), "created_at_ms": now}
    nxt["last_eval_coverage"] = summary.get("coverage")
    nxt["updated_at_ms"] = now
    write_stage_state(datas_dir, stage_id, nxt)


def _ensure_running_state(datas_dir: Path, stage_id: int) -> None:
    state = read_stage_state(datas_dir, stage_id)
    if str(state.get("ppo_status") or "") == "running":
        return
    now = int(time.time() * 1000)
    nxt = dict(state)
    nxt["current_phase"] = "ppo"
    nxt["ppo_status"] = "running"
    nxt["updated_at_ms"] = now
    nxt["last_status_change_at_ms"] = now
    write_stage_state(datas_dir, stage_id, nxt)


def _exit_if_backend_unhealthy(datas_dir: Path, stage_id: int, health_url: str) -> None:
    if is_backend_healthy(health_url):
        return
    _pause_and_exit(datas_dir, stage_id, "backend_not_listening")


def _pause_and_exit(datas_dir: Path, stage_id: int, reason: str, *, tb: str | None = None) -> None:
    now = int(time.time() * 1000)
    state = read_stage_state(datas_dir, stage_id)
    nxt = dict(state)
    nxt["current_phase"] = "ppo"
    nxt["ppo_status"] = "paused"
    nxt["ppo_last_error"] = str(reason)
    nxt["updated_at_ms"] = now
    nxt["last_status_change_at_ms"] = now
    write_stage_state(datas_dir, stage_id, nxt)
    patch = {"status": "exited", "exit_code": 0, "last_error": str(reason)}
    if tb:
        patch["last_traceback"] = str(tb)
        print(tb, flush=True)
    write_runtime(datas_dir, "ppo", stage_id, patch)
    raise SystemExit(0)


def _make_env(*, size: int, seed: int, cfg: dict[str, Any]) -> SnakeGymEnv:
    max_steps = int(cfg.get("rollout_max_steps") or 0) or int(size * size * 40)
    return SnakeGymEnv(size=size, seed=seed, max_steps=max_steps)


def _ppo_target_kl(cfg: dict[str, Any]) -> float | None:
    if not bool(int(cfg.get("target_kl_early_stop") or 0)):
        return None
    v = float(cfg.get("target_kl") or 0.0)
    return v if v > 0 else None


def _ppo_rollout_buffer_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    critic_lam = float(cfg.get("critic_gae_lambda") or 1.0)
    return {"critic_gae_lambda": float(critic_lam)}


def _ppo_policy_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    freeze_bn = bool(int(cfg.get("freeze_bn") or 0))
    share_features = bool(int(cfg.get("share_features_extractor") or 0))
    return {
        "features_extractor_class": CnnVitFeaturesExtractor,
        "features_extractor_kwargs": {"cfg": ModelCfg(), "freeze_bn": freeze_bn},
        "net_arch": [],
        "share_features_extractor": share_features,
    }


def _ppo_batch_size(cfg: dict[str, Any], n_steps: int, device: torch.device) -> int:
    batch_size = max(1, int(cfg.get("minibatch_size") or 512))
    batch_size = min(batch_size, int(n_steps))
    return min(batch_size, 2048) if str(device) in ("cuda", "mps") else batch_size


def _make_model(env: SnakeGymEnv, *, cfg: dict[str, Any], device: torch.device) -> PPO:
    n_steps = max(1, int(cfg.get("rollout_steps") or 8192))
    batch_size = _ppo_batch_size(cfg, n_steps, device)
    return PPO(
        "MlpPolicy",
        env,
        rollout_buffer_class=DecoupledGAERolloutBuffer,
        rollout_buffer_kwargs=_ppo_rollout_buffer_kwargs(cfg),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=int(cfg.get("ppo_epochs") or 4),
        learning_rate=float(cfg.get("lr") or 2.5e-4),
        gamma=float(cfg.get("gamma") or 0.99),
        gae_lambda=float(cfg.get("gae_lambda") or 0.95),
        clip_range=float(cfg.get("clip") or 0.2),
        ent_coef=float(cfg.get("ent_coef") or 0.01),
        vf_coef=float(cfg.get("vf_coef") or 0.5),
        max_grad_norm=float(cfg.get("max_grad_norm") or 0.5),
        target_kl=_ppo_target_kl(cfg),
        policy_kwargs=_ppo_policy_kwargs(cfg),
        device=str(device),
        verbose=0,
    )


def _pick_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _ppo_metrics(cb: _StopAndStatsCallback, log: dict[str, Any]) -> dict[str, Any]:
    loss = _pick_float(log.get("train/loss"))
    return {
        "ppo_loss": loss,
        "reward_mean": cb.reward_mean(),
        "step_count": int(cb.step_count),
        "approx_kl": _pick_float(log.get("train/approx_kl")),
        "clip_fraction": _pick_float(log.get("train/clip_fraction")),
        "entropy_loss": _pick_float(log.get("train/entropy_loss")),
        "policy_gradient_loss": _pick_float(log.get("train/policy_gradient_loss")),
        "value_loss": _pick_float(log.get("train/value_loss")),
        "explained_variance": _pick_float(log.get("train/explained_variance")),
    }


def _cfg_snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "lr",
        "gamma",
        "gae_lambda",
        "critic_gae_lambda",
        "clip",
        "ppo_epochs",
        "minibatch_size",
        "vf_coef",
        "ent_coef",
        "max_grad_norm",
        "rollout_steps",
        "rollout_max_steps",
        "eval_max_steps",
        "share_features_extractor",
        "target_kl",
        "target_kl_early_stop",
        "freeze_policy_rounds",
        "vf_freeze_rounds",
        "vf_unfreeze_vit_rounds",
        "vf_collapse_feature_std",
        "vf_collapse_refreeze_rounds",
        "freeze_bn",
        "value_warmup_steps",
        "value_warmup_lr",
        "value_warmup_epochs",
        "value_warmup_batch",
        "value_warmup_max_steps",
    )
    return {k: cfg.get(k) for k in keys}


def _ppo_dir(datas_dir: Path, stage_id: int) -> Path:
    return datas_dir / "stages" / str(stage_id) / "ppo"


def _ckpt_index_path(datas_dir: Path, stage_id: int) -> Path:
    return _ppo_dir(datas_dir, stage_id) / "checkpoints" / "index.json"


def _stage_size(datas_dir: Path, stage_id: int) -> int:
    meta = datas_dir / "stages" / str(stage_id) / "stage.json"
    if not meta.exists():
        return stage_id
    try:
        import json

        obj = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return stage_id
    return int(obj.get("size") or stage_id) if isinstance(obj, dict) else stage_id


def _run_id(stage_id: int) -> str:
    return f"ppo_{stage_id}_{int(time.time() * 1000)}"


def _install_signal_handlers() -> None:
    def on_stop(_sig, _frame):  # noqa: ANN001
        global _STOP
        _STOP = True

    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)
