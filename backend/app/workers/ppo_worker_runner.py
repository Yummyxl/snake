from __future__ import annotations

import os
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

from app.config import datas_dir as cfg_datas_dir, health_url as cfg_health_url, ppo_worker_cfg, worker_action
from app.data.runtime_repo import write_runtime
from app.data.stages_repo import read_stage_state, write_stage_state
from app.ml.checkpoints import InitPlan, load_weights, pick_device, select_ppo_init_plan
from app.ml.model import CnnVitActorCritic, ModelCfg
from app.ml.sb3_extractor import CnnVitFeaturesExtractor
from app.sim.snake_gym_env import SnakeGymEnv
from app.workers.worker_eval_rollouts import write_eval_latest
from app.workers.worker_health import is_backend_healthy
from app.workers.worker_index import pick_best_item, push_latest_items, read_index, write_index
from app.workers.worker_metrics import append_metrics_line

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

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self.reward_sum += float(torch.as_tensor(rewards).sum().item())
            self.step_count += int(torch.as_tensor(rewards).numel())
        return not self._stop_requested()

    def reward_mean(self) -> float:
        return float(self.reward_sum) / float(max(1, int(self.step_count)))


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

    from_bc = plan.ckpt_phase == "bc"
    if from_bc:
        info = _warmup_value_network(model, env, cfg=cfg, device=device)
        if info:
            write_runtime(datas_dir, "ppo", stage_id, {"value_warmup": info})

    ctl = _StopCtl()
    while True:
        _maybe_pause_and_exit(ctl, datas_dir, stage_id, model, cfg, size, device)
        _train_round(ctl, datas_dir, stage_id, model, cfg, size, from_bc=from_bc)
        if _STOP:
            _maybe_pause_and_exit(ctl, datas_dir, stage_id, model, cfg, size, device)
        _eval_and_checkpoint(datas_dir, stage_id, size, cfg, model, device, reason="round_end")
        ctl.dirty = False
        _exit_if_backend_unhealthy(datas_dir, stage_id, health_url)


def _warmup_value_network(model: PPO, env: SnakeGymEnv, *, cfg: dict[str, Any], device: torch.device) -> dict[str, Any] | None:
    steps = int(cfg.get("value_warmup_steps") or 0)
    epochs = int(cfg.get("value_warmup_epochs") or 0)
    if steps <= 0 or epochs <= 0:
        return None
    gamma = float(cfg.get("gamma") or 0.99)
    max_ep_steps = int(cfg.get("value_warmup_max_steps") or 5000)
    obs, returns = _collect_value_warmup_dataset(model, env, steps=steps, gamma=gamma, max_ep_steps=max_ep_steps)
    loss = _train_value_net(model, obs, returns, cfg=cfg, device=device)
    return {"steps": int(obs.shape[0]), "epochs": int(epochs), "loss": loss}


def _collect_value_warmup_dataset(model: PPO, env: SnakeGymEnv, *, steps: int, gamma: float, max_ep_steps: int) -> tuple[np.ndarray, np.ndarray]:
    obs_out: list[np.ndarray] = []
    ret_out: list[float] = []
    ep_obs: list[np.ndarray] = []
    ep_rewards: list[float] = []
    ep_steps = 0
    max_ep_steps = int(max(1, min(int(max_ep_steps), int(getattr(env, "max_steps", max_ep_steps)))))
    obs, _info = env.reset(seed=(int(time.time() * 1000) % 2_000_000_000))
    while len(obs_out) < int(steps):
        ep_obs.append(obs.copy())
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _info = env.step(action)
        ep_rewards.append(float(reward))
        ep_steps += 1
        if terminated or truncated or ep_steps >= max_ep_steps:
            _append_returns(ep_obs, ep_rewards, gamma, obs_out, ret_out)
            ep_obs, ep_rewards = [], []
            ep_steps = 0
            obs, _info = env.reset(seed=(int(time.time() * 1000) % 2_000_000_000))
    if ep_rewards:
        _append_returns(ep_obs, ep_rewards, gamma, obs_out, ret_out)
    obs_arr = np.stack(obs_out[:steps]).astype(np.float32, copy=False)
    ret_arr = np.asarray(ret_out[:steps], dtype=np.float32).reshape(-1, 1)
    return obs_arr, ret_arr


def _append_returns(ep_obs: list[np.ndarray], ep_rewards: list[float], gamma: float, obs_out: list[np.ndarray], ret_out: list[float]) -> None:
    g = 0.0
    for obs, reward in zip(reversed(ep_obs), reversed(ep_rewards)):
        g = float(reward) + float(gamma) * g
        obs_out.append(obs)
        ret_out.append(float(g))


def _train_value_net(model: PPO, obs: np.ndarray, returns: np.ndarray, *, cfg: dict[str, Any], device: torch.device) -> float | None:
    epochs = int(cfg.get("value_warmup_epochs") or 0)
    if obs.size == 0 or epochs <= 0:
        return None
    lr = float(cfg.get("value_warmup_lr") or 1e-4)
    batch = int(cfg.get("value_warmup_batch") or 1024)
    idx = np.arange(int(obs.shape[0]))
    share = bool(getattr(model.policy, "share_features_extractor", True))
    params = list(model.policy.value_net.parameters())
    if not share:
        params = [*model.policy.vf_features_extractor.parameters(), *model.policy.mlp_extractor.value_net.parameters(), *params]
    optimizer = torch.optim.AdamW(params, lr=lr)
    criterion = nn.MSELoss()
    last_loss: float | None = None
    model.policy.train()
    for _ in range(int(epochs)):
        np.random.shuffle(idx)
        for start in range(0, int(idx.size), int(batch)):
            j = idx[start : start + int(batch)]
            obs_t = torch.from_numpy(obs[j]).float().to(device)
            ret_t = torch.from_numpy(returns[j]).float().to(device)
            if share:
                with torch.no_grad():
                    features = model.policy.extract_features(obs_t)
                    latent_vf = model.policy.mlp_extractor(features)[1]
                pred = model.policy.value_net(latent_vf)
            else:
                pred = model.policy.predict_values(obs_t)
            loss = criterion(pred, ret_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
    model.policy.eval()
    return last_loss


def _train_round(ctl: _StopCtl, datas_dir: Path, stage_id: int, model: PPO, cfg: dict[str, Any], size: int, *, from_bc: bool) -> None:
    if int(cfg.get("episodes_per_train") or 1) <= 0:
        return
    for _k in range(int(cfg.get("episodes_per_train") or 1)):
        _apply_finetune_guard(model, datas_dir, stage_id, cfg, from_bc=from_bc)
        cb = _StopAndStatsCallback(stop_requested=lambda: _STOP)
        model.learn(total_timesteps=max(1, int(cfg.get("rollout_steps") or 8192)), reset_num_timesteps=False, callback=cb)
        log = model.logger.name_to_value
        loss = _pick_float(log.get("train/loss"))
        if cb.step_count > 0:
            ctl.mark_dirty()
        episode = _bump_episode(datas_dir, stage_id, cb.step_count)
        metrics = {
            "ppo_loss": loss,
            "reward_mean": cb.reward_mean(),
            "step_count": cb.step_count,
            "approx_kl": _pick_float(log.get("train/approx_kl")),
            "clip_fraction": _pick_float(log.get("train/clip_fraction")),
            "entropy_loss": _pick_float(log.get("train/entropy_loss")),
            "policy_gradient_loss": _pick_float(log.get("train/policy_gradient_loss")),
            "value_loss": _pick_float(log.get("train/value_loss")),
            "explained_variance": _pick_float(log.get("train/explained_variance")),
        }
        append_metrics_line(datas_dir, stage_id, "ppo", episode, size, cfg, metrics=metrics)
        if _STOP:
            return


def _apply_finetune_guard(model: PPO, datas_dir: Path, stage_id: int, cfg: dict[str, Any], *, from_bc: bool) -> None:
    freeze_rounds = int(cfg.get("freeze_policy_rounds") or 0)
    episode = int(read_stage_state(datas_dir, stage_id).get("ppo_episode") or 0)
    freeze = bool(from_bc and freeze_rounds > 0 and episode < freeze_rounds)
    pi_ex, vf_ex = _policy_pi_vf_extractors(model.policy)
    _set_trainable(vf_ex, True)
    _set_trainable(model.policy.mlp_extractor.value_net, True)
    _set_trainable(model.policy.value_net, True)
    _set_trainable(pi_ex, not freeze)
    _set_trainable(model.policy.mlp_extractor.policy_net, not freeze)
    _set_trainable(model.policy.action_net, not freeze)
    _set_bn_momentum(pi_ex, 0.0 if freeze else 0.1)
    _set_bn_momentum(vf_ex, 0.1)


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
    eval_model = _EvalModel(model.policy).to(device)
    res = write_eval_latest(
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
    ckpt_id = _save_latest_ckpt(datas_dir, stage_id, size, model, episode, res.summary, int(cfg.get("latest_keep") or 10))
    _maybe_update_best_ckpt(datas_dir, stage_id, size, model, episode, res.summary)
    _write_last_eval(datas_dir, stage_id, eval_id, ckpt_id, res.summary)
    write_runtime(datas_dir, "ppo", stage_id, {"last_eval_id": eval_id, "last_checkpoint_id": ckpt_id, "last_eval_reason": reason})


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


def _make_model(env: SnakeGymEnv, *, cfg: dict[str, Any], device: torch.device) -> PPO:
    freeze_bn = bool(int(cfg.get("freeze_bn") or 0))
    share_features = bool(int(cfg.get("share_features_extractor") or 0))
    policy_kwargs = {
        "features_extractor_class": CnnVitFeaturesExtractor,
        "features_extractor_kwargs": {"cfg": ModelCfg(), "freeze_bn": freeze_bn},
        "net_arch": [],
        "share_features_extractor": share_features,
    }
    target_kl = float(cfg.get("target_kl") or 0.0)
    n_steps = max(1, int(cfg.get("rollout_steps") or 8192))
    batch_size = max(1, int(cfg.get("minibatch_size") or 512))
    batch_size = min(batch_size, n_steps)
    batch_size = min(batch_size, 2048) if str(device) in ("cuda", "mps") else batch_size
    return PPO(
        "MlpPolicy",
        env,
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
        target_kl=target_kl if target_kl > 0 else None,
        policy_kwargs=policy_kwargs,
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
