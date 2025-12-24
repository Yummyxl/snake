from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
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
        latent_pi, latent_vf = self.policy.mlp_extractor(features)
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
        _pause_and_exit(datas_dir, stage_id, f"crash: {e}")


def _loop(datas_dir: Path, health_url: str, stage_id: int, cfg: dict[str, Any]) -> None:
    size = _stage_size(datas_dir, stage_id)
    device = pick_device()
    action = worker_action()
    env = _make_env(size=size, seed=stage_id, cfg=cfg)
    model = _make_model(env, cfg=cfg, device=device)
    plan, ckpt_episode = _init_weights(model, datas_dir, stage_id, device, action=action)
    write_runtime(datas_dir, "ppo", stage_id, {"device": str(device), "init_plan": plan.to_runtime()})
    _sync_episode_from_ckpt(datas_dir, stage_id, plan, ckpt_episode)
    ctl = _StopCtl()
    while True:
        _maybe_pause_and_exit(ctl, datas_dir, stage_id, model, cfg, size, device)
        _train_round(ctl, datas_dir, stage_id, model, cfg, size)
        if _STOP:
            _maybe_pause_and_exit(ctl, datas_dir, stage_id, model, cfg, size, device)
        _eval_and_checkpoint(datas_dir, stage_id, size, cfg, model, device, reason="round_end")
        ctl.dirty = False
        _exit_if_backend_unhealthy(datas_dir, stage_id, health_url)


def _train_round(ctl: _StopCtl, datas_dir: Path, stage_id: int, model: PPO, cfg: dict[str, Any], size: int) -> None:
    if int(cfg.get("episodes_per_train") or 1) <= 0:
        return
    for _k in range(int(cfg.get("episodes_per_train") or 1)):
        cb = _StopAndStatsCallback(stop_requested=lambda: _STOP)
        model.learn(total_timesteps=int(cfg.get("rollout_steps") or 8192), reset_num_timesteps=False, callback=cb)
        loss = _pick_float(model.logger.name_to_value.get("train/loss"))
        if cb.step_count > 0:
            ctl.mark_dirty()
        episode = _bump_episode(datas_dir, stage_id, cb.step_count)
        append_metrics_line(datas_dir, stage_id, "ppo", episode, size, cfg, metrics={"ppo_loss": loss, "reward_mean": cb.reward_mean(), "step_count": cb.step_count})
        if _STOP:
            return


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
    torch.save({"schema_version": 1, "arch": "cnn_vit_v1", "phase": "ppo", "stage_id": stage_id, "size": size, "episode": int(episode), "created_at_ms": int(now_ms), "model_state": state}, tmp)
    tmp.replace(path)


def _export_cnnvit_state(sb3_policy: Any, device: torch.device) -> dict[str, torch.Tensor]:
    tmp = CnnVitActorCritic(ModelCfg()).to(device)
    tmp.backbone.load_state_dict(sb3_policy.features_extractor.backbone.state_dict())
    tmp.vit.load_state_dict(sb3_policy.features_extractor.vit.state_dict())
    tmp.heads.policy.load_state_dict(sb3_policy.action_net.state_dict())
    tmp.heads.value.load_state_dict(sb3_policy.value_net.state_dict())
    return tmp.state_dict()


def _init_weights(model: PPO, datas_dir: Path, stage_id: int, device: torch.device, *, action: str) -> tuple[InitPlan, int | None]:
    plan = select_ppo_init_plan(datas_dir, stage_id, action)
    tmp = CnnVitActorCritic(ModelCfg()).to(device)
    plan2, episode = load_weights(tmp, datas_dir, plan, device)
    _apply_cnnvit_weights(model.policy, tmp)
    return plan2, episode


def _apply_cnnvit_weights(sb3_policy: Any, tmp: CnnVitActorCritic) -> None:
    sb3_policy.features_extractor.backbone.load_state_dict(tmp.backbone.state_dict())
    sb3_policy.features_extractor.vit.load_state_dict(tmp.vit.state_dict())
    sb3_policy.action_net.load_state_dict(tmp.heads.policy.state_dict())
    sb3_policy.value_net.load_state_dict(tmp.heads.value.state_dict())


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


def _pause_and_exit(datas_dir: Path, stage_id: int, reason: str) -> None:
    now = int(time.time() * 1000)
    state = read_stage_state(datas_dir, stage_id)
    nxt = dict(state)
    nxt["current_phase"] = "ppo"
    nxt["ppo_status"] = "paused"
    nxt["updated_at_ms"] = now
    nxt["last_status_change_at_ms"] = now
    write_stage_state(datas_dir, stage_id, nxt)
    write_runtime(datas_dir, "ppo", stage_id, {"status": "exited", "exit_code": 0, "last_error": reason})
    raise SystemExit(0)


def _make_env(*, size: int, seed: int, cfg: dict[str, Any]) -> SnakeGymEnv:
    max_steps = int(cfg.get("rollout_max_steps") or 0) or int(size * size * 8)
    return SnakeGymEnv(size=size, seed=seed, max_steps=max_steps)


def _make_model(env: SnakeGymEnv, *, cfg: dict[str, Any], device: torch.device) -> PPO:
    policy_kwargs = {"features_extractor_class": CnnVitFeaturesExtractor, "features_extractor_kwargs": {"cfg": ModelCfg()}, "net_arch": []}
    return PPO("MlpPolicy", env, n_steps=int(cfg.get("rollout_steps") or 8192), batch_size=int(cfg.get("minibatch_size") or 512), n_epochs=int(cfg.get("ppo_epochs") or 4), learning_rate=float(cfg.get("lr") or 2.5e-4), gamma=float(cfg.get("gamma") or 0.99), gae_lambda=float(cfg.get("gae_lambda") or 0.95), clip_range=float(cfg.get("clip") or 0.2), ent_coef=float(cfg.get("ent_coef") or 0.01), vf_coef=float(cfg.get("vf_coef") or 0.5), max_grad_norm=float(cfg.get("max_grad_norm") or 0.5), policy_kwargs=policy_kwargs, device=str(device), verbose=0)


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
