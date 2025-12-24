from __future__ import annotations

import os
import signal
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app.config import bc_worker_cfg, datas_dir as cfg_datas_dir, health_url as cfg_health_url
from app.data.json_store import read_json
from app.data.runtime_repo import write_runtime
from app.data.stages_repo import read_stage_state, write_stage_state
from app.ml.checkpoints import bc_action, load_weights, pick_device, select_bc_init_plan
from app.ml.model import CnnVitActorCritic, ModelCfg
from app.workers.worker_health import is_backend_healthy
from app.workers.worker_index import (
    pick_best_item,
    push_latest_items,
    read_index,
    write_index,
)
from app.workers.worker_metrics import append_metrics_line
from app.workers.worker_bc_train import TrainBatchSpec, rollouts_from_collect_summary, train_bc_episode, updates_per_episode
from app.workers.worker_eval_rollouts import write_eval_latest
from app.workers.worker_train_rollouts import collect_train_rollouts
from app.workers.worker_stop import StopRequested

_STOP = False


@dataclass
class _StopCtl:
    dirty: bool = False
    finalized: bool = False

    def mark_dirty(self) -> None:
        self.dirty = True


def run_bc_worker(stage_id: int) -> None:
    _install_signal_handlers()
    datas_dir = cfg_datas_dir()
    health_url = cfg_health_url()
    run_id = _run_id(stage_id)
    write_runtime(datas_dir, "bc", stage_id, {"status": "running", "pid": os.getpid(), "run_id": run_id})
    _ensure_running_state(datas_dir, stage_id)
    try:
        _loop(datas_dir, health_url, stage_id, bc_worker_cfg())
    except SystemExit:
        raise
    except Exception as e:
        reason = f"crash: {type(e).__name__}: {e}"
        _pause_and_exit(datas_dir, stage_id, reason, tb=traceback.format_exc())


def _finalize_stop_if_needed(
    ctl: _StopCtl,
    datas_dir: Path,
    stage_id: int,
    size: int,
    cfg: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    if ctl.finalized:
        return
    state = read_stage_state(datas_dir, stage_id)
    episode = int(state.get("bc_episode") or 0)
    eval_id = f"{episode:09d}"
    last = state.get("last_eval")
    has_eval = isinstance(last, dict) and str(last.get("phase") or "") == "bc" and str(last.get("eval_id") or "") == eval_id
    if not ctl.dirty and has_eval:
        return
    ctl.finalized = True
    _eval_and_checkpoint(datas_dir, stage_id, size, cfg, model, device, reason="stop")
    ctl.dirty = False


def _loop(datas_dir: Path, health_url: str, stage_id: int, cfg: dict[str, Any]) -> None:
    size = _stage_size(datas_dir, stage_id)
    device = pick_device()
    model = CnnVitActorCritic(ModelCfg()).to(device)
    optimizer = torch.optim.AdamW(_bc_parameters(model), lr=float(cfg.get("lr") or 3e-4))
    plan, ckpt_episode = _init_weights(datas_dir, stage_id, model, device)
    write_runtime(datas_dir, "bc", stage_id, {"device": str(device), "init_plan": plan.to_runtime()})
    _sync_episode_from_ckpt(datas_dir, stage_id, plan, ckpt_episode)
    ctl = _StopCtl()
    on_stop = lambda: _finalize_stop_if_needed(ctl, datas_dir, stage_id, size, cfg, model, device)

    while True:
        _maybe_pause_and_exit(datas_dir, stage_id, on_stop)
        episode_start = int(read_stage_state(datas_dir, stage_id).get("bc_episode") or 0) + 1
        try:
            summary = collect_train_rollouts(datas_dir, stage_id, "bc", episode_start, size, cfg, stop_requested=lambda: _STOP)
            _train_block(datas_dir, stage_id, size, cfg, model, optimizer, summary, device, on_stop, ctl.mark_dirty)
        except StopRequested:
            on_stop()
            _pause_and_exit(datas_dir, stage_id, "stop_requested")
        _maybe_pause_and_exit(datas_dir, stage_id, on_stop)
        _eval_and_checkpoint(datas_dir, stage_id, size, cfg, model, device, reason="round_end")
        ctl.dirty = False
        _exit_if_backend_unhealthy(datas_dir, stage_id, health_url)


def _train_block(datas_dir: Path, stage_id: int, size: int, cfg: dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer, collect_summary: dict[str, Any], device: torch.device, on_stop: Any, mark_dirty: Any) -> None:
    total_steps = int((collect_summary.get("result") or {}).get("step_count") or 0)
    batch_size = int(cfg.get("batch_size") or 1024)
    updates = updates_per_episode(total_steps, batch_size)
    rollouts = rollouts_from_collect_summary(datas_dir, stage_id, "bc", collect_summary)
    spec = TrainBatchSpec(size=size, batch_size=batch_size, updates=updates)
    for _ in range(int(cfg["episodes_per_train"])):
        _maybe_pause_and_exit(datas_dir, stage_id, on_stop)
        bc_loss, update_steps = train_bc_episode(
            model=model,
            optimizer=optimizer,
            device=device,
            rollouts=rollouts,
            spec=spec,
            stop_requested=lambda: _STOP,
        )
        if update_steps:
            mark_dirty()
        _maybe_pause_and_exit(datas_dir, stage_id, on_stop)
        episode = _bump_episode(datas_dir, stage_id)
        append_metrics_line(datas_dir, stage_id, "bc", episode, size, cfg, metrics={"bc_loss": bc_loss})
        _maybe_pause_and_exit(datas_dir, stage_id, on_stop)


def _eval_and_checkpoint(
    datas_dir: Path,
    stage_id: int,
    size: int,
    cfg: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    *,
    reason: str,
) -> None:
    episode = int(read_stage_state(datas_dir, stage_id).get("bc_episode") or 0)
    eval_id = f"{episode:09d}"
    max_steps_cfg = int(cfg.get("eval_max_steps") or 0)
    max_steps = max_steps_cfg if max_steps_cfg > 0 else int(size * size * 40)
    res = write_eval_latest(
        datas_dir=datas_dir,
        stage_id=stage_id,
        phase="bc",
        eval_id=eval_id,
        model=model,
        device=device,
        size=size,
        count=int(cfg.get("eval_rollouts") or 10),
        max_steps=max_steps,
        include_reward=False,
    )
    ckpt_id = _save_latest_ckpt(datas_dir, stage_id, size, model, episode, res.summary, int(cfg.get("latest_keep") or 10))
    _maybe_update_best_ckpt(datas_dir, stage_id, size, model, episode, res.summary)
    _write_last_eval(datas_dir, stage_id, eval_id, ckpt_id, res.summary)
    write_runtime(datas_dir, "bc", stage_id, {"last_eval_id": eval_id, "last_checkpoint_id": ckpt_id, "last_eval_reason": reason})


def _save_latest_ckpt(
    datas_dir: Path,
    stage_id: int,
    size: int,
    model: torch.nn.Module,
    episode: int,
    summary: dict[str, Any],
    keep: int,
) -> str:
    eval_id = f"{episode:09d}"
    ckpt_id = f"bc_latest_{eval_id}"
    rel = f"checkpoints/latest/{ckpt_id}.pt"
    now_ms = int(time.time() * 1000)
    _write_ckpt_pt(_bc_dir(datas_dir, stage_id) / rel, model, stage_id, size, episode, now_ms)
    _update_ckpt_latest_index(datas_dir, stage_id, ckpt_id, rel, episode, summary, now_ms, keep)
    return ckpt_id


def _maybe_update_best_ckpt(
    datas_dir: Path, stage_id: int, size: int, model: torch.nn.Module, episode: int, summary: dict[str, Any]
) -> None:
    idx_path = _ckpt_index_path(datas_dir, stage_id)
    idx = read_index(idx_path)
    cur = idx.get("best")
    prev = cur[0] if isinstance(cur, list) and cur and isinstance(cur[0], dict) else None
    new_item = _ckpt_item(f"bc_best_{episode:09d}", episode, summary, int(time.time() * 1000), best=True)
    best = pick_best_item([*( [prev] if isinstance(prev, dict) else [] ), new_item])
    if best and best.get("id") == new_item.get("id"):
        _write_ckpt_best_pt(datas_dir, stage_id, size, model, episode, summary)
        idx["best"] = [new_item]
        write_index(idx_path, idx)


def _write_ckpt_best_pt(
    datas_dir: Path, stage_id: int, size: int, model: torch.nn.Module, episode: int, summary: dict[str, Any]
) -> None:
    eval_id = f"{episode:09d}"
    best_id = f"bc_best_{eval_id}"
    dst_root = _bc_dir(datas_dir, stage_id) / "checkpoints" / "best"
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
    return {
        "id": ckpt_id,
        "episode": int(episode),
        "coverage": summary.get("coverage"),
        "step_count": summary.get("steps"),
        "created_at_ms": int(now_ms),
        "path": rel,
        "is_best": bool(best),
    }


def _update_ckpt_latest_index(
    datas_dir: Path,
    stage_id: int,
    ckpt_id: str,
    rel_path: str,
    episode: int,
    summary: dict[str, Any],
    now_ms: int,
    keep: int,
) -> None:
    idx_path = _ckpt_index_path(datas_dir, stage_id)
    idx = read_index(idx_path)
    item = {
        "id": ckpt_id,
        "episode": int(episode),
        "coverage": summary.get("coverage"),
        "step_count": summary.get("steps"),
        "created_at_ms": int(now_ms),
        "path": rel_path,
        "is_best": False,
    }
    latest, pruned = push_latest_items(idx.get("latest"), item, keep)
    idx["latest"] = latest
    _prune_ckpt_latest(_bc_dir(datas_dir, stage_id), pruned)
    write_index(idx_path, idx)


def _maybe_pause_and_exit(datas_dir: Path, stage_id: int, on_stop: Any | None = None) -> None:
    if not _STOP:
        return
    if on_stop:
        try:
            on_stop()
        except Exception as e:
            write_runtime(datas_dir, "bc", stage_id, {"last_error": f"finalize_stop_failed: {e}"})
    now = int(time.time() * 1000)
    state = read_stage_state(datas_dir, stage_id)
    nxt = dict(state)
    nxt["current_phase"] = "bc"
    nxt["bc_status"] = "paused"
    nxt["updated_at_ms"] = now
    nxt["last_status_change_at_ms"] = now
    write_stage_state(datas_dir, stage_id, nxt)
    write_runtime(datas_dir, "bc", stage_id, {"status": "exited", "exit_code": 0})
    raise SystemExit(0)


def _prune_ckpt_latest(phase_dir: Path, pruned: list[dict[str, Any]]) -> None:
    for it in pruned:
        rel = str(it.get("path") or "")
        if rel.startswith("checkpoints/latest/"):
            (phase_dir / rel).unlink(missing_ok=True)


def _write_last_eval(datas_dir: Path, stage_id: int, eval_id: str, ckpt_id: str, summary: dict[str, Any]) -> None:
    state = read_stage_state(datas_dir, stage_id)
    now = int(time.time() * 1000)
    nxt = dict(state)
    nxt["last_eval"] = {
        "phase": "bc",
        "eval_id": eval_id,
        "checkpoint_id": ckpt_id,
        "coverage": summary.get("coverage"),
        "length_max": summary.get("length_max"),
        "steps": summary.get("steps"),
        "reward_total": summary.get("reward_total"),
        "created_at_ms": now,
    }
    nxt["last_eval_coverage"] = summary.get("coverage")
    nxt["updated_at_ms"] = now
    write_stage_state(datas_dir, stage_id, nxt)


def _ensure_running_state(datas_dir: Path, stage_id: int) -> None:
    state = read_stage_state(datas_dir, stage_id)
    if str(state.get("bc_status") or "") == "running":
        return
    now = int(time.time() * 1000)
    nxt = dict(state)
    nxt["current_phase"] = "bc"
    nxt["bc_status"] = "running"
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
    nxt["current_phase"] = "bc"
    nxt["bc_status"] = "paused"
    nxt["bc_last_error"] = str(reason)
    nxt["updated_at_ms"] = now
    nxt["last_status_change_at_ms"] = now
    write_stage_state(datas_dir, stage_id, nxt)
    patch = {"status": "exited", "exit_code": 0, "last_error": str(reason)}
    if tb:
        patch["last_traceback"] = str(tb)
        print(tb, flush=True)
    write_runtime(datas_dir, "bc", stage_id, patch)
    raise SystemExit(0)


def _write_ckpt_pt(
    path: Path, model: torch.nn.Module, stage_id: int, size: int, episode: int, now_ms: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp_{os.getpid()}_{int(time.time() * 1000)}")
    torch.save(
        {
            "schema_version": 1,
            "arch": "cnn_vit_v1",
            "phase": "bc",
            "stage_id": stage_id,
            "size": size,
            "episode": int(episode),
            "created_at_ms": int(now_ms),
            "model_state": model.state_dict(),
        },
        tmp,
    )
    tmp.replace(path)


def _init_weights(
    datas_dir: Path, stage_id: int, model: torch.nn.Module, device: torch.device
) -> tuple[Any, int | None]:
    action = bc_action()
    plan = select_bc_init_plan(datas_dir, stage_id, action)
    return load_weights(model, datas_dir, plan, device)


def _sync_episode_from_ckpt(datas_dir: Path, stage_id: int, plan: Any, episode: int | None) -> None:
    if not str(getattr(plan, "mode", "")).startswith("resume_weights_only"):
        return
    if not isinstance(episode, int) or episode <= 0:
        raise ValueError("resume checkpoint episode 无效")
    state = read_stage_state(datas_dir, stage_id)
    nxt = dict(state)
    nxt["bc_episode"] = episode
    nxt["updated_at_ms"] = int(time.time() * 1000)
    write_stage_state(datas_dir, stage_id, nxt)


def _bc_parameters(model: Any) -> list[torch.nn.Parameter]:
    parts = [model.backbone, model.vit, model.heads.policy]
    out: list[torch.nn.Parameter] = []
    for p in parts:
        out.extend(list(p.parameters()))
    return out


def _bump_episode(datas_dir: Path, stage_id: int) -> int:
    state = read_stage_state(datas_dir, stage_id)
    now = int(time.time() * 1000)
    episode = int(state.get("bc_episode") or 0) + 1
    nxt = dict(state)
    nxt["bc_episode"] = episode
    nxt["updated_at_ms"] = now
    write_stage_state(datas_dir, stage_id, nxt)
    write_runtime(datas_dir, "bc", stage_id, {})
    return episode


def _bc_dir(datas_dir: Path, stage_id: int) -> Path:
    return datas_dir / "stages" / str(stage_id) / "bc"


def _ckpt_index_path(datas_dir: Path, stage_id: int) -> Path:
    return _bc_dir(datas_dir, stage_id) / "checkpoints" / "index.json"


def _stage_size(datas_dir: Path, stage_id: int) -> int:
    meta = read_json(datas_dir / "stages" / str(stage_id) / "stage.json", default={})
    return int(meta.get("size") or stage_id) if isinstance(meta, dict) else stage_id


def _run_id(stage_id: int) -> str:
    return f"bc_{stage_id}_{int(time.time() * 1000)}"


def _install_signal_handlers() -> None:
    def on_stop(_sig, _frame):  # noqa: ANN001
        global _STOP
        _STOP = True

    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)
