from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app.data.json_store import write_json
from app.config import SnakeEnvCfg
from app.ml.encoding import encode_grid
from app.sim.snake_env import SnakeEnv, eval_meta, eval_seed
from app.workers.worker_fs import clear_dir
from app.workers.worker_index import pick_best_item, read_index, write_index


@dataclass(frozen=True)
class EvalResult:
    summary: dict[str, Any]
    items: list[dict[str, Any]]
    best_item: dict[str, Any]


def write_eval_latest(
    *,
    datas_dir: Path,
    stage_id: int,
    phase: str,
    eval_id: str,
    model: torch.nn.Module,
    device: torch.device,
    size: int,
    count: int,
    max_steps: int,
    include_reward: bool,
) -> EvalResult:
    root = datas_dir / "stages" / str(stage_id) / phase / "evals" / "latest"
    root.mkdir(parents=True, exist_ok=True)
    now = int(time.time() * 1000)
    tmp = root / f".tmp_eval_{eval_id}_{now}"
    clear_dir(tmp)
    rollouts_dir = tmp / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    items, rollouts, agg = _eval_rollouts(stage_id, phase, eval_id, model, device, size, count, max_steps, include_reward)
    _write_rollout_files(rollouts_dir, rollouts)
    write_json(tmp / "summary.json", agg)
    base = root / f"eval_{eval_id}"
    shutil.rmtree(base, ignore_errors=True)
    tmp.rename(base)
    best_item = _pick_best(items)
    best_summary = _pick_rollout_summary(rollouts, best_item)
    _write_eval_index(datas_dir, stage_id, phase, items, best_item, best_summary, base / "rollouts")
    _prune_eval_dirs(root, keep=f"eval_{eval_id}")
    return EvalResult(summary=agg, items=items, best_item=best_item)


def _eval_rollouts(
    stage_id: int,
    phase: str,
    eval_id: str,
    model: torch.nn.Module,
    device: torch.device,
    size: int,
    count: int,
    max_steps: int,
    include_reward: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    now = int(time.time() * 1000)
    stats = _EvalStats(now=now)
    items: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []
    for k in range(1, int(count) + 1):
        item, rollout = _eval_one(stage_id, phase, eval_id, k, model, device, size, max_steps, include_reward, now)
        items.append(item)
        rollouts.append(rollout)
        stats.add(rollout["summary"])
    return items, rollouts, stats.agg()


def _rollout_one(
    stage_id: int,
    phase: str,
    eval_id: str,
    rollout_id: str,
    seed: int,
    model: torch.nn.Module,
    device: torch.device,
    size: int,
    max_steps: int,
    include_reward: bool,
) -> dict[str, Any]:
    env = SnakeEnv(SnakeEnvCfg(size=size, seed=seed, max_steps=max_steps))
    steps, best_len, flags, reward_total = _rollout_steps(env, model, device, size, max_steps, include_reward)
    cov = float(best_len) / float(size * size) if size > 0 else 0.0
    summary = _rollout_summary(cov, best_len, len(steps), flags, reward_total, include_reward)
    meta = eval_meta(stage_id, size, phase, eval_id, rollout_id, seed, max_steps)
    return {"meta": meta, "summary": summary, "steps": steps}


def _pick_action(
    model: torch.nn.Module,
    device: torch.device,
    size: int,
    pre: dict[str, Any],
    *,
    time_left: float,
    hunger: float,
    coverage_norm: float,
) -> str:
    x = encode_grid(
        size=size,
        snake=list(pre.get("snake") or []),
        food=list(pre.get("food") or [-1, -1]),
        dir_name=str(pre.get("dir") or ""),
        device=device,
        time_left=float(time_left),
        hunger=float(hunger),
        coverage_norm=float(coverage_norm),
    )[None, :, :, :]
    with torch.no_grad():
        logits, _v = model(x)
        a = int(torch.argmax(logits, dim=-1).item())
    return ["L", "S", "R"][a] if 0 <= a <= 2 else "S"


def _pick_best(items: list[dict[str, Any]]) -> dict[str, Any]:
    best = pick_best_item(items)
    if not best:
        raise ValueError("eval items empty")
    return best


def _write_eval_index(
    datas_dir: Path,
    stage_id: int,
    phase: str,
    latest: list[dict[str, Any]],
    best_item: dict[str, Any],
    best_summary: dict[str, Any],
    rollouts_dir: Path,
) -> None:
    idx_path = datas_dir / "stages" / str(stage_id) / phase / "evals" / "index.json"
    idx = read_index(idx_path)
    idx["latest"] = latest
    prev = _first_item(idx.get("best"))
    chosen = pick_best_item([x for x in (prev, best_item) if isinstance(x, dict)])
    if chosen and chosen.get("id") == best_item.get("id"):
        best_eval_id = str(best_item.get("episode") or 0).zfill(9)
        _sync_best(datas_dir, stage_id, phase, best_item, rollouts_dir, best_summary, best_eval_id)
        idx["best"] = [_mark_best(best_item)]
    write_index(idx_path, idx)


def _sync_best(
    datas_dir: Path,
    stage_id: int,
    phase: str,
    best_item: dict[str, Any],
    rollouts_dir: Path,
    best_summary: dict[str, Any],
    best_eval_id: str,
) -> None:
    dst_root = datas_dir / "stages" / str(stage_id) / phase / "evals" / "best"
    dst_root.mkdir(parents=True, exist_ok=True)
    tmp = dst_root / f".tmp_eval_{best_eval_id}_{int(time.time() * 1000)}"
    clear_dir(tmp)
    dst_eval = tmp
    (dst_eval / "rollouts").mkdir(parents=True, exist_ok=True)
    src = rollouts_dir / f"rollout_{best_item['id']}.json"
    dst = dst_eval / "rollouts" / src.name
    shutil.copyfile(src, dst)
    write_json(dst_eval / "summary.json", best_summary)
    final = dst_root / f"eval_{best_eval_id}"
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    _prune_eval_dirs(dst_root, keep=f"eval_{best_eval_id}")


def _mark_best(it: dict[str, Any]) -> dict[str, Any]:
    out = dict(it)
    out["is_best"] = True
    out["path"] = str(out.get("path") or "").replace("/latest/", "/best/")
    return out


def _first_item(v: Any) -> dict[str, Any] | None:
    if not isinstance(v, list) or not v or not isinstance(v[0], dict):
        return None
    return v[0]


def _pick_rollout_summary(rollouts: list[dict[str, Any]], best_item: dict[str, Any]) -> dict[str, Any]:
    rid = str(best_item.get("id") or "")
    for r in rollouts:
        meta = r.get("meta") if isinstance(r, dict) else None
        if isinstance(meta, dict) and str(meta.get("rollout_id") or "") == rid:
            s = r.get("summary")
            return s if isinstance(s, dict) else {}
    return {}


def _prune_eval_dirs(root: Path, *, keep: str) -> None:
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if name == keep:
            continue
        if name.startswith("eval_") or name.startswith(".tmp_eval_"):
            shutil.rmtree(entry, ignore_errors=True)


@dataclass
class _EvalStats:
    now: int
    covs: list[float] = None  # type: ignore[assignment]
    steps: list[int] = None  # type: ignore[assignment]
    lens: list[int] = None  # type: ignore[assignment]
    reward_total: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "covs", [])
        object.__setattr__(self, "steps", [])
        object.__setattr__(self, "lens", [])

    def add(self, summary: dict[str, Any]) -> None:
        self.covs.append(float(summary.get("coverage_max") or summary.get("coverage") or 0.0))
        self.steps.append(int(summary.get("steps") or 0))
        self.lens.append(int(summary.get("length_max") or 0))
        self.reward_total += float(summary.get("reward_total") or 0.0)

    def agg(self) -> dict[str, Any]:
        return {
            "coverage": float(sum(self.covs) / float(len(self.covs))) if self.covs else 0.0,
            "coverage_max": max(self.covs) if self.covs else 0.0,
            "steps": int(sum(self.steps) / float(len(self.steps))) if self.steps else 0,
            "length_max": max(self.lens) if self.lens else 0,
            "reward_total": float(self.reward_total),
            "created_at_ms": int(self.now),
        }


def _eval_one(
    stage_id: int,
    phase: str,
    eval_id: str,
    k: int,
    model: torch.nn.Module,
    device: torch.device,
    size: int,
    max_steps: int,
    include_reward: bool,
    now: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = eval_seed(stage_id, phase, eval_id, k)
    rid = f"{eval_id}-{k}"
    rollout = _rollout_one(stage_id, phase, eval_id, rid, seed, model, device, size, max_steps, include_reward)
    s = rollout["summary"]
    item = _eval_index_item(eval_id, rid, s, now)
    return item, rollout


def _eval_index_item(eval_id: str, rid: str, summary: dict[str, Any], now: int) -> dict[str, Any]:
    cov = float(summary.get("coverage_max") or summary.get("coverage") or 0.0)
    sc = int(summary.get("steps") or 0)
    length_max = summary.get("length_max")
    reward_total = summary.get("reward_total")
    return {
        "id": rid,
        "episode": int(eval_id),
        "coverage": cov,
        "step_count": sc,
        "length_max": int(length_max) if isinstance(length_max, (int, float)) else None,
        "reward_total": float(reward_total) if isinstance(reward_total, (int, float)) else None,
        "created_at_ms": int(summary.get("created_at_ms") or now),
        "path": f"evals/latest/eval_{eval_id}/rollouts/rollout_{rid}.json",
        "is_best": False,
    }


def _write_rollout_files(rollouts_dir: Path, rollouts: list[dict[str, Any]]) -> None:
    for r in rollouts:
        rid = str(r.get("meta", {}).get("rollout_id") or "")
        if rid:
            write_json(rollouts_dir / f"rollout_{rid}.json", r)


def _rollout_steps(
    env: SnakeEnv,
    model: torch.nn.Module,
    device: torch.device,
    size: int,
    max_steps: int,
    include_reward: bool,
) -> tuple[list[dict[str, Any]], int, dict[str, bool], float]:
    steps: list[dict[str, Any]] = []
    best_len = len(env.snake)
    reward_total = 0.0
    flags = {"terminated": False, "truncated": False}
    denom = float(max(1, int(env.max_steps)))
    for t in range(int(max_steps)):
        time_left, hunger, coverage_norm = _aux_scalars(env, size=size, denom=denom)
        pre = env.snapshot()
        action = _pick_action(model, device, size, pre, time_left=time_left, hunger=hunger, coverage_norm=coverage_norm)
        res = env.step(action)
        post = env.snapshot()
        reward_total += float(res.reward)
        best_len = max(best_len, len(env.snake))
        steps.append(_step_obj(t, pre, post, action, res, include_reward))
        if res.collision is not None:
            flags["terminated"] = True
            break
        if res.truncated or res.done:
            flags["truncated" if res.truncated else "terminated"] = True
            break
    return steps, best_len, flags, reward_total


def _aux_scalars(env: SnakeEnv, *, size: int, denom: float) -> tuple[float, float, float]:
    time_left = float(max(0.0, min(1.0, (float(env.max_steps) - float(env.steps)) / float(max(1.0, denom)))))
    hunger_steps = max(0, int(env.steps_since_last_eat) - int(env.hunger_grace_steps))
    hunger = float(max(0.0, min(1.0, float(hunger_steps) / float(max(1.0, denom)))))
    coverage = float(len(env.snake)) / float(max(1, int(size) * int(size)))
    return time_left, hunger, coverage


def _step_obj(t: int, pre: dict[str, Any], post: dict[str, Any], action: str, res: Any, include_reward: bool) -> dict[str, Any]:
    out = {
        "t": int(t),
        "snake": pre["snake"],
        "food": pre["food"],
        "dir": pre["dir"],
        "action": action,
        "done": bool(res.done),
        "info": {"ate": bool(res.ate), "collision": res.collision},
        "snake_next": post["snake"],
        "food_next": post["food"],
        "dir_next": post["dir"],
    }
    if include_reward:
        out["reward"] = float(res.reward)
    return out


def _rollout_summary(
    cov: float, best_len: int, step_count: int, flags: dict[str, bool], reward_total: float, include_reward: bool
) -> dict[str, Any]:
    return {
        "coverage": float(cov),
        "coverage_max": float(cov),
        "snake_length_max": int(best_len),
        "length_max": int(best_len),
        "steps": int(step_count),
        "terminated": bool(flags.get("terminated")),
        "truncated": bool(flags.get("truncated")),
        "reward_total": float(reward_total) if include_reward else 0.0,
        "created_at_ms": int(time.time() * 1000),
    }
