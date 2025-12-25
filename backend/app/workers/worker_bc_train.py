from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from app.config import snake_reward_cfg
from app.data.json_store import read_json
from app.ml.encoding import encode_grid


@dataclass(frozen=True)
class RolloutRef:
    path: Path
    step_count: int


@dataclass(frozen=True)
class TrainBatchSpec:
    size: int
    batch_size: int
    updates: int


def train_bc_episode(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rollouts: list[RolloutRef],
    spec: TrainBatchSpec,
    stop_requested: Callable[[], bool],
) -> tuple[float, int]:
    model.train()
    cache = _RolloutCache(cap=max(1, len(rollouts)))
    sampler = _WeightedSampler(rollouts, seed=random.randint(1, 1_000_000))
    total = 0.0
    steps = 0
    for _ in range(int(spec.updates)):
        if stop_requested():
            break
        loss = _train_update(model, optimizer, device, sampler, cache, spec)
        total += float(loss)
        steps += 1
    return ((total / float(max(1, steps))) if steps else 0.0), int(steps)


def rollouts_from_collect_summary(datas_dir: Path, stage_id: int, phase: str, summary: dict[str, Any]) -> list[RolloutRef]:
    base = datas_dir / "stages" / str(stage_id) / phase
    out: list[RolloutRef] = []
    for it in (summary.get("rollouts") or []):
        if not isinstance(it, dict):
            continue
        rel = str(it.get("path") or "")
        if not rel:
            continue
        out.append(RolloutRef(path=(base / rel), step_count=int(it.get("step_count") or 0)))
    return out


def updates_per_episode(total_steps: int, batch_size: int) -> int:
    if batch_size <= 0:
        return 1
    return max(1, int((int(total_steps) + int(batch_size) - 1) // int(batch_size)))


def _train_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sampler: _WeightedSampler,
    cache: _RolloutCache,
    spec: TrainBatchSpec,
) -> torch.Tensor:
    x, y = _sample_batch(device, sampler, cache, spec)
    logits, _v = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach()


def _sample_batch(
    device: torch.device, sampler: _WeightedSampler, cache: _RolloutCache, spec: TrainBatchSpec
) -> tuple[torch.Tensor, torch.Tensor]:
    picks = sampler.sample(int(spec.batch_size))
    by_rollout: dict[Path, list[int]] = {}
    for ref, step_idx in picks:
        by_rollout.setdefault(ref.path, []).append(step_idx)
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    for path, step_indices in by_rollout.items():
        steps = cache.steps(path)
        for i in step_indices:
            s = steps[i]
            xs.append(
                encode_grid(
                    size=int(spec.size),
                    snake=list(s.get("snake") or []),
                    food=list(s.get("food") or [-1, -1]),
                    dir_name=str(s.get("dir") or ""),
                    device=device,
                    time_left=float(s.get("_time_left") or 0.0),
                    hunger=float(s.get("_hunger") or 0.0),
                    coverage_norm=float(s.get("_coverage_norm") or 0.0),
                )
            )
            ys.append(_action_id(s.get("action")))
    return torch.stack(xs, dim=0), torch.tensor(ys, device=device, dtype=torch.long)


def _action_id(v: Any) -> int:
    m = {"L": 0, "S": 1, "R": 2}
    return int(m.get(str(v).strip().upper(), 1))


class _RolloutCache:
    def __init__(self, *, cap: int) -> None:
        self._cap = int(max(1, cap))
        self._cache: dict[Path, list[dict[str, Any]]] = {}
        self._order: list[Path] = []

    def steps(self, path: Path) -> list[dict[str, Any]]:
        if path in self._cache:
            return self._cache[path]
        obj = read_json(path, default={})
        if not isinstance(obj, dict):
            raise ValueError(f"rollout json invalid: {path}")
        meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
        steps = obj.get("steps")
        out = [s for s in (steps or []) if isinstance(s, dict)]
        if not out:
            raise ValueError(f"rollout steps missing: {path}")
        self._put(path, _with_aux_features(meta, out))
        return self._cache[path]

    def _put(self, path: Path, steps: list[dict[str, Any]]) -> None:
        self._cache[path] = steps
        self._order.append(path)
        while len(self._order) > self._cap:
            evict = self._order.pop(0)
            self._cache.pop(evict, None)


class _WeightedSampler:
    def __init__(self, rollouts: list[RolloutRef], *, seed: int) -> None:
        self._rng = random.Random(int(seed))
        self._items = [r for r in rollouts if r.step_count > 0]
        if not self._items:
            raise ValueError("no rollouts to sample")
        self._total = sum(int(r.step_count) for r in self._items)

    def sample(self, n: int) -> list[tuple[RolloutRef, int]]:
        out: list[tuple[RolloutRef, int]] = []
        for _ in range(int(n)):
            ref = self._pick_rollout()
            idx = self._rng.randrange(int(ref.step_count))
            out.append((ref, idx))
        return out

    def _pick_rollout(self) -> RolloutRef:
        r = self._rng.randrange(int(self._total))
        acc = 0
        for it in self._items:
            acc += int(it.step_count)
            if r < acc:
                return it
        return self._items[-1]


def _with_aux_features(meta: dict[str, Any], steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grace = int(snake_reward_cfg().get("hunger_grace_steps") or 0)
    max_steps = int(meta.get("max_steps") or 0)
    if max_steps <= 0:
        last_t = max((int(s.get("t") or 0) for s in steps if isinstance(s, dict)), default=len(steps) - 1)
        max_steps = max(1, int(last_t) + 1)
    denom = float(max(1, max_steps))
    size = int(meta.get("size") or meta.get("stage_id") or 0)
    board = float(max(1, int(size) * int(size)))
    since = 0
    for idx, s in enumerate(steps):
        t = int(s.get("t") if isinstance(s.get("t"), (int, float)) else idx)
        s["_time_left"] = float(max(0.0, min(1.0, (float(max_steps) - float(t)) / denom)))
        hunger_steps = max(0, int(since) - int(grace))
        s["_hunger"] = float(max(0.0, min(1.0, float(hunger_steps) / denom)))
        snake = s.get("snake") if isinstance(s.get("snake"), list) else []
        s["_coverage_norm"] = float(len(snake)) / board
        info = s.get("info") if isinstance(s.get("info"), dict) else {}
        since = 0 if bool(info.get("ate")) else (since + 1)
    return steps
