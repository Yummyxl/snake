from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from app.data.json_store import write_json
from app.data.runtime_repo import write_runtime
from app.sim.snake_rollout_gen import generate_rollout, now_ms
from app.workers.worker_fs import clear_dir
from app.workers.worker_stop import StopRequested


def collect_train_rollouts(
    datas_dir: Path,
    stage_id: int,
    phase: str,
    episode_start: int,
    size: int,
    cfg: dict[str, Any],
    *,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    train_dir = datas_dir / "stages" / str(stage_id) / phase / "train_rollouts"
    rollouts_dir = train_dir / "rollouts"
    clear_dir(rollouts_dir)
    collect_id = f"{episode_start:09d}"
    write_runtime(datas_dir, phase, stage_id, {"status": "collecting", "collect_id": collect_id})
    target = _target_cfg(cfg, size)
    items, rejected = _collect_rollout_files(rollouts_dir, stage_id, phase, size, collect_id, target, stop_requested)
    summary = _write_summary(train_dir, stage_id, phase, collect_id, episode_start, target, items, rejected)
    _write_index(train_dir, collect_id, episode_start, summary, rejected)
    write_runtime(datas_dir, phase, stage_id, {"last_collect_id": collect_id, "last_collect_coverage_max": summary["result"]["coverage_max"]})
    return summary


def _target_cfg(cfg: dict[str, Any], size: int) -> dict[str, Any]:
    max_steps = int(cfg.get("train_rollout_max_steps") or (size * size * 400))
    return {
        "rollout_count": int(cfg.get("train_rollout_count") or 2048),
        "min_rollout_coverage": float(cfg.get("train_min_rollout_coverage") or 0.70),
        "max_steps": max_steps,
        "max_attempts": int(cfg.get("train_max_attempts") or 100000),
        "steps_preview": max_steps,
    }


def _collect_rollout_files(
    rollouts_dir: Path,
    stage_id: int,
    phase: str,
    size: int,
    collect_id: str,
    target: dict[str, Any],
    stop_requested: Callable[[], bool] | None,
) -> tuple[list[dict[str, Any]], int]:
    want = int(target["rollout_count"])
    min_cov = float(target["min_rollout_coverage"])
    rejected = 0
    items: list[dict[str, Any]] = []
    for attempt in range(1, int(target["max_attempts"]) + 1):
        if stop_requested and stop_requested():
            raise StopRequested("stop requested during train rollout collection")
        if len(items) >= want:
            return items, rejected
        rid = f"{collect_id}-{attempt}"
        seed = _train_rollout_seed(stage_id, phase, collect_id, attempt)
        obj = _rollout_obj(stage_id, phase, size, collect_id, rid, seed, target, stop_requested)
        cov = float(obj["summary"].get("coverage_max") or 0.0)
        if cov < min_cov:
            rejected += 1
            continue
        path = rollouts_dir / f"rollout_{rid}.json"
        write_json(path, obj)
        items.append(_rollout_index_item(rid, obj["summary"], now_ms()))
    raise RuntimeError(f"collect train rollouts failed: accepted={len(items)}/{want}, rejected={rejected}")


def _rollout_obj(
    stage_id: int,
    phase: str,
    size: int,
    collect_id: str,
    rollout_id: str,
    seed: int,
    target: dict[str, Any],
    stop_requested: Callable[[], bool] | None,
) -> dict[str, Any]:
    rollout = generate_rollout(
        size=size,
        seed=seed,
        coverage_target=float(target["min_rollout_coverage"]),
        max_steps=int(target["max_steps"]),
        steps_saved=int(target["steps_preview"]),
        reject_below=False,
        max_attempts=1,
        stop_requested=stop_requested,
    )
    return {
        "meta": _meta(stage_id, phase, size, collect_id, rollout_id, seed, target, rollout.sim),
        "summary": rollout.summary,
        "steps": rollout.steps,
    }


def _train_rollout_seed(stage_id: int, phase: str, collect_id: str, attempt: int) -> int:
    phase_off = 0 if str(phase) == "bc" else 500
    return int(stage_id) * 1_000_000_000 + int(collect_id) * 1000 + int(attempt) + int(phase_off)


def _meta(
    stage_id: int,
    phase: str,
    size: int,
    collect_id: str,
    rollout_id: str,
    seed: int,
    target: dict[str, Any],
    sim: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage_id": stage_id,
        "size": size,
        "phase": phase,
        "kind": "train",
        "coord": "xy",
        "action_space": "relative_lsr",
        "action_map": ["L", "S", "R"],
        "step_state": "pre_action",
        "collect_id": collect_id,
        "rollout_id": rollout_id,
        "seed": seed,
        "created_at_ms": now_ms(),
        "max_steps": int(target["max_steps"]),
        "sim": sim,
    }


def _rollout_index_item(rollout_id: str, summary: dict[str, Any], created_at_ms: int) -> dict[str, Any]:
    return {
        "id": rollout_id,
        "coverage": summary.get("coverage_max"),
        "step_count": summary.get("steps"),
        "created_at_ms": created_at_ms,
        "path": f"train_rollouts/rollouts/rollout_{rollout_id}.json",
    }


def _write_summary(
    train_dir: Path,
    stage_id: int,
    phase: str,
    collect_id: str,
    episode_start: int,
    target: dict[str, Any],
    items: list[dict[str, Any]],
    rejected: int,
) -> dict[str, Any]:
    covs = [float(it.get("coverage") or 0.0) for it in items]
    steps = [int(it.get("step_count") or 0) for it in items]
    summary = {
        "meta": {
            "schema_version": 1,
            "stage_id": stage_id,
            "phase": phase,
            "kind": "train_collect",
            "collect_id": collect_id,
            "episode_start": episode_start,
            "created_at_ms": now_ms(),
            "target": dict(target),
        },
        "result": {
            "accepted": len(items),
            "rejected": rejected,
            "coverage_min": min(covs) if covs else None,
            "coverage_max": max(covs) if covs else None,
            "step_count": int(sum(steps)),
        },
        "rollouts": items,
    }
    write_json(train_dir / "summary.json", summary)
    return summary


def _write_index(train_dir: Path, collect_id: str, episode_start: int, summary: dict[str, Any], rejected: int) -> None:
    write_json(
        train_dir / "index.json",
        {
            "schema_version": 1,
            "latest": {
                "collect_id": collect_id,
                "episode_start": episode_start,
                "accepted": int(summary["result"]["accepted"]),
                "rejected": rejected,
                "created_at_ms": int(summary["meta"]["created_at_ms"]),
                "path": "train_rollouts/summary.json",
            },
        },
    )
