from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.data.json_store import read_json, write_json


def read_index(path: Path) -> dict[str, Any]:
    data = read_json(path, default={"best": [], "latest": []})
    return data if isinstance(data, dict) else {"best": [], "latest": []}


def write_index(path: Path, idx: dict[str, Any]) -> None:
    write_json(path, {"best": idx.get("best") or [], "latest": idx.get("latest") or []})


def push_latest_items(latest: Any, item: dict[str, Any], keep: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cur = [x for x in (latest or []) if isinstance(x, dict)] if isinstance(latest, list) else []
    out = [item, *cur]
    pruned = out[keep:]
    return out[:keep], pruned


def build_eval_item(eval_id: str, rollout_id: str, summary: dict[str, Any], now_ms: int | None = None) -> dict[str, Any]:
    now = int(now_ms or time.time() * 1000)
    return {
        "id": rollout_id,
        "episode": int(eval_id),
        "coverage": summary.get("coverage"),
        "step_count": summary.get("steps"),
        "created_at_ms": now,
        "path": f"evals/latest/eval_{eval_id}/rollouts/rollout_{rollout_id}.json",
        "is_best": False,
    }


def build_checkpoint_item(ckpt_id: str, eval_id: str, summary: dict[str, Any], now_ms: int | None = None) -> dict[str, Any]:
    now = int(now_ms or time.time() * 1000)
    return {
        "id": ckpt_id,
        "episode": int(eval_id),
        "coverage": summary.get("coverage"),
        "step_count": summary.get("steps"),
        "created_at_ms": now,
        "path": f"checkpoints/latest/{ckpt_id}.pt",
        "is_best": False,
    }


def pick_best_item(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for it in items:
        if not isinstance(it, dict) or not it.get("id"):
            continue
        if best is None or _better(it, best):
            best = it
    return best


def _better(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ac = float(a.get("coverage") or 0.0)
    bc = float(b.get("coverage") or 0.0)
    if ac != bc:
        return ac > bc
    as_ = int(a.get("step_count") or 0)
    bs_ = int(b.get("step_count") or 0)
    if as_ != bs_:
        return as_ < bs_
    return int(a.get("created_at_ms") or 0) > int(b.get("created_at_ms") or 0)
