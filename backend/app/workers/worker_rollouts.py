from __future__ import annotations

from pathlib import Path
from typing import Any

from app.data.json_store import write_json


def write_rollouts_placeholder(
    rollouts_dir: Path, stage_id: int, phase: str, eval_id: str, count: int, base_summary: dict[str, Any]
) -> list[dict[str, Any]]:
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    for k in range(1, count + 1):
        rollout_id = f"{eval_id}-{k}"
        summary = _vary_summary(base_summary, k)
        path = rollouts_dir / f"rollout_{rollout_id}.json"
        write_json(path, _rollout(stage_id, phase, eval_id, rollout_id, summary))
        out.append({"rollout_id": rollout_id, "summary": summary, "created_at_ms": int(base_summary.get("created_at_ms") or 0)})
    return out


def _rollout(stage_id: int, phase: str, eval_id: str, rollout_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "meta": {"stage_id": stage_id, "phase": phase, "eval_id": eval_id, "rollout_id": rollout_id, "step_state": "pre_action"},
        "summary": summary,
        "steps": [],
    }


def _vary_summary(base: dict[str, Any], k: int) -> dict[str, Any]:
    out = dict(base)
    cov = float(out.get("coverage") or 0.0)
    out["coverage"] = max(0.0, min(1.0, cov - (k - 1) * 0.001))
    steps = int(out.get("steps") or 0)
    out["steps"] = max(0, steps + (k - 1) * 3)
    return out
