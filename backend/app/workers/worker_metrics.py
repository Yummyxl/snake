from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.data.json_store import write_text


def append_metrics_line(
    datas_dir: Path,
    stage_id: int,
    phase: str,
    episode: int,
    size: int,
    cfg: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> None:
    path = datas_dir / "stages" / str(stage_id) / phase / "metrics" / "episodes.jsonl"
    obj = _metrics_obj(stage_id, phase, episode, size, metrics)
    _append_jsonl_capped(path, obj, int(cfg.get("metrics_keep") or 200))


def _metrics_obj(stage_id: int, phase: str, episode: int, size: int, metrics: dict[str, Any] | None) -> dict[str, Any]:
    base = {"stage_id": stage_id, "phase": phase, "episode": episode, "timestamp_ms": int(time.time() * 1000)}
    if phase == "bc":
        loss = None if not isinstance(metrics, dict) else metrics.get("bc_loss")
        return {**base, "bc_loss": float(loss) if isinstance(loss, (int, float)) else None}
    step_count = None if not isinstance(metrics, dict) else metrics.get("step_count")
    reward_mean = None if not isinstance(metrics, dict) else metrics.get("reward_mean")
    loss = None if not isinstance(metrics, dict) else metrics.get("ppo_loss")
    return {
        **base,
        "step_count": int(step_count) if isinstance(step_count, (int, float)) else None,
        "reward_mean": float(reward_mean) if isinstance(reward_mean, (int, float)) else None,
        "ppo_loss": float(loss) if isinstance(loss, (int, float)) else None,
    }


def _append_jsonl_capped(path: Path, obj: dict[str, Any], keep: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines(keepends=True)
    lines.append(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    if len(lines) > keep:
        lines = lines[-keep:]
    write_text(path, "".join(lines))
