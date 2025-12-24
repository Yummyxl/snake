from __future__ import annotations

import shutil
import time
from pathlib import Path

from app.data.json_store import write_json, write_text


def reset_stage(datas_dir: Path, stage_id: int) -> None:
    now_ms = int(time.time() * 1000)
    root = datas_dir / "stages" / str(stage_id)
    shutil.rmtree(root, ignore_errors=True)
    _write_skeleton(root, stage_id, now_ms)


def reset_stage_phase(datas_dir: Path, stage_id: int, phase: str) -> None:
    if phase not in ("bc", "ppo"):
        raise ValueError("phase must be bc/ppo")
    root = datas_dir / "stages" / str(stage_id)
    if not (root / "state.json").exists():
        reset_stage(datas_dir, stage_id)
        return
    phase_dir = root / phase
    shutil.rmtree(phase_dir, ignore_errors=True)
    _write_phase_skeleton(phase_dir)


def _write_skeleton(root: Path, stage_id: int, now_ms: int) -> None:
    size = stage_id
    write_json(root / "stage.json", {"stage_id": stage_id, "size": size})
    write_json(root / "state.json", _empty_state(stage_id, size, now_ms))
    for phase in ("bc", "ppo"):
        _write_phase_skeleton(root / phase)


def _write_phase_skeleton(phase_dir: Path) -> None:
    train_dir = phase_dir / "train_rollouts"
    (train_dir / "rollouts").mkdir(parents=True, exist_ok=True)
    write_json(train_dir / "index.json", {"schema_version": 1, "latest": None})
    write_json(train_dir / "summary.json", {"schema_version": 1, "meta": None, "result": None, "rollouts": []})
    _mkdir_phase_outputs(phase_dir)
    write_text(phase_dir / "metrics" / "episodes.jsonl", "")
    write_json(phase_dir / "evals" / "index.json", {"best": [], "latest": []})
    write_json(phase_dir / "checkpoints" / "index.json", {"best": [], "latest": []})
    write_json(phase_dir / "manual" / "index.json", {"best": [], "latest": []})


def _mkdir_phase_outputs(phase_dir: Path) -> None:
    for base in ("evals", "checkpoints", "manual"):
        for bucket in ("latest", "best"):
            (phase_dir / base / bucket).mkdir(parents=True, exist_ok=True)


def _empty_state(stage_id: int, size: int, now_ms: int) -> dict:
    return {
        "schema_version": 1,
        "stage_id": stage_id,
        "size": size,
        "current_phase": None,
        "bc_status": "not_started",
        "ppo_status": "not_started",
        "bc_episode": 0,
        "ppo_episode": 0,
        "last_eval": None,
        "last_eval_coverage": None,
        "created_at_ms": now_ms,
        "updated_at_ms": now_ms,
        "last_status_change_at_ms": now_ms,
    }
