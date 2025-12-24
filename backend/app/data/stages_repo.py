from __future__ import annotations

from pathlib import Path
from typing import Any

from app.data.json_store import read_json, write_json


def discover_stage_ids(datas_dir: Path) -> list[int]:
    stages_dir = datas_dir / "stages"
    if not stages_dir.exists():
        return [10, 20, 30]

    ids: list[int] = []
    for entry in stages_dir.iterdir():
        if entry.is_dir() and entry.name.isdigit():
            ids.append(int(entry.name))
    ids.sort()
    return ids or [10, 20, 30]


def read_stage_meta(datas_dir: Path, stage_id: int) -> dict[str, Any]:
    path = datas_dir / "stages" / str(stage_id) / "stage.json"
    data = read_json(path, default={"stage_id": stage_id, "size": stage_id})
    return data if isinstance(data, dict) else {"stage_id": stage_id, "size": stage_id}


def read_stage_state(datas_dir: Path, stage_id: int) -> dict[str, Any]:
    path = datas_dir / "stages" / str(stage_id) / "state.json"
    data = read_json(path, default=_empty_state(stage_id))
    return data if isinstance(data, dict) else _empty_state(stage_id)


def write_stage_state(datas_dir: Path, stage_id: int, state: dict[str, Any]) -> None:
    path = datas_dir / "stages" / str(stage_id) / "state.json"
    write_json(path, state)


def read_stage_phase_index(
    datas_dir: Path, stage_id: int, phase: str, kind: str
) -> dict[str, Any]:
    path = datas_dir / "stages" / str(stage_id) / phase / kind / "index.json"
    data = read_json(path, default={"best": [], "latest": []})
    return data if isinstance(data, dict) else {"best": [], "latest": []}


def read_stage_phase_json(datas_dir: Path, stage_id: int, phase: str, rel_path: str) -> Any:
    path = datas_dir / "stages" / str(stage_id) / phase / rel_path
    return read_json(path, default=None)


def _empty_state(stage_id: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage_id": stage_id,
        "size": stage_id,
        "current_phase": None,
        "bc_status": "not_started",
        "ppo_status": "not_started",
        "bc_episode": 0,
        "ppo_episode": 0,
        "last_eval": None,
        "last_eval_coverage": None,
        "created_at_ms": 0,
        "updated_at_ms": 0,
        "last_status_change_at_ms": 0,
    }
