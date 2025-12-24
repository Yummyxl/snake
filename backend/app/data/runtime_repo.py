from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.data.json_store import read_json, write_json


def runtime_path(datas_dir: Path, phase: str, stage_id: int) -> Path:
    return datas_dir / ".run" / f"{phase}_{stage_id}.runtime.json"


def pid_path(datas_dir: Path, phase: str, stage_id: int) -> Path:
    return datas_dir / ".run" / f"{phase}_{stage_id}.pid"


def write_runtime(datas_dir: Path, phase: str, stage_id: int, patch: dict[str, Any]) -> dict[str, Any]:
    now_ms = int(time.time() * 1000)
    out = dict(read_runtime(datas_dir, phase, stage_id))
    out.update(patch)
    out.setdefault("schema_version", 1)
    out.setdefault("stage_id", stage_id)
    out.setdefault("phase", phase)
    out["heartbeat_at_ms"] = now_ms
    write_json(runtime_path(datas_dir, phase, stage_id), out)
    return out


def read_runtime(datas_dir: Path, phase: str, stage_id: int) -> dict[str, Any]:
    path = runtime_path(datas_dir, phase, stage_id)
    data = read_json(path, default={})
    return data if isinstance(data, dict) else {}


def clear_stage_runtime(datas_dir: Path, stage_id: int) -> None:
    for phase in ("bc", "ppo"):
        _unlink(pid_path(datas_dir, phase, stage_id))
        _unlink(runtime_path(datas_dir, phase, stage_id))


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
