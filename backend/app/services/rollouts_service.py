from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.config import datas_dir as cfg_datas_dir
from app.data.json_store import read_json
from app.data.stages_repo import read_stage_phase_index

_DATAS_DIR = cfg_datas_dir()


def get_rollout_detail(stage_id: int, phase: str, source: str, rollout_id: str) -> dict[str, Any]:
    phase = _norm_phase(phase)
    source = _norm_source(source)
    rollout_id = str(rollout_id or "").strip()
    if not rollout_id:
        raise ValueError("rollout_id 不能为空")
    path = _find_rollout_path(stage_id, phase, source, rollout_id)
    abs_path = _safe_join(_DATAS_DIR / "stages" / str(stage_id) / phase, path)
    data = read_json(abs_path, default=None)
    if not isinstance(data, dict):
        raise ValueError("rollout 文件格式不正确")
    _assert_pre_action_steps(data)
    return {"ok": True, "stage_id": stage_id, "phase": phase, "source": source, "rollout_id": rollout_id, "rollout": data}


def _find_rollout_path(stage_id: int, phase: str, source: str, rollout_id: str) -> str:
    kind = "evals" if source == "eval" else "manual"
    idx = read_stage_phase_index(_DATAS_DIR, stage_id, phase, kind)
    item = _find_item(idx.get("best"), rollout_id) or _find_item(idx.get("latest"), rollout_id)
    if item is None:
        raise ValueError("未找到该 rollout")
    path = str(item.get("path") or "").strip()
    if not path:
        raise ValueError("rollout path 缺失")
    return path


def _find_item(items: Any, rollout_id: str) -> dict[str, Any] | None:
    if not isinstance(items, list):
        return None
    for it in items:
        if isinstance(it, dict) and str(it.get("id") or "") == rollout_id:
            return it
    return None


def _safe_join(root: Path, rel_path: str) -> Path:
    base = root.resolve()
    out = (base / rel_path).resolve()
    if not str(out).startswith(str(base) + os.sep):
        raise ValueError("非法路径")
    return out


def _assert_pre_action_steps(rollout: dict[str, Any]) -> None:
    meta = rollout.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("rollout meta 缺失")
    state = str(meta.get("step_state") or "").strip().lower()
    if state != "pre_action":
        raise ValueError("rollout step_state 必须为 pre_action")


def _norm_phase(phase: str) -> str:
    v = str(phase or "").strip().lower()
    if v not in ("bc", "ppo"):
        raise ValueError("phase 仅支持 bc/ppo")
    return v


def _norm_source(source: str) -> str:
    v = str(source or "").strip().lower()
    if v not in ("eval", "manual"):
        raise ValueError("source 仅支持 eval/manual")
    return v
