from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app.config import bc_action as cfg_bc_action
from app.data.json_store import read_json
from app.data.stages_repo import discover_stage_ids


@dataclass(frozen=True)
class InitPlan:
    mode: str
    ckpt_stage_id: int | None = None
    ckpt_phase: str | None = None
    ckpt_id: str | None = None
    ckpt_path: str | None = None
    missing_keys: list[str] | None = None
    unexpected_keys: list[str] | None = None

    def to_runtime(self) -> dict[str, Any]:
        out = {"mode": self.mode}
        if self.ckpt_stage_id is not None:
            out["from_stage_id"] = self.ckpt_stage_id
        if self.ckpt_id:
            out["ckpt"] = {"id": self.ckpt_id, "phase": self.ckpt_phase, "path": self.ckpt_path}
        if self.missing_keys:
            out["missing_keys"] = list(self.missing_keys)
        if self.unexpected_keys:
            out["unexpected_keys"] = list(self.unexpected_keys)
        return out


def bc_action() -> str:
    return cfg_bc_action()


def select_bc_init_plan(datas_dir: Path, stage_id: int, action: str) -> InitPlan:
    if action == "resume":
        latest = _read_index_item(datas_dir, stage_id, "bc", "checkpoints", "latest")
        if not _usable_item(latest):
            raise ValueError("resume 失败：未找到 BC latest checkpoint")
        return _plan_from_item("resume_weights_only", stage_id, "bc", latest)
    if stage_id in (20, 30):
        prev_id = _prev_stage_id(datas_dir, stage_id)
        if prev_id is None:
            raise ValueError("inherit 失败：未找到前置 Stage")
        best = _read_index_item(datas_dir, prev_id, "ppo", "checkpoints", "best")
        if not _usable_item(best):
            raise ValueError("inherit 失败：前置 Stage 缺少 PPO best .pt checkpoint")
        return _plan_from_item("inherit_prev_ppo_best", prev_id, "ppo", best)
    return InitPlan(mode="fresh_init")


def select_ppo_init_plan(datas_dir: Path, stage_id: int, action: str) -> InitPlan:
    action = str(action or "start")
    if action == "resume":
        latest = _read_index_item(datas_dir, stage_id, "ppo", "checkpoints", "latest")
        if not _usable_item(latest):
            raise ValueError("resume 失败：未找到 PPO latest checkpoint")
        return _plan_from_item("resume_weights_only", stage_id, "ppo", latest)
    best = _read_index_item(datas_dir, stage_id, "bc", "checkpoints", "best")
    if not _usable_item(best):
        raise ValueError("start 失败：未找到 BC best checkpoint（PPO 必须继承 BC best）")
    return _plan_from_item("inherit_bc_best", stage_id, "bc", best)


def load_weights(
    model: torch.nn.Module, datas_dir: Path, plan: InitPlan, device: torch.device
) -> tuple[InitPlan, int | None]:
    if not plan.ckpt_path or plan.ckpt_stage_id is None or not plan.ckpt_phase:
        return plan, None
    ckpt_file = datas_dir / "stages" / str(plan.ckpt_stage_id) / plan.ckpt_phase / plan.ckpt_path
    obj = torch.load(ckpt_file, map_location=device)
    if not isinstance(obj, dict) or "model_state" not in obj:
        raise ValueError("checkpoint 文件格式非法：缺少 model_state")
    model_state = obj["model_state"]
    strict = True
    try:
        info = model.load_state_dict(model_state, strict=strict)
    except RuntimeError:
        info = model.load_state_dict(model_state, strict=False)
        strict = False
    episode = int(obj.get("episode") or 0) if isinstance(obj.get("episode"), (int, float)) else None
    out = _plan_with_keys(plan, info)
    if not strict:
        out = InitPlan(**{**out.__dict__, "mode": f"{out.mode}_non_strict"})
    return out, episode


def pick_device() -> torch.device:
    forced = str(os.environ.get("CHICHI_TORCH_DEVICE") or "").strip().lower()
    if forced in ("cpu", "cuda", "mps"):
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _plan_from_item(mode: str, ckpt_stage_id: int, ckpt_phase: str, item: dict[str, Any]) -> InitPlan:
    ckpt_id = str(item.get("id") or "") or None
    path = str(item.get("path") or "") or None
    return InitPlan(mode=mode, ckpt_stage_id=ckpt_stage_id, ckpt_phase=ckpt_phase, ckpt_id=ckpt_id, ckpt_path=path)


def _plan_with_keys(plan: InitPlan, info: Any) -> InitPlan:
    missing = list(getattr(info, "missing_keys", []) or [])
    unexpected = list(getattr(info, "unexpected_keys", []) or [])
    return InitPlan(**{**plan.__dict__, "missing_keys": missing or None, "unexpected_keys": unexpected or None})


def _read_index_item(datas_dir: Path, stage_id: int, phase: str, kind: str, bucket: str) -> dict[str, Any] | None:
    idx = read_json(datas_dir / "stages" / str(stage_id) / phase / kind / "index.json", default={"best": [], "latest": []})
    if not isinstance(idx, dict):
        return None
    items = idx.get(bucket) or []
    return items[0] if isinstance(items, list) and items and isinstance(items[0], dict) else None


def _usable_item(item: dict[str, Any] | None) -> bool:
    if not isinstance(item, dict):
        return False
    path = str(item.get("path") or "")
    return bool(path) and path.endswith(".pt")


def _prev_stage_id(datas_dir: Path, stage_id: int) -> int | None:
    ids = discover_stage_ids(datas_dir)
    ids.sort()
    if stage_id not in ids:
        return None
    i = ids.index(stage_id)
    return ids[i - 1] if i > 0 else None
