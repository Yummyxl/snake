from __future__ import annotations

import time
from typing import Any

from app.config import api_base as cfg_api_base, datas_dir as cfg_datas_dir
from app.data.metrics_repo import read_metrics_tail
from app.data.stages_repo import (
    discover_stage_ids,
    read_stage_meta,
    read_stage_phase_index,
    read_stage_state,
    write_stage_state,
)
from app.data.runtime_repo import clear_stage_runtime
from app.data.stage_reset_repo import reset_stage
from app.external.training_process import probe_phase, start_bc, start_ppo, stop_phase

_DATAS_DIR = cfg_datas_dir()


def list_stages() -> list[dict[str, Any]]:
    ids = discover_stage_ids(_DATAS_DIR)
    items = [_get_stage_summary(stage_id) for stage_id in ids]
    items.sort(key=lambda x: int(x["stage_id"]))
    return items


def get_stage_detail(stage_id: int) -> dict[str, Any]:
    detail = dict(_load_stage_detail(stage_id))
    prev_id = _prev_stage_id(stage_id)
    detail["prev_stage"] = _prev_stage_payload(prev_id)
    detail["probe"] = _probe_training(detail)
    return detail


def reset_stage_data(stage_id: int) -> dict[str, Any]:
    state = read_stage_state(_DATAS_DIR, stage_id)
    bc_status = str(state.get("bc_status") or "not_started")
    ppo_status = str(state.get("ppo_status") or "not_started")
    _assert_can_reset(bc_status, ppo_status)
    phase = _pick_reset_phase(bc_status, ppo_status)
    if phase is None:
        reset_stage(_DATAS_DIR, stage_id)
    else:
        from app.data.stage_reset_repo import reset_stage_phase

        reset_stage_phase(_DATAS_DIR, stage_id, phase)
        _patch_state_after_phase_reset(stage_id, phase)
    clear_stage_runtime(_DATAS_DIR, stage_id)
    return {"ok": True, "stage_id": stage_id, "phase": phase, "reset_at_ms": int(time.time() * 1000)}


def _pick_reset_phase(bc_status: str, ppo_status: str) -> str | None:
    if str(ppo_status) != "not_started":
        return "ppo"
    if str(bc_status) != "not_started":
        return "bc"
    return None


def _patch_state_after_phase_reset(stage_id: int, phase: str) -> None:
    now_ms = int(time.time() * 1000)
    state = read_stage_state(_DATAS_DIR, stage_id)
    nxt = dict(state)
    if phase == "bc":
        nxt["current_phase"] = None
    else:
        bc_status = str(nxt.get("bc_status") or "not_started")
        nxt["current_phase"] = "bc" if bc_status != "not_started" else None
    nxt[f"{phase}_status"] = "not_started"
    nxt[f"{phase}_episode"] = 0
    last = state.get("last_eval")
    if isinstance(last, dict) and str(last.get("phase") or "") == phase:
        nxt["last_eval"] = None
        nxt["last_eval_coverage"] = None
    nxt["updated_at_ms"] = now_ms
    nxt["last_status_change_at_ms"] = now_ms
    write_stage_state(_DATAS_DIR, stage_id, nxt)


def start_stage_bc(stage_id: int) -> dict[str, Any]:
    return _start_or_resume_bc(stage_id, action="start")


def resume_stage_bc(stage_id: int) -> dict[str, Any]:
    return _start_or_resume_bc(stage_id, action="resume")


def _start_or_resume_bc(stage_id: int, *, action: str) -> dict[str, Any]:
    state = read_stage_state(_DATAS_DIR, stage_id)
    if action == "start":
        _ensure_stage_initialized(stage_id)
        state = read_stage_state(_DATAS_DIR, stage_id)
    _assert_can_start_or_resume_bc(stage_id, state, action=action)
    if probe_phase(_DATAS_DIR, "bc", stage_id).get("alive") or probe_phase(_DATAS_DIR, "ppo", stage_id).get("alive"):
        raise ValueError("检测到训练进程仍在运行，请先停止后再开始")
    clear_stage_runtime(_DATAS_DIR, stage_id)
    now_ms = int(time.time() * 1000)
    nxt = _patch_state_start_bc(state, now_ms)
    write_stage_state(_DATAS_DIR, stage_id, nxt)
    try:
        pid = start_bc(_DATAS_DIR, stage_id, cfg_api_base(), action=action)
    except Exception:
        write_stage_state(_DATAS_DIR, stage_id, state)
        raise
    return {"ok": True, "stage_id": stage_id, "action": action, "started_at_ms": now_ms, "pid": pid}


def _ensure_stage_initialized(stage_id: int) -> None:
    state_path = _DATAS_DIR / "stages" / str(stage_id) / "state.json"
    if state_path.exists():
        return
    reset_stage(_DATAS_DIR, stage_id)


def start_stage_ppo(stage_id: int) -> dict[str, Any]:
    return _start_or_resume_ppo(stage_id, action="start")


def resume_stage_ppo(stage_id: int) -> dict[str, Any]:
    return _start_or_resume_ppo(stage_id, action="resume")


def _start_or_resume_ppo(stage_id: int, *, action: str) -> dict[str, Any]:
    state = read_stage_state(_DATAS_DIR, stage_id)
    _assert_can_start_or_resume_ppo(stage_id, state, action=action)
    if probe_phase(_DATAS_DIR, "bc", stage_id).get("alive") or probe_phase(_DATAS_DIR, "ppo", stage_id).get("alive"):
        raise ValueError("检测到训练进程仍在运行，请先停止后再开始")
    clear_stage_runtime(_DATAS_DIR, stage_id)
    now_ms = int(time.time() * 1000)
    nxt = _patch_state_start_ppo(state, now_ms)
    write_stage_state(_DATAS_DIR, stage_id, nxt)
    try:
        pid = start_ppo(_DATAS_DIR, stage_id, cfg_api_base(), action=action)
    except Exception:
        write_stage_state(_DATAS_DIR, stage_id, state)
        raise
    return {"ok": True, "stage_id": stage_id, "action": action, "started_at_ms": now_ms, "pid": pid}


def stop_stage_bc(stage_id: int) -> dict[str, Any]:
    return stop_stage_phase(stage_id, phase="bc")


def stop_stage_ppo(stage_id: int) -> dict[str, Any]:
    return stop_stage_phase(stage_id, phase="ppo")


def stop_stage_phase(stage_id: int, phase: str | None) -> dict[str, Any]:
    state = read_stage_state(_DATAS_DIR, stage_id)
    target = phase or _pick_running_phase(state)
    if target is None:
        return {"ok": True, "stage_id": stage_id, "phase": None, "stopped": False, "reason": "not_running"}
    if target not in ("bc", "ppo"):
        raise ValueError("phase 仅支持 bc/ppo")
    if phase is not None and str(state.get(f"{target}_status") or "not_started") != "running":
        probe = probe_phase(_DATAS_DIR, target, stage_id)
        if not probe.get("alive") and not probe.get("has_pidfile"):
            return {"ok": True, "stage_id": stage_id, "phase": target, "stopped": False, "reason": "not_running"}
    info = stop_phase(_DATAS_DIR, target, stage_id)
    now_ms = int(time.time() * 1000)
    if info.get("signal_sent"):
        nxt = dict(state)
        nxt["current_phase"] = target
        nxt["updated_at_ms"] = now_ms
        write_stage_state(_DATAS_DIR, stage_id, nxt)
    else:
        nxt = dict(state)
        nxt["current_phase"] = target
        nxt[f"{target}_status"] = "paused"
        nxt["updated_at_ms"] = now_ms
        nxt["last_status_change_at_ms"] = now_ms
        write_stage_state(_DATAS_DIR, stage_id, nxt)
    return {
        "ok": True,
        "stage_id": stage_id,
        "phase": target,
        "stopped": bool(info.get("signal_sent")),
        "reason": "signal_sent" if info.get("signal_sent") else "no_process_fixed_state",
    }


def complete_stage_bc(stage_id: int) -> dict[str, Any]:
    return _complete_phase(stage_id, "bc")


def complete_stage_ppo(stage_id: int) -> dict[str, Any]:
    return _complete_phase(stage_id, "ppo")


def _complete_phase(stage_id: int, phase: str) -> dict[str, Any]:
    if phase not in ("bc", "ppo"):
        raise ValueError("phase 仅支持 bc/ppo")
    state = read_stage_state(_DATAS_DIR, stage_id)
    status = str(state.get(f"{phase}_status") or "not_started")
    if status != "paused":
        raise ValueError("仅允许 paused → completed（请先停止）")
    if probe_phase(_DATAS_DIR, phase, stage_id).get("alive"):
        raise ValueError("训练进程仍在停止中，请稍后再试")
    if phase == "bc":
        _assert_bc_best_checkpoint_ready(stage_id)
    if phase == "ppo":
        _assert_ppo_best_checkpoint_ready(stage_id)
    now_ms = int(time.time() * 1000)
    nxt = dict(state)
    nxt["current_phase"] = phase
    nxt[f"{phase}_status"] = "completed"
    nxt["updated_at_ms"] = now_ms
    nxt["last_status_change_at_ms"] = now_ms
    write_stage_state(_DATAS_DIR, stage_id, nxt)
    return {"ok": True, "stage_id": stage_id, "phase": phase, "completed_at_ms": now_ms}


def _assert_bc_best_checkpoint_ready(stage_id: int) -> None:
    idx = read_stage_phase_index(_DATAS_DIR, stage_id, "bc", "checkpoints")
    best = idx.get("best")
    if not isinstance(best, list) or not best or not isinstance(best[0], dict):
        raise ValueError("完成 BC 失败：缺少 BC best checkpoint（请先停止训练生成 checkpoint）")
    rel = str(best[0].get("path") or "")
    if not rel.endswith(".pt"):
        raise ValueError("完成 BC 失败：BC best checkpoint 不是 .pt 文件")
    if not (_DATAS_DIR / "stages" / str(stage_id) / "bc" / rel).exists():
        raise ValueError("完成 BC 失败：BC best checkpoint 文件不存在")


def _assert_ppo_best_checkpoint_ready(stage_id: int) -> None:
    idx = read_stage_phase_index(_DATAS_DIR, stage_id, "ppo", "checkpoints")
    best = idx.get("best")
    if not isinstance(best, list) or not best or not isinstance(best[0], dict):
        raise ValueError("完成 PPO 失败：缺少 PPO best checkpoint（请先停止训练生成 checkpoint）")
    rel = str(best[0].get("path") or "")
    if not rel.endswith(".pt"):
        raise ValueError("完成 PPO 失败：PPO best checkpoint 不是 .pt 文件")
    if not (_DATAS_DIR / "stages" / str(stage_id) / "ppo" / rel).exists():
        raise ValueError("完成 PPO 失败：PPO best checkpoint 文件不存在")


def _get_stage_summary(stage_id: int) -> dict[str, Any]:
    meta = read_stage_meta(_DATAS_DIR, stage_id)
    state = read_stage_state(_DATAS_DIR, stage_id)
    bc_episode = int(state.get("bc_episode") or 0)
    ppo_episode = int(state.get("ppo_episode") or 0)
    return {
        "stage_id": int(meta.get("stage_id") or stage_id),
        "size": int(meta.get("size") or stage_id),
        "current_phase": state.get("current_phase"),
        "bc_status": str(state.get("bc_status") or "not_started"),
        "ppo_status": str(state.get("ppo_status") or "not_started"),
        "bc_episode": bc_episode,
        "ppo_episode": ppo_episode,
        "total_episode": bc_episode + ppo_episode,
        "last_eval_coverage": state.get("last_eval_coverage"),
        "updated_at_ms": _pick_updated_at_ms(state),
    }


def _load_stage_detail(stage_id: int) -> dict[str, Any]:
    meta = read_stage_meta(_DATAS_DIR, stage_id)
    state = read_stage_state(_DATAS_DIR, stage_id)
    bc_status = str(state.get("bc_status") or "not_started")
    ppo_status = str(state.get("ppo_status") or "not_started")
    bc_episode = int(state.get("bc_episode") or 0)
    ppo_episode = int(state.get("ppo_episode") or 0)
    return {
        "stage_id": int(meta.get("stage_id") or stage_id),
        "size": int(meta.get("size") or stage_id),
        "current_phase": state.get("current_phase"),
        "bc_status": bc_status,
        "ppo_status": ppo_status,
        "bc_episode": bc_episode,
        "ppo_episode": ppo_episode,
        "total_episode": bc_episode + ppo_episode,
        "last_eval": state.get("last_eval"),
        "last_eval_coverage": state.get("last_eval_coverage"),
        "updated_at_ms": _pick_updated_at_ms(state),
        "has_bc_best_checkpoint": _has_bc_best_checkpoint(stage_id),
        "has_ppo_best_checkpoint": _has_ppo_best_checkpoint(stage_id),
        "eval_rollouts": _merge_eval_rollouts(stage_id),
        "metrics": _load_stage_metrics(stage_id),
    }


def _load_stage_metrics(stage_id: int) -> dict[str, Any]:
    return {
        "bc": _load_phase_metrics(stage_id, "bc"),
        "ppo": _load_phase_metrics(stage_id, "ppo"),
    }


def _load_phase_metrics(stage_id: int, phase: str) -> list[dict[str, Any]]:
    path = _DATAS_DIR / "stages" / str(stage_id) / phase / "metrics" / "episodes.jsonl"
    out: list[dict[str, Any]] = []
    for obj in read_metrics_tail(path, 50):
        item = _metrics_item(obj, phase)
        if item:
            out.append(item)
    return out


def _metrics_item(obj: Any, phase: str) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    base = {k: obj.get(k) for k in ("stage_id", "phase", "episode", "timestamp_ms")}
    if phase == "bc":
        base["bc_loss"] = obj.get("bc_loss")
        return base
    base["ppo_loss"] = obj.get("ppo_loss")
    base["reward_mean"] = obj.get("reward_mean")
    return base


def _has_bc_best_checkpoint(stage_id: int) -> bool:
    idx = read_stage_phase_index(_DATAS_DIR, stage_id, "bc", "checkpoints")
    best = idx.get("best")
    return isinstance(best, list) and len(best) > 0


def _has_ppo_best_checkpoint(stage_id: int) -> bool:
    idx = read_stage_phase_index(_DATAS_DIR, stage_id, "ppo", "checkpoints")
    best = idx.get("best")
    return isinstance(best, list) and len(best) > 0


def _merge_eval_rollouts(stage_id: int) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for phase in ("bc", "ppo"):
        idx = read_stage_phase_index(_DATAS_DIR, stage_id, phase, "evals")
        for kind, is_best in (("best", True), ("latest", False)):
            for raw in (idx.get(kind) or []):
                item = _eval_from_index(stage_id, phase, raw, is_best)
                if item is None:
                    continue
                key = (item["phase"], item["rollout_id"])
                merged[key] = _merge_eval_item(merged.get(key), item)
    return list(merged.values())


def _eval_from_index(
    stage_id: int, phase: str, raw: Any, is_best: bool
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    rollout_id = str(raw.get("id") or "").strip()
    if not rollout_id:
        return None
    rel = str(raw.get("path") or "")
    if not rel:
        return None
    if not (_DATAS_DIR / "stages" / str(stage_id) / phase / rel).exists():
        return None
    steps = raw.get("step_count")
    length_max = raw.get("length_max")
    reward_total = raw.get("reward_total")
    return {
        "rollout_id": rollout_id,
        "phase": phase,
        "source": "eval",
        "episode": int(raw.get("episode") or 0),
        "coverage": raw.get("coverage"),
        "steps": int(steps) if isinstance(steps, (int, float)) else None,
        "length_max": int(length_max) if isinstance(length_max, (int, float)) else None,
        "reward_total": float(reward_total) if isinstance(reward_total, (int, float)) else None,
        "created_at_ms": int(raw.get("created_at_ms") or 0),
        "is_best": bool(is_best or raw.get("is_best")),
    }


def _merge_eval_item(prev: dict[str, Any] | None, nxt: dict[str, Any]) -> dict[str, Any]:
    if prev is None:
        return nxt
    out = dict(prev)
    for k, v in nxt.items():
        if k == "is_best":
            out[k] = bool(out.get(k) or v)
        elif out.get(k) in (None, 0, "") and v not in (None, 0, ""):
            out[k] = v
    return out


def _prev_stage_id(stage_id: int) -> int | None:
    ids = discover_stage_ids(_DATAS_DIR)
    ids.sort()
    if stage_id not in ids:
        return None
    idx = ids.index(stage_id)
    return ids[idx - 1] if idx > 0 else None


def _prev_stage_payload(prev_id: int | None) -> dict[str, Any] | None:
    if prev_id is None:
        return None
    s = _load_stage_detail(prev_id)
    return {"stage_id": prev_id, "bc_status": s["bc_status"], "ppo_status": s["ppo_status"]}


def _assert_can_reset(bc_status: str, ppo_status: str) -> None:
    if bc_status == "running" or ppo_status == "running":
        raise ValueError("训练中不可初始化（reset）")
    if bc_status == "completed" and ppo_status == "completed":
        raise ValueError("已完成 Stage 不可初始化（reset）")


def _assert_can_start_or_resume_bc(stage_id: int, state: dict[str, Any], *, action: str) -> None:
    action = str(action or "start")
    bc = str(state.get("bc_status") or "not_started")
    ppo = str(state.get("ppo_status") or "not_started")
    if action == "resume":
        if bc != "paused":
            raise ValueError("仅当 BC=paused 时可恢复")
        if ppo == "running":
            raise ValueError("PPO 训练中不可恢复 BC")
        return
    if not (bc == "not_started" and ppo == "not_started"):
        raise ValueError("仅当 BC/PPO 都未开始时可开始 BC")
    _assert_inherit_checkpoint_ready(stage_id)
    prev_id = _prev_stage_id(stage_id)
    if prev_id is None:
        return
    prev = read_stage_state(_DATAS_DIR, prev_id)
    if not (_is_completed(prev.get("bc_status")) and _is_completed(prev.get("ppo_status"))):
        raise ValueError("需要先完成前置 Stage（BC+PPO）")


def _assert_can_start_or_resume_ppo(stage_id: int, state: dict[str, Any], *, action: str) -> None:
    action = str(action or "start")
    bc = str(state.get("bc_status") or "not_started")
    ppo = str(state.get("ppo_status") or "not_started")
    if action == "resume":
        if ppo != "paused":
            raise ValueError("仅当 PPO=paused 时可恢复")
        if bc == "running":
            raise ValueError("BC 训练中不可恢复 PPO")
        if bc != "completed":
            raise ValueError("需要先完成 BC 才可恢复 PPO")
        return
    if bc != "completed":
        raise ValueError("需要先完成 BC 才可开始 PPO")
    if ppo != "not_started":
        raise ValueError("仅当 PPO=not_started 时可开始 PPO")
    if not _has_bc_best_checkpoint(stage_id):
        raise ValueError("缺少 BC best checkpoint，无法开始 PPO")


def _assert_inherit_checkpoint_ready(stage_id: int) -> None:
    if stage_id not in (20, 30):
        return
    prev_id = _prev_stage_id(stage_id)
    if prev_id is None:
        raise ValueError("inherit 失败：未找到前置 Stage")
    idx = read_stage_phase_index(_DATAS_DIR, prev_id, "ppo", "checkpoints")
    best = idx.get("best")
    if not isinstance(best, list) or not best or not isinstance(best[0], dict):
        raise ValueError("inherit 失败：前置 Stage 缺少 PPO best .pt checkpoint")
    rel = str(best[0].get("path") or "")
    if not rel.endswith(".pt"):
        raise ValueError("inherit 失败：前置 Stage PPO best 不是 .pt checkpoint")
    if not (_DATAS_DIR / "stages" / str(prev_id) / "ppo" / rel).exists():
        raise ValueError("inherit 失败：前置 Stage PPO best checkpoint 文件不存在")


def _is_completed(status: Any) -> bool:
    return str(status or "not_started") == "completed"


def _patch_state_start_bc(state: dict[str, Any], now_ms: int) -> dict[str, Any]:
    out = dict(state)
    out["current_phase"] = "bc"
    out["bc_status"] = "running"
    out["updated_at_ms"] = now_ms
    out["last_status_change_at_ms"] = now_ms
    return out


def _patch_state_start_ppo(state: dict[str, Any], now_ms: int) -> dict[str, Any]:
    out = dict(state)
    out["current_phase"] = "ppo"
    out["ppo_status"] = "running"
    out["updated_at_ms"] = now_ms
    out["last_status_change_at_ms"] = now_ms
    return out


def _pick_running_phase(state: dict[str, Any]) -> str | None:
    bc = str(state.get("bc_status") or "not_started")
    ppo = str(state.get("ppo_status") or "not_started")
    if bc == "running":
        return "bc"
    if ppo == "running":
        return "ppo"
    return None


def _probe_training(detail: dict[str, Any]) -> dict[str, Any]:
    stage_id = int(detail.get("stage_id") or 0)
    bc_status = str(detail.get("bc_status") or "not_started")
    ppo_status = str(detail.get("ppo_status") or "not_started")
    bc = probe_phase(_DATAS_DIR, "bc", stage_id)
    ppo = probe_phase(_DATAS_DIR, "ppo", stage_id)
    effective = None
    if bc_status == "running" and bc.get("alive"):
        effective = "bc"
    elif ppo_status == "running" and ppo.get("alive"):
        effective = "ppo"
    stale = (bc_status == "running" and not bc.get("alive")) or (ppo_status == "running" and not ppo.get("alive"))
    return {
        "bc": {"pid": bc.get("pid"), "alive": bool(bc.get("alive")), "runtime": _runtime_public(bc.get("runtime"))},
        "ppo": {"pid": ppo.get("pid"), "alive": bool(ppo.get("alive")), "runtime": _runtime_public(ppo.get("runtime"))},
        "effective_training": effective,
        "stale_running": stale,
    }


def _runtime_public(runtime: Any) -> dict[str, Any] | None:
    if not isinstance(runtime, dict):
        return None
    out: dict[str, Any] = {}
    for k in ("status", "run_id", "last_error", "exit_code", "heartbeat_at_ms", "stop_requested_at_ms"):
        if k in runtime:
            out[k] = runtime.get(k)
    return out or None


def _pick_updated_at_ms(state: dict[str, Any]) -> int:
    last_eval = state.get("last_eval")
    if isinstance(last_eval, dict):
        val = last_eval.get("created_at_ms")
        if isinstance(val, (int, float)):
            return int(val)
    for key in ("last_status_change_at_ms", "updated_at_ms"):
        val = state.get(key)
        if isinstance(val, (int, float)):
            return int(val)
    return 0
