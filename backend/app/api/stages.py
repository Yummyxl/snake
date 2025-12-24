from __future__ import annotations

from fastapi import APIRouter

from app.services.stages_service import (
    complete_stage_bc,
    complete_stage_ppo,
    get_stage_detail,
    list_stages,
    reset_stage_data,
    resume_stage_ppo,
    resume_stage_bc,
    start_stage_ppo,
    start_stage_bc,
    stop_stage_bc,
    stop_stage_ppo,
)

router = APIRouter()


@router.get("/api/stages")
def get_stages() -> list[dict]:
    return list_stages()


@router.get("/api/cache/stages/reset")
def reset_stage_cache() -> dict:
    return {"ok": True}


@router.get("/api/stages/{stage_id}")
def get_stage(stage_id: int) -> dict:
    return get_stage_detail(stage_id)


@router.get("/api/stages/{stage_id}/reset")
def reset_stage(stage_id: int) -> dict:
    try:
        return reset_stage_data(stage_id)
    except ValueError as e:
        return {"ok": False, "error": str(e), "stage_id": stage_id}


@router.get("/api/stages/{stage_id}/bc/start")
def start_stage_bc_api(stage_id: int) -> dict:
    try:
        return start_stage_bc(stage_id)
    except ValueError as e:
        return {"ok": False, "error": str(e), "stage_id": stage_id}
    except Exception as e:
        return {"ok": False, "error": f"start bc failed: {e}", "stage_id": stage_id}


@router.get("/api/stages/{stage_id}/bc/resume")
def resume_stage_bc_api(stage_id: int) -> dict:
    try:
        return resume_stage_bc(stage_id)
    except ValueError as e:
        return {"ok": False, "error": str(e), "stage_id": stage_id}
    except Exception as e:
        return {"ok": False, "error": f"resume bc failed: {e}", "stage_id": stage_id}


@router.get("/api/stages/{stage_id}/ppo/start")
def start_stage_ppo_api(stage_id: int) -> dict:
    try:
        return start_stage_ppo(stage_id)
    except ValueError as e:
        return {"ok": False, "error": str(e), "stage_id": stage_id}
    except Exception as e:
        return {"ok": False, "error": f"start ppo failed: {e}", "stage_id": stage_id}


@router.get("/api/stages/{stage_id}/ppo/resume")
def resume_stage_ppo_api(stage_id: int) -> dict:
    try:
        return resume_stage_ppo(stage_id)
    except ValueError as e:
        return {"ok": False, "error": str(e), "stage_id": stage_id}
    except Exception as e:
        return {"ok": False, "error": f"resume ppo failed: {e}", "stage_id": stage_id}


@router.get("/api/stages/{stage_id}/bc/stop")
def stop_stage_bc_api(stage_id: int) -> dict:
    try:
        return stop_stage_bc(stage_id)
    except Exception as e:
        return {"ok": False, "error": f"stop bc failed: {e}", "stage_id": stage_id}


@router.get("/api/stages/{stage_id}/ppo/stop")
def stop_stage_ppo_api(stage_id: int) -> dict:
    try:
        return stop_stage_ppo(stage_id)
    except Exception as e:
        return {"ok": False, "error": f"stop ppo failed: {e}", "stage_id": stage_id}


@router.get("/api/stages/{stage_id}/bc/complete")
def complete_stage_bc_api(stage_id: int) -> dict:
    try:
        return complete_stage_bc(stage_id)
    except ValueError as e:
        return {"ok": False, "error": str(e), "stage_id": stage_id}
    except Exception as e:
        return {"ok": False, "error": f"complete bc failed: {e}", "stage_id": stage_id}


@router.get("/api/stages/{stage_id}/ppo/complete")
def complete_stage_ppo_api(stage_id: int) -> dict:
    try:
        return complete_stage_ppo(stage_id)
    except ValueError as e:
        return {"ok": False, "error": str(e), "stage_id": stage_id}
    except Exception as e:
        return {"ok": False, "error": f"complete ppo failed: {e}", "stage_id": stage_id}
