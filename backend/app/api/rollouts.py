from __future__ import annotations

from fastapi import APIRouter

from app.services.rollouts_service import get_rollout_detail

router = APIRouter()


@router.get("/api/rollouts/{rollout_id}")
def get_rollout_api(rollout_id: str, stage_id: int, phase: str, source: str) -> dict:
    try:
        return get_rollout_detail(stage_id=stage_id, phase=phase, source=source, rollout_id=rollout_id)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"read rollout failed: {e}"}

