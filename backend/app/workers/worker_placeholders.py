from __future__ import annotations

from typing import Any


def train_one_episode(stage_id: int, size: int, cfg: dict[str, Any], init_ckpt: dict[str, Any] | None) -> None:
    _ = (stage_id, size, cfg, init_ckpt)


def eval_rollouts(stage_id: int, size: int, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    _ = (stage_id, size, cfg)
    return []
