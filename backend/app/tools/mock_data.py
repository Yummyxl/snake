from __future__ import annotations

import shutil
from pathlib import Path

from app.data.stage_reset_repo import reset_stage


def main() -> None:
    datas_dir = _repo_root() / "datas"
    _clear_stage_dirs(datas_dir / "stages")
    for stage_id in (10, 20, 30):
        reset_stage(datas_dir, stage_id)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _clear_stage_dirs(stages_dir: Path) -> None:
    stages_dir.mkdir(parents=True, exist_ok=True)
    for entry in stages_dir.iterdir():
        if entry.is_dir() and entry.name.isdigit():
            shutil.rmtree(entry, ignore_errors=True)


if __name__ == "__main__":
    main()

