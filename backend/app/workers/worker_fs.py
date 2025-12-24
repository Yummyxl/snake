from __future__ import annotations

import shutil
from pathlib import Path


def clear_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def copy_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)


def remove_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
