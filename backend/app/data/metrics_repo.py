from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_metrics_tail(path: Path, keep: int) -> list[dict[str, Any]]:
    raw = _tail_lines(path, int(keep))
    out: list[dict[str, Any]] = []
    for line in raw:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _tail_lines(path: Path, keep: int) -> list[str]:
    if keep <= 0 or not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(0, 2)
        pos = f.tell()
        buf = b""
        while pos > 0 and buf.count(b"\n") <= keep:
            step = min(4096, pos)
            pos -= step
            f.seek(pos)
            buf = f.read(step) + buf
    return [b.decode("utf-8") for b in buf.splitlines()[-keep:]]

