from __future__ import annotations

import urllib.request


def is_backend_healthy(health_url: str, timeout_s: float = 0.5) -> bool:
    try:
        with urllib.request.urlopen(health_url, timeout=timeout_s) as resp:
            return int(getattr(resp, "status", 0)) == 200
    except Exception:
        return False

