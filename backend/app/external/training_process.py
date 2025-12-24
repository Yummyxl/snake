from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from app.config import worker_env
from app.data.runtime_repo import pid_path, read_runtime, runtime_path, write_runtime


def start_bc(datas_dir: Path, stage_id: int, api_base: str, *, action: str = "start") -> int:
    _ensure_dirs(datas_dir)
    pidfile = pid_path(datas_dir, "bc", stage_id)
    alive_pid = _read_pid(pidfile)
    if alive_pid and _is_pid_alive(alive_pid) and _matches_phase(alive_pid, "bc"):
        return alive_pid
    _cleanup_stale_runfiles(datas_dir, "bc", stage_id)
    log = _open_log(datas_dir, f"bc_{stage_id}.log")
    env = _worker_env(datas_dir, api_base, action=action)
    cmd = ["uv", "run", "python", "-m", "app.workers.bc_worker", "--stage-id", str(stage_id)]
    write_runtime(datas_dir, "bc", stage_id, {"status": "starting"})
    try:
        p = subprocess.Popen(cmd, cwd=_backend_dir(), env=env, stdout=log, stderr=log, start_new_session=True)
    except Exception:
        log.close()
        _cleanup_stale_runfiles(datas_dir, "bc", stage_id)
        raise
    log.close()
    worker_pid = _pick_worker_pid(p.pid, phase="bc") or p.pid
    pidfile.write_text(str(worker_pid), encoding="utf-8")
    write_runtime(datas_dir, "bc", stage_id, {"status": "running", "pid": worker_pid, "wrapper_pid": p.pid})
    return worker_pid


def start_ppo(datas_dir: Path, stage_id: int, api_base: str, *, action: str = "start") -> int:
    _ensure_dirs(datas_dir)
    pidfile = pid_path(datas_dir, "ppo", stage_id)
    alive_pid = _read_pid(pidfile)
    if alive_pid and _is_pid_alive(alive_pid) and _matches_phase(alive_pid, "ppo"):
        return alive_pid
    _cleanup_stale_runfiles(datas_dir, "ppo", stage_id)
    log = _open_log(datas_dir, f"ppo_{stage_id}.log")
    env = _worker_env(datas_dir, api_base, action=action)
    cmd = ["uv", "run", "python", "-m", "app.workers.ppo_worker", "--stage-id", str(stage_id)]
    write_runtime(datas_dir, "ppo", stage_id, {"status": "starting"})
    try:
        p = subprocess.Popen(cmd, cwd=_backend_dir(), env=env, stdout=log, stderr=log, start_new_session=True)
    except Exception:
        log.close()
        _cleanup_stale_runfiles(datas_dir, "ppo", stage_id)
        raise
    log.close()
    worker_pid = _pick_worker_pid(p.pid, phase="ppo") or p.pid
    pidfile.write_text(str(worker_pid), encoding="utf-8")
    write_runtime(datas_dir, "ppo", stage_id, {"status": "running", "pid": worker_pid, "wrapper_pid": p.pid})
    return worker_pid


def probe_phase(datas_dir: Path, phase: str, stage_id: int) -> dict:
    pidfile = pid_path(datas_dir, phase, stage_id)
    pid = _read_pid(pidfile)
    alive = bool(pid and _is_pid_alive(pid) and _matches_phase(pid, phase))
    if pid and not alive and pidfile.exists():
        _cleanup_stale_runfiles(datas_dir, phase, stage_id)
    runtime = read_runtime(datas_dir, phase, stage_id)
    return {"pid": pid, "alive": alive, "has_pidfile": pidfile.exists(), "runtime": runtime or None}


def stop_phase(datas_dir: Path, phase: str, stage_id: int) -> dict:
    pidfile = pid_path(datas_dir, phase, stage_id)
    pid = _read_pid(pidfile)
    if not pid or not _is_pid_alive(pid) or not _matches_phase(pid, phase):
        _cleanup_stale_runfiles(datas_dir, phase, stage_id)
        return {"pid": pid, "alive": False, "signal_sent": False}
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _cleanup_stale_runfiles(datas_dir, phase, stage_id)
        return {"pid": pid, "alive": False, "signal_sent": False}
    except PermissionError:
        os.kill(pid, signal.SIGTERM)
    write_runtime(
        datas_dir,
        phase,
        stage_id,
        {"status": "stopping", "stop_requested_at_ms": int(time.time() * 1000), "pid": pid},
    )
    return {"pid": pid, "alive": True, "signal_sent": True}


def _backend_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_dirs(datas_dir: Path) -> None:
    (datas_dir / ".run").mkdir(parents=True, exist_ok=True)
    (datas_dir / "logs").mkdir(parents=True, exist_ok=True)


def _open_log(datas_dir: Path, name: str):
    return open(datas_dir / "logs" / name, "a", encoding="utf-8")


def _worker_env(datas_dir: Path, api_base: str, *, action: str) -> dict[str, str]:
    return worker_env(datas_dir, api_base, backend_pid=os.getpid(), action=action)


def _pidfile_alive(pidfile: Path) -> bool:
    if not pidfile.exists():
        return False
    try:
        pid = int(pidfile.read_text(encoding="utf-8").strip())
    except ValueError:
        return False
    return _is_pid_alive(pid)


def _read_pid(pidfile: Path) -> int | None:
    if not pidfile.exists():
        return None
    try:
        pid = int(pidfile.read_text(encoding="utf-8").strip())
    except ValueError:
        return None
    return pid if pid > 0 else None


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _matches_phase(pid: int, phase: str) -> bool:
    cmd = _proc_cmd(pid)
    if not cmd:
        return False
    phase = str(phase or "").strip().lower()
    if phase == "bc":
        return "app.workers.bc_worker" in cmd
    if phase == "ppo":
        return "app.workers.ppo_worker" in cmd
    return False


def _proc_cmd(pid: int) -> str | None:
    try:
        out = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    s = (out.stdout or "").strip()
    return s or None


def _pick_worker_pid(wrapper_pid: int, *, phase: str) -> int | None:
    for _ in range(15):
        pid = _first_child_pid(wrapper_pid)
        if pid and _matches_phase(pid, phase):
            return pid
        time.sleep(0.05)
    return None


def _first_child_pid(ppid: int) -> int | None:
    try:
        out = subprocess.run(
            ["ps", "-o", "pid=", "-P", str(ppid)],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    for tok in (out.stdout or "").strip().split():
        try:
            pid = int(tok)
        except ValueError:
            continue
        return pid if pid > 0 else None
    return None


def _cleanup_stale_runfiles(datas_dir: Path, phase: str, stage_id: int) -> None:
    try:
        pid_path(datas_dir, phase, stage_id).unlink()
    except FileNotFoundError:
        pass
